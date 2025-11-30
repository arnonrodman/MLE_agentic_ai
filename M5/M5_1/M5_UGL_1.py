#!/usr/bin/env python
# coding: utf-8

# # M5 Agentic AI - Customer Service Pipeline
# 
# ## 1. Introduction
# 
# ### 1.1 Lab overview
# 
# In this lab, you will experience how **agentic workflows** coordinate multiple specialized roles to handle customer requests in a retail scenario.  
# 
# For example, imagine a walk-in customer who wants to return **two Aviator sunglasses**: you’ll look up the item, compute the refund, update stock, and record the transaction.  
# But if another customer tries to buy **five Mystique sunglasses**, the system must check inventory. If stock would go negative, it should **flag the issue and stop**.  
# 
# This sets the stage for the lab, where you’ll see how a **Planner**, a **Reflection Agent**, and an **Executor** work together with validations to keep transactions safe.
# 
# ### 🎯 1.2 Learning outcome
# 
# By the end of this lab, you will see how multiple agents coordinate roles, share context, and adapt their behavior to complete complex customer service tasks.
# 
# 
# 

# ## 2. Setup: Import libraries and load environment
# 
# As in previous labs, you now import the required libraries, load environment variables, and set up helper utilities.
# 

# In[2]:


# =========================
# Imports & utilities
# =========================

# --- Standard library ---
from __future__ import annotations
from typing import Any, Callable, Optional
import json
import re

# --- Third-party ---
import pandas as pd
import duckdb
from openai import OpenAI
from dotenv import load_dotenv

# --- Local ---
import inventory_utils
import utils
import tools

import os
# API key will be loaded from .env file via load_dotenv()


_ = load_dotenv()
client = OpenAI()


# ### 2.1 Data setup
# 
# Create the initial **inventory** and **transactions** tables for the sunglasses store. Then, display a preview of both so you can see the starting point before running any workflows.
# 

# In[3]:


inventory_df = inventory_utils.create_inventory_dataframe()
transaction_df = inventory_utils.create_transaction_dataframe()

utils.print_html(inventory_df.head(), title="Inventory DataFrame")
utils.print_html(transaction_df.head(), title="Transaction DataFrame")


# ### 2.2 DuckDB helpers
# 
# To make the data easier to query, you will now create a helper function that sets up a **DuckDB connection** and registers both DataFrames as SQL views:
# 
# - `inventory_df`  
# - `transaction_df`  

# In[ ]:


# =========================
# DuckDB helpers
# =========================
def create_duckdb_with_views(inventory_df: pd.DataFrame, transaction_df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.register("inventory_df", inventory_df)
    con.register("transaction_df", transaction_df)
    return con

con = create_duckdb_with_views(inventory_df, transaction_df)


# Once the views are in place, you can treat the in-memory DataFrames like database tables and run SQL directly against them. For instance:

# In[ ]:


result = con.sql("SELECT * FROM inventory_df")
utils.print_html(result.df().head(), title="Inventory DataFrame via SQL")


# By doing this, you have ensured that the rest of the workflow can reason about inventory and transactions using familiar database operations.

# ## 3. Tool functions
# 
# Since you want to handle inventory and transactions, the first step is to provide the system with tools that make those operations possible. Before you build the agents, you need to give them something concrete to work with. Agents on their own can’t query a database or update records — they rely on tools that bridge the gap between natural instructions and actual operations.
# 
# You’ll now define a set of helper functions bundled in the `tools` module. This toolkit provides:
# 
# * **READ** — fetch data with lightweight DuckDB SQL (e.g., look up inventory or transactions).  
# * **WRITE** — update in-memory DataFrames (adjust inventory, append new transactions).  
# * **Propose-only** — compute outcomes such as totals or balances without mutating state.  
# * **Helpers** — perform quick math (e.g., totals, refunds) and simple assertions.  
# * **Validations** — check conditions like non-null values or non-negative stock.  
# * **Registry** — map string names (including aliases) to callables that the plan or LLM can execute.  
# 
# All tool functions are accessible via the registry. By exposing them this way, you can later call tools directly from the planning and execution workflow.
# 
# ### 3.1 Trying out tools
# 
# Before moving to agents, you should try out a few of these functions manually.   This will give you confidence that the registry is wired correctly.
# 
# For example:

# In[ ]:


# Lookup a product (READ)
prod = tools.TOOL_REGISTRY["get_inventory_data"](con=con, product_name="Aviator")
utils.print_html(prod, title="Product Lookup")

# Compute a purchase total (HELPER)
total = tools.TOOL_REGISTRY["compute_total"](qty=3, price=prod["item"]["price"])
utils.print_html(total, title="Purchase Total")


# ## 4. Plan execution  
# 
# You now define how a **plan of tool calls** can be executed from start to finish. Instead of writing raw SQL, each step is carried out by invoking registered tools and validations. 
# 

# In[ ]:


def execute_plan_tools_only(
    plan: dict[str, Any],
    inventory_df: pd.DataFrame,
    transaction_df: pd.DataFrame,
    return_updated_frames: bool = True,
    stop_on_failed_validation: bool = True  # <-- new flag
) -> dict[str, Any]:
    """
    Executes a plan based ONLY on tools (no SQL visible in the plan).
    - Runs tools step by step and stores results in the context.
    - Performs validations using tools.
    - Write tools return updated DataFrames and the executor applies them automatically.
    - If stop_on_failed_validation=True, execution halts at the first failed validation.
    """
    con = create_duckdb_with_views(inventory_df, transaction_df)
    ctx: dict[str, Any] = {
        "__con__": con,
        "__frames__": {"inventory_df": inventory_df.copy(), "transaction_df": transaction_df.copy()}
    }

    report: dict[str, Any] = {"ok": True, "steps": []}
    try:
        for step in plan.get("steps", []):
            step_number = step.get("step_number")
            description = step.get("description", "")

            # tools
            tool_error = None
            try:
                ran = tools.run_tools_for_step(step, ctx)
            except Exception as e:
                ran = {}
                tool_error = str(e)
                report["ok"] = False

            # validations
            validations = [tools.run_tool_validation(v, ctx) for v in step.get("validations", [])]
            step_ok = (tool_error is None) and all(v.get("ok", False) for v in validations)
            if not step_ok:
                report["ok"] = False

            # record the step
            report["steps"].append({
                "step_number": step_number,
                "description": description,
                "tools_run": list(ran.keys()),
                "tool_error": tool_error,
                "validations": validations,
            })

            # stop execution if a validation failed
            if stop_on_failed_validation and any(not v.get("ok", False) for v in validations):
                report["aborted"] = True
                report["abort_step"] = step_number
                report["abort_reason"] = "validation_failed"
                break

    finally:
        con.close()

    if return_updated_frames:
        report["updated_frames"] = {
            "inventory_df": ctx["__frames__"]["inventory_df"],
            "transaction_df": ctx["__frames__"]["transaction_df"],
        }

    return report


# <div style="background-color:#ffe4e1; padding:12px; border-radius:6px; color:black;">  
# <strong>Note:</strong> The execution process follows these stages:  
# <ol type="a">  
# <li>A DuckDB connection is created and working copies of the DataFrames are stored in a <code>context</code>.</li>  
# <li>Each step in the plan is executed in sequence, with tools run and results captured in the report.</li>  
# <li>Validations are applied as tool calls (instead of exceptions).</li>  
# <li>If any tool or validation fails, the report is marked as <code>False</code>, and—if <code>stop_on_failed_validation=True</code>—execution halts immediately.</li>  
# <li>Write tools update the DataFrames automatically, and the updated frames can optionally be returned at the end.</li>  
# </ol>  
# </div>  
# 
# By structuring execution this way, you make workflows **safe, testable, and extensible**. It also becomes easier to add new tools, validations, or checks later without breaking the overall process. 
# 
# 
# Now you can try the next example. To keep it simple, we’ll start with a **single-step plan** that just looks up a product in the inventory. This way, you can see how `execute_plan_tools_only()` processes the step, records results, and returns a structured execution report. 

# In[ ]:


# Minimal one-step plan: lookup the product "Aviator"
simple_plan = {
    "reasoning": "User wants to check availability of Aviator sunglasses.",
    "steps": [
        {
            "step_number": 1,
            "description": "Lookup Aviator sunglasses in inventory",
            "tools": [
                {"use": "get_inventory_data", "args": {"product_name": "Aviator"}, "result_key": "prod"}
            ],
            "validations": [
                {"name": "product_found", "use_tool": "assert_true", "args": {"value_from": "context.prod.item"}}
            ]
        }
    ]
}

# Run the plan with current inventory and transactions
report = execute_plan_tools_only(
    plan=simple_plan,
    inventory_df=inventory_df,
    transaction_df=transaction_df
)

utils.print_html(report, title="Execution Report: Single-step Plan")


# ## 5. Agentic Workflow Steps
# 
# Up to this point, you've seen how to define tools and execute plans directly. The next step is to introduce **agentic workflow steps** that intelligently generate, review, and explain plans. Together, these workflow components make the system more **autonomous** and **resilient**.

# ### 5.1 Planning Step
# 
# The planning step takes a **customer query** (e.g., *"Buy 2 Aviators"*) and transforms it into a structured plan made up entirely of tool calls — similar to the single-step example you saw earlier, but typically spanning **multiple coordinated steps**.  
# 
# Instead of you writing out the plan manually, the workflow decides:  
# - Which tools to use  
# - How to sequence them  
# - What validations to apply  
# 
# The planner follows a strict `TOOLS-ONLY` spec, ensuring no raw SQL appears in plans and all steps are auditable.
# 
# The output is a JSON object containing `reasoning` and `steps`.  This structured format ensures the plan is both **machine-readable** and **auditable**.

# In[ ]:


# =========================
# Planning spec (TOOLS-ONLY) and planning workflow
# =========================

# Shared planning spec: TOOLS ONLY (no raw SQL in the plan)
PLANNING_SPEC_TOOLS_ONLY = """
You are a planning system for a sunglasses store. Produce a FULL, AUTONOMOUS plan using TOOLS ONLY.
We will run this plan against two pandas DataFrames registered in DuckDB as views:
- inventory_df(name, item_id, description, quantity_in_stock, price)
- transaction_df(transaction_id, customer_name, transaction_summary, transaction_amount, balance_after_transaction)

Customer intents include:
- Purchase: "I want to buy 3 Aviators"
- Return: "I'd like to return two Sport sunglasses"
- Inquiry: "Do you have Mystique glasses?"
- Browse: "Show me what's available"

IMPORTANT: ALLOWED TOOLS ONLY (do NOT invent new tools)
Tool catalog (names, exact args, outputs):
1) get_inventory_data
   - args: { product_name?: string, item_id?: string }
   - returns: { rows: DataFrame, match_count: int, item: dict|null }
   - notes: Use this for product lookup (case-insensitive by name) or by item_id.
2) get_transaction_data
   - args: { mode?: "last_balance" }
   - returns: { mode: string, last_txn_id: string|null, last_balance: number }
3) compute_total
   - args: { qty: number, price: number }
   - returns: { amount: number }
4) compute_refund
   - args: { qty: number, price: number }   # Refund is negative by design
   - returns: { amount: number }
5) update_inventory
   - args: { item_id: string, delta?: number, quantity_new?: number }
   - returns: { inventory_df: DataFrame, updated: { item_id: string, quantity_in_stock: number } }
   - notes: For purchase use delta = -qty. For return use delta = +qty.
6) append_transaction
   - args: { customer_name: string, summary: string, amount: number }
   - returns: { transaction_df: DataFrame, transaction: { ... } }
7) assert_true
   - args: { value: any }                    # passes if truthy (non-null/non-zero/non-empty)
   - returns: { ok: boolean }
8) assert_nonnegative_stock
   - args: { inventory_df: DataFrame, item_id: string }
   - returns: { ok: boolean, qty: number }

STRICT RULES:
1) Return VALID JSON ONLY with keys: reasoning, steps.
2) Each step MUST contain:
   - "step_number": integer
   - "description": short human text
   - "tools": an array of tool calls in order. Each tool call is:
       {"use": "<tool_name>", "args": {...}, "result_key": "<context_key>"}
     * You MAY reference previous results using dotted paths starting with "context.", e.g., "context.prod.item.price".
     * Use *_from to resolve from context, e.g., {"price_from": "context.prod.item.price"}.
     * Use ONLY the tools listed above. Do NOT use names like assert_one, assert_gt, assert_contains, format_return_summary, lookup_product, propose_transaction, etc.
     * Strings like the transaction summary MUST be composed inline by you (e.g., "Return 2 Sport sunglasses").
   - "validations": array of tool validations:
       {"name": "...", "use_tool": "<tool_name>", "args": {...}}
     * Allowed validation tools: assert_true, assert_nonnegative_stock ONLY.
     * Examples:
         - product_found: assert_true with {"value_from": "context.prod.item"} (non-null)
         - nonnegative_stock_after_update: assert_nonnegative_stock with {"inventory_df_from": "context.__frames__.inventory_df", "item_id_from": "context.prod.item.item_id"}
3) Do NOT include raw SQL in the plan. Tools run any needed SQL internally.
4) For purchases/returns, include tool calls to:
   - Lookup product via get_inventory_data (case-insensitive by name)
   - (Purchase only) compute_total; (Return only) compute_refund
   - Create a clear summary STRING inline (e.g., "Purchase 3 Aviator sunglasses" / "Return 2 Sport sunglasses")
   - Update inventory via update_inventory (delta = -qty for purchases, +qty for returns)
   - Append the transaction via append_transaction (amount from compute_total/compute_refund)
5) Use canonical arg names exactly as in the Tool catalog:
   - quantity -> use qty
   - unit_price -> use price
   - Do NOT add extra args like sign; compute_refund already returns negative amounts.

OUTPUT JSON SHAPE:
{
  "reasoning": "...",
  "steps": [
    {
      "step_number": 1,
      "description": "...",
      "tools": [ {"use": "...", "args": {...}, "result_key": "..."} ],
      "validations": [ {"name": "...", "use_tool": "...", "args": {...}} ]
    }
  ]
}

EXAMPLE (Return 2 Sport sunglasses for a walk-in):
{
  "reasoning": "User requests a return of 2 Sport units. We'll lookup product, compute a negative refund, update stock (+2), and append a refund transaction.",
  "steps": [
    {
      "step_number": 1,
      "description": "Lookup product 'Sport' and capture item details",
      "tools": [
        {"use": "get_inventory_data", "args": {"product_name": "Sport"}, "result_key": "prod"}
      ],
      "validations": [
        {"name": "product_found", "use_tool": "assert_true", "args": {"value_from": "context.prod.item"}}
      ]
    },
    {
      "step_number": 2,
      "description": "Compute refund amount for qty=2",
      "tools": [
        {"use": "compute_refund", "args": {"qty": 2, "price_from": "context.prod.item.price"}, "result_key": "refund"}
      ],
      "validations": []
    },
    {
      "step_number": 3,
      "description": "Update inventory by adding returned quantity",
      "tools": [
        {"use": "update_inventory", "args": {"item_id_from": "context.prod.item.item_id", "delta": 2}, "result_key": "inv_after"}
      ],
      "validations": [
        {"name": "stock_nonnegative", "use_tool": "assert_nonnegative_stock",
         "args": {"inventory_df_from": "context.__frames__.inventory_df", "item_id_from": "context.prod.item.item_id"}}
      ]
    },
    {
      "step_number": 4,
      "description": "Append the refund transaction for a walk-in customer",
      "tools": [
        {"use": "append_transaction",
         "args": {
           "customer_name": "WALK_IN_CUSTOMER",
           "summary": "Return 2 Sport sunglasses",
           "amount_from": "context.refund.amount"
         },
         "result_key": "txn"}
      ],
      "validations": [
        {"name": "transaction_created", "use_tool": "assert_true",
         "args": {"value_from": "context.txn.transaction.transaction_id"}}
      ]
    }
  ]
}
"""

def generate_plan(user_query: str, model: str = "o4-mini") -> dict[str, Any]:
    context = f"{PLANNING_SPEC_TOOLS_ONLY}\n\nCustomer query: {user_query}\nProduce the plan now."
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return JSON ONLY following the TOOLS-ONLY planning spec."},
            {"role": "user", "content": context},
        ],
        response_format={"type": "json_object"},
    )
    return json.loads(resp.choices[0].message.content)


# Now you can try the **planning step** with a simple query:  
# 
# > "I'd like to return two Aviator sunglasses."  
# 
# Run the next cell to see how it generates a structured JSON plan together with its reasoning.

# In[ ]:


# =========================
# Try the planning step
# =========================
user_query = "I'd like to return two Aviator sunglasses"

draft_plan = generate_plan(user_query, model="o4-mini")
utils.print_html(json.dumps(draft_plan, indent=2), title="Draft Plan from Planning Step")


# ### 5.2 Reflection Step
# 
# Even the best plans may contain formatting errors or small inconsistencies. The reflection step acts as a **quality control process**:  
# 
# - It repairs **minor issues** (e.g., invalid JSON escapes).  
# - It critiques the draft and produces a corrected `revised_plan` that follows the **TOOLS-ONLY spec**.  
# - It guarantees structure by always returning a JSON object with two keys:  
#   - `critique` → feedback on the draft  
#   - `revised_plan` → the corrected plan  
# 
# If the reflection output is malformed, the system gracefully falls back to the original draft.

# In[ ]:


# =========================
# Reflection step (enforces the same TOOLS-ONLY spec)
# =========================
_ALLOWED_ESC = r'["\\/bfnrtu]'
def _repair_invalid_json_escapes(s: str) -> str:
    s = s.replace("\\'", "'")
    return re.sub(rf'\\(?!{_ALLOWED_ESC})', r'', s)

def _parse_json_or_repair(s: str) -> dict[str, Any]:
    try:
        return json.loads(s)
    except Exception:
        return json.loads(_repair_invalid_json_escapes(s))

def reflect_on_plan(user_query: str, draft_plan: dict[str, Any], model: str = "o4-mini") -> dict[str, Any]:
    sys = (
        "You are a senior plan reviewer. Return STRICT JSON with keys "
        "'critique' (string) and 'revised_plan' (object). The revised_plan MUST follow the TOOLS-ONLY spec."
    )
    user = (
        "TOOLS-ONLY PLANNING SPEC (enforce exactly):\n"
        f"{PLANNING_SPEC_TOOLS_ONLY}\n\n"
        "Customer query:\n"
        f"{user_query}\n\n"
        "Draft plan (JSON):\n"
        f"{json.dumps(draft_plan, ensure_ascii=False)}\n\n"
        "Task: Critique the draft against the spec and return a corrected 'revised_plan' if needed. "
        "Ensure valid JSON and that no raw SQL appears in the plan (only tool calls)."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    data = _parse_json_or_repair(resp.choices[0].message.content)
    if "revised_plan" not in data or not isinstance(data["revised_plan"], dict):
        if "steps" in data:
            data = {"critique": "No explicit critique provided.", "revised_plan": data}
        else:
            data = {"critique": "Malformed reflection output; falling back to draft.", "revised_plan": draft_plan}
    return data


# Now you can try the **reflection step**.  For this, you will intentionally craft a slightly flawed draft plan (e.g., wrong argument names and missing validations). Run the next cell to see how this step produces a **critique** and a **revised_plan** that complies with the TOOLS-ONLY spec.

# In[ ]:


# A purposely imperfect draft plan (arg mismatch + missing validation)
draft_plan = {
    "reasoning": "User wants to buy 2 Aviators.",
    "steps": [
        {
            "step_number": 1,
            "description": "Lookup product 'Aviator'",
            "tools": [
                {"use": "get_inventory_data", "args": {"product_name": "Aviator"}, "result_key": "prod"}
            ],
            "validations": []  # (missing product_found validation)
        },
        {
            "step_number": 2,
            "description": "Compute total for purchase",
            "tools": [
                # <-- wrong arg name: should be qty
                {"use": "compute_total", "args": {"quantity": 2, "price_from": "context.prod.item.price"}, "result_key": "total"}
            ],
            "validations": []
        }
    ]
}

user_query = "I'd like to buy 2 Aviator sunglasses for a walk-in customer"

# Run the reflection step to critique and revise the draft
reflection = reflect_on_plan(user_query=user_query, draft_plan=draft_plan, model="o4-mini")

# Display results
utils.print_html(reflection.get("critique"), title="Reflection Critique")
utils.print_html(json.dumps(reflection.get("revised_plan"), indent=2), title="Revised Plan (TOOLS-ONLY compliant)")


# ### 5.3 Error Explanation Step
# 
# Even with validations in place, execution may still fail — perhaps due to insufficient stock or a malformed plan. The error explanation step translates these failures into **human-readable guidance**.  
# 
# - It takes the **customer query** and the **execution report** as input.  
# - It uses a model to **summarize errors in plain language**.  
# - It outputs clear feedback, e.g.: *"Stock would go negative. Try reducing the quantity."*  
# 
# This closes the loop, helping learners connect technical validation errors to **actionable insights**.

# In[ ]:


def explain_execution_error(user_query: str, execution_report: dict[str, Any], model: str = "o4-mini") -> str:
    sys = (
        "You are a senior plan reviewer. Given a json with the report, "
        "explain in simple terms what went wrong and how to fix it."
    )
    user = (
        "Customer query:\n"
        f"{user_query}\n\n"
        "Execution report (JSON):\n"
        f"{execution_report}\n\n"
        "Task: Explain in simple terms what went wrong and how to fix it."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": sys},
                  {"role": "user", "content": user}],
        response_format={"type": "text"},
    )
    return resp.choices[0].message.content


# Now you can try the **Error Explanation Step**. To demonstrate, we'll simulate a failed execution where a purchase request would make the stock go negative.  The step will take the **customer query** and the **execution report**, then translate the technical error into simple guidance.
# 
# Run the next cell to see how it summarizes the problem and suggests a fix.

# In[ ]:


# Example: simulate a failed execution report
failed_report = {
    "ok": False,
    "steps": [
        {
            "step_number": 1,
            "description": "Lookup product 'Aviator'",
            "tools_run": ["get_inventory_data"],
            "tool_error": None,
            "validations": [{"name": "product_found", "ok": True}],
        },
        {
            "step_number": 2,
            "description": "Update inventory by subtracting 999 units",
            "tools_run": ["update_inventory"],
            "tool_error": None,
            "validations": [{"name": "stock_nonnegative", "ok": False, "qty": -995}],
        },
    ],
    "aborted": True,
    "abort_step": 2,
    "abort_reason": "validation_failed",
}

# Use the error explanation step to explain the error
explanation = explain_execution_error(
    user_query="Buy 999 Aviator sunglasses",
    execution_report=failed_report
)

utils.print_html(content=explanation, title="Error Explanation")


# <div style="background-color:#ffe4e1; padding:12px; border-radius:6px; color:black;">
#   <strong>Note:</strong> Use the <em>Error Explanation Step</em> when <code>report['ok']</code> is <code>False</code>.  
#   This allows you to turn a failed execution report into a clear, plain-English explanation of what went wrong and how to fix it.
# </div>

# ## 6. End-to-End Example
# 
# Now that you have tools, an executor, and workflow steps in place, you will run the whole pipeline from a single prompt. This walkthrough helps you see the full flow and where each piece fits.
# 
# What you will do:
# a) Preview the initial data.  
# b) Build a **tools-only** plan from a natural-language request.  
# c) Run a **reflection** pass to enforce the spec and fix small issues.  
# d) Execute the final plan safely.  
# e) Inspect the step-by-step report and updated tables.  
# f) (If something fails) Generate a plain-English error explanation.
# 
# ### Model options
# 
# You can mix and match OpenAI models depending on capability, cost, and speed:
# - `gpt-4o`  
# - `gpt-4.1`  
# - `gpt-4.1-mini`  
# - `o4-mini`
# 
# In practice, **self-reflection with `o4-mini`** often gives strong results for this workflow. Feel free to try different pairings (e.g., planning with `o4-mini` and reflecting with `gpt-4o`) to compare plans and validation outcomes.
# 
# ### Prompt ideas (expected behavior)
# 
# — "I'd like to **buy five Mystique** sunglasses for a walk-in customer" → **Fail** (would drive stock negative)  
# — "I'd like to **return two Aviator** sunglasses for a walk-in customer" → **Pass**  
# — "I'd like to **buy ten Sport** sunglasses for a walk-in customer" → **Pass**
# 
# > Results may vary across models and runs. Try slightly different phrasing, swap models for planning vs. reflection, and observe how plans and validations change.
# 
# **Note:** Because LLMs are stochastic, two runs with the same prompt may differ. Experiment with model choices and parameters to find the setup that best fits your needs.

# In[ ]:


user_prompt = "I'd like to return two Aviator sunglasses for a walk-in customer"
utils.print_html(user_prompt, title="User Prompt")

# Assumes you already created inventory_df and transaction_df.
utils.print_html(inventory_df, title="Initial Inventory (sample)")
utils.print_html(transaction_df, title="Initial Transactions (tail)")


# 1) Create a plan (TOOLS-ONLY)
draft_plan = generate_plan(user_prompt, model="o4-mini")
utils.print_html(json.dumps(draft_plan, indent=2), title="Draft Plan")

# 2) Reflect and possibly revise
reflection = reflect_on_plan(user_prompt, draft_plan, model="o4-mini")
utils.print_html(reflection.get("critique"), title="Reflection Critique")
final_plan = reflection["revised_plan"]
utils.print_html(json.dumps(final_plan, indent=2), title="Final Plan (After Reflection)")

# 3) Execute the final plan (no 'apply' arg — executor auto-applies when tools return updated DataFrames)
report = execute_plan_tools_only(
    final_plan,
    inventory_df,
    transaction_df,
    return_updated_frames=True
)

# 4) Show the execution report
utils.print_html(f"Overall execution status: {'SUCCESS' if report['ok'] else 'FAILED'}", title="Execution Status")
for step in report["steps"]:
    utils.print_html(step, title=f"Execution Report Step {step['step_number']}: {step['description']}")

if report["ok"]:
    inventory_df = report["updated_frames"]["inventory_df"]
    transaction_df = report["updated_frames"]["transaction_df"]
    utils.print_html(inventory_df, title="Updated Inventory")
    utils.print_html(transaction_df, title="Updated Transactions")
else:
    utils.print_html("Some validations failed, no transactions were made — check the message below.", title="Execution Status")
    error_ = explain_execution_error(user_prompt, report, model="o4-mini")
    utils.print_html(f"<pre>{error_}</pre>", title="Error Explanation")


# ## 7. Final Takeaways
# 
# In this lab you learned how to:
# 
# - Break down customer requests into **planning, reflection, execution, and error explanation** workflow steps.  
# - Use a **tools-only registry** to enforce safe, structured execution.  
# - Add **validation hooks** (e.g., non-negative stock) to prevent unsafe updates.  
# - Provide **human-readable feedback** when execution fails.

# <div style="border:1px solid #22c55e; border-left:6px solid #16a34a; background:#dcfce7; border-radius:6px; padding:14px 16px; color:#064e3b; font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,Cantarell,Noto Sans,sans-serif;">
# 
# 🎉 <strong>Congratulations!</strong>  
# 
# You have now completed the lab on building an **agentic customer service workflow**. Along the way, you experienced how planning, reflection, and execution steps can work together to turn natural requests into safe and structured tool calls. You also saw how validations keep updates reliable, how failures can be explained in plain English, and how tools-only execution makes the workflow transparent and extensible.  
# 
# With these skills, you are ready to design your own agentic workflows that not only handle tasks automatically, but also give you confidence in their safety, explainability, and adaptability. 🌟  
# </div>

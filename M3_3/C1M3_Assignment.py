#!/usr/bin/env python
# coding: utf-8

# # Graded Lab : Tool Use and Reflective Agents
# 
# In this lab, you will explore how AI agents can enhance research workflows by leveraging external tools and engaging in critical self-reflection. You'll learn how to build and integrate callable tools—such as web and academic search functions, and connect them to a language model using OpenAI's tool-calling API. Then, you’ll guide the agent to not only generate content but also **reflect** on its own output, improving the quality and depth of the final report. By the end of this lab, you will have implemented a mini agent capable of searching, reasoning, and publishing structured reports in HTML—laying the foundation for more advanced multi-step and autonomous AI systems.

# ### 🎯 Learning Objectives
# 
# By the end of this lab, you can:
# - Chain steps into a research pipeline (**search → reflection → formatting**).
# - Convert natural-language output into **styled HTML** suitable for sharing.

# ---
# <a name='submission'></a>
# 
# <h4 style="color:green; font-weight:bold;">TIPS FOR SUCCESSFUL GRADING OF YOUR ASSIGNMENT:</h4>
# 
# * All cells are frozen except for the ones where you need to write your solution code or when explicitly mentioned you can interact with it.
# 
# * In each exercise cell, look for comments `### START CODE HERE ###` and `### END CODE HERE ###`. These show you where to write the solution code. **Do not add or change any code that is outside these comments**.
# 
# * You can add new cells to experiment but these will be omitted by the grader, so don't rely on newly created cells to host your solution code, use the provided places for this.
# 
# * Avoid using global variables unless you absolutely have to. The grader tests your code in an isolated environment without running all cells from the top. As a result, global variables may be unavailable when scoring your submission. Global variables that are meant to be used will be defined in UPPERCASE.
# 
# * To submit your notebook for grading, first save it by clicking the 💾 icon on the top left of the page and then click on the <span style="background-color: red; color: white; padding: 3px 5px; font-size: 16px; border-radius: 5px;">Submit assignment</span> button on the top right of the page.
# ---

# In[ ]:


# ================================
# Standard library imports
# ================================
import json

# ================================
# Third-party imports
# ================================
from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import display, HTML
import os
import httpx
import research_tools

# API keys will be loaded from .env file via load_dotenv()

# Configure OpenAI client with SSL verification disabled
http_client = httpx.Client(verify=False)
CLIENT = OpenAI(http_client=http_client)


# In[ ]:


import unittests


# ## Using Tools
# 
# You’ll use two research tools exposed in the `research_tools` module:
# - **`arxiv_search_tool(query, max_results)`** – academic papers via arXiv API.
# - **`tavily_search_tool(query, max_results, include_images)`** – general web search via Tavily.

# Let's explore how the `arxiv_search_tool` works.
# 
# This tool searches arXiv and returns a list of papers with:
# - `title`, `authors`, `published`, `summary`, `url`, and (if available) `link_pdf`.
# 
# Below, we run a quick test and print the results in a readable format. Next cell is editable so feel free to try some search queries:
# 

# In[ ]:


# Test the arXiv search tool
topic = "linear algebra"

arxiv_results = research_tools.arxiv_search_tool(topic, max_results=3)

# Show formatted arxiv_results
for i, paper in enumerate(arxiv_results, 1):
    if "error" in paper:
        print(f"❌ Error: {paper['error']}")
    else:
        print(f"📄 Paper {i}")
        print(f"  Title     : {paper['title']}")
        print(f"  Authors   : {', '.join(paper['authors'])}")
        print(f"  Published : {paper['published']}")
        print(f"  URL       : {paper['url']}\n")


print("\n🧾 Raw arxiv_Results:\n")
print(json.dumps(arxiv_results, indent=2))


# The `tavily_search_tool` calls the Tavily API to fetch web results. Returns a list of dicts:
# - `title`, `content`, `url` (and optional image URLs when `include_images=True`).
# 
# Run the cell to inspect sample output. Next cell is editable so feel free to try some search queries:

# In[ ]:


# Test the Tavily search tool
topic = "retrieval-augmented generation applications"

tavily_results = research_tools.tavily_search_tool(topic)
for item in tavily_results:
    print(item)


# ## Tool Mapping
# 
# In the next cell you will define a dictionary that maps tool names (strings) to the actual Python functions. This allows the model to call tools by name during tool-calling. This dictionary will be used in your first graded function:

# In[ ]:


# Tool mapping
TOOL_MAPPING = {
    "tavily_search_tool": research_tools.tavily_search_tool,
    "arxiv_search_tool": research_tools.arxiv_search_tool,
}


# ## Exercise 1: Generate Research Report with Tools
# **Goal:** Implement `generate_research_report_with_tools(prompt)`.
# In this exercise, you'll work on a function that generates a detailed research report with the assistance of online tools. Focus on setting up interaction with the language model and handling the responses effectively.
# 
# ## Key Hints
# 
# ### 1. Setting Up the Chat with the Language Model
# - **Tool Selection**: Ensure that the tools are automatically selected by the model. Look into how to set `tool_choice` to "auto" within the function call. A helpful resource can be found in [OpenAI’s Function Calling Documentation](https://platform.openai.com/docs/guides/function-calling#tool-choice).
# - **Parameter Configuration**: Consider the parameters already defined in your function, such as model, messages, and tools. Think about how these might be used in your setup.
# 
# ### 2. Recording Tool Call Results
# - **Understanding the `ChatCompletionMessage`** object will help you access the required attributes to save the messages. An example of `ChatCompletionMessage` looks like this: 
# 
# ```python
# ChatCompletionMessage(
#     content=None,
#     refusal=None,
#     role='assistant',
#     annotations=[],
#     audio=None,
#     function_call=None,
#     tool_calls=[
#         ChatCompletionMessageFunctionToolCall(
#             id='call_ymMki5TBB91efJhMPjgoqjop',
#             function=Function(
#                 arguments='{"query":"radio observations of recurrent novae","max_results":5}',
#                 name='arxiv_search_tool'
#             ),
#             type='function'
#         )
#     ]
# )
# ```
# Assuming that `msg` if of type `ChatCompletionMessage`, if you wanted to get the `name` of a `tool_call` you can do something like:
# ```python
#  for call in msg.tool_calls:
#     tool_name = call.function.name
# ```
# Finally, the `result` variable will be created by actually calling the function associated with each tool (`tool_func`).
# 
# By leveraging these hints, you'll work towards an implementation that enables robust data gathering and report generation through smart tool integration.

# In[ ]:


# GRADED FUNCTION: generate_research_report_with_tools
def generate_research_report_with_tools(prompt: str, model: str = "gpt-4o") -> str:
    """
    Generates a research report using OpenAI's tool-calling with arXiv and Tavily tools.

    Args:
        prompt (str): The user prompt.
        model (str): OpenAI model name.

    Returns:
        str: Final assistant research report text.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a research assistant that can search the web and arXiv to write detailed, "
                "accurate, and properly sourced research reports.\n\n"
                "🔍 Use tools when appropriate (e.g., to find scientific papers or web content).\n"
                "📚 Cite sources whenever relevant. Do NOT omit citations for brevity.\n"
                "🌐 When possible, include full URLs (arXiv links, web sources, etc.).\n"
                "✍️ Use an academic tone, organize output into clearly labeled sections, and include "
                "inline citations or footnotes as needed.\n"
                "🚫 Do not include placeholder text such as '(citation needed)' or '(citations omitted)'."
            )
        },
        {"role": "user", "content": prompt}
    ]

    # List of available tools
    tools = [research_tools.arxiv_tool_def, research_tools.tavily_tool_def]

    # Maximum number of turns
    max_turns = 10
    
    # Iterate for max_turns iterations
    for _ in range(max_turns):

        ### START CODE HERE ###

        # Chat with the LLM via the client and set the correct arguments. Hint: Their names match names of variables already defined.
        # Make sure to let the LLM choose tools automatically. Hint: Look at the docs provided earlier!
        response = CLIENT.chat.completions.create( 
            model=None,
            messages=None,
            tools=None,
            tool_choice=None,
            temperature=1, 
        ) 

        ### END CODE HERE ###

        # Get the response from the LLM and append to messages
        msg = response.choices[0].message 
        messages.append(msg) 

        # Stop when the assistant returns a final answer (no tool calls)
        if not msg.tool_calls:      
            final_text = msg.content
            print("✅ Final answer:")
            print(final_text)
            break

        # Execute tool calls and append results
        for call in msg.tool_calls:
            tool_name = call.function.name
            args = json.loads(call.function.arguments)
            print(f"🛠️ {tool_name}({args})")

            try:
                tool_func = TOOL_MAPPING[tool_name]
                result = tool_func(**args)
            except Exception as e:
                result = {"error": str(e)}

            ### START CODE HERE ###

            # Keep track of tool use in a new message
            new_msg = { 
                # Set role to "tool" (plain string) to signal a tool was used
                "role": None,
                # As stated in the markdown when inspecting the ChatCompletionMessage object 
                # every call has an attribute called id
                "tool_call_id": None,
                # The name of the tool was already defined above, use that variable
                "name": None,
                # Pass the result of calling the tool to json.dumps
                "content": json.dumps(None)
            }

            ### END CODE HERE ###

            # Append to messages
            messages.append(new_msg)

    return final_text


# Run the following cell to check the correctness of your code. It might take a while so don't worry if it takes a couple of minutes to run:

# In[ ]:


# Test your code!
unittests.test_generate_research_report_with_tools(generate_research_report_with_tools)


# ## Exercise 2: Reflection + Rewrite
# 
# **Goal:** Implement `reflection_and_rewrite(report)`.
# 
# In this task, your goal is to develop a function that takes a report, analyzes it, generates a structured reflection, and produces an improved version of the report. This involves two main tasks: crafting a precise prompt and setting up a correctly configured response call to the language model.
# 
# ## Key Steps
# 
# ### 1. Create a User Prompt
# 
# - **Objective**: Guide the language model to output a structured response in JSON format.
# - **Format**: Ensure the output includes two keys, `"reflection"` and `"revised_report"`.
# - **Details**: Your reflection should cover strengths, limitations, suggestions, and opportunities. The revised report should incorporate these elements to improve clarity and academic tone.
# 
# ### 2. Configure the Response Call
# 
# - **Parameters**: Use the specified model (e.g., `"gpt-4o-mini"`) and set the temperature equal to the `temperature` parameter of the graded function.
# - **Structure**: Make sure the response setup directs the model properly, ensuring the JSON format is adhered to without additional commentary.
# 
# 
# By implementing these steps, your function will effectively transform and improve the given reports. Handle JSON parsing carefully to ensure the output is valid and reliable. Happy coding!

# In[ ]:


# GRADED FUNCTION: reflection_and_rewrite
def reflection_and_rewrite(report, model: str = "gpt-4o-mini", temperature: float = 0.3) -> dict:
    """
    Generates a structured reflection AND a revised research report.
    Accepts raw text OR the messages list returned by generate_research_report_with_tools.

    Returns:
        dict with keys:
          - "reflection": structured reflection text
          - "revised_report": improved version of the input report
    """

    # Input can be plain text or a list of messages, this function detects and parses accordingly
    report = research_tools.parse_input(report)

    ### START CODE HERE ###

    # Define the prompt. A multi-line f-string is typically used for this.
    # Remember it should ask the model to output ONLY valid JSON with this structure:
    # {{ "reflection": "<text>", "revised_report": "<text>" }}
    user_prompt = None

    # Get a response from the LLM
    response = CLIENT.chat.completions.create( 
        # Pass in the model
        model=None,
        messages=[ 
            # System prompt is already defined
            {"role": "system", "content": "You are an academic reviewer and editor."},
            # Add user prompt
            {"role": "user", "content": None},
        ],
        # Set the temperature equal to the temperature parameter passed to the function
        temperature=None
    )

    ### END CODE HERE ###

    # Extract output
    llm_output = response.choices[0].message.content.strip()

    # Check if output is valid JSON
    try:
        data = json.loads(llm_output)
    except json.JSONDecodeError:
        raise Exception("The output of the LLM was not valid JSON. Adjust your prompt.")

    return {
        "reflection": str(data.get("reflection", "")).strip(),
        "revised_report": str(data.get("revised_report", "")).strip(),
    }


# In[ ]:


# Test your code!
unittests.test_reflection_and_rewrite(reflection_and_rewrite)


# ## Exercise 3: Convert Report to HTML
# **Goal:** Implement `convert_report_to_html(report)`.
# This exercise focuses on transforming a plain text research report into a well-structured HTML document. You will build a function to facilitate this conversion using a language model.
# 
# ## Key Steps
# 
# ### 1. Create a User Prompt
# - **Objective**: Instruct the model to transform plain text into HTML structure.
# - **Format**: Ensure the output is valid, clean HTML with appropriate section headers, formatted paragraphs, and clickable links.
# - **Details**: Preserve the citation style and request that the model responds only with HTML, without additional commentary.
# 
# ### 2. Configure the Response Call
# - **Parameters**: Use the specified model (e.g., `"gpt-4o"`) and set an appropriate temperature to balance creativity and accuracy.
# - **Structure**: Configure the `CLIENT.chat.completions.create` call properly, using both system and user prompts to ensure a clear and focused task description.
# 
# By following these steps, you'll effectively convert plaintext reports into formatted HTML documents.

# In[ ]:


# GRADED FUNCTION: convert_report_to_html
def convert_report_to_html(report, model: str = "gpt-4o", temperature: float = 0.5) -> str:
    """
    Converts a plaintext research report into a styled HTML page using OpenAI.
    Accepts raw text OR the messages list from the tool-calling step.
    """

    # Input can be plain text or a list of messages, this function detects and parses accordingly
    report = research_tools.parse_input(report)

    # System prompt is already provided
    system_prompt = "You convert plaintext reports into full clean HTML documents."

    ### START CODE HERE ###
    
    # Build the user prompt instructing the model to return ONLY valid HTML
    user_prompt = None

    # Call the LLM by interacting with the CLIENT. 
    # Remember to set the correct values for the model, messages (system and user prompts) and temperature
    response = None

    ### END CODE HERE ###

    # Extract the HTML from the assistant message
    html = response.choices[0].message.content.strip()  

    return html


# In[ ]:


# Test your code!
unittests.test_convert_report_to_html(convert_report_to_html)


# ### 🚀 End-to-End Pipeline
# 
# Run this cell to execute the full workflow:
# 
# 1. Generate a research report (tools).
# 2. Reflect on the report.
# 3. Convert the report to HTML.
# 
# > You should see the rendered HTML below and two concise reflections in the console.

# In[ ]:


# 1) Research with tools
prompt_ = "Radio observations of recurrent novae"
preliminary_report = generate_research_report_with_tools(prompt_)
print("=== Research Report (preliminary) ===\n")
print(preliminary_report)

# 2) Reflection on the report (use the final TEXT to avoid ambiguity)
reflection_text = reflection_and_rewrite(preliminary_report)   # <-- pass text, not messages
print("=== Reflection on Report ===\n")
print(reflection_text['reflection'], "\n")
print("=== Revised Report ===\n")
print(reflection_text['revised_report'], "\n")


# 3) Convert the report to HTML (use the TEXT and correct function name)
html = convert_report_to_html(reflection_text['revised_report'])

print("=== Generated HTML (preview) ===\n")
print((html or "")[:600], "\n... [truncated]\n")

# 4) Display full HTML
display(HTML(html))


# ### 📌 “Expected Output” note (for the notebook text cell)
# 
# - `generate_research_report_with_tools` should return a **non-trivial string** (> 50 chars).
# 
# - `reflection_and_rewrite` should return a **dict** with **'reflection'** and **'revised\_report'** (both strings). The reflection should **mention** the four sections (Strengths, Limitations, Suggestions, Opportunities).
# 
# - `convert_report_to_html` should return a **string that looks like HTML** (e.g., includes `<html>`, `<h1>`, `<p>`, or closing tags).
# 

# ---
# 
# ## ✅ Wrap-Up
# 
# You built a mini research agent that can:
# - 🔎 call tools (arXiv + Tavily),
# - 🧠 reflect on its own output,
# - 📰 publish a clean HTML report.
# 
# Great job!
# 
# ### What to Submit
# - Your notebook with Exercise 1–3 completed.
# 
# ### Troubleshooting (quick)
# - **Model/tool-call loop stalls?** Lower `max_turns` or print intermediate messages.
# - **HTML looks odd?** Re-run conversion with a fresh assistant response.
# 
# **You’re done—nice work!** 🚀
# 

# ## Check grading feedback
# 
# If you have collapsed the right panel to have more screen space for your code, as shown below:
# 
# <img src="./images/collapsed.png" alt="Collapsed Image" width="800" height="400"/>
# 
# You can click on the left-facing arrow button (highlighted in red) to view feedback for your submission after submitting it for grading. Once expanded, it should display like this:
# 
# <img src="./images/expanded.png" alt="Expanded Image" width="800" height="400"/>

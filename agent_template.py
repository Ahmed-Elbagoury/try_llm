# Necessary Imports
import csv
import pandas as pd
import math
import numpy as np
import os
from langchain_core.output_parsers import StrOutputParser

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_openai_functions_agent
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

# Perform necessary imports for creating agents
from langchain import hub # Used to pull predefined prompts from LangChain Hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.chat_message_histories import ChatMessageHistory # Used to store chat history in memory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import OpenAI
import os

## Set the OpenAI API key and model name
Open_API_Key = os.getenv("Open_API_Key")
os.environ["OPENAI_API_KEY"] = Open_API_Key
MODEL="gpt-4o-mini"
client = OpenAI(api_key=Open_API_Key)

## Set the Tavily API key
os.environ["TAVILY_API_KEY"] = os.getenv("Tavily_API_Key")

## Load the vectorstore
embeddings = OpenAIEmbeddings()
vector = FAISS.load_local(
    "./faiss_index", embeddings, allow_dangerous_deserialization=True
)


## Create the conversational agent

# Creating a retriever
# See https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/vectorstore/
retriever = # Create retriever from the loaded vector 


# import the 'tool' decorator
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool

# Define a tool for Amazon product search using @tool decorator
@tool
def amazon_product_search(query: str):
    """Search for information about Amazon products.
    For any questions related to Amazon products, this tool must be used."""

    ## TODO create retriever_tool based on the retriever object

# Import necessary classes from LangChain for Tavily integration
from langchain_community.tools.tavily_search import TavilySearchResults

@tool
def search_tavily(query: str):
    """
    Executes a web search using the TavilySearchResults tool.

    Parameters:
        query (str): The search query entered by the user.

    Returns:
        list: A list of search results containing answers, raw content, and images.
    """
    # TODO: Create an instance of TavilySearchResults with customized parameters
    

# hwchase17/react is a prompt template designed for ReAct-style
# conversational agents.
prompt = # pull "hwchase17/react" prompt from langchain hub

## Create a list of tools: retriever_tool and search_tool
tools = # TODO: Create a list of tools based on search_tavily and amazon_product_search.

# Enable memory optimization with ConversationSummaryMemory
# This ensures that older conversations are summarized instead of keeping full history,
# preventing excessive context length that slows down responses.
summary_memory = ## Create summary_memory using ConversationSummaryMemory()

# Initialize OpenAI model with streaming enabled
# Streaming allows tokens to be processed in real-time, reducing response latency.
summary_llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, streaming=True)

# Create a ReAct agent
# The agent will reason and take actions based on retrieved tools and memory.
summary_react_agent = ## TODO create a react agent. Check create_react_agent() from langchain

# Configure the AgentExecutor to manage reasoning steps
summary_agent_executor = # Create an agent executor. Check AgentExecutor() from langchain. Make sure to pass the summary_memory to it



# Building an UI for the chatbot with agents
import gradio as gr

# Initialize session-based chat history
session_memory = {}

def get_memory(session_id):
    """Fetch or create a chat history instance for a given session."""
    # Create a new element in the dictionary that corresponds to the session memory if it does not exit

# Wrap agent with session-based chat history
agent_with_chat_history = # Create agent with memory using RunnableWithMessageHistory()

# Define function for Gradio interface
def chat_with_agent(user_input, session_id):
    """Processes user input and maintains session-based chat history."""
    memory = get_memory(session_id)  # Fetch chat history for the session
    #memory.clear()
    # Invoke the agent with session memory
    response = agent_with_chat_history.invoke(
        {"input": user_input, "chat_history": memory.messages},
        config={"configurable": {"session_id": session_id}}
    )

    # Extract only the 'output' field from the response
    if isinstance(response, dict) and "output" in response:
        return response["output"]  # Return clean text response
    else:
        return "Error: Unexpected response format"

# Create Gradio app interface
with gr.Blocks() as app:
    gr.Markdown("# 🤖 Review Genie - Agents & ReAct Framework")
    gr.Markdown("Enter your query below and get AI-powered responses with session memory.")

    with gr.Row():
        input_box = gr.Textbox(label="Enter your query:", placeholder="Ask something...")
        output_box = gr.Textbox(label="Response:", lines=10)

    submit_button = gr.Button("Submit")

    submit_button.click(chat_with_agent, inputs=input_box, outputs=output_box)

# Launch the Gradio app
app.launch(debug=True, share=True)

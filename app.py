import streamlit as st
from haystack_integrations.tools.mcp import MCPToolset, StdioServerInfo
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack_integrations.components.generators.anthropic import AnthropicChatGenerator
from haystack.components.tools import ToolInvoker
from haystack.components.converters import OutputAdapter
from haystack.dataclasses import ChatMessage
import os
import nest_asyncio
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested asyncio event loops
nest_asyncio.apply()

# Set page config
st.set_page_config(
    page_title="Medical Research Assistant",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for API key
if "anthropic_api_key" not in st.session_state:
    st.session_state.anthropic_api_key = None

# Function to initialize the Haystack pipeline
@st.cache_resource
def initialize_pipeline(api_key):
    # Set the API key
    os.environ['ANTHROPIC_API_KEY'] = api_key
    
    # Initialize MCP tools
    server_info = StdioServerInfo(command="uv", args=["run", "--with", "biomcp-python", "biomcp", "run"])
    mcp_toolset = MCPToolset(server_info)
    
    # Create the pipeline
    pipeline = Pipeline()
    pipeline.add_component("llm", AnthropicChatGenerator(model="claude-3-7-sonnet-20250219", tools=mcp_toolset))
    pipeline.add_component("tool_invoker", ToolInvoker(tools=mcp_toolset))
    pipeline.add_component(
        "adapter",
        OutputAdapter(
            template="{{ initial_msg + initial_tool_messages + tool_messages }}",
            output_type=list[ChatMessage],
            unsafe=True,
        ),
    )
    pipeline.add_component("response_llm", AnthropicChatGenerator(model="claude-3-7-sonnet-20250219"))
    
    # Connect the components
    pipeline.connect("llm.replies", "tool_invoker.messages")
    pipeline.connect("llm.replies", "adapter.initial_tool_messages")
    pipeline.connect("tool_invoker.tool_messages", "adapter.tool_messages")
    pipeline.connect("adapter.output", "response_llm.messages")
    
    return pipeline, mcp_toolset

# Sidebar for API key input
with st.sidebar:
    st.title("Configuration")
    st.markdown("---")
    
    # API key input
    api_key = st.text_input("Anthropic API Key", type="password", value=st.session_state.anthropic_api_key or "")
    if api_key:
        st.session_state.anthropic_api_key = api_key
    
    st.markdown("---")
    st.title("Available Tools")
    
    # Initialize pipeline if API key is provided
    if st.session_state.anthropic_api_key:
        try:
            pipeline, mcp_toolset = initialize_pipeline(st.session_state.anthropic_api_key)
            
            # Display tools in sidebar
            for tool in mcp_toolset.tools:
                with st.expander(f"🔧 {tool.name}"):
                    st.markdown(tool.description)
                    st.markdown("**Parameters:**")
                    st.json(tool.parameters)
        except Exception as e:
            error_msg = f"Error initializing pipeline: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)
    else:
        st.info("Please enter your Anthropic API key to see available tools.")

# Main chat interface
st.title("Medical Research Assistant")
st.markdown("""
This assistant can help you with:
- Searching clinical trials
- Finding variant information
- Retrieving article details
- And more!
""")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Check if API key is provided
    if not st.session_state.anthropic_api_key:
        st.error("Please enter your Anthropic API key in the sidebar to use the chat.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        try:
            # Create a placeholder for streaming the response
            response_placeholder = st.empty()
            
            # Create a class to hold the response
            class ResponseAccumulator:
                def __init__(self):
                    self.text = ""
                
                def add_chunk(self, chunk):
                    self.text += chunk.content
                    response_placeholder.markdown(self.text + "▌")
            
            # Create an instance of the accumulator
            accumulator = ResponseAccumulator()
            
            # Create a new response_llm component with the streaming callback
            pipeline.remove_component("response_llm")
            pipeline.add_component("response_llm", AnthropicChatGenerator(
                model="claude-3-7-sonnet-20250219",
                streaming_callback=accumulator.add_chunk
            ))
            
            # Reconnect the components
            pipeline.connect("adapter.output", "response_llm.messages")
            
            # Create messages for the pipeline
            logger.info(f"Prompt: {prompt}")
            messages = [ChatMessage.from_user(prompt)]
            
            # Log the pipeline inputs for debugging
            logger.info(f"Running pipeline with inputs: messages={messages}")
            
            # Run the pipeline with all required inputs
            result = pipeline.run({
                "llm": {"messages": messages},
                "adapter": {"initial_msg": messages}
            })
            
            # Log the pipeline result for debugging
            logger.info(f"Pipeline result: {result}")
            
            # Update the placeholder with the final response
            response_placeholder.markdown(accumulator.text)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": accumulator.text})
        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg}) 
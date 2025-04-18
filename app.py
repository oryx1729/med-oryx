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
import json

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested asyncio event loops
nest_asyncio.apply()

# Set page config
st.set_page_config(
    page_title="Medical Research Agent",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize session state for tool invocations
if "tool_invocations" not in st.session_state:
    st.session_state.tool_invocations = []

# Function to initialize the Haystack pipeline
@st.cache_resource
def initialize_pipeline():
    # Get the API key from environment variable
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")
    
    # Initialize MCP tools
    server_info = StdioServerInfo(command="uv", args=["run", "--with", "biomcp-python", "biomcp", "run"])
    mcp_toolset = MCPToolset(server_info)
    
    # Initialize the Agent
    # The AnthropicChatGenerator will use the env var by default
    agent = Agent(
        chat_generator=AnthropicChatGenerator(model="claude-3-5-sonnet-latest"), 
        tools=mcp_toolset
    )
    
    # Create the pipeline
    pipeline = Pipeline()
    pipeline.add_component("agent", agent)
    
    return pipeline, mcp_toolset


with st.sidebar:
    st.title("Available Tools")
    
    # Initialize pipeline only if API key env var is set
    if os.environ.get('ANTHROPIC_API_KEY'):
        try:
            # Pass API key to initialize_pipeline
            pipeline, mcp_toolset = initialize_pipeline()
            
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
            # Display error in sidebar or main area? Sidebar for config error seems ok.
            st.error(error_msg) 
    else:
        st.info("ANTHROPIC_API_KEY environment variable not set. Please configure it to use the application.")

# Main chat interface
st.title("Med Oryx -- Medical Research Agent")


st.info("""
**Med Oryx is your intelligent medical research companion powered by BioMCP and Haystack.**

Leveraging authoritative biomedical data sources, Med Oryx can help you:
- Search and analyze **clinical trials** from ClinicalTrials.gov (including protocols, outcomes, locations)
- Find **genomic variant** information from MyVariant.info
- Access **scientific literature** through PubMed/PubTator3
- Answer complex biomedical questions using natural language

Built with [BioMCP](https://github.com/genomoncology/biomcp) for biomedical data access and [Haystack](https://github.com/deepset-ai/haystack) for AI orchestration.
""")

# Display chat messages
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # Display tool invocations for this assistant message if they exist
        invocations = st.session_state.tool_invocations[idx] if idx < len(st.session_state.tool_invocations) else None
        if message["role"] == "assistant" and invocations:
            with st.expander("🔧 View Tool Invocations", expanded=False):
                for tool_invocation in invocations:
                    tool_name = tool_invocation.get("name", "Unknown Tool")
                    tool_args = tool_invocation.get("args", {})
                    tool_result = tool_invocation.get("result", "No result")
                    
                    # Display using Streamlit columns
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.markdown(f"🔧 **{tool_name}**")
                    with col2:
                        st.markdown("**Input:**")
                        st.json(tool_args)
                        st.markdown("**Output:**")
                        # Handle potential non-JSON results gracefully
                        try:
                            # Attempt to load string results as JSON first
                            if isinstance(tool_result, str):
                                try:
                                    parsed_result = json.loads(tool_result)
                                    st.json(parsed_result)
                                except json.JSONDecodeError:
                                    # If not a JSON string, display as preformatted text/markdown
                                    st.markdown(f"```\n{tool_result}\n```") 
                            else:
                                 st.json(tool_result) # If already object/dict, display as JSON
                        except Exception as json_ex:
                            logger.error(f"Error displaying tool result: {json_ex}")
                            st.text(str(tool_result)) # Fallback to text
                    st.markdown("---")

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Check if API key environment variable is set
    if not os.environ.get('ANTHROPIC_API_KEY'):
        st.error("ANTHROPIC_API_KEY environment variable not set. Please configure it to use the application.")
        st.stop()
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.tool_invocations.append(None) # Add placeholder for user message tools
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        try:
            logger.info(f"Running agent with query: {prompt}")
            
            # --- Streaming Setup --- 
            response_placeholder = st.empty() # Placeholder for streaming output
            
            # Define the streaming callback handler
            class ResponseAccumulator:
                def __init__(self):
                    self.text = ""
                
                def add_chunk(self, chunk):
                    # Assuming chunk has a .content attribute like before
                    # Adapt this based on the actual chunk structure if needed
                    if hasattr(chunk, 'content'): 
                        self.text += chunk.content
                        response_placeholder.markdown(self.text + "▌")
                    elif isinstance(chunk, str): # Fallback if chunk is just a string
                        self.text += chunk
                        response_placeholder.markdown(self.text + "▌")
                    # Log chunk structure if needed for debugging
                    # logger.info(f"Stream chunk: {chunk}") 

            accumulator = ResponseAccumulator()
            
            # Get the agent component and set its generator's streaming callback
            try:
                agent_component = pipeline.get_component("agent")
                # Assuming the generator is accessible like this, adjust if needed
                if hasattr(agent_component, 'chat_generator') and agent_component.chat_generator:
                    agent_component.chat_generator.streaming_callback = accumulator.add_chunk
                else:
                    logger.warning("Could not find chat_generator on agent component to set streaming callback.")
            except Exception as e:
                 logger.error(f"Error setting streaming callback: {e}")
            # --- End Streaming Setup ---
            
            # Prepare the initial message for the agent pipeline
            user_input_msg = ChatMessage.from_user(prompt)
            
            # Run the agent pipeline
            result = pipeline.run({"agent": {"messages": [user_input_msg]}})
            
            # logger.info(f"Agent result: {result}")
            
            # --- Extract Data from Pipeline Result --- 
            agent_output = result.get("agent", {})
            full_transcript_messages = agent_output.get("messages", [])
            
            # Final answer might be fully captured by accumulator, or get from last message
            final_answer = accumulator.text if accumulator.text else "Sorry, I couldn't get a final answer." 
            # Optionally, double-check with the last message if accumulator is empty
            if not final_answer and full_transcript_messages:
                 for msg in reversed(full_transcript_messages):
                     if isinstance(msg, ChatMessage) and msg.role == "assistant":
                         final_answer = msg.text
                         break

            current_tool_invocations = []
            if full_transcript_messages:
                # Parse the transcript for tool calls and results
                for msg in full_transcript_messages:
                    # Tool messages contain ToolCallResult(s) in tool_call_results list
                    if isinstance(msg, ChatMessage) and msg.role == "tool" and msg.tool_call_results:
                        for tool_call_result in msg.tool_call_results:
                            tool_origin = tool_call_result.origin # This is the original ToolCall
                            if tool_origin:
                                current_tool_invocations.append({
                                    "name": tool_origin.tool_name,
                                    "args": tool_origin.arguments,
                                    "result": tool_call_result.result # The actual string/object result
                                })
                            else:
                                logger.warning(f"ToolCallResult missing origin: {tool_call_result}")
                        
            # --- End Extraction Logic --- 

            # Display final answer (update placeholder without cursor)
            response_placeholder.markdown(final_answer)
            
            # Add assistant response and tools to history
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            st.session_state.tool_invocations.append(current_tool_invocations)
            
            # Display current tool invocations if any (immediately below the answer)
            if current_tool_invocations:
                with st.expander("🔧 View Tool Invocations", expanded=False): # Collapsed by default now
                    for tool_invocation in current_tool_invocations:
                        tool_name = tool_invocation.get("name", "Unknown Tool")
                        tool_args = tool_invocation.get("args", {})
                        tool_result = tool_invocation.get("result", "No result")
                        # ipdb.set_trace()
                        # Display using Streamlit columns
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.markdown(f"🔧 **{tool_name}**")
                        with col2:
                            st.markdown("**Input:**")
                            st.json(tool_args)
                            st.markdown("**Output:**")
                            # Handle potential non-JSON results gracefully
                            try:
                                # Attempt to load string results as JSON first
                                if isinstance(tool_result, str):
                                    try:
                                        parsed_result = json.loads(tool_result)
                                        st.json(parsed_result)
                                    except json.JSONDecodeError:
                                         # Display as markdown/code block if not JSON string
                                        st.markdown(f"```\n{tool_result}\n```") 
                                else:
                                     st.json(tool_result) # If already object/dict, display as JSON
                            except Exception as json_ex:
                                logger.error(f"Error displaying tool result: {json_ex}")
                                st.text(str(tool_result)) # Fallback to text
                        st.markdown("---")

        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            st.session_state.tool_invocations.append([]) # Add empty tools for error message 
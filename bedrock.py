import boto3  # AWS SDK for Python to interact with AWS services
from langgraph_checkpoint_aws.saver import BedrockSessionSaver  # Saver for Bedrock session state
from botocore.config import Config  # Configuration for AWS clients
from langchain_aws import ChatBedrockConverse  # Conversational AI model interface
from langchain_core.tools import tool  # Decorator for tools (commented out here)
from langgraph.graph import StateGraph  # Graph structure for state management
from langchain_core.runnables import RunnableLambda  # Runnable lambda functions
from langchain_core.messages import BaseMessage  # Base class for messages


# from langgraph_checkpoint_aws.client import BedrockSessionClient  # (commented out)

# AWS region for the Bedrock client
region_name='us-east-1'  
# Create Bedrock runtime client
bedrock_client = boto3.client(service_name='bedrock-runtime',  
                                    region_name=region_name,
                                    config=Config(read_timeout=2000))


context = "you are a conversational AI, and the user is asking you a question"  # Context for the conversation


# Initialize conversational AI model
llm = ChatBedrockConverse(  
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    temperature=0,
    max_tokens=None,
    client=bedrock_client,
)

# @tool
# def search_shoes(preference):
#     """Search for shoes based on user preferences and interests."""  # Example tool function (commented out)
#     return 


# Saver for session state
session_saver = BedrockSessionSaver(  
    region_name=region_name,
   )




def inject_context(state):  # Function to inject context into the state
    messages = state.get("messages", [])
    return {
        "context": context,
        "messages": messages
    }

context_loader = RunnableLambda(inject_context)  # Runnable lambda for context injection


def extract_messages(state: dict):  # Function to extract messages from state
    return state["messages"]

extract_messages_node = RunnableLambda(extract_messages)  # Runnable lambda for extracting messages




from langgraph.graph import END  # Import END constant for graph

# Build graph
graph_builder = StateGraph(dict)  # Initialize state graph builder

# Add nodes
graph_builder.add_node("load_context", context_loader)  # Node to load context
graph_builder.add_node("extract_messages", extract_messages_node)  # Node to extract messages
graph_builder.add_node("llm", llm)  # Node for language model

# Connect nodes
graph_builder.add_edge("load_context", "extract_messages")  # Edge from context loader to message extractor
graph_builder.add_edge("extract_messages", "llm")  # Edge from message extractor to language model

# Set entry/exit
graph_builder.set_entry_point("load_context")  # Entry point of the graph
graph_builder.set_finish_point("llm")  # Finish point of the graph

# Compile graph
graph = graph_builder.compile(checkpointer=session_saver)  # Compile the graph with session saver



client = session_saver.session_client.client  # AWS client for session saver
# Create a new session
session_id = session_saver.session_client.client.create_session()["sessionId"]  # Create a new session ID

print(f"Session Created {session_id}")  # Print session creation confirmation


config = {"configurable": {"thread_id": session_id}}  # Configuration with thread ID



chat_history = []  # Initialize chat history
while True:
    user_input = input("User: ")  # Get user input
    if user_input.lower() in ["quit", "exit", "q"]:  # Check for exit commands
        print("Goodbye!")  # Print goodbye message
        break

    # Append user message
    chat_history.append(("user", user_input))  # Add user message to history

    # Run graph with current history
    for event in graph.stream(
        {"messages": chat_history}, 
        config
    ):
        for value in event.values():
            # print("Raw value from event:", value,)  # Print raw event value

            if isinstance(value, BaseMessage):  # Check if value is a message
                print("Assistant:", value.content)  # Print assistant response
                chat_history.append(("ai", value.content))  # Add assistant response to history

# for i in graph.get_state_history(config, limit=5):
#     print(i,"graph state")

# List all invocation steps
# steps = client.list_invocation_steps(
#     sessionIdentifier=session_id,
# )
# if steps["invocationSteps"]:
#     first_step = steps["invocationSteps"][0]
#     invocationIdentifier = first_step["invocationIdentifier"]
#     invocationStepId = first_step["invocationStepId"]
#     # Get specific step details
#     step_details = client.get_invocation_step(
#         sessionIdentifier=session_id,
#         invocationIdentifier=invocationIdentifier,
#         invocationStepId=invocationStepId,
#     )

# # ✅ Optional: retrieve a valid checkpoint_id (you may replace this with an actual ID dynamically)
# checkpoint_id = session_saver.get_latest_checkpoint_id(session_id=session_id)

# config_replay = {
#     "configurable": {
#         "thread_id": session_id,
#         "checkpoint_id": checkpoint_id,  # ✅ Inject actual checkpoint_id
#     }
# }
# for event in graph.stream(None, config_replay, stream_mode="values"):
#     print(event)


# config = {
#     "configurable": {
#         "thread_id": session_id,
#         "checkpoint_id": "<checkpoint_id>",
#     }
# }
# graph.update_state(config, {"state": "updated state"})

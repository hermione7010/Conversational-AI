import boto3
from langgraph_checkpoint_aws.saver import BedrockSessionSaver
from botocore.config import Config
from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langgraph.graph import StateGraph  
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import BaseMessage  # ✅ Added to avoid NameError
# from langgraph_checkpoint_aws.client import BedrockSessionClient  


region_name='us-east-1'
bedrock_client = boto3.client(service_name='bedrock-runtime',
                                    region_name=region_name,
                                    config=Config(read_timeout=2000))


context = "you are a conversational AI, and the user is asking you a question"

llm = ChatBedrockConverse(
    model="anthropic.claude-3-sonnet-20240229-v1:0",
    temperature=0,
    max_tokens=None,
    client=bedrock_client,
)

@tool
def search_shoes(preference):
    """Search for shoes based on user preferences and interests."""
    return 

session_saver = BedrockSessionSaver(
    region_name=region_name,
   )

def inject_context(state):
        return {"context": context}
 
context_loader = RunnableLambda(inject_context)
graph_builder = StateGraph(dict)  # or a more specific schema
graph_builder.add_node("load_context", llm)
graph_builder.set_entry_point("load_context")
graph = graph_builder.compile(checkpointer=session_saver)
 



client = session_saver.session_client.client # Same region as above
# Create a new session
session_id = session_saver.session_client.client.create_session()["sessionId"]




config = {"configurable": {"thread_id": session_id}}

while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break
    for event in graph.stream(
        {"messages": [("user", user_input)]}, 
        config
    ):
        for value in event.values():
            print(value)
            continue
            if isinstance(value["messages"][-1], BaseMessage):
                print("Assistant:", value["messages"][-1].content)


for i in graph.get_state_history(config, limit=5):
    print(i)

# List all invocation steps
steps = client.list_invocation_steps(
    sessionIdentifier=session_id,
)
if steps["invocationSteps"]:
    first_step = steps["invocationSteps"][0]
    invocationIdentifier = first_step["invocationIdentifier"]
    invocationStepId = first_step["invocationStepId"]
    # Get specific step details
    step_details = client.get_invocation_step(
        sessionIdentifier=session_id,
        invocationIdentifier=invocationIdentifier,
        invocationStepId=invocationStepId,
    )

# ✅ Optional: retrieve a valid checkpoint_id (you may replace this with an actual ID dynamically)
checkpoint_id = session_saver.get_latest_checkpoint_id(session_id=session_id)

config_replay = {
    "configurable": {
        "thread_id": session_id,
        "checkpoint_id": checkpoint_id,  # ✅ Inject actual checkpoint_id
    }
}
for event in graph.stream(None, config_replay, stream_mode="values"):
    print(event)


config = {
    "configurable": {
        "thread_id": session_id,
        "checkpoint_id": "<checkpoint_id>",
    }
}
graph.update_state(config, {"state": "updated state"})
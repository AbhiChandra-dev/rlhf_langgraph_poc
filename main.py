from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from utils import get_human_feedback, save_feedback

# Define the state
class AgentState(dict):
    pass

# Initialize the graph
graph = StateGraph(AgentState)

# Node: Generate response
def generate_response(state):
    llm = ChatOpenAI(model_name="gpt-4")
    response = llm.predict(state["query"])
    state["response"] = response
    return state

graph.add_node("generate_response", generate_response)

# Node: Get human feedback
def collect_feedback(state):
    feedback = get_human_feedback(state["query"], state["response"])
    state["feedback"] = feedback
    save_feedback(state["query"], state["response"], feedback)
    return state

graph.add_node("collect_feedback", collect_feedback)

# Define the workflow
graph.set_entry_point("generate_response")
graph.add_edge("generate_response", "collect_feedback")
graph.add_edge("collect_feedback", END)

# Compile and run
workflow = graph.compile()
initial_state = AgentState({"query": "Explain the theory of relativity."})
workflow.invoke(initial_state)

import json

def get_human_feedback(query, response):
    print(f"Query: {query}")
    print(f"Response: {response}")
    feedback = input("Is the response helpful? (yes/no): ")
    return feedback

def save_feedback(query, response, feedback):
    data = {
        "query": query,
        "response": response,
        "feedback": feedback
    }
    with open("data/feedback.jsonl", "a") as f:
        f.write(json.dumps(data) + "\n")

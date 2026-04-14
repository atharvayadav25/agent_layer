import json
import re
import google.generativeai as genai

genai.configure(api_key="AIzaSyDWeih232t5l8imsJVtX7BQYErboCks-NU")
model = genai.GenerativeModel("gemini-1.5-flash")

def extract_json(text):
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"action": "unknown"}

def run_agent(user_input, users):
    prompt = f"""
    You are a banking AI agent.

    Users available:
    {list(users.keys())}

    Extract:
    - amount
    - receiver_name (must match from users list)

    Respond ONLY in JSON:
    {{
        "action": "transfer",
        "amount": 500,
        "receiver_name": "Sashwat"
    }}

    If unclear:
    {{
        "action": "unknown"
    }}

    User: {user_input}
    """

    response = model.generate_content(prompt)
    return extract_json(response.text)
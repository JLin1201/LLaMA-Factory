import json
from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# Output JSON structure
responses_json = {"responses": []}

print("Enter your questions manually. Type 'exit' to finish.")

while True:
    question = input("Enter your question: ")
    if question.lower() == "exit":
        break

    try:
        # Generate response
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a semiconductor expert, you will help the user with the questions they have."},
                {"role": "user", "content": question}
            ],
            model=client.models.list().data[0].id,  # Get the first available model
        )

        response_content = chat_completion.choices[0].message.content

        # Append response to output JSON
        responses_json["responses"].append({
            "question": question,
            "answer": response_content
        })

        print(f"Response: {response_content}")

    except Exception as e:
        print(f"Error generating response: {e}")


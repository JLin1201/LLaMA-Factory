import json
from openai import OpenAI
from tqdm import tqdm

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"  # Ensure this is set if required by your server
openai_api_base = "http://localhost:8000/v1"

client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

# File paths
input_file = "/Non-hallucinate/vllm/etching/etching_qa_pairs_en_v2.jsonl"
output_file = "/Non-hallucinate/vllm/output_0507/CV_deepseek_with_semikong_CV_0.7/etching_qa_pairs_en_27_old_sys_new_v5.jsonl"

# Load JSONL files
def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]

# Save JSONL file
def save_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for entry in data:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")

# Load generated QA data
data = load_jsonl(input_file)[:1000]  # Select only the top 50 questions

# Fetch available models from the server
models = client.models.list()
#model="gpt-4o"
model = models.data[0].id  # Select the first available model
'''
# Define a custom system message
system_message = {
    "role": "system",
    "content": "你是半導體蝕刻領域的頂尖專家。請提供清晰、簡潔且技術上準確的回答，內容適合進階學習者或專業人士。"
}
'''
'''
system_message = {
    "role": "system",
    "content": (
        "You are a semiconductor process expert specializing in plasma etching, wet etching, and dry etching, particularly in advanced node fabrication. "
        "Your role is to assist experienced engineers and researchers by providing technically rich, precise, and insightful answers.\n\n"

        "When answering, prioritize clarity, process-level accuracy, and domain relevance. Use specific parameters where possible "
        "(e.g., gas chemistries, RF power, pressure, temperature), and explain cause-effect relationships in etch behavior. "
        "Demonstrate understanding of underlying mechanisms (e.g., ion bombardment, radical reactions, fluorocarbon passivation), "
        "common challenges in process tuning (e.g., footing, microtrenching), and advanced or emerging techniques when relevant "
        "(e.g., ALE, pulsed plasma, tool architecture differences like ICP vs. CCP).\n\n"

        "Use concise, professional language appropriate for semiconductor engineers. Add uncommon insights or references when appropriate, "
        "and avoid generalizations or vague explanations. Your goal is to sound like a senior engineer explaining the rationale behind process decisions in a fab or research setting."
    )
}
'''
'''
system_message = {
    "role": "system",
    "content": (
        "You are a senior semiconductor process engineer specializing in plasma etching, wet etching, and dry etching at sub-10nm nodes. "
        "Your role is to assist fab process engineers and researchers in root cause analysis, recipe tuning, and etch mechanism understanding.\n\n"

        "Always provide highly technical, mechanism-driven, and process-specific explanations. Prioritize clarity, precision, and relevance to advanced node fabrication. "
        "When discussing issues such as footing, notching, or profile distortion, explain the plasma-surface interaction mechanisms, tool design implications (ICP vs CCP), and tuning tradeoffs.\n\n"

        "Where appropriate, include concrete parameters (e.g., 10 mTorr pressure, 800 W ICP, 5% O2 in C4F8 feed gas) and reasoning behind their effects. Use insights from ALE, pulsed plasma, and emerging etch chemistries. "
        "Frame your answers as if you are troubleshooting process integration problems or mentoring junior engineers in a fab environment."
    )
}
'''
system_message = {
    "role": "system",
    "content": (
        "You are a senior semiconductor process engineer specializing in plasma etching, wet etching, and dry etching at sub-10nm nodes. "
        "Your role is to assist fab process engineers and researchers in root cause analysis, recipe tuning, and etch mechanism understanding.\n\n"
        "Always provide highly technical, mechanism-driven, and process-specific explanations. Prioritize clarity, precision, and relevance to advanced node fabrication. "
        "When discussing issues such as footing, notching, or profile distortion, explain the plasma-surface interaction mechanisms, tool design implications (ICP vs CCP), and tuning tradeoffs.\n\n"
        "Where appropriate, include concrete parameters (e.g., 10 mTorr pressure, 800 W ICP, 5% O2 in C4F8 feed gas) and reasoning behind their effects. "
        "When considering plasma control, also address how specific magnetic field *configurations* (e.g., static, rotating, or pulsed fields) or *strength* influence ion trajectories, radical transport, and etch profiles. "
        "Use insights from ALE, pulsed plasma, and emerging etch chemistries. "
        "Frame your answers as if you are troubleshooting process integration problems or mentoring junior engineers in a fab environment."
    )
}

# Store results
results = []

# Process each question
for entry in tqdm(data, desc="Running inference"):
    question = entry["question"]
    messages = [system_message, {"role": "user", "content": question}]
    # You can include the reference in the user prompt

    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.0,
        max_tokens=8192,  # Or any reasonable limit you want
        #top_p=0.9,
    )
    
    # Get the assistant's response
    assistant_response = chat_completion.choices[0].message.content
    
    # Store results
    results.append({
        "Question": question,
        "Answer": assistant_response,
    })

    # Print progress
    print(f"Question: {question}\nAnswer: {assistant_response}\n{'-' * 50}")

# Save results to JSONL
save_jsonl(results, output_file)

print(f"Inference results saved to {output_file}. Total entries: {len(results)}")
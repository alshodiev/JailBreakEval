import json
import os
from tqdm import tqdm
from llama_cpp import Llama  # Import llama-cpp-python
from metrics import measure_harmfulness


def load_model(model_path, n_ctx=4096):
    print(f"Loading model from: {model_path}...")
    model = Llama(model_path=model_path, n_ctx=n_ctx)
    print("Model loaded successfully!")
    return model


def load_prompts(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    prompts = [entry['prompt'] for entry in data]
    return prompts


def evaluate_prompt(model, prompt, max_tokens=100):
    output = model(prompt, max_tokens=max_tokens)
    response = output['choices'][0]['text'].strip()
    return response



def evaluate_prompts(model, prompts, output_file="results/evaluation_results.json"):
    results = []
    
    for prompt in tqdm(prompts, desc="Evaluating prompts"):
        response = evaluate_prompt(model, prompt)
        harmfullness_score = measure_harmfulness(response)
        results.append({
            "prompt": prompt,
            "response": response,
            "harmfulness_score": harmfullness_score
        })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}.")
    return results

if __name__ == '__main__':
    model_path = "/Users/alshodiev/.cache/huggingface/hub/models--TheBloke--Mistral-7B-v0.1-GGUF/snapshots/d4ae605152c8de0d6570cf624c083fa57dd0d551/mistral-7b-v0.1.Q4_K_M.gguf"

    # Load the model
    llm = load_model(model_path)

    # Load jailbreak prompts
    json_file_path = "data/jailbreaks/jailbreaks.json"
    prompts = load_prompts(json_file_path)
    print(f"Loaded {len(prompts)} prompts.")

    # Run evaluations
    evaluate_prompts(llm, prompts)

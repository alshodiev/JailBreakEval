from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from llama_cpp import Llama
import os

model_path = "/Users/alshodiev/.cache/huggingface/hub/models--TheBloke--Mistral-7B-v0.1-GGUF/snapshots/d4ae605152c8de0d6570cf624c083fa57dd0d551/mistral-7b-v0.1.Q4_K_M.gguf"


llm = Llama(model_path=model_path, n_ctx=4096)

# Run inference
prompt = "Explain how AI safety works."
output = llm(prompt)
print(output["choices"][0]["text"])


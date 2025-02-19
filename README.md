# JailBreakEval
LLM-generated Jailbreak Evaluation Tool

## Setting Up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/jailbreak-evaluation-tool.git
   cd jailbreak-evaluation-tool
   ```

2. Create a virtual environment
    ```
    python -m venv jailbreak-env
    source jailbreak-env/bin/activate  # On macOS/Linux
    jailbreak-env\Scripts\activate    # On Windows
    ```
3. Install Dependencies
    ```
    pip install -r requirements.txt
    ```
4. Run the project
    ```
    streamlit run src/app.py
    ```
5. To run adversarial test, download model weights of a given model (since my laptop doesn't have GPU, I'm going to just download 1 model) from HuggingFace
    ```
    mistral-7b-v0.1.Q4_K_M.gguf 
    ```
    



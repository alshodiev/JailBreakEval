# JailBreakEval
LLM-generated Jailbreak Evaluation Tool

## Project Structure

```
JailBreakDefense/
│── 📁 data/                      # Store datasets (raw and processed)
│    ├── 📁 jailbreaks/
│        ├── jailbreaks.json
│   ├── red_team_small.json       
    ├── red_team_attempts.jsonl
│   ├── jailbreak_dataset.json    # Processed dataset for RL training
│── 📁 embeddings/                 # Store embeddings
│   ├── X_train.npy
│   ├── y_train.npy
│   ├── X_test.npy
│   ├── y_test.npy
│── 📁 models/                     # Checkpoints and trained models
│   ├── best_xgboost_model.pkl         # Trained XGBoost model
│   ├── rl_agent_checkpoint/      # RL Agent model weights
│── 📁 scripts/                    # Utility scripts
│   ├── extract_embeddings.py     # Extracts embeddings from RoBERTa
│   ├── train_xgboost.py          # Trains XGBoost model
│   ├── train_rl_agent.py         # Trains RL agent using PPO
│   ├── adversarial_filter.py     # Deploys the adversarial filter
    ├── run_aws.sh                 # Shell script to train and evaluate models
│── 📁 notebooks/                  # Jupyter Notebooks for EDA & Testing
│   ├── train_models.ipynb
│── 📁 tests/                      # Unit tests for debugging
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_rl_agent.py
│── 📁 results/                      # Unit tests for debugging
│   ├── evaluation_results.json
│── 📁 configs/                    # Configuration files
│   ├── ppo_config.json            # RL Agent hyperparameters
│── .gitignore                      # Ignore large files and logs
│── requirements.txt                # List of dependencies
│── README.md                       # Project documentation
│── setup.py                        # Setup script for installation

```

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
    



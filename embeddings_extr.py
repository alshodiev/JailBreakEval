import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import os


def parse_args():
    """Parses input arguments."""
    parser = argparse.ArgumentParser(description="Extract embeddings from a Transformer model for harmlessness scoring")
    parser.add_argument("--batch_size", "-bs", type=int, default=16, help="Batch size for DataLoader")
    parser.add_argument("--model_name", "-m", type=str, default="roberta-large-openai-detector",
                        help="HuggingFace model name (e.g., roberta-large-openai-detector, distilbert-base-uncased)")
    return parser.parse_args()


class HarmfulnessDataset(Dataset):
    def __init__(self, transcripts, scores, tokenizer, max_length):
        self.transcripts = transcripts
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.transcripts)

    def __getitem__(self, idx):
        transcript = self.transcripts[idx]
        score = self.scores[idx]

        try:
            encoding = self.tokenizer(transcript,
                                      return_tensors='pt',
                                      max_length=self.max_length,
                                      padding='max_length',
                                      truncation=True)
        except Exception as e:
            print(f"Error tokenizing transcript at index {idx}: {e}")
            raise

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(score, dtype=torch.float)
        }


class TransformerEmbeddingExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        try:
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.last_hidden_state.mean(dim=1)  # Mean-pooling for fixed-size embeddings
        except Exception as e:
            print(f"Error during model forward pass: {e}")
            raise


def extract_embeddings(loader, model, device):
    embeddings, values = [], []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            try:
                print(f"Processing batch {batch_idx + 1}/{len(loader)}...")
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(outputs.cpu().numpy())
                values.append(labels.cpu().numpy())

            except Exception as batch_error:
                print(f"Error in batch {batch_idx + 1}: {batch_error}")
                raise

    return np.vstack(embeddings), np.concatenate(values)


if __name__ == "__main__":
    try:
        args = parse_args()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        save_path = os.getcwd()
        print(f"Files will be saved in: {os.path.join(save_path, "embeddings")}")

        # load the JSON file
        try:
            with open('red_team_small.json', 'r') as f:
                data = json.load(f)
            print("Successfully loaded JSON file.")
        except Exception as json_error:
            print(f"Error loading JSON file: {json_error}")
            raise

        # extract relevant fields
        try:
            transcripts = [entry["transcript"] for entry in data]
            scores = [entry["min_harmlessness_score_transcript"] for entry in data]
            print("Extracted transcripts and harmlessness scores.")
        except KeyError as key_error:
            print(f"Missing expected keys in JSON data: {key_error}")
            raise
        except Exception as extract_error:
            print(f"Error extracting data from JSON: {extract_error}")
            raise

        # load tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            print(f"Loaded tokenizer for model: {args.model_name}")
        except Exception as tokenizer_error:
            print(f"Error loading tokenizer: {tokenizer_error}")
            raise

        # determine dynamic max_length
        try:
            tokenized_lengths = [len(tokenizer.tokenize(text)) for text in transcripts]
            max_length = min(max(tokenized_lengths), tokenizer.model_max_length)  # Ensure it doesn’t exceed model limit
            print(f"Determined max_length: {max_length}")
        except Exception as length_error:
            print(f"Error computing max_length: {length_error}")
            raise

        # train-test split
        try:
            train_texts, test_texts, train_scores, test_scores = train_test_split(
                transcripts, scores, test_size=0.2, random_state=42
            )
            print("Successfully split data into training and testing sets.")
        except Exception as split_error:
            print(f"Error splitting data: {split_error}")
            raise

        # create datasets
        try:
            train_dataset = HarmfulnessDataset(train_texts, train_scores, tokenizer, max_length)
            test_dataset = HarmfulnessDataset(test_texts, test_scores, tokenizer, max_length)

            train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
            print("Created datasets and dataloaders.")
        except Exception as dataset_error:
            print(f"Error creating datasets: {dataset_error}")
            raise

        # load the transformer model
        try:
            print("Loading Transformer model...")
            pretrained_model = AutoModel.from_pretrained(args.model_name)
            embedding_extractor = TransformerEmbeddingExtractor(pretrained_model)
            print(f"Successfully loaded model: {args.model_name}")
        except Exception as model_error:
            print(f"Error loading model: {model_error}")
            raise

        # extract embeddings
        try:
            print("Extracting embeddings for training data...")
            X_train, y_train = extract_embeddings(train_loader, embedding_extractor, device)
            print("Training embeddings extracted successfully.")

            print("Extracting embeddings for test data...")
            X_test, y_test = extract_embeddings(test_loader, embedding_extractor, device)
            print("Test embeddings extracted successfully.")
        except Exception as embedding_error:
            print(f"Error extracting embeddings: {embedding_error}")
            raise

        # save embeddings
        try:
            print(f"Saving outputs to {save_path}...")
            np.save(os.path.join(save_path, 'X_train.npy'), X_train)
            np.save(os.path.join(save_path, 'y_train.npy'), y_train)
            np.save(os.path.join(save_path, 'X_test.npy'), X_test)
            np.save(os.path.join(save_path, 'y_test.npy'), y_test)
            print("Embeddings and labels saved successfully.")
        except Exception as save_error:
            print(f"Error saving outputs: {save_error}")
            raise

    except Exception as e:
        print(f"An error occurred during execution: {e}")
        raise

import os
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import snapshot_download

class EmbeddingModel:
    def __init__(self, model_name="optimum/all-MiniLM-L6-v2"):
        """
        Loads the ONNX model and tokenizer.
        """
        print("Loading ONNX model...")
        
        # Define model path - use a persistent location or tmp
        self.model_path = os.path.join(os.getcwd(), "model_onnx")
        
        # Download if not exists
        if not os.path.exists(os.path.join(self.model_path, "model.onnx")):
            print(f"Downloading model {model_name} to {self.model_path}...")
            snapshot_download(repo_id=model_name, local_dir=self.model_path, 
                              allow_patterns=["*.onnx", "*.json", "*.txt"])
        
        # Load Tokenizer
        self.tokenizer = Tokenizer.from_file(os.path.join(self.model_path, "tokenizer.json"))
        
        # Load ONNX Session
        self.session = ort.InferenceSession(os.path.join(self.model_path, "model.onnx"))
        print("ONNX model loaded successfully.")

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = np.expand_dims(attention_mask, -1).repeat(token_embeddings.shape[-1], axis=-1)
        sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
        sum_mask = np.clip(input_mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def encode(self, texts):
        """
        Encodes a list of texts into embeddings using ONNX.
        """
        if isinstance(texts, str):
            texts = [texts]
            
        # Tokenize
        encoded_input = self.tokenizer.encode_batch(texts)
        
        # Prepare inputs for ONNX
        input_ids = np.array([e.ids for e in encoded_input], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded_input], dtype=np.int64)
        token_type_ids = np.array([e.type_ids for e in encoded_input], dtype=np.int64)
        
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids
        }
        
        # Run inference
        output = self.session.run(None, ort_inputs)
        
        # Mean Pooling - simplified for this model
        # Note: optimum models usually output last_hidden_state as first element
        embeddings = self.mean_pooling(output, attention_mask)
        
        # Normalize
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norm

    def get_score(self, user_text: str, ideal_text: str) -> float:
        """
        Computes cosine similarity between user answer and ideal answer.
        Returns score between 0 and 1.
        """
        embeddings = self.encode([user_text, ideal_text])
        user_emb, ideal_emb = embeddings
        
        score = cosine_similarity([user_emb], [ideal_emb])[0][0]
        return round(float(score), 2)

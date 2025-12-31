from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Loads the SBERT model once during initialization.
        """
        print("Loading SBERT model...")
        self.model = SentenceTransformer(model_name)
        print("SBERT model loaded successfully.")

    def get_score(self, user_text: str, ideal_text: str) -> float:
        """
        Computes cosine similarity between user answer and ideal answer.
        Returns score between 0 and 1.
        """
        embeddings = self.model.encode(
            [user_text, ideal_text],
            normalize_embeddings=True  # improves cosine stability
        )

        user_emb, ideal_emb = embeddings

        score = cosine_similarity(
            [user_emb], [ideal_emb]
        )[0][0]

        return round(float(score), 2)

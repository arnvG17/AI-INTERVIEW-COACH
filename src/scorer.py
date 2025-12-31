from .features import (
    semantic_similarity_score,
    keyword_coverage_score,
    star_structure_score,
    answer_length_score,
    redundancy_score
)

def final_scoring(user_text, ideal_text, keywords, embedding_model):
    # Feature-wise scores
    features = {
        "semantic_similarity": semantic_similarity_score(user_text, ideal_text, embedding_model),
        "keywords": keyword_coverage_score(user_text, keywords, embedding_model),
        "STAR": star_structure_score(user_text, embedding_model),
        "length": answer_length_score(user_text),
        "redundancy": redundancy_score(user_text)
    }

    # Weighted scoring - Favoring Semantic & Structure over Keywords
    weights = {
        "semantic_similarity": 0.45,  # Increased from 0.4
        "keywords": 0.15,             # Decreased from 0.2
        "STAR": 0.25,                 # Kept high, now semantic
        "length": 0.1,
        "redundancy": 0.05
    }

    final_score = sum(features[f]["score"] * weights[f] for f in features)
    return {
        "final_score": round(final_score),
        "explanation": features
    }

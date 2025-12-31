from typing import List, Dict
import spacy
import re
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load Spacy model with fallback
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Spacy model 'en_core_web_sm' not found. Please run 'python -m spacy download en_core_web_sm'")
    # Fallback to a blank model to prevent crash during import, though functionality will be degraded
    nlp = spacy.blank("en")

def preprocess_text(text: str) -> List[str]:
    """Lemmatize and remove stop words/punctuation"""
    doc = nlp(text.lower())
    return [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

def semantic_similarity_score(user_text: str, ideal_text: str, embedding_model) -> Dict:
    score = embedding_model.get_score(user_text, ideal_text)
    reason = f"Cosine similarity between user and ideal answer: {score}"
    return {"score": round(score*100), "reason": reason}

def keyword_coverage_score(user_text: str, keywords: List[str], embedding_model=None) -> Dict:
    if not keywords:
        return {"score": 0, "reason": "No keywords provided"}

    # Lemmatize user text
    user_lemmas = set(preprocess_text(user_text))
    user_text_lower = user_text.lower()
    
    matched = []
    semantic_matches = []
    
    keywords_lemmatized = {kw: preprocess_text(kw) for kw in keywords}
    
    score_count = 0

    for kw, kw_lemmas in keywords_lemmatized.items():
        matched_this = False
        
        # Strategy 1: Check if ANY lemma from the keyword appears in user text
        if any(lemma in user_lemmas for lemma in kw_lemmas):
            matched.append(kw)
            score_count += 1
            matched_this = True
            continue
            
        # Strategy 2: Check if the full keyword phrase appears as substring
        if kw.lower() in user_text_lower:
            matched.append(kw)
            score_count += 1
            matched_this = True
            continue
            
        # Strategy 3: AGGRESSIVE SEMANTIC MATCHING
        # Split user text into sentences and check each sentence against the keyword
        # This handles verb→noun mappings like "prioritized" → "prioritization"
        if embedding_model and not matched_this:
            doc = nlp(user_text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
            
            if sentences:
                # Encode keyword and all sentences
                kw_embedding = embedding_model.encode([kw])[0]
                sent_embeddings = embedding_model.encode(sentences)
                
                # Find max similarity across all sentences
                similarities = cosine_similarity([kw_embedding], sent_embeddings)[0]
                max_sim = np.max(similarities)
                
                # Lower threshold for better detection
                if max_sim > 0.30:
                    semantic_matches.append(kw)
                    score_count += 0.9  # High partial credit for semantic match
                    continue
                
    total_score_norm = min(score_count / len(keywords), 1.0)
    
    reason_parts = []
    if matched: reason_parts.append(f"Matched: {matched}")
    if semantic_matches: reason_parts.append(f"Semantic matches: {semantic_matches}")
    
    reason = "; ".join(reason_parts) if reason_parts else "No significant keywords found"
    
    return {"score": round(total_score_norm*100), "reason": reason}

def star_structure_score(user_text: str, embedding_model=None) -> Dict:
    """
    Detect STAR structure using context-aware semantic anchors.
    Checks if the ENTIRE answer contains elements related to each component.
    """
    # Split into sentences for granular analysis
    doc = nlp(user_text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5]
    
    if not sentences:
        return {"score": 0, "reason": "No text provided"}

    components = {"Situation": 0, "Task": 0, "Action": 0, "Result": 0}
    
    # Expanded Semantic Anchors (more comprehensive)
    anchors = {
        "Situation": [
            "situation", "context", "background", "when I was", "the problem was", 
            "faced a challenge", "time constraint", "deadline", "given a task",
            "assigned to", "working on", "during", "at the time"
        ],
        "Task": [
            "task", "goal", "objective", "needed to", "required to", "had to",
            "my responsibility", "deliver", "ensure", "achieve"
        ],
        "Action": [
            "action", "I did", "I implemented", "I started", "steps I took", "decided to",
            "I broke", "I prioritized", "I coordinated", "I created", "I analyzed",
            "I worked", "I collaborated", "I developed"
        ],
        "Result": [
            "result", "outcome", "consequence", "achieved", "learned", "improved", 
            "revenue", "delivered", "completed", "success", "impact", "on time"
        ]
    }

    detected_components = []
    
    if embedding_model:
        # Encode all sentences once
        sent_embeddings = embedding_model.encode(sentences)
        
        for comp, phrases in anchors.items():
            # Encode anchors for this component
            anchor_embeddings = embedding_model.encode(phrases)
            
            # Compute similarity matrix: (num_sentences x num_anchors)
            sim_matrix = cosine_similarity(sent_embeddings, anchor_embeddings)
            
            # Check if ANY sentence has high similarity to ANY anchor
            max_sim = np.max(sim_matrix)
            
            # Lowered threshold for better detection
            if max_sim > 0.28:
                components[comp] = 1
                detected_components.append(comp)
    else:
        # Fallback to string matching
        lower_text = user_text.lower()
        for comp, phrases in anchors.items():
            if any(p in lower_text for p in phrases):
                components[comp] = 1
                detected_components.append(comp)

    score = sum(components.values()) / 4.0
    reason = f"STAR structure found: {', '.join(detected_components)}" if detected_components else "No clear STAR structure detected"
    return {"score": round(score*100), "reason": reason}

def answer_length_score(user_text: str, min_len=40, max_len=400) -> Dict:
    length = len(user_text.split())
    if length < min_len:
        score = 0.5
        reason = f"Answer too short ({length} words), aim for {min_len}+"
    elif length > max_len:
        score = 0.9
        reason = f"Answer detailed ({length} words)"
    else:
        score = 1.0
        reason = f"Answer length optimal ({length} words)"
    return {"score": round(score*100), "reason": reason}

def redundancy_score(user_text: str) -> Dict:
    # Use lemma uniqueness instead of exact word uniqueness
    lemmas = preprocess_text(user_text)
    if not lemmas:
        return {"score": 100, "reason": "No content"}
        
    unique_lemmas = set(lemmas)
    ratio = len(unique_lemmas) / len(lemmas)
    
    # Penalize only very low ratios (repetitive)
    # Typical ratio is 0.5-0.7 for normal text. Below 0.3 is suspicious.
    
    score = 1.0
    if ratio < 0.35:
        score = 0.5
        reason = f"High repetition detected (diversity index: {ratio:.2f})"
    else:
        reason = f"Vocabulary diversity is good (index: {ratio:.2f})"
        
    return {"score": round(score*100), "reason": reason}

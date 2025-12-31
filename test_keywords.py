from src.embeddings import EmbeddingModel
from src.features import keyword_coverage_score

# Initialize model
model = EmbeddingModel()

# Test case from user's example
user_answer = """I was given a data analysis task with a 48-hour deadline. 
I broke the work into milestones, prioritized key metrics, and coordinated quick reviews. 
I delivered on time, and the insights were used in the final presentation."""

keywords = ["prioritization", "time management", "planning", "execution", "results"]

print("Testing Keyword Matching:")
print(f"User Answer: {user_answer}")
print(f"Keywords: {keywords}")
print()

result = keyword_coverage_score(user_answer, keywords, model)

print(f"Score: {result['score']}%")
print(f"Reason: {result['reason']}")
print()
print("Expected matches:")
print("- 'prioritization' should match 'prioritized' (lemma)")
print("- 'planning' should match 'milestones' (semantic) or direct lemma")
print("- 'execution' should match 'coordinated' (semantic)")
print("- 'time management' should match 'delivered on time' (semantic)")
print("- 'results' should match 'results' (exact)")

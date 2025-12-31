from src.embeddings import EmbeddingModel
from src.scorer import final_scoring

# Initialize model
print("Loading model...")
model = EmbeddingModel()

# User's actual answer from the problem description
user_answer = """I was given a data analysis task with a 48-hour deadline. 
I broke the work into milestones, prioritized key metrics, and coordinated quick reviews. 
I delivered on time, and the insights were used in the final presentation."""

ideal_answer = """Situation: I was given a data analysis task with a 48-hour deadline. 
Task: Deliver accurate insights without compromising quality. 
Action: I broke the work into milestones, prioritized key metrics, and coordinated quick reviews with stakeholders. 
Result: I delivered on time, and the insights were used in the final presentation."""

keywords = ["prioritization", "time management", "planning", "execution", "results"]

print("\n" + "="*60)
print("TESTING IMPROVED SEMANTIC EVALUATION")
print("="*60)
print(f"\nUser Answer:\n{user_answer}")
print(f"\nKeywords to match: {keywords}")
print("\n" + "-"*60)

result = final_scoring(user_answer, ideal_answer, keywords, model)

print(f"\n🎯 FINAL SCORE: {result['final_score']}/100")
print("\n📊 Feature-wise Breakdown:")
for feature, data in result['explanation'].items():
    print(f"  • {feature.upper()}: {data['score']}% - {data['reason']}")

print("\n" + "="*60)
print("EXPECTED IMPROVEMENTS:")
print("="*60)
print("✓ 'prioritization' should match 'prioritized' (verb→noun)")
print("✓ 'planning' should match 'broke work into milestones' (semantic)")
print("✓ 'execution' should match 'coordinated' (semantic)")
print("✓ 'time management' should match 'delivered on time' (semantic)")
print("✓ STAR should detect all 4 components (Situation has 'deadline')")

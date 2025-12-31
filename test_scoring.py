from src.embeddings import EmbeddingModel
from src.scorer import final_scoring
import time

def run_tests():
    print("Loading model...")
    s = time.time()
    model = EmbeddingModel()
    print(f"Model loaded in {time.time() - s:.2f}s")
    
    keywords = ["collaboration", "conflict resolution", "communication", "deadline"]
    ideal_answer = "I handle conflict by communicating openly with the team, identifying the root cause, and finding a compromise that meets our deadlines."
    
    # Case 1: Keyword Stuffing (Robot)
    # Contains all keywords but no structure or natural flow.
    robot_answer = "Collaboration conflict resolution communication deadline. Situation task action result. I did these things."
    
    # Case 2: Natural Semantic Answer (Human)
    # Uses synonyms and implicit structure.
    # Situation: "We faced a tight schedule" (deadline)
    # Task: "needed to align the team"
    # Action: "I spoke with everyone individually" (communication)
    # Result: "we delivered on time"
    human_answer = "In my last project, we faced a tight schedule where team members disagreed on the approach. My goal was to align everyone. I spoke with everyone individually to understand their concerns and facilitated a meeting to reach a consensus. As a result, we delivered the project on time and improved our working relationship."
    
    print("\n--- TEST: Keyword Stuffing vs Natural Semantic ---")
    
    print(f"\nIdeal Answer: {ideal_answer}")
    print(f"Keywords: {keywords}")
    
    print("\n1. Scoring Robot Answer:")
    score_robot = final_scoring(robot_answer, ideal_answer, keywords, model)
    print(score_robot)
    
    print("\n2. Scoring Human Answer:")
    score_human = final_scoring(human_answer, ideal_answer, keywords, model)
    print(score_human)
    
    print("\n--- RESULTS ---")
    print(f"Robot Score: {score_robot['final_score']}")
    print(f"Human Score: {score_human['final_score']}")
    
    if score_human['final_score'] >= score_robot['final_score']:
        print("SUCCESS: Human answer scored higher or equal to keyword stuffing.")
    else:
        print("FAILURE: Robot answer outscored human answer.")

    # Check nuances
    print("\nHuman Breakdown:")
    print(f"STAR Score: {score_human['explanation']['STAR']['score']} (Reason: {score_human['explanation']['STAR']['reason']})")
    print(f"Keywords Score: {score_human['explanation']['keywords']['score']} (Reason: {score_human['explanation']['keywords']['reason']})")

if __name__ == "__main__":
    run_tests()

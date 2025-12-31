from src.embeddings import EmbeddingModel
from src.data_loader import load_questions


def main():
    print("Initializing AI Interview Coach...")
    ai_engine = EmbeddingModel()

    print("\nLoading questions...")
    questions = load_questions()

    print("\n--- TEST RUN ---")
    selected_q = questions[0]
    print(f"Question: {selected_q['question']}")

    user_answer = (
        "I had to deliver a project in 48 hours. "
        "I prioritized tasks, communicated with the team, "
        "and successfully completed it on time."
    )

    score = ai_engine.get_score(
        user_answer,
        selected_q["ideal_answer"]
    )

    print(f"\nUser Answer: {user_answer}")
    print(f"AI Score: {score}")

    if score > 0.7:
        print("Feedback: Great job! Detailed and relevant.")
    elif score > 0.4:
        print("Feedback: Good start, but try to add more STAR details.")
    else:
        print("Feedback: Your answer seems off-topic.")


if __name__ == "__main__":
    main()

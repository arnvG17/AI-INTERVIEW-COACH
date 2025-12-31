from flask import Flask, render_template, request, jsonify
from src.embeddings import EmbeddingModel
from src.scorer import final_scoring
from src.data_loader import load_questions

app = Flask(__name__)

# Initialize embedding model
embedding_model = EmbeddingModel()
questions_data = load_questions()

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", questions=questions_data)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    user_answer = request.form.get("answer", "")
    question_id = int(request.form.get("question_id", 0))
    
    question = questions_data[question_id]
    ideal_answer = question["ideal_answer"]
    keywords = question.get("keywords", [])

    result = final_scoring(user_answer, ideal_answer, keywords, embedding_model)
    return render_template("index.html", questions=questions_data,
                           selected_question_id=question_id,
                           user_answer=user_answer,
                           result=result)

if __name__ == "__main__":
    app.run(debug=True)

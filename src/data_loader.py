import json
from pathlib import Path


def load_questions(data_path: str = "data/questions.json"):
    """
    Loads interview questions from a JSON file.
    """
    path = Path(data_path)

    if not path.exists():
        raise FileNotFoundError(f"Questions file not found at {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

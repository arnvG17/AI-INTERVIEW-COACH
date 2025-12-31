import sys
import os

print(f"Python Executable: {sys.executable}")
print(f"CWD: {os.getcwd()}")

try:
    import spacy
    print(f"Spacy Version: {spacy.__version__}")
    print(f"Spacy Location: {os.path.dirname(spacy.__file__)}")
    try:
        nlp = spacy.load("en_core_web_sm")
        print("Model 'en_core_web_sm' loaded successfully.")
    except Exception as e:
        print(f"Model load failed: {e}")
except ImportError:
    print("Spacy NOT found.")
except Exception as e:
    print(f"Error importing spacy: {e}")

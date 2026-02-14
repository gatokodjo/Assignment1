# Transition-Based Dependency Parser (Python)

This project implements a **transition-based dependency parser** that parses English sentences into **CoNLL-style dependency graphs** using a trained `TransitionParser`. It supports both **file input** and **stdin input**, and leverages **spaCy** to initialize sentences for parsing.

---

## Requirements

- Python 3.12 (or compatible)
- pip packages:
  - `numpy`
  - `scipy`
  - `scikit-learn >= 1.2`
  - `joblib`
  - `spacy`
  - `nltk`

- SpaCy model: `en_core_web_sm` (automatically downloaded if missing)

---

## File Structure

Assignment1/
├── code/
│ ├── parse_sentences.py # Main script for parsing sentences
│ ├── badfeaturesmodel.joblib # Trained TransitionParser model
│ ├── providedcode/ # Folder containing dependency parser code
│ │ ├── dependencygraph.py
│ │ ├── transitionparser.py
│ │ ├── featureextractor.py
│ │ └── transition.py
│ └── my_text.txt # Example text input (optional)
├── README.md
└── submission.zip # Zip for submission

---

## Usage

### From a text file

```bash
python parse_sentences.py <model_file> <input_file> > output.conll

python parse_sentences.py badfeaturesmodel.joblib my_text.txt > parsed.conll


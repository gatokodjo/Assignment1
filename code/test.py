# test.py
import random
from featureextractor import FeatureExtractor
from providedcode import dataset
from providedcode.evaluate import DependencyEvaluator
from providedcode.transitionparser import TransitionParser
from transition import Transition
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # ---------------- Load corpus ----------------
    all_train_data = dataset.get_danish_train_corpus().parsed_sents()
    all_test_data = dataset.get_danish_test_corpus().parsed_sents()

    print(f"Total training sentences available: {len(all_train_data)}")
    print(f"Total test sentences available: {len(all_test_data)}")

    # ---------------- Subsample for practical runtime ----------------
    # Use up to 1000 training sentences (adjust if you want larger)
    train_data = all_train_data[:2000]
    # Use up to 200 test sentences
    test_data = all_test_data[:400]

    print(f"Training sentences: {len(train_data)}")
    print(f"Test sentences: {len(test_data)}")

    try:
        # ---------------- Train parser ----------------
        tp = TransitionParser(Transition, FeatureExtractor, classifier="sgd")
        tp.train(train_data)
        tp.save("dan_model.joblib")
        print("Model saved to dan_model.joblib")

        # ---------------- Load parser ----------------
        tp = TransitionParser.load("dan_model.joblib", Transition, FeatureExtractor)

        # ---------------- Parse test data ----------------
        parsed = tp.parse(test_data)

        # ---------------- Write CoNLL output ----------------
        with open("test.conll", "w", encoding="utf-8") as f:
            for p in parsed:
                f.write(p.to_conll(10))
                f.write("\n")

        # ---------------- Evaluate ----------------
        ev = DependencyEvaluator(test_data, parsed)
        las, uas = ev.eval()
        print(f"LAS: {las:.4f} \nUAS: {uas:.4f}")

    except NotImplementedError as e:
        print(f"NotImplementedError encountered: {e}")

"""
Arc-Eager Transition-Based Dependency Parser (Modernized)
This module implements an arc-eager transition-based dependency parser using an SGD classifier.
Works with Python 3.12 and scikit-learn >= 1.2.
Author: Modernized version by Gato
"""
import copy
import os
import tempfile
from typing import List, Tuple, Optional

import numpy as np
from scipy import sparse
from sklearn.datasets import load_svmlight_file
from sklearn.linear_model import SGDClassifier


class Configuration:
    """Parser state holding stack, buffer, and arcs."""

    def __init__(self, dep_graph, feature_extractor):
        self.stack = [0]  # Root element
        self.buffer = list(range(1, len(dep_graph.nodes)))
        self.arcs = []
        self._tokens = dep_graph.nodes
        self._user_feature_extractor = feature_extractor

    def __str__(self):
        return f"Stack: {self.stack} Buffer: {self.buffer} Arcs: {self.arcs}"

    def extract_features(self):
        return self._user_feature_extractor(self._tokens, self.buffer, self.stack, self.arcs)


class TransitionParser:
    """Arc-eager transition-based parser with linear classifier."""

    # ---------------- Initialization ----------------
    def __init__(self, transition, feature_extractor, classifier="sgd"):
        self._dictionary = {}
        self._transition = {}
        self._match_transition = {}
        self._model = None
        self._user_feature_extractor = feature_extractor
        self.transitions = transition
        self._clf_type = classifier

    # ---------------- Feature Processing ----------------
    def _convert_to_binary_features(self, features: List[str]) -> str:
        indices = []
        for feat in features:
            self._dictionary.setdefault(feat, len(self._dictionary))
            indices.append(self._dictionary[feat])
        return " ".join(f"{i}:1.0" for i in sorted(indices))

    def _feature_row_sparse(self, features: List[str]):
        cols = [self._dictionary[f] for f in features if f in self._dictionary]
        if not cols:
            return sparse.csr_matrix((1, len(self._dictionary)), dtype=np.float64)

        cols = np.array(sorted(cols), dtype=np.int32)
        data = np.ones_like(cols, dtype=np.float64)
        rows = np.zeros_like(cols, dtype=np.int32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(1, len(self._dictionary)), dtype=np.float64)

    # ---------------- Projectivity ----------------
    @staticmethod
    def _is_projective(depgraph) -> bool:
        arc_list = set()
        for node in depgraph.nodes.values():
            if "head" in node:
                arc_list.add((node["head"], node["address"]))

        for parent, child in arc_list:
            if child > parent:
                child, parent = parent, child
            for k in range(child + 1, parent):
                for m in range(len(depgraph.nodes)):
                    if (k, m) in arc_list or (m, k) in arc_list:
                        return False
        return True

    # ---------------- Training ----------------
    def _write_to_file(self, key: str, binary_features: str, input_file):
        self._transition.setdefault(key, len(self._transition) + 1)
        self._match_transition[self._transition[key]] = key
        input_file.write(f"{self._transition[key]} {binary_features}\n".encode("utf-8"))

    def _create_training_examples_arc_eager(self, depgraphs, input_file):
        training_seq = []
        projective_graphs = [dg for dg in depgraphs if self._is_projective(dg)]

        for depgraph in projective_graphs:
            conf = Configuration(depgraph, self._user_feature_extractor.extract_features)

            while conf.buffer:
                b0 = conf.buffer[0]
                features = conf.extract_features()
                binary_features = self._convert_to_binary_features(features)

                s0 = conf.stack[-1] if conf.stack else None

                # Left-arc
                rel = self._get_dep_relation(b0, s0, depgraph) if s0 is not None else None
                if rel is not None:
                    key = self.transitions.LEFT_ARC + ":" + rel
                    self._write_to_file(key, binary_features, input_file)
                    self.transitions.left_arc(conf, rel)
                    training_seq.append(key)
                    continue

                # Right-arc
                rel = self._get_dep_relation(s0, b0, depgraph) if s0 is not None else None
                if rel is not None:
                    key = self.transitions.RIGHT_ARC + ":" + rel
                    self._write_to_file(key, binary_features, input_file)
                    self.transitions.right_arc(conf, rel)
                    training_seq.append(key)
                    continue

                # Reduce
                if s0 is not None and self._can_reduce(Configuration(depgraph, self._user_feature_extractor.extract_features)):
                    key = self.transitions.REDUCE
                    self._write_to_file(key, binary_features, input_file)
                    self.transitions.reduce(conf)
                    training_seq.append(key)
                    continue

                # Default: shift
                key = self.transitions.SHIFT
                self._write_to_file(key, binary_features, input_file)
                self.transitions.shift(conf)
                training_seq.append(key)

        print(f"Number of sentences: {len(depgraphs)}")
        print(f"Number of valid (projective) sentences: {len(projective_graphs)}")
        print(f"Number of transition instances: {len(training_seq)}")
        return training_seq

    def train(self, depgraphs):
        with tempfile.NamedTemporaryFile(prefix="train", delete=False) as tf:
            self._create_training_examples_arc_eager(depgraphs, tf)
            temp_name = tf.name

        x_train, y_train = load_svmlight_file(temp_name)
        y_train = y_train.astype(int)

        self._model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=2000,
            tol=1e-4,
            random_state=42,
        )
        print("Training classifier (SGD, log_loss)...")
        self._model.fit(x_train, y_train)
        print("Training complete.")

        if os.path.exists(temp_name):
            os.remove(temp_name)

    # ---------------- Parsing ----------------
    def parse(self, depgraphs):
        if not self._model:
            raise ValueError("No model trained!")

        results = []
        class_index_to_key = {i: self._match_transition[c] for i, c in enumerate(self._model.classes_)}

        for gold_graph in depgraphs:
            depgraph = copy.deepcopy(gold_graph)
            conf = Configuration(depgraph, self._user_feature_extractor.extract_features)

            safety = 5 * len(depgraph.nodes) + 10
            while (conf.buffer or len(conf.stack) > 1) and safety > 0:
                safety -= 1
                feats = conf.extract_features()
                x_test = self._feature_row_sparse(feats)

                scores = self._model.predict_proba(x_test)[0]
                ranked = np.argsort(-scores)

                applied = False
                for idx in ranked:
                    key = class_index_to_key.get(idx)
                    if key is None:
                        continue
                    action, rel = self._decode_key(key)
                    if self._is_action_valid(conf, action):
                        self._apply_action(conf, action, rel)
                        applied = True
                        break

                if not applied:
                    if self._can_shift(conf):
                        self.transitions.shift(conf)
                    elif self._can_reduce(conf):
                        self.transitions.reduce(conf)
                    else:
                        break

            self._finalize_graph_with_arcs(depgraph, conf.arcs)
            results.append(depgraph)

        return results

    # ---------------- Helper Methods ----------------
    def _decode_key(self, key: str) -> Tuple[str, Optional[str]]:
        if ":" in key:
            action, rel = key.split(":", 1)
        else:
            action, rel = key, None
        return action, rel

    def _apply_action(self, conf: Configuration, action: str, rel: Optional[str]):
        if action == self.transitions.LEFT_ARC:
            self.transitions.left_arc(conf, rel)
        elif action == self.transitions.RIGHT_ARC:
            self.transitions.right_arc(conf, rel)
        elif action == self.transitions.SHIFT:
            self.transitions.shift(conf)
        elif action == self.transitions.REDUCE:
            self.transitions.reduce(conf)

    def _is_action_valid(self, conf: Configuration, action: str) -> bool:
        if action == self.transitions.LEFT_ARC:
            return bool(conf.stack and conf.buffer)
        elif action == self.transitions.RIGHT_ARC:
            return bool(conf.stack and conf.buffer)
        elif action == self.transitions.SHIFT:
            return bool(conf.buffer)
        elif action == self.transitions.REDUCE:
            return bool(conf.stack)
        return False

    def _can_shift(self, conf: Configuration) -> bool:
        return bool(conf.buffer)

    def _can_reduce(self, conf: Configuration) -> bool:
        return bool(conf.stack)

    def _finalize_graph_with_arcs(self, depgraph, arcs: List[Tuple[int, str, int]]):
        for head, rel, dep in arcs:
            depgraph.nodes[head]["deps"].setdefault(rel, [])
            depgraph.nodes[head]["deps"][rel].append(dep)
            depgraph.nodes[dep]["head"] = head
            depgraph.nodes[dep]["rel"] = rel

    def _get_dep_relation(self, head_idx, dep_idx, depgraph) -> Optional[str]:
        dep_node = depgraph.get_by_address(dep_idx)
        if dep_node["head"] == head_idx:
            return dep_node["rel"]
        return None

    # ---------------- Save / Load ----------------
    def save(self, filepath):
        from joblib import dump
        dump(
            {
                "model": self._model,
                "dictionary": self._dictionary,
                "transition": self._transition,
                "match_transition": self._match_transition,
            },
            filepath,
            compress=3,
        )
        print(f"Model bundle saved to {filepath}")

    @staticmethod
    def load(filepath, transition_class, feature_extractor_class):
        from joblib import load as joblib_load
        bundle = joblib_load(filepath)
        tp = TransitionParser(transition_class, feature_extractor_class, classifier="sgd")
        tp._model = bundle["model"]
        tp._dictionary = bundle["dictionary"]
        tp._transition = bundle["transition"]
        tp._match_transition = bundle["match_transition"]
        return tp

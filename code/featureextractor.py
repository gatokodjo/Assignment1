from collections import defaultdict
from itertools import chain

class FeatureExtractor:
    """Feature extractor for arc-eager transition-based dependency parser."""

    @staticmethod
    def _check_informative(feat, underscore_is_informative=False):
        """Check whether a feature is informative (non-empty and optionally not '_')."""
        if feat is None or feat == "":
            return False
        if not underscore_is_informative and feat == "_":
            return False
        return True

    @staticmethod
    def find_left_right_dependencies(idx, arcs):
        """
        Returns left-most and right-most dependency relation for a node.
        arcs: list of (head, rel, dep)
        """
        left_most = 10**6
        right_most = -1
        dep_left_most = ""
        dep_right_most = ""
        for wi, r, wj in arcs:
            if wi == idx:
                if (wj > wi) and (wj > right_most):
                    right_most = wj
                    dep_right_most = r
                if (wj < wi) and (wj < left_most):
                    left_most = wj
                    dep_left_most = r
        return dep_left_most, dep_right_most

    @staticmethod
    def extract_features(tokens, buffer, stack, arcs):
        """
        Extract features for the classifier.
        Returns a list of string features.
        """
        result = []

        # ---------------- Stack features ----------------
        if stack:
            stk0 = tokens[stack[-1]]
            if FeatureExtractor._check_informative(stk0.get("word"), True):
                result.append("STK_0_WORD_" + stk0["word"])
            if FeatureExtractor._check_informative(stk0.get("tag"), True):
                result.append("STK_0_POS_" + stk0["tag"])
        if len(stack) > 1:
            stk1 = tokens[stack[-2]]
            if FeatureExtractor._check_informative(stk1.get("word"), True):
                result.append("STK_1_WORD_" + stk1["word"])
            if FeatureExtractor._check_informative(stk1.get("tag"), True):
                result.append("STK_1_POS_" + stk1["tag"])

        # ---------------- Buffer features ----------------
        if buffer:
            buf0 = tokens[buffer[0]]
            if FeatureExtractor._check_informative(buf0.get("word"), True):
                result.append("BUF_0_WORD_" + buf0["word"])
            if FeatureExtractor._check_informative(buf0.get("tag"), True):
                result.append("BUF_0_POS_" + buf0["tag"])
        if len(buffer) > 1:
            buf1 = tokens[buffer[1]]
            if FeatureExtractor._check_informative(buf1.get("word"), True):
                result.append("BUF_1_WORD_" + buf1["word"])
            if FeatureExtractor._check_informative(buf1.get("tag"), True):
                result.append("BUF_1_POS_" + buf1["tag"])

        # ---------------- Dependency features ----------------
        if stack:
            left_dep, right_dep = FeatureExtractor.find_left_right_dependencies(stack[-1], arcs)
            if FeatureExtractor._check_informative(left_dep):
                result.append("STK_0_LEFT_DEP_" + left_dep)
            if FeatureExtractor._check_informative(right_dep):
                result.append("STK_0_RIGHT_DEP_" + right_dep)
        if len(stack) > 1:
            left_dep, right_dep = FeatureExtractor.find_left_right_dependencies(stack[-2], arcs)
            if FeatureExtractor._check_informative(left_dep):
                result.append("STK_1_LEFT_DEP_" + left_dep)
            if FeatureExtractor._check_informative(right_dep):
                result.append("STK_1_RIGHT_DEP_" + right_dep)

        # ---------------- Structural features ----------------
        result.append("STACK_SIZE_" + str(len(stack)))
        result.append("BUFFER_SIZE_" + str(len(buffer)))

        # ---------------- Interaction / Distance features ----------------
        if stack and buffer:
            stk0 = tokens[stack[-1]]
            buf0 = tokens[buffer[0]]
            # POS bigram
            if FeatureExtractor._check_informative(stk0.get("tag")) and FeatureExtractor._check_informative(buf0.get("tag")):
                result.append("STK_0_POS_BUF_0_POS_" + stk0["tag"] + "_" + buf0["tag"])
            # Distance between stack0 and buffer0
            dist = abs(stack[-1] - buffer[0])
            result.append("STK_0_BUF_0_DIST_" + str(dist))

        return result

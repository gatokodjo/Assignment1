# evaluate.py
# Natural Language Toolkit: evaluation of dependency parser
# Author: Long Duong <longdt219@gmail.com>
# Adapted and fixed for Assignment 1

from __future__ import absolute_import, division
import unicodedata


class DependencyEvaluator(object):
    """
    Class for measuring labelled and unlabelled attachment score for
    dependency parsing. Evaluation ignores punctuation.
    """

    def __init__(self, parsed_sents, gold_sents):
        """
        :param parsed_sents: list of parsed sentences (DependencyGraph objects)
        :param gold_sents: list of gold standard sentences (DependencyGraph objects)
        """
        self._parsed_sents = parsed_sents
        self._gold_sents = gold_sents

    def _remove_punct(self, inStr):
        """
        Remove punctuation from a string.
        """
        punc_cat = set(["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"])
        return "".join(x for x in inStr if unicodedata.category(x) not in punc_cat)

    def eval(self):
        """
        Compute LAS and UAS.

        :return: (UAS, LAS) tuple as floats
        """
        if len(self._parsed_sents) != len(self._gold_sents):
            raise ValueError(
                "Number of parsed sentences differs from number of gold sentences."
            )

        total = 0
        corr = 0    # UAS
        corrL = 0   # LAS

        for parsed_sent, gold_sent in zip(self._parsed_sents, self._gold_sents):
            parsed_nodes = parsed_sent.nodes
            gold_nodes = gold_sent.nodes

            if len(parsed_nodes) != len(gold_nodes):
                raise ValueError("Parsed and gold sentences must have equal length.")

            for addr, parsed_node in parsed_nodes.items():
                gold_node = gold_nodes[addr]

                if parsed_node["word"] is None:
                    continue
                if parsed_node["word"] != gold_node["word"]:
                    raise ValueError("Sentence words do not match at address {}".format(addr))

                # Skip punctuation
                if self._remove_punct(parsed_node["word"]) == "":
                    continue

                total += 1
                if parsed_node["head"] == gold_node["head"]:
                    corr += 1
                    if parsed_node["rel"] == gold_node["rel"]:
                        corrL += 1

        uas = corr / total if total > 0 else 0.0
        las = corrL / total if total > 0 else 0.0
        return las, uas


if __name__ == "__main__":
    import doctest
    doctest.testmod(optionflags=doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS)

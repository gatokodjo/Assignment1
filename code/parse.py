#!/usr/bin/env python3
import sys
from providedcode.transitionparser import TransitionParser
from providedcode.featureextractor import FeatureExtractor
from providedcode.transition import Transition
from providedcode.dependencygraph import DependencyGraph, DependencyGraphError


def read_sentences_from_stdin():
    """
    Reads sentences from stdin. Sentences are separated by blank lines.
    Returns a list of sentences, each sentence as list of CoNLL lines.
    """
    sentences = []
    current = []
    for line in sys.stdin:
        line = line.strip()
        if line == "":
            if current:
                sentences.append(current)
                current = []
        else:
            # If the line does not start with an ID, add proper ID
            if not line[0].isdigit():
                tokens = line.split()
                conll_lines = [
                    f"{i+1}\t{token}\t{token}\tX\tX\t_\t0\troot\t_\t_"
                    for i, token in enumerate(tokens)
                ]
                current.extend(conll_lines)
            else:
                current.append(line)
    if current:
        sentences.append(current)
    return sentences


def main():
    if len(sys.argv) != 2:
        print("Usage: python parse.py <model_file>", file=sys.stderr)
        sys.exit(1)

    model_file = sys.argv[1]

    # Load the trained model with the proper feature extractor
    tp = TransitionParser.load(model_file, Transition, FeatureExtractor)

    sentences = read_sentences_from_stdin()

    for sent_lines in sentences:
        try:
            dep_graph = DependencyGraph(sent_lines)
            parsed_graphs = tp.parse([dep_graph])
            parsed = parsed_graphs[0]

            # Output in CoNLL format (10-column)
            for idx in range(1, len(parsed.nodes)):
                node = parsed.nodes[idx]
                cols = [
                    str(node["address"]),
                    node["word"],
                    node.get("lemma", node["word"]),
                    node.get("ctag", "X"),
                    node.get("tag", "X"),
                    "_",  # feats
                    str(node.get("head", 0)),
                    node.get("rel", "dep"),
                    "_", "_"
                ]
                print("\t".join(cols))
            print()  # Blank line between sentences

        except DependencyGraphError as e:
            print(f"# Skipping invalid sentence: {e}", file=sys.stderr)
            continue


if __name__ == "__main__":
    main()

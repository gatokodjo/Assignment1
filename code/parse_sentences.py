#!/usr/bin/env python3
"""
Parse normal English sentences using a trained TransitionParser.

Usage:
    python parse_sentences.py <model_file> [input_file]

If input_file is omitted, reads from stdin.

This version uses spaCy to convert raw text into dependency graphs.
"""

import sys
from providedcode.transitionparser import TransitionParser
from providedcode.dependencygraph import DependencyGraph, DependencyGraphError
from providedcode.featureextractor import FeatureExtractor
from providedcode.transition import Transition
import spacy

# Load English model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy en_core_web_sm model...")
    import spacy.cli
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


def read_sentences(input_file=None):
    """Read raw text from file or stdin and split into sentences."""
    if input_file:
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = sys.stdin.read()

    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    return sentences


def sentences_to_depgraphs(sentences):
    """Convert raw sentences to DependencyGraph objects using spaCy."""
    depgraphs = []

    for sent in sentences:
        doc = nlp(sent)
        conll_lines = []

        # Build CoNLL 10-column lines
        for i, token in enumerate(doc, start=1):
            head = token.head.i + 1 if token.head != token else 0
            rel = token.dep_ if token.dep_ else "dep"
            conll_lines.append(
                f"{i}\t{token.text}\t{token.lemma_}\t{token.pos_}\t{token.tag_}\t_\t{head}\t{rel}\t_\t_"
            )

        try:
            dg = DependencyGraph()  # create empty graph
            dg._parse(conll_lines)  # patch original parser with spaCy lines
            depgraphs.append(dg)
        except DependencyGraphError as e:
            print(f"# Skipping invalid sentence: {e}", file=sys.stderr)
            continue

    return depgraphs


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_sentences.py <model_file> [input_file]", file=sys.stderr)
        sys.exit(1)

    model_file = sys.argv[1]
    input_file = sys.argv[2] if len(sys.argv) > 2 else None

    # Load trained TransitionParser model
    tp = TransitionParser.load(model_file, Transition, FeatureExtractor)

    # Read sentences and convert to DependencyGraph objects
    sentences = read_sentences(input_file)
    depgraphs = sentences_to_depgraphs(sentences)

    if not depgraphs:
        print("# No valid sentences to parse.", file=sys.stderr)
        sys.exit(0)

    # Parse sentences
    parsed_graphs = tp.parse(depgraphs)

    # Output in CoNLL format
    for parsed in parsed_graphs:
        for idx in range(1, len(parsed.nodes)):
            node = parsed.nodes[idx]
            cols = [
                str(node["address"]),
                node["word"],
                node.get("lemma", node["word"]),
                node.get("upos", "X"),
                node.get("xpos", "X"),
                "_",
                str(node.get("head", 0)),
                node.get("rel", "dep"),
                "_", "_"
            ]
            print("\t".join(cols))
        print()  # blank line between sentences


if __name__ == "__main__":
    main()

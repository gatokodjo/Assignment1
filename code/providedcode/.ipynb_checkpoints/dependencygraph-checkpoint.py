# Natural Language Toolkit: Dependency Grammars
from __future__ import absolute_import, print_function, unicode_literals
from collections import defaultdict
from itertools import chain
from pprint import pformat
import nltk
from nltk.tree import Tree


class DependencyGraphError(Exception):
    """Dependency graph exception."""


class DependencyGraph(object):
    """Full dependency graph with nodes and labeled arcs."""

    @staticmethod
    def from_sentence(sent):
        """Create a DependencyGraph from a raw sentence with NLTK tokenization."""
        tokens = nltk.word_tokenize(sent)
        tagged = nltk.pos_tag(tokens)
        dg = DependencyGraph()
        for index, (word, tag) in enumerate(tagged):
            dg.nodes[index + 1] = {
                "word": word,
                "lemma": "_",
                "ctag": tag,
                "tag": tag,
                "feats": "_",
                "rel": "_",
                "deps": defaultdict(list),
                "head": 0,
                "address": index + 1,
            }
        dg.connect_graph()
        return dg

    def __init__(self, tree_str=None, cell_extractor=None, zero_based=False, cell_separator=None):
        self.nodes = defaultdict(lambda: {"deps": defaultdict(list), "head": 0, "rel": "dep"})
        self.nodes[0].update({
            "word": None,
            "lemma": None,
            "ctag": "TOP",
            "tag": "TOP",
            "feats": None,
            "rel": "TOP",
            "address": 0,
        })
        self.root = None
        if tree_str:
            self._parse(tree_str, cell_extractor=cell_extractor, zero_based=zero_based, cell_separator=cell_separator)

    def get_by_address(self, addr):
        return self.nodes[addr]

    def contains_address(self, addr):
        return addr in self.nodes

    def add_arc(self, head_address, mod_address):
        rel = self.nodes[mod_address].get("rel", "dep")
        self.nodes[head_address]["deps"].setdefault(rel, [])
        self.nodes[head_address]["deps"][rel].append(mod_address)
        self.nodes[mod_address]["head"] = head_address
        self.nodes[mod_address]["rel"] = rel

    def connect_graph(self):
        for idx, node in self.nodes.items():
            if idx == 0:
                continue
            head = node.get("head", 0)
            rel = node.get("rel", "dep")
            self.nodes[head]["deps"].setdefault(rel, [])
            self.nodes[head]["deps"][rel].append(idx)
        self.root = self.nodes[0]

    def _parse(self, input_, cell_extractor=None, zero_based=False, cell_separator=None):
        if isinstance(input_, str):
            input_ = (line for line in input_.split("\n") if line.strip())
        for index, line in enumerate(input_, start=1):
            cells = line.split(cell_separator)
            if len(cells) == 10:
                _, word, lemma, ctag, tag, feats, head, rel, _, _ = cells
            elif len(cells) == 4:
                word, tag, head, rel = cells
                lemma, ctag, feats = word, tag, "", ""
            else:
                word, tag, head = cells[:3]
                lemma, ctag, feats, rel = word, tag, "", "dep"
            head = int(head) if head else 0
            self.nodes[index].update({
                "address": index,
                "word": word,
                "lemma": lemma,
                "ctag": ctag,
                "tag": tag,
                "feats": feats,
                "head": head,
                "rel": rel,
            })
            self.nodes[head]["deps"].setdefault(rel, []).append(index)
        self.root = self.nodes[0]

    def to_conll(self, style=10):
        """
        Export the dependency graph in CoNLL format.

        style: 3, 4, or 10 column format.
        """
        if style == 3:
            template = "{word}\t{tag}\t{head}\n"
        elif style == 4:
            template = "{word}\t{tag}\t{head}\t{rel}\n"
        elif style == 10:
            template = "{i}\t{word}\t{lemma}\t{ctag}\t{tag}\t{feats}\t{head}\t{rel}\t_\t_\n"
        else:
            raise ValueError(f"Unsupported CoNLL style: {style}")

        lines = []
        for i, node in sorted(self.nodes.items()):
            if node["tag"] == "TOP":
                continue
            lines.append(template.format(i=i, **node))
        return "".join(lines)

    def __str__(self):
        return pformat(self.nodes)

    def __repr__(self):
        return f"<DependencyGraph with {len(self.nodes)} nodes>"

    def tree(self):
        """Convert the dependency graph to an NLTK Tree."""
        def _tree(i):
            node = self.get_by_address(i)
            children = list(chain.from_iterable(node["deps"].values()))
            if children:
                return Tree(node["word"], [_tree(dep) for dep in children])
            else:
                return node["word"]

        return _tree(0)

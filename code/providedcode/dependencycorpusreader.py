#!/usr/bin/env python3
# Natural Language Toolkit: Dependency Corpus Reader
#
# Copyright (C) 2001-2015 NLTK Project
# Author: Kepa Sarasola <kepa.sarasola@ehu.es>
#         Iker Manterola <returntothehangar@hotmail.com>
#
# URL: <http://nltk.org/>
# For license information, see LICENSE.TXT

from __future__ import absolute_import

import codecs
from nltk.corpus.reader.api import *
from nltk.corpus.reader.util import *
from nltk.tokenize import *
from providedcode.dependencygraph import DependencyGraph  # ✅ absolute import


class DependencyCorpusReader(SyntaxCorpusReader):
    """Corpus reader for dependency treebanks."""

    def __init__(
        self,
        root,
        fileids,
        encoding="utf8",
        word_tokenizer=TabTokenizer(),
        sent_tokenizer=RegexpTokenizer("\n", gaps=True),
        para_block_reader=read_blankline_block,
    ):
        super().__init__(root, fileids, encoding)

    def raw(self, fileids=None):
        """Return raw text of given file(s) as a single string."""
        result = []
        for fileid, encoding in self.abspaths(fileids, include_encoding=True):
            if isinstance(fileid, PathPointer):
                result.append(fileid.open(encoding=encoding).read())
            else:
                with codecs.open(fileid, "r", encoding) as fp:
                    result.append(fp.read())
        return concat(result)

    def words(self, fileids=None):
        """Return words from corpus."""
        return concat(
            [
                DependencyCorpusView(fileid, False, False, False, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )

    def tagged_words(self, fileids=None):
        """Return words with POS tags from corpus."""
        return concat(
            [
                DependencyCorpusView(fileid, True, False, False, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )

    def sents(self, fileids=None):
        """Return sentences from corpus."""
        return concat(
            [
                DependencyCorpusView(fileid, False, True, False, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )

    def tagged_sents(self, fileids=None):
        """Return sentences with POS tags."""
        return concat(
            [
                DependencyCorpusView(fileid, True, True, False, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )

    def parsed_sents(self, fileids=None):
        """Return sentences parsed as DependencyGraph objects."""
        sents = concat(
            [
                DependencyCorpusView(fileid, False, True, True, encoding=enc)
                for fileid, enc in self.abspaths(fileids, include_encoding=True)
            ]
        )
        return [DependencyGraph(sent) for sent in sents]


class DependencyCorpusView(StreamBackedCorpusView):
    """View of corpus file, optionally tokenized and parsed."""

    _DOCSTART = "-DOCSTART- -DOCSTART- O\n"

    def __init__(
        self,
        corpus_file,
        tagged,
        group_by_sent,
        dependencies,
        chunk_types=None,
        encoding="utf8",
    ):
        self._tagged = tagged
        self._dependencies = dependencies
        self._group_by_sent = group_by_sent
        self._chunk_types = chunk_types
        super().__init__(corpus_file, encoding=encoding)

    def read_block(self, stream):
        """Read the next sentence or sentence block from the corpus file."""
        sent = read_blankline_block(stream)[0].strip()
        if sent.startswith(self._DOCSTART):
            sent = sent[len(self._DOCSTART):].lstrip()

        # Extract word/tag pairs or dependency structures
        if not self._dependencies:
            lines = [line.split("\t") for line in sent.split("\n")]
            if len(lines[0]) in {3, 4}:
                sent = [(line[0], line[1]) for line in lines]
            elif len(lines[0]) == 10:
                sent = [(line[1], line[4]) for line in lines]
            else:
                raise ValueError("Unexpected number of fields in dependency tree file")

            if not self._tagged:
                sent = [word for (word, tag) in sent]

        # Return as list of sentences or flattened
        return [sent] if self._group_by_sent else list(sent)

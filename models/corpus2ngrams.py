#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import interfaces, utils
from itertools import tee, izip, islice, chain


class Corpus2Ngrams(interfaces.TransformationABC):
    """
        Read a corpus, seen as a list of tokens, and return them as
        a N-tuple of tokens.
    """
    def __init__(self, n, start_char="^", end_char="$", emit_padding=True):
        """
            Configure returned ngram
        :param n: # of token per -gram
        :param start_char: special char for doc start
        :param end_char: special char for doc end
        :param emit_padding: emit n-gram with start/end chars
        """
        assert n > 0    # 0-gram not allowed
        self.n = n
        self.start_char, self.end_char = start_char, end_char
        self.emit_padding = emit_padding

    def __getitem__(self, doc):
        # if doc is an iterable apply to all
        is_corpus, doc = utils.is_corpus(doc)
        if is_corpus:
            return self._apply(doc)

        # introduce start/end chars to "n-grammize" head and tail of a doc
        doc = chain([self.start_char] * (self.n-1),
                    doc,
                    [self.end_char] * (self.n-1))
        # create n independent iterators and consume them to prepare for zipping
        iterators = tee(doc, self.n)
        [next(islice(it, i, i), None) for i, it in enumerate(iterators)]
        ngrams = [ngram for ngram in izip(*iterators)]
        if not self.emit_padding and self.n > 1:
            ngrams = ngrams[self.n-1:-(self.n-1)] # Prune
        return ngrams
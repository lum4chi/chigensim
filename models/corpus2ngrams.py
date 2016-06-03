#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import interfaces, utils
from itertools import tee, izip, islice, chain


class Corpus2Ngrams(interfaces.TransformationABC):
    """
        Read a corpus, seen as a list of tokens, and return them as
        a concatenation of N-grams.
    """
    def __init__(self, n, sep=" ", start_char="^", end_char="$", emit_head=True, emit_tail=True):
        """
            Configure returned ngram
        :param n: # of token per -gram
        :param sep: n-gram is a string, set a convenient separator
        :param start_char: special char for doc start
        :param end_char: special char for doc end
        :param emit_head: emit n-gram with start char?
        :param emit_tail: emit n-gram with end char?
        """
        self.n = n
        self.sep = sep
        self.start_char, self.end_char = start_char, end_char
        self.emit_head, self.emit_tail = emit_head, emit_tail

    def __getitem__(self, doc):
        # if doc is an iterable apply to all
        is_corpus, doc = utils.is_corpus(doc)
        if is_corpus:
            return self._apply(doc)

        # introduce start/end chars to "n-grammize" head and tail of a doc
        doc = chain([self.start_char] * (self.n-1), doc, [self.end_char] * (self.n-1))
        # create n independent iterators and consume them to prepare for zipping
        iterators = tee(doc, self.n)
        [next(islice(it, i, i), None) for i, it in enumerate(iterators)]
        ngrams = [self.sep.join(ngram) for ngram in izip(*iterators)]
        if not self.emit_head: ngrams = ngrams[self.n-1:]   # prune head
        if not self.emit_tail: ngrams = ngrams[:-self.n-1]  # prune tail
        return ngrams

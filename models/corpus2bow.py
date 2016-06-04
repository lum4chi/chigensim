#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import interfaces, corpora, utils


class Corpus2Bow(interfaces.TransformationABC):
    """
        Collect and index feautures of a doc (seen as list of token/word).
        If doc is a corpus, then apply transformation to all.
        Note: if "grouped" return a vector, ordered by feature id (not by
        appearance in doc), otherwise doc is simply translated by dictionary
        ids (as string, to preserve compatibility with other chains of bow)
    """
    def __init__(self, dictionary=corpora.Dictionary(), group=True, to_string=False):
        self._dict = dictionary
        self._grouped = group
        self.to_string = to_string
        # Optional: collect statistics if dictionary is empty
        self._allow_update = False if len(self._dict) > 0 else True

    def __getitem__(self, doc):
        # if doc is an iterable apply to all
        is_corpus, doc = utils.is_corpus(doc)
        if is_corpus:
            return self._apply(doc)

        # Appling transformation, return doc as a bag-of-words list
        doc = list(doc)  # if doc is an iterable, read all to be processed more than once
        grouped = self._dict.doc2bow(doc, self._allow_update)   # = [(id, freq),...]
        result = grouped if self._grouped \
            else [self._dict.token2id[w] for w in doc if w in self._dict.token2id]  # = [1, 2, 1, ...]
        if self.to_string: result = [str(r) for r in result]    # handy to chain with other text transformations
        return result

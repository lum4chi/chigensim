#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import corpora, utils
from mygensim.corpora import SimpleCorpus


class CorpusByDictionary:
    """
        Identity transformation that update a dictionary
    """
    def __init__(self, dictionary=None, to_bow=False):
        self.dict = corpora.Dictionary() if dictionary is None else dictionary
        # Optimize trasformation by setting right fuction at init time
        if to_bow: self.__getitem__ = self._getbow

    def _getbow(self, doc):
        # if doc is an iterable apply to all
        is_corpus, doc = utils.is_corpus(doc)
        if is_corpus:
            return SimpleCorpus(self._apply(doc))

        return self.dict.doc2bow(doc, allow_update=True)

    def __getitem__(self, doc):
        # if doc is an iterable apply to all
        is_corpus, doc = utils.is_corpus(doc)
        if is_corpus:
            return SimpleCorpus(self._apply(doc))

        self.dict.doc2bow(doc, allow_update=True)

        return doc

    def _apply(self, corpus):
        for doc in corpus:
            yield self[doc]
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import interfaces, utils
from mygensim import corpora


class Corpus2Bob(interfaces.TransformationABC):
    """
        Collect and index feautures of a doc (seen as list of token/word).
        If doc is a corpus, then apply transformation to all.
        Note: doc2bob return a vector, ordered by feature id (not by
        appearance in doc.
    """
    def __init__(self, bidictionary=corpora.Bidictionary()):
        self.bidict = bidictionary

    def __getitem__(self, doc):
        # if doc is an iterable apply to all
        is_corpus, doc = utils.is_corpus(doc)
        if is_corpus:
            return self._apply(doc)

        # appling transformation, return doc as a bag-of-bitokens list
        allow_update = False if len(self.bidict) > 0 else True
        return self.bidict.doc2bob(doc, allow_update)

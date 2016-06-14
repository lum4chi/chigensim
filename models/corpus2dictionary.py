#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import interfaces, corpora, utils


class Corpus2Dictionary(interfaces.TransformationABC):
    """
        Collect and index feautures of a doc (seen as list of token/word).
        If doc is a corpus, then apply transformation to all.
    """
    def __init__(self, dictionary=corpora.Dictionary(), flatten=False):
        self._dict = dictionary
        self._flatten = flatten
        # Optional: collect statistics if dictionary is empty
        self._allow_update = False if len(self._dict) > 0 else True

    def __getitem__(self, doc):
        # if doc is an iterable apply to all
        is_corpus, doc = utils.is_corpus(doc)
        if is_corpus:
            return self._apply(doc)

        # Appling transformation, return doc as a bag-of-words list
        doc = list(doc)
        # if required, flatten ids: [(1,1), (2,2), (3,1),...] = [1, 2, 2, 3,...]
        bow = self._dict.doc2bow(doc, self._allow_update)
        return bow if not self._flatten else\
                    [e for sub in [[_id]*rep for _id, rep in bow] for e in sub]

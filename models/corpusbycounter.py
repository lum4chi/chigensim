#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import utils, interfaces
from mygensim.corpora import RawCounter


class CorpusByCounter(interfaces.TransformationABC):
    """
        Identity transformation that update a counter.
    """
    def __init__(self, counter=None):
        self.counter = RawCounter() if counter is None else counter

    def __getitem__(self, doc):
        # if doc is an iterable apply to all
        is_corpus, doc = utils.is_corpus(doc)
        if is_corpus:
            return self._apply(doc)

        self.counter.update(doc)

        return doc
#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import interfaces
import itertools


class SimpleCorpus(interfaces.CorpusABC):
    """
        Simple class to wrap any iterable as a gensim corpus.
        gensim.utils.is_corpus recognize as corpus any obj that:
        - has __iter__ AND __len__ implemented (known from gensim.utils.is_corpus)
        - has *Corpus in class name !!!
    """
    def __init__(self, iterable):
        self.iterable = iterable
        self.len = -1

    def __len__(self):
        return self.len + 1

    def __iter__(self):
        for n, e in enumerate(self.iterable):
            self.len = n
            yield e

    def __getitem__(self, i): # TODO: support only positive slicing
        if isinstance(i, int): i = slice(i, i+1)
        return SimpleCorpus(itertools.islice(self.__iter__(), i.start, i.stop, i.step))

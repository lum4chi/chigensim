#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import interfaces, utils


class Corpus2(interfaces.TransformationABC):
    """
        This is a "generic" transformation.
        Apply to a doc the provided function.
        If doc is a corpus, then apply transformation to all.
        Return doc(corpus) is based upon supplied function.
    """
    def __init__(self, function, *fargs, **fkwargs):
        """
            Initialize function.
        :param function: function to be applied
        :param fargs: args of the function
        :param fkwargs: kwargs of the function
        :return:
        """
        self.funct = function
        self.fargs, self.fkwargs = fargs, fkwargs

    def __getitem__(self, doc):
        # if doc is an iterable apply to all
        is_corpus, doc = utils.is_corpus(doc)
        if is_corpus:
            return self._apply(doc)

        # return transformed doc according to function
        return self.funct(doc, *self.fargs, **self.fkwargs)

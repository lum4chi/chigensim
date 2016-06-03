#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from gensim import corpora

# TODO: to be done!!
class ClassifierDictionary(corpora.Dictionary):
    """
        This dictionary receive documents seen as bow vectors (like [(<t1>,<tf1>),...])
        and harvest statistics to compute probabilistic text classification.
    """
    def __init__(self, documents=None, doc_labels=None, prune_at=2000000):
        """
            Train the dictionary with documents and labels
        :param documents:
        :param doc_labels:
        :param prune_at:
        :return:
        """
        super(ClassifierDictionary, self).__init__(documents, prune_at)

    def doc2class(self):
        pass

#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from __future__ import division
from gensim import models, utils


def okapi_factor(tf, d_length, avg_d_length, k=1.2, b=0.75):
    """ Right factor of BM25 formula (left is IDF).
    :param tf: term frequency
    :param d_length: document length (how many terms)
    :param avg_d_length: from corpus
    :param k: 0.0 = BIM, large = use of raw tf
    :param b: no normalization 0.0 <= b <= 1.0 full normalization
    """
    return (k + 1) * tf / (k * ((1 - b) + b * (d_length / avg_d_length)) + tf)

class BM25Model(models.TfidfModel):
    """ Use of models.TfidfModel as base to build BM25 Model (11.32) appeared in
        "An introduction to Information Retrieval" by Manning, Raghavan and Schütze
    """
    def __init__(self, *args, **kwargs):
        super(BM25Model, self).__init__(*args, normalize=False, **kwargs)
        # Computing once and for all average document length with statistics from dict
        self.avg_d_length = self.num_nnz / self.num_docs

    def __str__(self):
        return "BM25Model(num_docs=%s, num_nnz=%s)" % (self.num_docs, self.num_nnz)

    def __getitem__(self, bow, eps=1e-12):
        """ Overwrite weight calculus with a sum, according to Retrieval Status Value:
            RSVd(t) = IDF(t)*okapi_f(...) # Component to sum
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        # --- only vector component calculation has changed from original method ---
        # unknown (new) terms will be given zero weight
        vector = [(termid, self.idfs.get(termid) * okapi_factor(tf, len(bow), self.avg_d_length))
                  for termid, tf in bow if self.idfs.get(termid, 0.0) != 0.0]

        # --- no need to normalize ---

        # make sure there are no explicit zeroes in the vector (must be sparse)
        vector = [(termid, weight) for termid, weight in vector if abs(weight) > eps]
        return vector

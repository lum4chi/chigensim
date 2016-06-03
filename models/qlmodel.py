#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from __future__ import division
from gensim import models, utils
import math


class QLModel(models.TfidfModel):
    """ Use of models.TfidfModel as base to build Query Likelihood Model (12.9) appeared in
        "An introduction to Information Retrieval" by Manning, Raghavan and Schütze
    """
    def __init__(self, *args, **kwargs):
        super(QLModel, self).__init__(*args, normalize=False, **kwargs)

    def __str__(self):
        return "QueryLikelihoodModel(num_docs=%s, num_nnz=%s)" % (self.num_docs, self.num_nnz)

    def __getitem__(self, bog, eps=1e-12):
        """ Overwrite weight calculus with estimation of a Model of d, based on its own "gram"
            (we can see bag-of-word as bag-of-gram based upon what tokenize policy to adopt):
            P(q|d) ≈ prod( P(g|d) for g in q )  # product of only the gram present in query
            P(g|d) ≈ tf(g,d) / len(d)           # compute prob of every gram
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bog = utils.is_corpus(bog)
        if is_corpus:
            return self._apply(bog)

        # --- only vector component calculation has changed from original method ---
        # unknown (new) terms will be given zero weight
        # 0 < P(g|d) <= 1, then -1 * log() to avoid negative
        vector = [(gramid, -math.log(tf / len(bog)))
                  for gramid, tf in bog if self.idfs.get(gramid, 0.0) != 0.0]

        # --- no need to normalize ---

        # make sure there are no explicit zeroes in the vector (must be sparse)
        vector = [(termid, weight) for termid, weight in vector if abs(weight) > eps]
        return vector
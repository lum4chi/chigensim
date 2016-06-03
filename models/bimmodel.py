#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from __future__ import division
from gensim import models, utils
import math


def croff_harper_pt():
    """ pt is random = 0.5 -> pt/(1-pt) = 1 """
    return 1.0

def greiff_pt(docfreq, totaldocs):
    return 1/3 + 2/3 * docfreq / totaldocs

class BimModel(models.TfidfModel):
    """ Use of models.TfidfModel as base to build Binary Independence Model appeared in
        "An introduction to Information Retrieval" by Manning, Raghavan and Schütze
    """
    def __init__(self, *args, **kwargs):
        super(BimModel, self).__init__(*args, normalize=False, **kwargs)

    def __str__(self):
        return "BimModel(num_docs=%s, num_nnz=%s)" % (self.num_docs, self.num_nnz)

    def __getitem__(self, bow, eps=1e-12):
        """ Overwrite weight calculus with a sum, according to Retrieval Status Value:
                RSVd = sum(ct(t) for t in bow(d))       # Scoring formula for doc d
                ct = log(pt/(1-pt)) + log((1-ut)/ut)    # Weight of vector component
        """
        # if the input vector is in fact a corpus, return a transformed corpus as a result
        is_corpus, bow = utils.is_corpus(bow)
        if is_corpus:
            return self._apply(bow)

        # --- only vector component calculation has changed from original method ---
        # unknown (new) terms will be given zero weight
        vector = [(termid, math.log(greiff_pt(self.dfs[termid], self.num_docs)) + self.idfs.get(termid))
                  for termid, _ in bow if self.idfs.get(termid, 0.0) != 0.0]

        # --- no need to normalize ---

        # make sure there are no explicit zeroes in the vector (must be sparse)
        vector = [(termid, weight) for termid, weight in vector if abs(weight) > eps]
        return vector

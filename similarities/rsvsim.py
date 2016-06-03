#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import division
from gensim import similarities, utils

class SparseMatrixRSV(similarities.SparseMatrixSimilarity):
    """ Overwriting SparseMatrixSimilarities to replace Cosine Similarity
        to match Binary Independence Model, where a term in a query weight
        only for its presence (1) or absence (0), and RSV(d) is always
        a sum of its selected (by query) weight.
        (If you need the product, consider:
                prod(w1,w2,...) ≈ sum(log(w1),log(w2),...)
        in a sense that both increase monotonically and so we can take
        advantage of original get_similarities() dot product)
    """
    def get_similarities(self, query):
        """ Compute RSV against every other document in the collection
            by weight every query component = 1 (dot product inside original
            "get_similarities()" become a mask to select only the document
            component referring to queried terms) an then, result ends up
            as sum of only query term weights.
            ( RSV(d) = ... + tx(d) * tx(q) + ty(d) * ty(q) + ...
                     = ... + tx(d) * 1 + ty(d) * 1 + ...
              "..." are all the term NOT in query:
                    tz(d) * tz(q) = tz(d) * 0 = 0 )
        """
        is_corpus, query = utils.is_corpus(query)
        if is_corpus: # gensim allow multiple queries as list of lists
            query = [[(t, 1) for t, w in q] for q in query]
        else:
            query = [(t, 1) for t, w in query]
        return super(SparseMatrixRSV, self).get_similarities(query)

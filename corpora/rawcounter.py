#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from collections import Counter
from gensim import utils
import cPickle


class RawCounter(Counter):
    """
        A collections.Counter that can be saved/loaded.
        Unexplainably "num_docs" can't be written to file normally and
        need to be init at -1 because at creation "update" is called
    """
    def __init__(self, *args, **kwargs):
        self.num_docs = -1  # TODO this is "special"
        self.num_pos = 0
        super(RawCounter, self).__init__(*args, **kwargs)

    def update(self, *args, **kwds):
        partial = Counter(*args, **kwds)
        self.num_docs += 1
        self.num_pos += sum(partial.values())
        super(RawCounter, self).update(partial)

    def save(self, fname):
        # TODO: self.num_docs is the only need this threatment?!
        cPickle.dump(self, open(fname, 'wb'))
        with open(fname, 'ab') as f: f.write(str(self.num_docs))

    @staticmethod
    def load(fname):
        # TODO: self.num_docs is the only need this threatment?!
        with open(fname, 'rb') as f:
            instance = cPickle.load(f)
            instance.num_docs = int(f.readline())
            return instance
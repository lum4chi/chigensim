#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from simplecorpus import SimpleCorpus
from pymongo import MongoClient


class MongoCorpus(SimpleCorpus):
    """
        Corpus wrapper around a MongoDB collection.
        Subset corpus by setting a query here.
    """
    def __init__(self, db, collection, query={}):
        self.client = MongoClient()[db][collection]
        self.query = query

    def __iter__(self):
        """
            _obj_ is a dictionary: you can filter the right
            key to feed only docs text.
        """
        for obj in self.client.find(self.query):
            yield obj

    def __len__(self):
        return self.client.find(self.query).count()



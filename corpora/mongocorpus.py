#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from simplecorpus import SimpleCorpus
from pymongo import MongoClient


class MongoCorpus(SimpleCorpus):
    """
        Corpus wrapper around a MongoDB collection.
        Subset corpus by setting a query. If "aggregate" is used,
        this will override "query". In this case use "$match" in
        aggregation method.
    """
    def __init__(self, db, collection, aggregate=[], query={}):
        self.client = MongoClient()[db][collection]
        self.aggregate_arg = aggregate
        self.find_arg = query

    def __iter__(self):
        """
            _obj_ is a dictionary: you can filter the right
            key to feed only docs text.
        """
        collection = self.client.find(self.find_arg, no_cursor_timeout=True) \
                        if len(self.aggregate_arg) == 0 \
                else self.client.aggregate(self.aggregate_arg)
        for doc in collection:
            yield doc

        collection.close()

    def __len__(self):
        if len(self.aggregate_arg) == 0:
            return self.client.find(self.find_arg).count()
        else:
            d = next(self.client.aggregate(self.aggregate_arg +
                        [{"$group": {"_id": "null", "count": {"$sum": 1}}}]))
            return d['count']


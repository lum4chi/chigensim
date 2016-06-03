#!/usr/bin/python
# -*- coding: utf-8 -*-

def docs2corpus(docs, dictionary, tokenizer):
    """ Read a document and transform into BoW vector, while compiling a dictionary.
    :param docs: an iterable collection of doc
    :param dictionary: dictionary to be filled with token
    :param tokenizer: tokening function
    :return: a generator of BoW vectors
    """
    for d in docs:
        yield dictionary.doc2bow(tokenizer(d), allow_update=True)

def list2val_id(a_list):
    """ Given a list, return a dict as an inverted index.
    :param a_list:
    :return:
    """
    a_dict = {}
    for n, elem in enumerate(a_list): a_dict[elem] = n
    return a_dict

def is_generator(obj):
    import types as t
    return isinstance(obj, t.GeneratorType)

def is_list(obj):
    return isinstance(obj, list)

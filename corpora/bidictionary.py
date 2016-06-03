#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Francesco Lumachi <francesco.lumachi@gmail.com>
from __future__ import division
from gensim import corpora, utils
from six import iteritems, itervalues, iterkeys
from collections import defaultdict
from itertools import tee, izip
from scipy.stats import rv_discrete

class Bidictionary(utils.SaveLoad):
    # TODO not completely implemented all methods as a gensim dictionary!
    """
        This object provide a convenient way to parse document (seen as list of tokens
        in appearance order!) while extracting bigrams frequencies, returning gensim-like
        "bag-of-bigrams" vector, which in features (bi-tokens) are identified by a tuple
        (firstId, secondId).
        Ex: bd = Bidictionary(documents=<some_corpus>)
            bd[<token_id>]          # return id->plain_token (from corpora.Dictionary)
            bd.token2id[<token>]    # return plain_token->id (from corpora.Dictionary)
            bd.fid_sid2id[<firstId>, <secondId>]    # return tokenid, tokenid -> bitokenid
            bd.dfs[bitokenid]       # return document frequency of bitoken
    """
    def __init__(self, documents=None, prune_at=2000000, doc_start='^', doc_end='$'):
        """
            Choose doc_start, doc_end to some char ignore by tokenizer, otherwise statistics
            about start/end token will be compromised.
        """
        self._unidict = corpora.Dictionary()
        # add dummy doc to map start/end chars (will produce len(unidict)+1: nevermind)
        self._unidict.doc2bow([doc_start, doc_end], allow_update=True)
        self.doc_start, self.doc_end = doc_start, doc_end

        # Statistics gensim-like
        self.fid_sid2bid = {}  # (firstid, secondid) -> tokenId
        self.bid2fid_sid = {}  # TODO: reverse mapping for fid_sid2bid; only formed on request, to save memory
        self.dfs = {}  # document frequencies: tokensId -> in how many documents those tokens appeared
        self.num_pos = 0  # total number of corpus positions
        self.num_nnz = 0  # total number of non-zeroes in the BOW matrix

        if documents is not None:
            self.add_documents(documents, prune_at=prune_at)

    num_docs = property(lambda self: self._unidict.num_docs - 1) # 1 is the dummy doc ['^','$']

    def doc2bob(self, document, allow_update=False, return_missing=False):
        """ Document tokens are parsed pairwise to produce bag-of-bitokens features """
        positional_doc = [self.doc_start] + document + [self.doc_end]

        # Index single tokens
        self._unidict.doc2bow(positional_doc, allow_update, return_missing)

        # Construct ((firstid, secondid), frequency) mapping.
        d1, d2 = tee(positional_doc)
        next(d2, None)  # step ahead second iterator
        counter = defaultdict(int)
        for first, second in izip(d1, d2):
            # saving space using same indexes as unidict
            try:
                firstid = self._unidict.token2id[first]
                secondid = self._unidict.token2id[second]
                counter[firstid, secondid] += 1
            except KeyError: # 1 or both token aren't indexed: skip.
                continue

        fid_sid2bid = self.fid_sid2bid
        if allow_update or return_missing:
            missing = dict((f_s, freq) for f_s, freq in iteritems(counter) if f_s not in fid_sid2bid)
            if allow_update:
                for w in missing:
                    # new id = number of ids made so far;
                    # NOTE this assumes there are no gaps in the id sequence!
                    fid_sid2bid[w] = len(fid_sid2bid)

        result = dict((fid_sid2bid[w], freq) for w, freq in iteritems(counter) if w in fid_sid2bid)

        if allow_update:
            self.num_pos += sum(itervalues(counter))
            self.num_nnz += len(result)
            # increase document count for each unique token that appeared in the document
            dfs = self.dfs
            for bid in iterkeys(result):
                dfs[bid] = dfs.get(bid, 0) + 1

        # return tokensids, in ascending id order
        result = sorted(iteritems(result))
        if return_missing:
            return result, missing
        else:
            return result

    def add_documents(self, docs, prune_at=2000000):
        for d in docs:
            self.doc2bob(d, allow_update=True)

    def tokens2bid(self, tokens):
        """
        :param tokens: need to be a tuple ('a','b')
        """
        fid, sid = self._unidict.token2id[tokens[0]], self._unidict.token2id[tokens[1]]
        return self.fid_sid2bid[(fid, sid)]

    def __getitem__(self, ids):
        # If you want the frequency, you need to ask for a "bid" and then to self.dfs[bid]
        if isinstance(ids, int): return self._unidict.__getitem__(ids)         # tid -> 'token'
        if isinstance(ids, str): return self._unidict.token2id[ids]     # 'token' -> id
        if isinstance(ids, tuple):
            if isinstance(ids[0], int): return self.fid_sid2bid[ids]    # fid, sid -> bid
            if isinstance(ids[0], str): return self.tokens2bid(ids)     # 'a', 'b' -> bid

    @staticmethod
    def load_from_text(fname):
        return super(Bidictionary, fname).load_from_text(fname)

    def save_as_text(self, fname, sort_by_word=True):
        """
        Save this Dictionary to a text file, in format:
        `id[TAB]fid[TAB]sid[TAB]document frequency[NEWLINE]`
        and _unidict has an usual gensim dictionary
        """
        self._unidict.save_as_text(fname + '.index', sort_by_word)
        with utils.smart_open(fname, 'wb') as fout:
            # no word to display in bidict
            for fid_sid, id in sorted(iteritems(self.fid_sid2bid)):
                line = "%i\t%i\t%i\t%i\n" % (id, fid_sid[0], fid_sid[1], self.dfs.get(id, 0))
                fout.write(utils.to_utf8(line))

    @staticmethod
    def load_from_text(fname):
        """
        Load a previously stored Dictionary from a text file.
        Mirror function to `save_as_text`.
        """
        result = Bidictionary()
        # restore _unidict as gensim dictionary
        result._unidict = corpora.Dictionary.load_from_text(fname + '.index')

        with utils.smart_open(fname) as f:
            for lineno, line in enumerate(f):
                line = utils.to_unicode(line)
                try:
                    bid, fid, sid, docfreq = line[:-1].split('\t')
                    fid_sid = (int(fid), int(sid))
                except Exception:
                    raise ValueError("invalid line in dictionary file %s: %s"
                                     % (fname, line.strip()))
                bid = int(bid)
                if fid_sid in result.fid_sid2bid:
                    raise KeyError('token %s is defined as ID %d and as ID %d' % (fid_sid, bid, result.fid_sid2bid[fid_sid]))
                result.fid_sid2bid[fid_sid] = bid
                result.dfs[bid] = int(docfreq)

        return result

    def mle(self, estimated, given):
        """ Compute Maximum Likelihood Estimation probability
            to extract the second token given first. """
        try:
            firstid = self._unidict.token2id[given]
            secondid = self._unidict.token2id[estimated]
            return self.dfs[self.fid_sid2bid[firstid, secondid]] / self._unidict.dfs[firstid]
        except KeyError:
            return 0.0

    def mlebyids(self, estimated, given):
        """ Compute Maximum Likelihood Estimation probability
            to extract the second token id given first id. """
        try:
            return self.dfs[self.fid_sid2bid[given, estimated]] / self._unidict.dfs[given]
        except KeyError:
            return 0.0

    def generate_text(self, seed, n):
        """ Given a seed token, produce n likelihood tokens to follow. """
        def nexttokenid(seedid):
            candidates = [sid for fid, sid in self.fid_sid2bid.keys() if fid == seedid]
            if len(candidates) == 0: raise StopIteration
            probs = [self.mlebyids(probid, seedid) for probid in candidates]
            return rv_discrete(values=(candidates, probs)).rvs()

        seedid = self._unidict.token2id[seed]
        text = [seed]
        for n in range(0, n):
            try:
                seedid = nexttokenid(seedid)
                text.append(self._unidict[seedid])
            except StopIteration:
                break

        return ' '.join(text)

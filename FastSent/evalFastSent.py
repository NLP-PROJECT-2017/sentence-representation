# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

"""
Example of file to compare skipthought vectors with our InferSent model
"""
import logging
from exutil import dotdict
import sys
import numpy as np
reload(sys)
sys.setdefaultencoding('utf8')


# Set PATHs
PATH_TO_SENTEVAL = 'SentEval/'
PATH_TO_DATA = 'SentEval/data/senteval_data/'
PATH_TO_FASTSENT = 'sentence-representation/FastSent'
#assert PATH_TO_SKIPTHOUGHT != '', 'Download skipthought and set correct PATH'
PATH_TO_GENSIM = 'py2.7/lib/python2.7/site-packages'
PATH_TO_MODEL = 'out/FastSent_autoencoding_512_3_0'
# import skipthought and Senteval
sys.path.insert(0, PATH_TO_FASTSENT)
sys.path.insert(0, PATH_TO_SENTEVAL)
sys.path.insert(0, PATH_TO_GENSIM)
import fastsent
import senteval
model = fastsent.FastSent.load(PATH_TO_MODEL)

def prepare(params, samples):
    return

def batcher(params, batch):
    embeddings = []
    for sent in batch:
        try:
            ss = ''
            for s in sent:
                if s in model:
                    ss += s + ' '
            if ss == '':
                ss = '.'
            embeddings.append(model[ss.strip()])
        except KeyError:
            embeddings.append(np.random.rand(512)) 
    return np.array(embeddings)


# Set params for SentEval
params_senteval = {'usepytorch': True,
                   'task_path': PATH_TO_DATA,
                   'batch_size': 512}
params_senteval = dotdict(params_senteval)

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    params_senteval.encoder = model
    se = senteval.SentEval(params_senteval, batcher, prepare)
    se.eval(['CR', 'MR', 'MPQA', 'SUBJ', 'SST', 'TREC', 'MRPC', 'SNLI', 'SICKEntailment', 'SICKRelatedness', 'STSBenchmark', 'STS12', 'STS13', 'STS14', 'STS15', 'STS16'])

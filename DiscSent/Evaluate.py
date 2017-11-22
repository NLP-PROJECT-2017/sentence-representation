import argparse
import cPickle as pickle
import json
import logging
from os.path import join as pjoin
import os
import sys

import numpy as np

from Utils import *

SENTEVAL_PATH = '/misc/vlgscratch4/BowmanGroup/haoyue/code/SentEval/'
# https://github.com/facebookresearch/SentEval
assert SENTEVAL_PATH, 'Set SentEval Path'
sys.path.insert(0, SENTEVAL_PATH)
import senteval


class DotDict(dict):
  __getattr__ = dict.get
  setattr__ = dict.__setitem__
  delattr__ = dict.__delitem__


def prepare(params, samples):
  pass


def batcher_bow(params, batch):
  sents = map(lambda sen: [word.lower() for word in sen], batch)
  embeddings = list()
  for sent in sents:
    sent_vec = list()
    for w in sent:
      if w in params.w2v:
        sent_vec.append(params.w2v[w])
    if len(sent_vec) == 0:
      embeddings.append([0 for _ in range(300)])
    else:
      embeddings.append(np.mean(sent_vec, 0))
  return np.vstack(embeddings)


def batcher_self_bow(params, batch):
  sents = map(lambda sen: ['<S>'] + [word.lower() for word in sen], batch)
  sents, lengths = batched_data.batch_to_vars(sents)
  model_embedding = params.encoder.word_embed(sents).mean(1).squeeze()
  try:
    combine_embedding = params.encoder2.word_embed(sents).mean(1).squeeze()
    model_embedding = np.concatenate((model_embedding, combine_embedding), axis=1)
  except AttributeError:
    pass
  return model_embedding


def batcher(params, batch):
  sents = map(lambda sen: ['<S>'] + [word.lower() for word in sen], batch)
  sents, lengths = batched_data.batch_to_vars(sents)
  embedding, h_for, h_back = params.encoder.forward(sents, lengths=lengths)
  model_embedding = embedding.data.cpu().numpy()
  if params.add_external_bow:
    bow = batcher_bow(params, batch)
    model_embedding = np.concatenate((model_embedding, bow), axis=1)
  if params.add_self_bow:
    bow = batcher_self_bow(params, batch)
    model_embedding = np.concatenate((model_embedding, bow), axis=1)
  return model_embedding


def batcher_random(params, batch):
  length = len(batch)
  return np.random.rand(length, 600)


def batcher_combine(params, batch):
  bow = None
  add_external_bow = params.add_external_bow
  if add_external_bow:
    params.add_external_bow = False
    bow = batcher_bow(params, batch)
  sents = map(lambda sen: ['<S>'] + [word.lower() for word in sen], batch)
  sents, lengths = batched_data.batch_to_vars(sents)
  embedding_1, h_for, h_back = params.encoder.forward(sents, lengths=lengths)
  embedding_1 = embedding_1.data.cpu().numpy()
  embedding_2, h_for, h_back = params.encoder2.forward(sents, lengths=lengths)
  embedding_2 = embedding_2.data.cpu().numpy()
  if bow is not None:
    model_embedding = np.concatenate((embedding_1, embedding_2, bow), axis=1)
  else:
    model_embedding = np.concatenate((embedding_1, embedding_2), axis=1)
  if add_external_bow:
    params.add_external_bow = True
  if params.add_self_bow:
    bow = batcher_self_bow(params, batch)
    model_embedding = np.concatenate((model_embedding, bow), axis=1)
  return model_embedding


def load_w2v(path, cut=-1):
  f = open(path)
  cnt = 0
  word_vec = dict()
  if path[-4:] == ".txt":  # glove
    for i, line in enumerate(f):
      word, vec = line.split(' ', 1)
      word_vec[word] = np.fromstring(vec, sep=' ')
      if i == cut - 1:
        break
  elif path[-3:] == ".pk":
    vocab, vec = pickle.load(open(path))
    for i, word in enumerate(vocab):
      word_vec[word] = vec[i]
      if i == cut - 1:
        break
  return word_vec


if __name__ == "__main__":
  parser = argparse.ArgumentParser("Sentence Evaluation")
  parser.add_argument("-data", "--data_folder", default='Data/Gutenberg',
                      help="location of the data")
  parser.add_argument("-o", "--output_folder", default='Data/Logs',
                      help="Where to save the model and evaluation results")
  parser.add_argument("-model", "--model_prefix", default='',
                      help="path to saved model, without suffix")
  parser.add_argument("-nm", "--name", default='evaluate',
                      help="prefix for the result")
  parser.add_argument("-sf", "--suffix", default='',
                      help="suffix(epoch number) of the model to evaluate")
  parser.add_argument("-nc", "--nocuda", action='store_false', dest='cuda',
                        help="not to use CUDA")
  parser.add_argument("-max_l", "--max_length", default=128, type=int,
                      help="truncate sentences longer than")
  parser.add_argument("-rand", "--random", action='store_true',
                      help="get result for random embeddings")
  parser.add_argument("-bow", "--bow", action='store_true',
                      help="get result for only BoW embeddings")
  parser.add_argument("-w2v", "--word_embedding",
                      default="/misc/vlgscratch4/BowmanGroup/haoyue/shelve/SortedFastTextUD.pk",
                      help="path to word vector")
  parser.add_argument("-aeb", "--add_external_bow", action="store_true",
                      help="whether to add BoW to obtained sentence embeddings")
  parser.add_argument("-asb", "--add_self_bow", action="store_true",
                      help="whether to add self BoW (updated) to obtained sentence embeddings")
  parser.add_argument("-cut", "--cut_voc", type=int, default=-1,
                      help="cut vocabulary size")
  parser.add_argument("-comb", "--combine_prefix", type=str, default="",
                      help="combine another model")
  parser.add_argument("-sfc", "--combine_suffix", default="",
                      help="suffix(epoch number) of the model to evaluate")
  options = parser.parse_args()
  # config for log file
  log_file = pjoin(options.output_folder, options.name + '_' + get_time_str() + '.log')
  logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG, filename=log_file)
  logging.info("ARGUMENTS<<<<<")
  for arg, value in sorted(vars(options).items()):
    print arg, value
    logging.info("Argument %s: %r", arg, value)
  logging.info(">>>>>ARGUMENTS")
  # config for transfer tasks
  transfer_tasks = ['MR', 'CR', 'SUBJ', 'MPQA', 'SST', 'TREC', 'SICKRelatedness', 'SICKEntailment', 'MRPC', 'STS14']
  if options.random:
    params_senteval = DotDict({'usepytorch': True,
                               'task_path': pjoin(SENTEVAL_PATH, 'data/senteval_data')
                               })
    evaluator = senteval.SentEval(params_senteval, batcher_random, prepare)
  elif options.bow:
    params_senteval = DotDict({'usepytorch': True,
                               'task_path': pjoin(SENTEVAL_PATH, 'data/senteval_data'),
                               'w2v': load_w2v(options.word_embedding, options.cut_voc)
                               })
    evaluator = senteval.SentEval(params_senteval, batcher_bow, prepare)
  else:
    # load saved model
    args_path = pjoin(options.output_folder, options.model_prefix + '.args.pk')
    args = pickle.load(open(args_path))
    args.data_folder = options.data_folder
    args.max_length = options.max_length
    encoder = Encoder(args)
    model_file = pjoin(options.output_folder, options.model_prefix + options.suffix + '.pt')
    encoder.load_state_dict(torch.load(model_file))
    if options.cuda:
      encoder = encoder.cuda()
    # config for evaluator
    batched_data  = BatchedData(args)
    params_senteval = DotDict({'usepytorch': True,
                               'task_path': pjoin(SENTEVAL_PATH, 'data/senteval_data'),
                               'add_external_bow': options.add_external_bow,
                               'add_self_bow': options.add_self_bow,
                               'w2v': load_w2v(options.word_embedding, options.cut_voc)
                               })
    params_senteval.encoder = encoder
    combine_args_path = pjoin(options.output_folder, options.combine_prefix + '.args.pk')
    if os.path.exists(combine_args_path):
      args = pickle.load(open(combine_args_path))
      args.data_folder = options.data_folder
      args.max_length = options.max_length
      encoder_combine = Encoder(args)
      model_file = pjoin(options.output_folder, options.combine_prefix + options.combine_suffix + '.pt')
      encoder_combine.load_state_dict(torch.load(model_file))
      if options.cuda:
        encoder_combine = encoder_combine.cuda()
      params_senteval.encoder2 = encoder_combine
      evaluator = senteval.SentEval(params_senteval, batcher_combine, prepare)
    else:
      evaluator = senteval.SentEval(params_senteval, batcher, prepare)
  # evaluate
  results_transfer = evaluator.eval(transfer_tasks)
  logging.info("RESULTS<<<<<")
  for item in results_transfer:
    logging.info(str(item))
    logging.info(str(results_transfer[item]))
  logging.info(">>>>>RESULTS")


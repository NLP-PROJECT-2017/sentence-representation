import argparse
import shelve
import logging
import cPickle as pickle
from random import shuffle
from os.path import join as pjoin

import torch.nn as nn
import torch.optim as optim

from Utils import *


# make model, criterion, training ops
def make_modules(options):
  # model
  sent_encode     = Encoder(options).cuda()
  print 'loaded model from file'
  compare 	      = Comparator(options.task_type, options).cuda()
  # initialization
  compare.init_params(options.init_range)    
  # criteria and optimization
  if options.task_type == 'msrp' or options.task_type == 'sst2' or options.task_type == 'subj':
    crit      = nn.BCELoss()
  elif options.task_type == 'sick':
    crit      = nn.MSELoss()
  elif options.task_type == 'trec' or options.task_type == 'snli':
    crit      = nn.NLLLoss()
  params = [p for p in compare.parameters()]
  # optionally: additional BoW representation
  sent_encode_bow = None
  if options.use_bow:
    options.encode_type = "BoW"
    options.pre_mode    = options.bow_pre_mode
    options.model_file  = None
    sent_encode_bow     = Encoder(options).cuda()
    sent_encode_bow.init_params(options.init_range)
    params += [p for p in sent_encode_bow.parameters()]
  # learning modules
  if options.optimizer == 'sgd':
    opt = optim.SGD(params, lr=options.learning_rate, weight_decay=options.l2_reg)
  else:
    opt = optim.Adagrad(params, lr=options.learning_rate, weight_decay=options.l2_reg)
  modules = {'sentences'      : sent_encode,
             'sentences_bow'  : sent_encode_bow,
             'compare'        : compare,
             'optim'          : opt,
             'crit'           : crit}
  return modules


# train or evaluate on a specific task
def run_epoch(data, modules, options, training=False):
  print 'starting epoch', e, '\t', get_time_str()
  tot_loss      = 0.
  tot_accu      = 0.
  bs            = options.batch_size
  mode          = 'train' if training else 'valid'
  if training:
    shuffle(data.data[options.task_type][mode])
  for b in range(len(data.data[options.task_type][mode]) / bs):
    b_xl, b_xr, b_ll, b_lr, b_y = data.next_batch(options.task_type, mode)
    # make sentence representation
    sl    = modules['sentences'](b_xl, lengths=b_ll).detach()
    if options.use_bow:
      sl  = torch.cat([sl, modules['sentences_bow'](b_xl) * 0.001], 1)
    if options.task_type in ['msrp', 'sick', 'snli']:
      sr    = modules['sentences'](b_xr, lengths=b_lr).detach()
      if options.use_bow:
        sr  = torch.cat([sr, modules['sentences_bow'](b_xr) * 0.001], 1)
      b_scores  = modules['compare'](sl, sr, training=training)
    elif options.task_type == 'trec' or options.task_type == 'sst2' or options.task_type == 'subj':
      b_scores  = modules['compare'](sl, training=training)
    # compute accuracy and loss
    loss  = modules['crit'](b_scores, b_y)
    if options.task_type == 'msrp' or options.task_type == 'sst2' or options.task_type == 'subj':
      accu  = ((b_scores > 0.5) == (b_y > 0.5)).sum()
    elif options.task_type == 'sick':
      accu  = loss * bs
    elif options.task_type == 'trec' or options.task_type == 'snli':
      _, preds  = b_scores.max(1)
      accu      = (preds == b_y).sum()
    # compute loss and (optionally) backprop
    if training:
      modules['optim'].zero_grad()
      loss.backward()
      modules['optim'].step()
    # accumulate loss and accuracy
    tot_accu  += float(accu.data[0]) / bs
    tot_loss  += loss.data[0]
    if b % 100 == 0 and b > 0 and training:
      print b, '\t', tot_loss / (b + 1), '\t', tot_accu / (b + 1), '\t', get_time_str()
  # end of epoch
  b = len(data.data[options.task_type][mode]) / bs
  print b, '\t LOSS:', tot_loss / b, '\t ACCU:', tot_accu / b, '\t', get_time_str()
  return tot_accu / b


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='This program \
                 trains a model on a external validation tasks.')
  # task
  parser.add_argument("-task", "--task_type", default='msrp',
                      help="semantic evaluation task [msrp|sick|sst2|trec|subj|snli]")
  parser.add_argument("-cst", "--choose_subj_test", default=0, type=int,
                      help="test split for subj task")
  # model: general
  parser.add_argument("-mod_args", "--model_args_file", default='/data/ml2/jernite/TextData/Models/bigru_512_learn_lin.args.pk',
                      help="saved model")
  parser.add_argument("-model", "--model_file", default='/data/ml2/jernite/TextData/Models/bigru_512_learn_lin.pt',
                      help="saved model")
  parser.add_argument("-ir", "--init_range", default=1, type=float,
                      help="random initialization parameter")
  parser.add_argument("-hw", "--highway", action="store_true",
                      help="use highway network in comparator")
  parser.add_argument("-comp_m", "--comp_mode", default='hw',
                      help="sentence transformation for comparator [hw|lin|gate]")
  parser.add_argument("-comp_nl", "--comp_nlayers", default=0, type=int,
                      help="layers of non-linearity in comparator")
  parser.add_argument("-bow", "--use_bow", action="store_true",
                      help="also use BoW embedding of the sentence")# optimization
  parser.add_argument("-bow_pre_m", "--bow_pre_mode", default='learn',
                      help="learn from scratch or adapt GLoVE")# optimization
  parser.add_argument("-pre", "--pre_trained", default='/data/ml2/jernite/TextData/Embeddings/SortedGlove6B.pk',
                      help="location of pre-trained word2vec model")
  parser.add_argument("-bs", "--batch_size", default=32, type=int,
                      help="batch size for training")
  parser.add_argument("-epochs", "--epochs", default=5, type=int,
                      help="training epochs")
  parser.add_argument("-lr", "--learning_rate", default=1e-2, type=float,
                      help="learning rate for training")
  parser.add_argument("-optim", "--optimizer", default='adagrad',
                      help="optimization algorithm [adagrad|adam|sgd]")
  parser.add_argument("-do", "--dropout", default=0, type=float,
                      help="dropout")
  parser.add_argument("-l2", "--l2_reg", default=0, type=float,
                      help="l2 regularizer")
  # data and saving
  parser.add_argument("-data", "--data_folder", default='/data/ml2/jernite/TextData/TaskData',
                      help="location of the data")
  #-------# Starting
  args = parser.parse_args()
  loaded_args = pickle.load(open(args.model_args_file))
  for k, v in vars(loaded_args).items():
    if k not in vars(args):
      vars(args)[k] = v
  #~ for arg, value in sorted(vars(args).items()):
    #~ print arg, value
  # make model
  print 'starting time', '\t', get_time_str()
  modules = make_modules(args)
  print 'made model', '\t', get_time_str()
  print 'dropout', args.dropout
  # loading data
  batched_data  = BatchedDataSemText(args)
  for e in range(args.epochs):
    run_epoch(batched_data, modules, args, training=True)
    if e % 1 == 0:
      run_epoch(batched_data, modules, args)

# CUDA_VISIBLE_DEVICES=1 python Train.py -order -next -conj -encode GRU -bid -h_dim 128 -lr 1e-2 -ir 1e-2 -epochs 50 -nm log_joint_bigru
# CUDA_VISIBLE_DEVICES=0 python TestMulti.py -bid -h_dim 256 -epochs 201 -bs 8 -model /data/ml2/jernite/TextData/model_bigru_256_10.pk -lr 0.1 -ir 0.1 -task msrp

# MSRP - eval_msrp_nhidval_10_10_hw_2_0.log:215 	    LOSS: 0.648882429614 	 ACCU: 0.712209302326 	2017-03-27_133815_314874
# TREC - eval_trec_nhidval_100_100_hw_0_10.log:681 	  LOSS: 0.620209315581 	 ACCU: 0.783406754772 	2017-03-25_035946_449091
# SUBJ - eval_subj_nhidval_10_10_hw_0_1.log:125 	    LOSS: 0.361709058344 	 ACCU: 0.847 	2017-03-25_012935_678955

# CUDA_VISIBLE_DEVICES=0 python TestMulti.py -pre ../SentOrderData/Embeddings/SortedGlove6B.pk -mod_args ../SentOrderData/Models/model_learn_gate_bigru.args.pk -model ../SentOrderData/Models/model_learn_gate_bigru.pt -epochs 50 -bs 16 -bow_pre_m lin -lr 0.05 -task msrp -comp_m hw -bow -ir 2e-3 -l2 2.5e-4
# MSRP-73.0

# CUDA_VISIBLE_DEVICES=0 python TestMulti.py -pre ../SentOrderData/Embeddings/SortedGlove840B.pk -mod_args ../SentOrderData/Models/model_learn_gate_bigru.args.pk -model ../SentOrderData/Models/model_learn_gate_bigru.pt -epochs 200 -bs 16 -bow_pre_m hw -lr 0.05 -task trec -comp_m hw -bow -ir 1e-2 -l2 0
# TREC 82.5


# cat slurm_219773_eval_msrp_50_none_lin_model_learn_lin_bigru.out | head -n 111
# MSRP none 71.6
# cat slurm_219770_eval_msrp_10_lin_lin_model_learn_lin_bigru.out | head -n 31
# MSRP lin  72.7
# cat slurm_219755_eval_msrp_1_gate_lin_model_learn_gate_bigru.out | head -n 331
# MSRP gate 73.3


# all    - 71.6
# order  - 68.5
# next   - 69.8
# conj   - 70.0
# books  - 70.0

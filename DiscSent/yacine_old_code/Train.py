import argparse
import shelve
import logging
import cPickle as pickle
from os.path import join as pjoin

import torch.nn as nn
import torch.optim as optim

from Utils import *


# make model, criterion, training ops
def make_modules(options):
  # model
  sent_encode   = Encoder(options).cuda()
  order_compare = Comparator('order', options).cuda()
  next_compare  = Comparator('next', options).cuda()
  conj_compare  = Comparator('conj', options,
                             nconj=9 if options.conj_coarse else 43).cuda()
  # initialization
  sent_encode.init_params(options.init_range)
  order_compare.init_params(options.init_range)
  next_compare.init_params(options.init_range)
  conj_compare.init_params(options.init_range)
  # criteria and optimization
  bce_crit      = nn.BCELoss()
  nll_crit      = nn.NLLLoss()
  params = [p for p in sent_encode.parameters()] + \
           [p for p in order_compare.parameters() if options.order_task] + \
           [p for p in next_compare.parameters() if options.next_task] + \
           [p for p in conj_compare.parameters() if options.conj_task]
  if options.optimizer == 'sgd':
    opt = optim.SGD(params, lr=options.learning_rate)
  elif options.optimizer == 'adam':
    opt = optim.Adam(params, lr=options.learning_rate)
  else:
    opt = optim.Adagrad(params, lr=options.learning_rate)
  modules = {'sentences'      : sent_encode,
             'order_compare'  : order_compare,
             'next_compare'   : next_compare,
             'conj_compare'   : conj_compare,
             'optim'          : opt,
             'order_crit'     : bce_crit,
             'next_crit'      : nll_crit,
             'conj_crit'      : nll_crit}
  return modules


# train or evaluate on a specific task
def run_task_epoch(task, data, modules, options, training=False):
  print 'starting epoch', e, '\t', get_time_str()
  logging.info("%s \t starting epoch %d", get_time_str(), e)
  tot_loss      = 0.
  tot_accu      = 0.
  bs            = options.batch_size
  mode          = 'train' if training else 'valid'
  scores        = []
  for b in range(len(data.data[task][mode]) / bs):
    b_xl, b_xr, b_ll, b_lr, b_y = data.next_batch(task, mode)
    # make sentence representation
    if task == 'order' or task == 'conj':
      sl  = modules['sentences'](b_xl, lengths=b_ll)
      sr  = modules['sentences'](b_xr, lengths=b_lr)
    elif task == 'next':
      sl  = modules['sentences'](b_xl, lengths=b_ll.view(-1)).view(options.batch_size,
                                                                   options.next_context, -1)
      sr  = modules['sentences'](b_xr, lengths=b_lr.view(-1)).view(options.batch_size,
                                                                   options.next_proposals, -1)
    # compute scores
    b_scores  = modules[task + '_compare'](sl, sr)
    scores    += [b_scores.data.cpu().numpy()]
    # compute accuracy
    if task == 'order':
      accu  = ((b_scores > 0.5) == (b_y > 0.5)).sum()
    elif task == 'conj' or task == 'next':
      _, preds  = b_scores.max(1)
      accu      = (preds == b_y).sum()
    # compute loss and (optionally) backprop
    loss      =  modules[task + '_crit'](b_scores, b_y)
    if training:
      modules['optim'].zero_grad()
      loss.backward()
      modules['optim'].step()
    # accumulate loss and accuracy
    tot_accu  += float(accu.data[0]) / bs
    tot_loss  += loss.data[0]
    if b % 400 == 0 and b > 0 and training:
      print b, '\t', tot_loss / (b + 1), '\t', tot_accu / (b + 1), '\t', get_time_str()
  # end of epoch
  b = len(data.data[task][mode]) / bs
  print b, '\t LOSS:', tot_loss / b, '\t ACCU:', tot_accu / b, '\t', get_time_str()
  if training:
    logging.info("%s \t Loss: %f \t Accu: %f", get_time_str(), tot_loss / b, tot_accu / b)
  else:
    logging.info("%s \t VALIDATING Loss: %f \t Accu: %f", get_time_str(), tot_loss / b, tot_accu / b)
  return (tot_accu / b, scores)


# joint training
def run_joint_epoch(data, modules, options):
  print 'starting epoch', e, '\t', get_time_str()
  logging.info("%s \t starting epoch %d", get_time_str(), e)
  tot_loss      = {'order': 0.,
                   'next' : 0.,
                   'conj' : 0.}
  tot_accu      = {'order': 0.,
                   'next' : 0.,
                   'conj' : 0.}
  tot_counts    = {'order': 0,
                   'next' : 0,
                   'conj' : 0}
  bs            = options.batch_size
  for b in range(options.n_batches):
    try:
      modules['optim'].zero_grad()
      # at most one batch per task
      if options.order_task:
        b_xl, b_xr, b_ll, b_lr, b_y = data.next_batch('order', 'train')
        sl  = modules['sentences'](b_xl, lengths=b_ll)
        sr  = modules['sentences'](b_xr, lengths=b_lr)
        b_scores  = modules['order_compare'](sl, sr)
        accu      = ((b_scores > 0.5) == (b_y > 0.5)).sum()
        loss      = modules['order_crit'](b_scores, b_y)
        loss.backward()
        tot_counts['order'] += 1
        tot_accu['order']   += float(accu.data[0]) / bs
        tot_loss['order']   += loss.data[0]
      if options.conj_task and (tot_counts['order'] % options.ns_conj == 0):
        b_xl, b_xr, b_ll, b_lr, b_y = data.next_batch('conj', 'train')
        sl  = modules['sentences'](b_xl, lengths=b_ll)
        sr  = modules['sentences'](b_xr, lengths=b_lr)
        b_scores  = modules['conj_compare'](sl, sr)
        _, preds  = b_scores.max(1)
        accu      = (preds == b_y).sum()
        loss      = modules['conj_crit'](b_scores, b_y)
        loss.backward()
        tot_counts['conj'] += 1
        tot_accu['conj']   += float(accu.data[0]) / bs
        tot_loss['conj']   += loss.data[0]
      if options.next_task and (tot_counts['order'] % options.ns_next == 0):
        b_xl, b_xr, b_ll, b_lr, b_y = data.next_batch('next', 'train')
        sl  = modules['sentences'](b_xl, lengths=b_ll.view(-1)).view(options.batch_size,
                                                                     options.next_context, -1)
        sr  = modules['sentences'](b_xr, lengths=b_lr.view(-1)).view(options.batch_size,
                                                                     options.next_proposals, -1)
        b_scores  = modules['next_compare'](sl, sr)
        _, preds  = b_scores.max(1)
        accu      = (preds == b_y).sum()
        loss      = modules['next_crit'](b_scores, b_y)
        loss.backward()
        tot_counts['next'] += 1
        tot_accu['next']   += float(accu.data[0]) / bs
        tot_loss['next']   += loss.data[0]
      modules['optim'].step()
      # accumulate loss and accuracy
      if b % 1000 == 0 and b > 0:
        print b, '\t',
        if options.order_task:
          print tot_accu['order'] / tot_counts['order'], '\t',
        if options.next_task:
          print tot_accu['next'] / tot_counts['next'], '\t',
        if options.conj_task:
          print tot_accu['conj'] / tot_counts['conj'], '\t',
        print ''
    except:
      print "MISSED BATCH", b
      logging.info("MISSED BATCH %d", b)
  # end of epoch
  if options.order_task:
    print "Order:", get_time_str(), tot_loss['order'] / tot_counts['order'], tot_accu['order'] / tot_counts['order']
    logging.info("%s \t Loss order: %f \t Accu order: %f", get_time_str(), tot_loss['order'] / tot_counts['order'], tot_accu['order'] / tot_counts['order'])
  if options.next_task:
    print "Next:", get_time_str(), tot_loss['next'] / tot_counts['next'], tot_accu['next'] / tot_counts['next']
    logging.info("%s \t Loss next: %f \t Accu next: %f", get_time_str(), tot_loss['next'] / tot_counts['next'], tot_accu['next'] / tot_counts['next'])
  if options.conj_task:
    print "Conjunction:", get_time_str(), tot_loss['conj'] / tot_counts['conj'], tot_accu['conj'] / tot_counts['conj']
    logging.info("%s \t Loss conj: %f \t Accu conj: %f", get_time_str(), tot_loss['conj'] / tot_counts['conj'], tot_accu['conj'] / tot_counts['conj'])

# main
if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='This program \
                 trains a model on a variety discourse modeling tasks.')
  # task choice
  parser.add_argument("-order", "--order_task", action="store_true",
                      help="train for order task")
  parser.add_argument("-next", "--next_task", action="store_true",
                      help="train for next task")
  parser.add_argument("-conj", "--conj_task", action="store_true",
                      help="train for conj task")
  parser.add_argument("-all", "--all_tasks", action="store_true",
                      help="train for all tasks")
  parser.add_argument("-ns_next", "--ns_next", default=1, type=int,
                      help="sub-sampling for next task")
  parser.add_argument("-ns_conj", "--ns_conj", default=1, type=int,
                      help="sub-sampling for conjunction task")
  parser.add_argument("-next_co", "--next_context", default=3, type=int,
                      help="size of context for next sentence prediction")
  parser.add_argument("-next_pro", "--next_proposals", default=5, type=int,
                      help="number of proposals for next sentence prediction")
  parser.add_argument("-conj_coarse", "--conj_coarse", action="store_true",
                      help="use grouping of conjunctions")
  # word embeddings
  parser.add_argument("-in_dim", "--embedding_size", default=256, type=int,
                      help="word embedding dimension")
  parser.add_argument("-pre", "--pre_trained", default='',
                      help="location of pre-trained word2vec model")
  parser.add_argument("-pre_m", "--pre_mode", default='learn',
                      help="fine-tuning strategy for word embeddings [hw|lin|gate|learn]")
  parser.add_argument("-chars", "--use_chars", action="store_true",
                      help="character aware word embeddings")
  parser.add_argument("-c_dim", "--char_embedding_size", default=16, type=int,
                      help="character embedding dimension")
  parser.add_argument("-c_win", "--max_char_window", default=6, type=int,
                      help="character embedding dimension convolutional window")
  # model: general
  parser.add_argument("-encode", "--encode_type", default='BoW',
                      help="type of sentence encoder to learn [BoW|GRU]")
  parser.add_argument("-h_dim", "--hidden_size", default=512, type=int,
                      help="sentence rerpresentation dimension")
  parser.add_argument("-ir", "--init_range", default=1, type=float,
                      help="random initialization parameter")
  parser.add_argument("-nvoc", "--voc_size", default=100000, type=int,
                      help="vocabulary size to use")
  parser.add_argument("-bow", "--use_bow", action="store_true",
                      help="also use BoW embedding of the sentence")
  parser.add_argument("-do", "--dropout", default=0, type=float,
                      help="dropout for task")
  parser.add_argument("-hw", "--highway", action="store_true",
                      help="use highway network in comparator")
  parser.add_argument("-comp_m", "--comp_mode", default='hw',
                      help="sentence transformation for comparator [hw|lin|gate]")
  parser.add_argument("-comp_nl", "--comp_nlayers", default=0, type=int,
                      help="layers of non-linearity in comparator")
  # model: GRU
  parser.add_argument("-nl_rnn", "--nlayers_rnn", default=1, type=int,
                      help="number of rnn layers")
  parser.add_argument("-do_rnn", "--rnn_dropout", default=0., type=float,
                      help="RNN dropout")
  parser.add_argument("-bid", "--bidirectional", action="store_true",
                      help="bidirectional RNN")
  # optimization
  parser.add_argument("-bs", "--batch_size", default=32, type=int,
                      help="batch size for training")
  parser.add_argument("-epochs", "--epochs", default=5, type=int,
                      help="training epochs")
  parser.add_argument("-n_batches", "--n_batches", default=16000, type=int,
                      help="training batches per epoch")
  parser.add_argument("-lr", "--learning_rate", default=0.01, type=float,
                      help="learning rate for training")
  parser.add_argument("-optim", "--optimizer", default='adagrad',
                      help="optimization algorithm [adagrad|adam|sgd]")
  # data and saving
  parser.add_argument("-data", "--data_folder", default='SentOrderData/TaskData',
                      help="location of the data")
  parser.add_argument("-o", "--output_folder", default='SentOrderData/Logs',
                      help="Where to save the model")
  parser.add_argument("-nm", "--name", default='model',
                      help="prefix for the saved model")
  #-------# Starting
  args = parser.parse_args()
  args.model_file = None
  if args.all_tasks:
    args.order_task = True
    args.next_task  = True
    args.conj_task  = True
  log_file = pjoin(args.output_folder, args.name + '_' + get_time_str() + '.log')
  logging.basicConfig(filename=log_file,level=logging.DEBUG)
  logging.info("ARGUMENTS<<<<<")
  for arg, value in sorted(vars(args).items()):
    print arg, value
    logging.info("Argument %s: %r", arg, value)
  logging.info(">>>>>ARGUMENTS")
  pickle.dump(args, open(log_file[:-4] + '.args.pk', 'wb'))
  # make model
  print 'starting time', '\t', get_time_str()
  logging.info("%s \t starting time", get_time_str())
  modules = make_modules(args)
  print 'made model', '\t', get_time_str()
  logging.info("%s \t made model", get_time_str())
  # loading data  
  batched_data  = BatchedDataText(args)
  # joint training
  best_score = 0
  for e in range(args.epochs): 
    run_joint_epoch(batched_data, modules, args)
    # validate
    if e % 1 == 0:
      tot_score = 0.
      tot_div   = 0.
      if args.order_task:
        acc, preds = run_epoch('order', batched_data, modules, args)
        tot_score += acc
        pickle.dump(preds, open('order_preds_%2d.pk' % (e,), 'wb'))
        tot_div   += 1.
      if args.next_task:
        acc, preds = run_epoch('next', batched_data, modules, args)
        tot_score += acc / args.ns_next
        pickle.dump(preds, open('next_preds_%2d.pk' % (e,), 'wb'))
        tot_div   += 1. / args.ns_next
      if args.conj_task:
        acc, preds = run_epoch('conj', batched_data, modules, args)
        tot_score += acc / args.ns_conj
        pickle.dump(preds, open('conj_preds_%2d.pk' % (e,), 'wb'))
        tot_div   += 1. / args.ns_conj
      tot_score /= tot_div
      if tot_score > best_score:
        best_score  = tot_score
        model_file  = log_file[:-4] + '_' + str(e) + '.pt'
        print "SAVING MODEL: epoch", e, "weighted", tot_score
        torch.save(modules["sentences"].state_dict(), model_file)

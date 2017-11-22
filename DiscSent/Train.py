import argparse
import logging
import cPickle as pickle
from os.path import join as pjoin
import random

import torch.nn as nn
import torch.optim as optim

from Utils import *

# make model, criterion, training ops
def make_modules(options):
  # model
  modules         = {}
  sent_encode     = Encoder(options).cuda()
  params          = [p for p in sent_encode.parameters()]
  sent_encode.init_params(options.init_range)
  modules['sentences']  = sent_encode
  all_scores      = AllScores(options, sent_encode)
  modules['all_compare']    = all_scores
  modules['order_compare']  = all_scores.order_compare
  modules['next_compare']   = all_scores.next_compare
  modules['conj_compare']   = all_scores.conj_compare
  modules['disclm_compare'] = all_scores.disclm_compare
  modules['genlm_compare']  = all_scores.genlm_compare
  modules['skip_compare']   = all_scores.skip_compare
  params              += [p for p in all_scores.parameters()]
  # criteria and optimization
  bce_crit      = nn.BCELoss()
  nll_crit      = nn.NLLLoss()
  nll_crit_lm   = nn.NLLLoss(ignore_index=0)
  nll_crit_lm_2 = nn.NLLLoss2d(ignore_index=0)
  if options.optimizer == 'sgd':
    opt = optim.SGD(params, lr=options.learning_rate)
  elif options.optimizer == 'adam':
    opt = optim.Adam(params, lr=options.learning_rate)
  else:
    opt = optim.Adagrad(params, lr=options.learning_rate)
  modules['optim']          = opt
  modules['order_crit']     = bce_crit
  modules['next_crit']      = nll_crit
  modules['conj_crit']      = nll_crit
  modules['disclm_crit']    = bce_crit
  modules['genlm_crit']     = nll_crit_lm
  modules['skip_crit']      = nll_crit_lm
  modules['fastsent_crit']  = nll_crit_lm_2
  return modules


# joint training
def run_joint_epoch(data, modules, options):
  print 'starting epoch', e, '\t', get_time_str()
  logging.info("%s \t starting epoch %d", get_time_str(), e)
  tot_loss      = {'order'    : 0.,
                   'next'     : 0.,
                   'conj'     : 0.,
                   'disclm'   : 0.,
                   'genlm'    : 0.,
                   'skip'     : 0.,
                   'fastsent' : 0.}
  tot_accu      = {'order'    : 0.,
                   'next'     : 0.,
                   'conj'     : 0.,
                   'disclm'   : 0.}
  tot_counts    = {'order'    : 0.,
                   'next'     : 0.,
                   'conj'     : 0.,
                   'disclm'   : 0.,
                   'genlm'    : 0.,
                   'skip'     : 0.,
                   'fastsent' : 0.}
  bs            = options.batch_size
  for b in range(options.n_batches):
    if True:
    #~ try:
      modules['optim'].zero_grad()
      # at most one batch per task
      if options.order_task:
        b_xl, b_xr, b_ll, b_lr, b_y = data.next_batch('order', 'train')
        sl, hl_for, hl_back	= modules['sentences'](b_xl, lengths=b_ll)
        sr, hr_for, hr_back	= modules['sentences'](b_xr, lengths=b_lr)
        b_scores  = modules['order_compare'](sl, sr)
        accu      = ((b_scores > 0.5) == (b_y > 0.5)).sum()
        loss      = modules['order_crit'](b_scores, b_y)
        loss.backward()
        tot_counts['order'] += 1
        tot_accu['order']   += float(accu.data[0]) / bs
        tot_loss['order']   += loss.data[0]
      if options.conj_task and (tot_counts['order'] % options.ns_conj == 0):
        b_xl, b_xr, b_ll, b_lr, b_y = data.next_batch('conj', 'train')
        sl, hl_for, hl_back	= modules['sentences'](b_xl, lengths=b_ll)
        sr, hr_for, hr_back	= modules['sentences'](b_xr, lengths=b_lr)
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
        sl, hl_for, hl_back = modules['sentences'](b_xl,
                                                   lengths=b_ll.view(-1))
        sl = sl.view(options.batch_size, options.next_context, -1)
        sr, hr_for, hr_back = modules['sentences'](b_xr,
                                                   lengths=b_lr.view(-1))
        sr = sr.view(options.batch_size, options.next_proposals, -1)
        b_scores  = modules['next_compare'](sl, sr)
        _, preds  = b_scores.max(1)
        accu      = (preds == b_y).sum()
        loss      = modules['next_crit'](b_scores, b_y)
        loss.backward()
        tot_counts['next'] += 1
        tot_accu['next']   += float(accu.data[0]) / bs
        tot_loss['next']   += loss.data[0]
      if options.disclm_task:
        b_x, b_l, b_y     = data.next_batch('disclm', 'train')
        if b_x is not None:
          s, h_for, h_back  = modules['sentences'](b_x, lengths=b_l)
          b_scores  = modules['disclm_compare'](s)
          accu      = ((b_scores > 0.5) == (b_y > 0.5)).sum()
          loss      = modules['disclm_crit'](b_scores, b_y)
          loss.backward()
          tot_counts['disclm']  += 1
          # TODO: fix hack
          if not options.order_task:
            tot_counts['order'] += 1
          tot_accu['disclm']    += float(accu.data[0]) / bs
          tot_loss['disclm']    += loss.data[0]
      if options.genlm_task and (tot_counts['order'] % options.ns_gen == 0):
        b_x, b_l, b_yf, b_yb  = data.next_batch('genlm', 'train')
        s, h_for, h_back      = modules['sentences'](b_x, lengths=b_l)
        b_scores_for, b_scores_back  = modules['genlm_compare'](h_for, h_back)
        loss          = modules['genlm_crit'](b_scores_for, b_yf.view(-1))
        tot_counts['genlm']  += 1
        if h_back is not None:
          loss        += modules['genlm_crit'](b_scores_back, b_yb.view(-1))
          tot_counts['genlm']+= 1
        loss                 *= options.mul_gen
        loss.backward()
        tot_loss['genlm']    += loss.data[0] / options.mul_gen
      if options.skip_task:
        b_x, b_xl, b_xr, b_l, b_ll, b_lr, b_yl, b_yr = data.next_batch('skip', 'train')
        s, h_for, h_back        = modules['sentences'](b_x, lengths=b_l)
        b_scores_l, b_scores_r  = modules['skip_compare'](s, s_left=b_xl, s_right=b_xr, l_left=b_ll, l_right=b_lr)
        loss = modules['skip_crit'](b_scores_l, b_yl.view(-1)) + modules['skip_crit'](b_scores_r, b_yr.view(-1))
        loss.backward()
        tot_counts['skip']  += 2
        tot_loss['skip']    += loss.data[0]
      if options.fastsent_task:
        b_x, b_l, b_xl, b_xr    = data.next_batch('fastsent', 'train')
        s, h_for, h_back        = modules['sentences'](b_x, lengths=b_l)
        b_scores_l, b_scores_r  = modules['fastsent_compare'](s, s_left=b_xl, s_right=b_xr)
        loss = modules['fastsent_crit'](b_scores_l, b_xl.view(1, b_xl.size(0), b_xl.size(1))) + \
               modules['fastsent_crit'](b_scores_r, b_xr.view(1, b_xr.size(0), b_xr.size(1)))
        loss.backward()
        tot_counts['fastsent']  += 2
        tot_loss['fastsent']    += loss.data[0]
      nn.utils.clip_grad_norm(modules['sentences'].parameters(), args.clip_norm)
      modules['optim'].step()
      # accumulate loss and accuracy
      if (b % 1000 == 0 and b > 0) or (b % 10 == 0 and 0 < b <= 100):
        print b, '\t',
        if options.order_task:
          print 'o', tot_accu['order'] / tot_counts['order'], '\t',  tot_loss['order'] / tot_counts['order'], '\t',
        if options.next_task:
          print 'n', tot_accu['next'] / tot_counts['next'], '\t',  tot_loss['next'] / tot_counts['order'],'\t',
        if options.conj_task:
          print 'c', tot_accu['conj'] / tot_counts['conj'], '\t',  tot_loss['conj'] / tot_counts['order'],'\t',
        if options.disclm_task and tot_counts['disclm'] != 0:
          print 'd', tot_accu['disclm'] / tot_counts['disclm'], '\t',  tot_loss['disclm'] / tot_counts['order'], '\t',
        if options.genlm_task:
          print 'g', tot_loss['genlm'] / tot_counts['genlm'], '\t',
          print 'ppl', math.exp(tot_loss['genlm'] / tot_counts['genlm']), '\t',
        if options.skip_task:
          print tot_loss['skip'] / tot_counts['skip'], '\t',
        if options.fastsent_task:
          print tot_loss['fastsent'] / tot_counts['fastsent'], '\t',
        print ''
    #~ except:
      #~ print "MISSED BATCH", b
      #~ logging.info("MISSED BATCH %d", b)
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
  if options.disclm_task and tot_counts['disclm'] != 0:
    print "DiscLM:", get_time_str(), tot_loss['disclm'] / tot_counts['disclm'], tot_accu['disclm'] / tot_counts['disclm']
    logging.info("%s \t Loss disclm: %f \t Accu disclm: %f", get_time_str(), tot_loss['disclm'] / tot_counts['disclm'],
                 tot_accu['disclm'] / tot_counts['disclm'])
  if options.genlm_task:
    print "GenLM:", get_time_str(), tot_loss['genlm'] / tot_counts['genlm']
    logging.info("%s \t Loss genlm: %f", get_time_str(), tot_loss['genlm'] / tot_counts['genlm'])
  if options.skip_task:
    print "Skip-thought:", get_time_str(), tot_loss['skip'] / tot_counts['skip']
    logging.info("%s \t Loss skip: %f", get_time_str(), tot_loss['skip'] / tot_counts['skip'])
  if options.fastsent_task:
    print "FastSent:", get_time_str(), tot_loss['fastsent'] / tot_counts['fastsent']
    logging.info("%s \t Loss fastsent: %f", get_time_str(), tot_loss['fastsent'] / tot_counts['fastsent'])


# train or evaluate on a specific task
def run_task_epoch(task, data, modules, options, training=False):
  print 'starting epoch', e, '\t', get_time_str()
  logging.info("%s \t starting epoch %d", get_time_str(), e)
  tot_loss      = 0.
  tot_accu      = 0.
  bs            = options.batch_size
  mode          = 'train' if training else 'valid'
  scores        = []
  for b in range(len(data.data[task]['0']) / bs):
    if task in ['order', 'conj', 'next']:
      b_xl, b_xr, b_ll, b_lr, b_y = data.next_batch(task, mode)
    elif task == 'disclm':
      b_x, b_l, b_y = data.next_batch('disclm', mode)
    elif task == 'genlm':
      b_x, b_l, b_yf, b_yb = data.next_batch('genlm', mode)
    # make sentence representation
    if task == 'order' or task == 'conj':
      sl, hl_for, hl_back	= modules['sentences'](b_xl, lengths=b_ll)
      sr, hr_for, hr_back	= modules['sentences'](b_xr, lengths=b_lr)
    elif task == 'next':
      sl, hl_for, hl_back = modules['sentences'](b_xl,
                                                 lengths=b_ll.view(-1))
      sl = sl.view(options.batch_size, options.next_context, -1)
      sr, hr_for, hr_back = modules['sentences'](b_xr,
                                                 lengths=b_lr.view(-1))
      sr = sr.view(options.batch_size, options.next_proposals, -1)
    elif task == 'disclm' or task == 'genlm':
      s, h_for, h_back    = modules['sentences'](b_x, lengths=b_l)
    # compute scores
    if task in ['order', 'conj', 'next']:
      b_scores  = modules[task + '_compare'](sl, sr)
      scores += [b_scores.data.cpu().numpy()]
    elif task == 'disclm':
      b_scores  = modules['disclm_compare'](s)
      scores += [b_scores.data.cpu().numpy()]
    elif task == 'genlm':
      b_scores_for, b_scores_back  = modules['genlm_compare'](h_for, h_back)
      # Don't save all scores, too slow
    # compute accuracy
    if task == 'order' or task == 'disclm':
      accu  = ((b_scores > 0.5) == (b_y > 0.5)).sum()
      loss = modules[task + '_crit'](b_scores, b_y)
    elif task == 'conj' or task == 'next':
      _, preds  = b_scores.max(1)
      accu      = (preds == b_y).sum()
      loss = modules[task + '_crit'](b_scores, b_y)
    elif task == 'genlm':
      loss          = modules['genlm_crit'](b_scores_for, b_yf.view(-1))
      if h_back:
        loss        += modules['genlm_crit'](b_scores_back, b_yb.view(-1))
    # compute loss and (optionally) backprop
    if training:
      modules['optim'].zero_grad()
      loss.backward()
      nn.utils.clip_grad_norm(modules['sentences'].parameters(), args.clip_norm)
      modules['optim'].step()
    # accumulate loss and accuracy
    if task != 'genlm':
      tot_accu  += float(accu.data[0]) / bs
    tot_loss  += loss.data[0]
    if b % 400 == 0 and b > 0 and training:
      print b, '\t', tot_loss / (b + 1), '\t', tot_accu / (b + 1), '\t', get_time_str()
  # end of epoch
  b = len(data.data[task]['0']) / bs
  print b, '\t LOSS:', tot_loss / b, '\t ACCU:', tot_accu / b, '\t', get_time_str()
  if task == 'genlm':
    print 'PPL:', math.exp(tot_loss / b)
  if training:
    if task == 'genlm':
      logging.info("%s \t Loss: %f \t Ppl: %f", get_time_str(), tot_loss / b, math.exp(tot_loss / b))
    else:
      logging.info("%s \t Loss: %f \t Accu: %f", get_time_str(), tot_loss / b, tot_accu / b)
  else:
    if task == 'genlm':
      logging.info("%s \t VALIDATING Loss: %f \t Ppl: %f", get_time_str(), tot_loss / b, math.exp(tot_loss / b))
    else:
      logging.info("%s \t VALIDATING Loss: %f \t Accu: %f", get_time_str(), tot_loss / b, tot_accu / b)
  return (tot_accu / b, scores)


def generate(options, generate_size=320, temperature=1):
  begins = [['<S>'] for _ in range(generate_size)]
  result = [[] for _ in range(generate_size)]
  hidden = None
  for i in range(options.max_length):
    batch_var, batch_len = batched_data.batch_to_vars(begins, mode='generate')
    s, hidden, _ = modules['sentences'](batch_var, lengths=batch_len, hidden=hidden)
    b_scores, _ = modules['genlm_compare'](hidden, _)
    word_weights = b_scores.squeeze().data.div(temperature).exp()
    word_idx = torch.multinomial(word_weights, 1)
    for j in range(generate_size):
      begins[j] = [batched_data.vocab[word_idx[j][0]]]
      result[j].append(begins[j][0])
    hidden = hidden.unsqueeze(0)
  for j in range(generate_size):
    for p in range(len(result[j])):
      if result[j][p] == '</S>':
        result[j] = ' '.join(result[j][:p])
        break
    if type(result[j]) == list:
      result[j] = ' '.join(result[j])
  return result


def make_disc_shelve(options, large_batch_size=64000):
  print 'begin to generate'
  batch_number = options.n_batches * options.batch_size // large_batch_size + 2
  batched_data.data['disclm']['len'] = batch_number
  # valid
  sent_list = list()
  while len(sent_list) < large_batch_size // 2:
      sent_list.extend(generate(options))
  sent_list = sent_list[:large_batch_size // 2]
  sent_list = map(lambda x: (x, 0), sent_list)
  sent_list.extend(batched_data.data['genlm']['0'][:large_batch_size-len(sent_list)])
  random.shuffle(sent_list)
  batched_data.data['disclm']['0'] = sent_list
  # train
  for i in range(1, batch_number):
    sent_list = list()
    while len(sent_list) < large_batch_size // 2:
      sent_list.extend(generate(options))
    sent_list = sent_list[:large_batch_size // 2]
    sent_list = map(lambda x: (x, 0), sent_list)
    sent_list.extend(batched_data.get_values('genlm', large_batch_size - len(sent_list)))
    random.shuffle(sent_list)
    batched_data.data['disclm'][str(i)] = sent_list
  batched_data.data['disclm'].sync()
  batched_data.cur_batch_idx['disclm'] = {'nbatches': batched_data.data['disclm']['len'],
                                          'train': 0,
                                          'valid': 0,
                                          'shlf_idx': 1}


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
  parser.add_argument("-disc", "--disclm_task", action="store_true",
                      help="train for discriminative LM task")
  parser.add_argument("-dynamic", "--dynamic", action="store_true",
                      help="whether generate fake sentences dynamically")
  parser.add_argument("-gen", "--genlm_task", action="store_true",
                      help="train for language modeling task")
  parser.add_argument("-skip", "--skip_task", action="store_true",
                      help="train for skip thought task")
  parser.add_argument("-fastsent", "--fastsent_task", action="store_true",
                      help="train for fastsent task")
  parser.add_argument("-all", "--all_tasks", action="store_true",
                      help="train for all tasks")
  parser.add_argument("-ns_next", "--ns_next", default=4, type=int,
                      help="sub-sampling for next task")
  parser.add_argument("-ns_conj", "--ns_conj", default=4, type=int,
                      help="sub-sampling for conjunction task")
  parser.add_argument("-ns_gen", "--ns_gen", default=4, type=int,
                      help="sub-sampling for language modeling task")
  parser.add_argument("-m_gen", "--mul_gen", default=1., type=float,
                      help="weight loss for language modeling task")
  parser.add_argument("-ns_disc", "--ns_disc", default=1, type=int,
                      help="sub-sampling for disclm task")
  parser.add_argument("-next_co", "--next_context", default=3, type=int,
                      help="size of context for next sentence prediction")
  parser.add_argument("-next_pro", "--next_proposals", default=5, type=int,
                      help="number of proposals for next sentence prediction")
  parser.add_argument("-conj_coarse", "--conj_coarse", action="store_true",
                      help="use grouping of conjunctions")
  parser.add_argument("-max_l", "--max_length", default=64, type=int,
                      help="truncate sentences longer than")
  # word embeddings
  parser.add_argument("-in_dim", "--embedding_size", default=256, type=int,
                      help="word embedding dimension")
  parser.add_argument("-voc", "--voc_file", default='/data/ml2/jernite/TextData/Embeddings/Glove840BVocab500.pk',
                      help="location of file with vocabulary")
  parser.add_argument("-pre", "--pre_trained", default='/data/ml2/jernite/TextData/Embeddings/Glove840BVectors500.pk',
                      help="location of pre-trained word2vec model")
  parser.add_argument("-pre_m", "--pre_mode", default='learn',
                      help="fine-tuning strategy for word embeddings [hw|lin|gate|learn]")
  # model: general
  parser.add_argument("-encode", "--encode_type", default='BoW',
                      help="type of sentence encoder to learn [BoW|GRU|LSTM]")
  parser.add_argument("-h_dim", "--hidden_size", default=512, type=int,
                      help="sentence rerpresentation dimension")
  parser.add_argument("-h_dim_st", "--hidden_size_st", default=512, type=int,
                      help="hidden dimenstion dimension for skip-thought rep")
  parser.add_argument("-attn_dim", "--attention_dim", default=64, type=int,
                      help="sentence rerpresentation dimension")
  parser.add_argument("-ir", "--init_range", default=1, type=float,
                      help="random initialization parameter")
  parser.add_argument("-evoc", "--encode_voc_size", default=500000, type=int,
                      help="vocabulary size to use")
  parser.add_argument("-dvoc", "--decode_voc_size", default=20000, type=int,
                      help="vocabulary size to use")
  parser.add_argument("-bow", "--use_bow", action="store_true",
                      help="also use BoW embedding of the sentence")
  parser.add_argument("-combine", "--combine_hidden", default='sum',
                      help="sentence transformation for comparator [sum|max|attn]")
  parser.add_argument("-do", "--dropout", default=0, type=float,
                      help="dropout for task")
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
  parser.add_argument("-epochs", "--epochs", default=100, type=int,
                      help="training epochs")
  parser.add_argument("-n_batches", "--n_batches", default=32000, type=int,
                      help="training batches per epoch")
  parser.add_argument("-lr", "--learning_rate", default=0.001, type=float,
                      help="learning rate for training")
  parser.add_argument("-optim", "--optimizer", default='adagrad',
                      help="optimization algorithm [adagrad|adam|sgd]")
  parser.add_argument("-clip", "--clip_norm", default=5, type=float,
                      help="clip norm for LSTM")
  # data and saving
  parser.add_argument("-data", "--data_folder", default='Data/Gutenberg',
                      help="location of the data")
  parser.add_argument("-o", "--output_folder", default='Data/Logs',
                      help="Where to save the model")
  parser.add_argument("-nm", "--name", default='model',
                      help="prefix for the saved model")
  parser.add_argument("-save_preds", "--save_preds", action="store_true",
                      help="record predictions")
  # continue training
  parser.add_argument("-cont", "--cont_model", default='',
                      help="continue training a saved model, encoder module")
  parser.add_argument("-cont_c", "--cont_compare", default='',
                      help="continue training a saved model, compare module")
  parser.add_argument("-cont_o", "--cont_optim", default='',
                      help="continue training a saved model, optimization module")
  parser.add_argument("-cont_b", "--cont_batch", default=0, type=int,
                      help="continue training from batch")
  #-------# Starting
  args = parser.parse_args()
  args.model_file = None
  if args.all_tasks:
    args.order_task   = True
    args.next_task    = True
    args.conj_task    = True
    args.disclm_task  = True
    # all_tasks means all discriminative tasks
    #~ args.genlm_task   = True
  args.make_disc_shelve = True
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
  batched_data  = BatchedData(args)
  # (optional) continue training
  if args.cont_model != '':
    modules['sentences'].load_state_dict(torch.load(args.cont_model))
    modules['all_compare'].load_state_dict(torch.load(args.cont_compare))
    modules['optim'].load_state_dict(torch.load(args.cont_optim))
    if args.order_task:
      seen_o  = (args.cont_batch) * 32000
      batched_data.cur_batch_idx['order']['shlf_idx'] = seen_o / 32000 + 1
      if batched_data.cur_batch_idx['order']['shlf_idx'] >= batched_data.cur_batch_idx['order']['nbatches']:
        batched_data.cur_batch_idx['order']['shlf_idx'] = 1
      str_id = str(batched_data.cur_batch_idx['order']['shlf_idx'])
      batched_data.cur_batch_idx['order']['train'] = seen_o % (len(batched_data.data['order'][str_id]) / args.batch_size)
    if args.next_task:
      seen_n  = seen_o / args.ns_next
      batched_data.cur_batch_idx['next']['shlf_idx'] = seen_n / 32000 + 1
      if batched_data.cur_batch_idx['next']['shlf_idx'] >= batched_data.cur_batch_idx['next']['nbatches']:
        batched_data.cur_batch_idx['next']['shlf_idx'] = 1
      str_id = str(batched_data.cur_batch_idx['next']['shlf_idx'])
      batched_data.cur_batch_idx['next']['train'] = seen_n % (len(batched_data.data['next'][str_id]) / args.batch_size)
    if args.conj_task:
      seen_c  = seen_o / args.ns_conj
      batched_data.cur_batch_idx['conj']['shlf_idx'] = seen_c / 32000 + 1
      if batched_data.cur_batch_idx['conj']['shlf_idx'] >= batched_data.cur_batch_idx['conj']['nbatches']:
        batched_data.cur_batch_idx['conj']['shlf_idx'] = 1
      str_id = str(batched_data.cur_batch_idx['conj']['shlf_idx'])
      batched_data.cur_batch_idx['conj']['train'] = seen_c % (len(batched_data.data['conj'][str_id]) / args.batch_size)
    if args.disclm_task:
      seen_d  = seen_o
      batched_data.cur_batch_idx['disclm']['shlf_idx'] = seen_d / 32000 + 1
      if batched_data.cur_batch_idx['disclm']['shlf_idx'] >= batched_data.cur_batch_idx['disclm']['nbatches']:
        batched_data.cur_batch_idx['disclm']['shlf_idx'] = 1
      str_id = str(batched_data.cur_batch_idx['disclm']['shlf_idx'])
      batched_data.cur_batch_idx['disclm']['train'] = seen_d % (len(batched_data.data['disclm'][str_id]) / args.batch_size)
    if args.genlm_task:
      seen_g  = seen_o / args.ns_gen
      batched_data.cur_batch_idx['genlm']['shlf_idx'] = seen_g / 32000 + 1
      if batched_data.cur_batch_idx['genlm']['shlf_idx'] >= batched_data.cur_batch_idx['genlm']['nbatches']:
        batched_data.cur_batch_idx['genlm']['shlf_idx'] = 1
      str_id = str(batched_data.cur_batch_idx['genlm']['shlf_idx'])
      batched_data.cur_batch_idx['genlm']['train'] = seen_g % (len(batched_data.data['genlm'][str_id]) / args.batch_size)
    if args.skip_task:
      seen_t  = (args.cont_batch) * 32000
      batched_data.cur_batch_idx['skip']['shlf_idx'] = seen_t / 32000 + 1
      if batched_data.cur_batch_idx['skip']['shlf_idx'] >= batched_data.cur_batch_idx['skip']['nbatches']:
        batched_data.cur_batch_idx['skip']['shlf_idx'] = 1
      str_id = str(batched_data.cur_batch_idx['skip']['shlf_idx'])
      batched_data.cur_batch_idx['skip']['train'] = seen_t % (len(batched_data.data['skip'][str_id]) / args.batch_size)
  # joint training
  best_score = 0.
  tot_score  = 0.
  for e in range(args.cont_batch, args.epochs):
    run_joint_epoch(batched_data, modules, args)
    if args.dynamic and args.make_disc_shelve:
        make_disc_shelve(args)
        args.make_disc_shelve = False
    # validate
    if e % 1 == 0:
      tot_score = 0.
      tot_div   = 0.
      if True:
      # if not (args.skip_task or args.fastsent_task or args.genlm_task):
        if args.order_task:
          acc, preds = run_task_epoch('order', batched_data, modules, args)
          tot_score += acc
          if args.save_preds:
            pickle.dump(preds, open(log_file[:-4] + '_order_preds_%02d.pk' % (e,), 'wb'))
          tot_div   += 1.
        if args.next_task:
          acc, preds = run_task_epoch('next', batched_data, modules, args)
          tot_score += acc / args.ns_next
          if args.save_preds:
            pickle.dump(preds, open(log_file[:-4] + '_next_preds_%02d.pk' % (e,), 'wb'))
          tot_div   += 1. / args.ns_next
        if args.conj_task:
          acc, preds = run_task_epoch('conj', batched_data, modules, args)
          tot_score += acc / args.ns_conj
          if args.save_preds:
            pickle.dump(preds, open(log_file[:-4] + '_conj_preds_%02d.pk' % (e,), 'wb'))
          tot_div   += 1. / args.ns_conj
        if args.disclm_task:
          acc, preds = run_task_epoch('disclm', batched_data, modules, args)
          tot_score += acc / args.ns_disc
          if args.save_preds:
            pickle.dump(preds, open(log_file[:-4] + '_disclm_preds_%02d.pk' % (e,), 'wb'))
          tot_div   += 1. / args.ns_disc
          if args.dynamic and acc > 0.7:
            args.make_disc_shelve = True
        if args.genlm_task:
          _, _ = run_task_epoch('genlm', batched_data, modules, args)
        if tot_div > 0:
          tot_score /= tot_div

      print "SAVING MODEL: epoch", e, "weighted", tot_score
      model_file  = log_file[:-4] + '_' + str(e) + '.pt'
      torch.save(modules["sentences"].state_dict(), model_file)
      compare_file  = log_file[:-4] + '_comp_' + str(e) + '.pt'
      torch.save(modules["all_compare"].state_dict(), compare_file)
      optim_file  = log_file[:-4] + '_optim_' + str(e) + '.pt'
      torch.save(modules["optim"].state_dict(), compare_file)

# model_onc_joint_2017-09-28_115204_063300_10.pt
# model_a_joint_2017-09-28_115205_220045_7.pt
# model_ag_joint_2017-09-28_115211_988592_8.pt
# model_skip_joint_2017-09-28_115216_099235_19.pt

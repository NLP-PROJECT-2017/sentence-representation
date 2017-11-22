import shelve
import codecs
import numpy as np
import cPickle as pickle
from os.path import join as pjoin
from datetime import datetime

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Highway network
class Highway(nn.Module):
  
  
  def __init__(self, in_dim, out_dim, bias=True):
    super(Highway, self).__init__()
    self.hw_gate  = nn.Linear(in_dim, out_dim, bias)
    self.hw_tran  = nn.Linear(in_dim, out_dim, bias)
    self.proj     = nn.Linear(in_dim, out_dim, bias)

  
  def forward(self, x):
    gate  = F.sigmoid(self.hw_gate(x))
    tran  = F.tanh(self.hw_tran(x))
    proj  = self.proj(x)
    res   = gate * tran + (1 - gate) * proj
    return res

# Gated transformation network
class Gated(nn.Module):
  
  
  def __init__(self, in_dim, out_dim, bias=True):
    super(Gated, self).__init__()
    self.hw_gate  = nn.Linear(in_dim, out_dim, bias)
    self.hw_tran  = nn.Linear(in_dim, out_dim, bias)

  
  def forward(self, x):
    gate  = F.sigmoid(self.hw_gate(x))
    res   = gate * self.hw_tran(x)
    return res

# List conjunctions, and coarse grouping
conjunct_dic  = {'addition'  : ['again', 'also', 'besides', 'finally',
                                'further', 'furthermore', 'moreover',
                                'in addition'],
                 'contrast'  : ['anyway', 'however', 'instead', 
                                'nevertherless', 'otherwise', 'contrarily',
                                'conversely', 'nonetheless', 'in contrast',
                                'rather'], # 'on the other hand', 
                 'time'      : ['meanwhile', 'next', 'then',
                                'now', 'thereafter'],
                 'result'    : ['accordingly', 'consequently', 'hence',
                                'henceforth', 'therefore', 'thus',
                                'incidentally', 'subsequently'],
                 'specific'  : ['namely', 'specifically', 'notably',
                                'that is', 'for example'],
                 'compare'   : ['likewise', 'similarly'],
                 'strengthen': ['indeed', 'in fact'],
                 'return'    : ['still'], # 'nevertheless'],
                 'recognize' : ['undoubtedly', 'certainly']}

conj_list_1 = [conj for ls in conjunct_dic.values() for conj in ls]
conj_map_1  = dict([(conj, i) for i, conj in enumerate(conj_list_1)])
conj_list_2 = conjunct_dic.keys()
conj_map_2  = dict([(conj, i) for i, conj in enumerate(conj_list_2)])

# Data helper for unsupervised training
class BatchedDataText():
  
  # load data from files
  def __init__(self, options):
    self.options  = options
    vocab_file    = pjoin(options.data_folder, 'SortedVocab.txt')
    f = codecs.open(vocab_file, 'r', encoding='utf8')
    vocab = ['</S>', '<S>'] + [line.strip() for line in f]
    f.close()
    self.rev_voc = dict([(w, i)
                         for i, w in enumerate(vocab[:options.voc_size])])
    self.batches        = {}
    self.cur_batch_idx  = {}
    self.data           = {}
    if options.order_task:
      # load shelve files
      self.data['order']          = {}
      self.data['order']['shlf']  = shelve.open(pjoin(options.data_folder,
                                                      'OrderShelve.shlf'))
      self.data['order']['train'] = self.data['order']['shlf']['0']
      self.data['order']['valid'] = pickle.load(open(pjoin(options.data_folder,
                                                           'OrderValid.pk')))[:32000]
      # prepare batch variables
      self.batches['order']       = {}
      self.cur_batch_idx['order'] = {'train'     : 0,
                                     'valid'     : 0,
                                     'next_shlf' : 0}
    if options.conj_task:
      # load shelve files
      self.data['conj']           = {}
      self.data['conj']['shlf']   = shelve.open(pjoin(options.data_folder,
                                                      'ConjShelve.shlf'))
      self.data['conj']['train']  = self.data['conj']['shlf']['0']
      self.data['conj']['valid']  = pickle.load(open(pjoin(options.data_folder,
                                                           'ConjValid.pk')))[:32000]
      # prepare batch variables
      self.batches['conj']        = {}
      self.cur_batch_idx['conj']  = {'train'     : 0,
                                     'valid'     : 0,
                                     'next_shlf' : 0}
    if options.next_task:
      # load shelve files
      self.data['next']           = {}
      self.data['next']['shlf']   = shelve.open(pjoin(options.data_folder,
                                                      'NextShelve.shlf'))
      self.data['next']['train']  = self.data['next']['shlf']['0']
      self.data['next']['valid']  = pickle.load(open(pjoin(options.data_folder,
                                                           'NextValid.pk')))[:32000]
      # prepare batch variables
      self.batches['next']        = {}
      self.cur_batch_idx['next']  = {'train'     : 0,
                                     'valid'     : 0,
                                     'next_shlf' : 0}
  
  
  # words to Variables
  def batch_to_vars(self, batch_sen):
    lengths = [len(sen) for sen in batch_sen]
    max_len = max(lengths) + 1
    padded_batch_sen_l = [sen + ['</S>'] * (max_len - lengths[i])
                          for i, sen in enumerate(batch_sen)]
    batch_array = np.array([[self.rev_voc.get(w, 1) for w in sen]
                            for sen in padded_batch_sen_l])
    batch_var   = Variable(torch.Tensor(batch_array), requires_grad=False).long().cuda()
    batch_len   = Variable(torch.Tensor(lengths), requires_grad=False).long().cuda()
    return (batch_var, batch_len)
  
  
  def word_to_chars(self, w, max_len):
    res = [15] + [ord(c) for c in w] + [14]
    res = res + [0] * (max_len - len(res))
    return res
  
  # chars to Variables
  def batch_chars_to_vars(self, batch_sen):
    lengths = [len(sen) for sen in batch_sen]
    max_len = max(lengths) + 1
    char_lengths  = [len(w) + 2 for sen in batch_sen for w in sen]
    max_char_len  = max(char_lengths) + 1
    # TODO
    padded_batch_sen_l = [[self.word_to_chars(w, max_char_len)
                           for w in (sen + [''] * (max_len - lengths[i]))]
                          for i, sen in enumerate(batch_sen)]
    batch_array = np.array(padded_batch_sen_l)
    batch_var   = Variable(torch.Tensor(batch_array), requires_grad=False).long().cuda()
    batch_len   = Variable(torch.Tensor(lengths), requires_grad=False).long().cuda()
    return (batch_var, batch_len)


  # prepare next batch
  def next_batch(self, task, mode):
    b   = self.cur_batch_idx[task][mode]
    bs  = self.options.batch_size
    batch = self.data[task][mode][b * bs:(b+1) * bs]
    # input
    if task == 'next':
      batch_sen_l = [['<S>'] + sen.split() for ex in batch for sen in ex[0]]
      batch_sen_r = [['<S>'] + sen.split() for ex in batch for sen in ex[1]]
    else:
      batch_sen_l = [['<S>'] + ex[0].split() for ex in batch]
      batch_sen_r = [['<S>'] + ex[1].split() for ex in batch]
    if self.options.use_chars:
      b_xl, b_ll  = self.batch_chars_to_vars(batch_sen_l)
    else:
      b_xl, b_ll  = self.batch_to_vars(batch_sen_l)
    self.batches[task]['b_xl']  = b_xl
    self.batches[task]['b_ll']  = b_ll
    if self.options.use_chars:
      b_xr, b_lr  = self.batch_chars_to_vars(batch_sen_r)
    else:
      b_xr, b_lr  = self.batch_to_vars(batch_sen_r)
    self.batches[task]['b_xr']  = b_xr
    self.batches[task]['b_lr']  = b_lr
    # label
    if task == 'conj':
      if self.options.conj_coarse:
        b_y = Variable(torch.Tensor([conj_map_2[ex[3]] for ex in batch]),
                       requires_grad=False).long().cuda()
      else:
        b_y = Variable(torch.Tensor([conj_map_1[ex[2]] for ex in batch]),
                       requires_grad=False).long().cuda()
    elif task == 'order':
      b_y = Variable(torch.Tensor([ex[2] for ex in batch]), requires_grad=False).cuda()
    elif task == 'next':
      b_y = Variable(torch.Tensor([ex[2] for ex in batch]), requires_grad=False).long().cuda()
    self.batches[task]['b_y']  = b_y
    # move relevant batch pointer
    self.cur_batch_idx[task][mode]  = (b + 1) % (len(self.data[task][mode]) / bs)
    if (mode == 'train') and self.cur_batch_idx[task][mode] == 0:
      try:
        self.cur_batch_idx[task]['next_shlf'] += 1
        str_id = str(self.cur_batch_idx[task]['next_shlf'])
        self.data[task]['train'] = self.data[task]['shlf'][str_id]
      except:
        self.cur_batch_idx[task]['next_shlf'] = 0
        str_id = str(self.cur_batch_idx[task]['next_shlf'])
        self.data[task]['train'] = self.data[task]['shlf'][str_id]
      self.data[task]['shlf'].sync()
    return (b_xl, b_xr, b_ll, b_lr, b_y)

# Data helper for external evaluation
class BatchedDataSemText():
  
  # load data from files
  def __init__(self, options):
    self.options  = options
    vocab_file    = pjoin(options.data_folder, 'SortedVocab.txt')
    f = codecs.open(vocab_file, 'r', encoding='utf8')
    vocab = ['</S>', '<S>'] + [line.strip() for line in f]
    f.close()
    self.rev_voc = dict([(w, i)
                         for i, w in enumerate(vocab[:options.voc_size])])
    self.batches        = {}
    self.cur_batch_idx  = {}
    self.data           = {}
    
    # MSR paraphrase
    msrp_train, msrp_test     = pickle.load(open(pjoin(options.data_folder,
                                                       'msrp_tt.pk')))
    # load shelve files
    self.data['msrp']           = {}
    self.data['msrp']['train']  = msrp_train
    self.data['msrp']['valid']  = msrp_test
    self.batches['msrp']       = {}
    self.cur_batch_idx['msrp'] = {'train'     : 0,
                                  'valid'     : 0}
    # TREC question classification
    trec_train, trec_test     = pickle.load(open(pjoin(options.data_folder,
                                                       'trec_tt.pk')))
    self.data['trec']           = {}
    self.data['trec']['train']  = trec_train
    self.data['trec']['valid']  = trec_test
    self.batches['trec']       = {}
    self.cur_batch_idx['trec'] = {'train'     : 0,
                                  'valid'     : 0}
    # Stanford Sentiment Treebank
    sst2_train, _, sst2_test  = pickle.load(open(pjoin(options.data_folder,
                                                       'sst2_tvt.pk')))
    self.data['sst2']           = {}
    self.data['sst2']['train']  = sst2_train
    self.data['sst2']['valid']  = sst2_test
    self.batches['sst2']       = {}
    self.cur_batch_idx['sst2'] = {'train'     : 0,
                                  'valid'     : 0}
    # SNLI
    snli_train, _, snli_test  = pickle.load(open(pjoin(options.data_folder,
                                                       'snli_tvt.pk')))
    self.data['snli']           = {}
    self.data['snli']['train']  = snli_train
    self.data['snli']['valid']  = snli_test
    self.batches['snli']       = {}
    self.cur_batch_idx['snli'] = {'train'     : 0,
                                  'valid'     : 0}
    # SICK sentence similarity
    sick_train, _, sick_test  = pickle.load(open(pjoin(options.data_folder,
                                                       'sick_tvt.pk')))
    self.data['sick']           = {}
    self.data['sick']['train']  = sick_train
    self.data['sick']['valid']  = sick_test
    self.batches['sick']       = {}
    self.cur_batch_idx['sick'] = {'train'     : 0,
                                  'valid'     : 0}
    
    # SUBJ subjectivity evaluation
    # TODO: 10-fold
    subj_folds = pickle.load(open(pjoin(options.data_folder, 'subj_folds.pk')))
    self.data['subj']          = {}
    self.data['subj']['train'] = [x for i, fold in enumerate(subj_folds) if i != options.choose_subj_test for x in fold]
    self.data['subj']['valid'] = subj_folds[options.choose_subj_test]
    self.batches['subj']       = {}
    self.cur_batch_idx['subj'] = {'train'     : 0,
                                  'valid'     : 0}
  
  
  # words to Variables
  def batch_to_vars(self, batch_sen):
    lengths = [len(sen) for sen in batch_sen]
    max_len = max(lengths) + 1
    padded_batch_sen_l = [sen + ['</S>'] * (max_len - lengths[i])
                          for i, sen in enumerate(batch_sen)]
    batch_array = np.array([[self.rev_voc.get(w, 1) for w in sen]
                            for sen in padded_batch_sen_l])
    batch_var   = Variable(torch.Tensor(batch_array), requires_grad=False).long().cuda()
    batch_len   = Variable(torch.Tensor(lengths), requires_grad=False).long().cuda()
    return (batch_var, batch_len)
  
  
  # prepare next batch
  def next_batch(self, task, mode):
    b   = self.cur_batch_idx[task][mode]
    bs  = self.options.batch_size
    batch = self.data[task][mode][b * bs:(b+1) * bs]
    # input
    if task in ['msrp', 'sick', 'snli']:
      batch_sen_l = [['<S>'] + ex[0].split() for ex in batch]
      batch_sen_r = [['<S>'] + ex[1].split() for ex in batch]
      b_xl, b_ll  = self.batch_to_vars(batch_sen_l)
      b_xr, b_lr  = self.batch_to_vars(batch_sen_r)
      if task == 'msrp' or task == 'sick':
        b_y = Variable(torch.Tensor([ex[2] for ex in batch]), requires_grad=False).cuda()
      elif task == 'snli':
        b_y = Variable(torch.Tensor([ex[2] for ex in batch]), requires_grad=False).long().cuda()
    elif task in ['sst2', 'subj', 'trec']:
      batch_sen   = [['<S>'] + ex[0].split() for ex in batch]
      b_xl, b_ll  = self.batch_to_vars(batch_sen)
      b_xr, b_lr = (None, None)
      if task == 'sst2' or task == 'subj':
        b_y = Variable(torch.Tensor([ex[1] for ex in batch]), requires_grad=False).cuda()
      elif  task == 'trec':
        b_y = Variable(torch.Tensor([ex[1] for ex in batch]), requires_grad=False).long().cuda()
    self.batches[task]['b_xl']  = b_xl
    self.batches[task]['b_ll']  = b_ll
    self.batches[task]['b_xr']  = b_xr
    self.batches[task]['b_lr']  = b_lr
    self.batches[task]['b_y']  = b_y
    # move relevant batch pointer
    self.cur_batch_idx[task][mode]  = (b + 1) % (len(self.data[task][mode]) / bs)
    return (b_xl, b_xr, b_ll, b_lr, b_y)


# print time
def get_time_str():
  time_str = str(datetime.now())
  time_str = '_'.join(time_str.split())
  time_str = '_'.join(time_str.split('.'))
  time_str = ''.join(time_str.split(':'))
  return time_str

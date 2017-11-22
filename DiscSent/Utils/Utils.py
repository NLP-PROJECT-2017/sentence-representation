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

# TODO general use functions
# print time
def get_time_str():
  time_str = str(datetime.now())
  time_str = '_'.join(time_str.split())
  time_str = '_'.join(time_str.split('.'))
  time_str = ''.join(time_str.split(':'))
  return time_str

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

# Highway network (no projection)
class HighwaySquare(nn.Module):
  
  
  def __init__(self, dim, bias=True):
    super(HighwaySquare, self).__init__()
    self.hw_gate  = nn.Linear(dim, dim, bias)
    self.hw_tran  = nn.Linear(dim, dim, bias)

  
  def forward(self, x):
    gate  = F.sigmoid(self.hw_gate(x))
    tran  = F.tanh(self.hw_tran(x))
    res   = gate * tran + (1 - gate) * x
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


# make vocab and prepare embeddings
#~ embed_file = '/data/ml2/jernite/TextData/Embeddings/glove.840B.300d.txt'
#~ vocab = ['<PAD>', '</S>', '<S>', '<UNK>']
#~ embeds = [np.random.normal(0, 1, (300,)) for _ in range(4)]

#~ f = open(embed_file)
#~ ct = 0
#~ for l in f:
  #~ ct += 1
  #~ if ct % 100000 == 0:
    #~ print ct
  #~ tb = l.strip().split()
  #~ if len(tb) == 301:
    #~ vocab += [tb[0]]
    #~ embeds += [np.array([float(x) for x in tb[1:]])]
  #~ else:
    #~ print 'error', tb, l

#~ f.close()

#~ embed_array = np.array(embeds)



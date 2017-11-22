from Utils import *
from scipy.stats import ortho_group

class HighwaySquare(nn.Module):
  
  
  def __init__(self, dim, bias=True):
    super(Highway, self).__init__()
    self.hw_gate  = nn.Linear(dim, dim, bias)
    self.hw_tran  = nn.Linear(dim, dim, bias)

  
  def forward(self, x):
    gate  = F.sigmoid(self.hw_gate(x))
    tran  = F.tanh(self.hw_tran(x))
    res   = gate * tran + (1 - gate) * x
    return res


# Word embedding: either pre-trained or fully learned
class WordEmbed(nn.Module):
  
  
  def __init__(self, options):
    super(WordEmbed, self).__init__()
    self.voc_size     = options.voc_size
    self.pre_trained  = (options.pre_mode != 'learn')
    if self.pre_trained:
      with open(options.pre_trained) as f:
        voc, self.pre_embeddings  = pickle.load(f)
        self.pre_embeddings[0]    = 0
        self.pre_lut              = nn.Embedding(self.voc_size, self.pre_embeddings.shape[1])
      if options.pre_mode == 'hw':
        self.tune = Highway(self.pre_embeddings.shape[1], options.embedding_size, False)
      elif options.pre_mode == 'lin':
        self.tune = nn.Linear(self.pre_embeddings.shape[1], options.embedding_size, False)
      elif options.pre_mode == 'gate':
        self.tune = Gated(self.pre_embeddings.shape[1], options.embedding_size, False)
    else:
      self.lut  = nn.Embedding(options.voc_size, options.embedding_size)
  
  
  def forward(self, x):
    if self.pre_trained:
      pre_embed = self.pre_lut(x).detach()
      pre_size  = pre_embed.size()
      post_size = list(pre_size[:])
      post_size[-1] = -1
      res = self.tune(pre_embed.view(-1, pre_size[2])).view(post_size)  * 1e-2
    else:
      res = self.lut(x)
    return res
  
  
  def init_params(self, x):
    if self.pre_trained:
      self.pre_lut.weight.data.copy_(torch.FloatTensor(self.pre_embeddings[:self.voc_size]))
      self.pre_embeddings      = None
    else:
      self.lut.weight.data.uniform_(-x / math.sqrt(self.lut.weight.data.size(-1)),
                                    x / math.sqrt(self.lut.weight.data.size(-1)))

# Character-Aware NLM
# takes (N, max_char) input, returns (N, embedding_size) word reps
class CharsToWord(nn.Module):
  
  
  def __init__(self, options, nchars=256):
    super(CharsToWord, self).__init__()
    self.char_to_rep  = nn.Embedding(nchars,
                                     options.char_embedding_size,
                                     padding_idx=0)
    self.max_window   = options.max_char_window
    self.char_dim     = options.char_embedding_size
    self.out_dim      = options.embedding_size
    self.pre_dim      = 0
    self.convos       = []
    for ks in range(1, self.max_window + 1):
      self.convos += [nn.Conv1d(options.char_embedding_size, 25 * ks, ks,
                                bias=False)]
      self.pre_dim  += 25 * ks
    self.transform    = Highway(self.pre_dim, self.out_dim)
    # register parameters (not automatic apparently)
    for i, convo in enumerate(self.convos):
      for j, param in enumerate(convo.parameters()):
        self.register_parameter('convo_' + str(i) + '_' + str(j), param)
  
  
  def forward(self, x):
    pre_x     = x.view(-1, x.size(-1))
    pre_chrep = self.char_to_rep(pre_x)
    char_reps = torch.transpose(pre_chrep, 1, 2)
    conv_reps = [torch.max(F.tanh(conv(char_reps)), 2)[0].squeeze()
                 for conv in self.convos]
    pre_rep   = torch.cat(conv_reps, 1)
    pre_res   = self.transform(pre_rep)
    res       = pre_res.view(x.size(0), x.size(1), -1)
    return res
  
  
  def init_params(self, x):
    for p in self.parameters():
      p.data.uniform_(-x / math.sqrt(p.data.size(-1)),
                      x / math.sqrt(p.data.size(-1)))

# Sentence encoder: BoW, convolution, GRU, biGRU
class Encoder(nn.Module):
  
  
  def __init__(self, options):
    super(Encoder, self).__init__()
    self.encode_type  = options.encode_type
    self.embed_dim    = options.embedding_size
    self.rep_dim      = options.hidden_size
    if options.use_chars:
      self.word_embed   = CharsToWord(options)
    else:
      self.voc_size     = options.voc_size
      self.word_embed   = WordEmbed(options)
    if self.encode_type == 'GRU':
      self.hidden_size  = options.hidden_size
      self.nlayers_rnn  = options.nlayers_rnn
      self.num_directions = 2 if options.bidirectional else 1
      self.rnn  = nn.GRU(input_size    = self.embed_dim,
                         hidden_size   = options.hidden_size,
                         num_layers    = options.nlayers_rnn,
                         dropout       = options.rnn_dropout,
                         bidirectional = options.bidirectional)
    elif self.encode_type == 'BoW':
      self.bow_lin      = nn.Linear(self.embed_dim, options.hidden_size)
    if options.model_file is not None:
      print 'loading model from file', options.model_file
      self.load_state_dict(torch.load(options.model_file))
  
  
  def forward(self, x, hidden=None, lengths=None):
    if self.encode_type == 'GRU':
      if lengths is None:
        raise ValueError("the RNN doesn't work without lengths")
      # The Pytorch RNN takes a PackedSequence item,
      # which is created from a batch sorted by sequence lengths
      sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
      _, reverse_idx  = sorted_idx.sort()
      sorted_x        = x.index_select(0, sorted_idx)
      sorted_reps     = self.word_embed(sorted_x)
      # make initial hidden state
      if hidden is None:
        h_size = (self.nlayers_rnn * self.num_directions,
                  sorted_reps.size(0), self.hidden_size)
        h_0 = Variable(sorted_reps.data.new(*h_size).zero_(),
                       requires_grad=False)
      else:
        h_0 = hidden.index_select(0, sorted_idx)
      # make PackedSequence input
      rnn_input = nn.utils.rnn.pack_padded_sequence(sorted_reps,
                                                    sorted_lengths.data.tolist(),
                                                    batch_first=True)
      y, h_n  = self.rnn(rnn_input, h_0)
      # put batch dimension first, reshape and restore initial order
      last_h  = h_n.transpose(0, 1).contiguous()
      last_h  = last_h.view(sorted_reps.size(0), -1)
      last_h  = last_h.index_select(0, reverse_idx)
      return last_h
    elif self.encode_type == 'BoW':
      y = self.bow_lin(self.word_embed(x).sum(1).squeeze())
      return y
  
  
  def init_params(self, x):
    for p in self.parameters():
      p.data.uniform_(-x / math.sqrt(p.data.size(-1)), x / math.sqrt(p.data.size(-1)))
    self.word_embed.init_params(x)
    # orthonormal initialization for GRUs
    if self.encode_type == 'GRU':
      for p in self.rnn.parameters():
        if p.data.size(-1) == self.rep_dim:
          m = np.concatenate([ortho_group.rvs(dim=self.rep_dim) for _ in range(3)])
          p.data.copy_(torch.Tensor(m))
          m = None

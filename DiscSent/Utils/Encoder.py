from Utils import *
from scipy.stats import ortho_group

# Word embedding: either pre-trained or fully learned
class WordEmbed(nn.Module):


  def __init__(self, options):
    super(WordEmbed, self).__init__()
    self.voc_size     = options.encode_voc_size
    self.pre_trained  = (options.pre_mode != 'learn')
    if self.pre_trained:
      print 'loading embeddings from', options.pre_trained
      with open(options.pre_trained) as f:
        voc, self.pre_embeddings  = pickle.load(f)
        self.pre_embeddings[0]    = 0
        self.pre_lut              = nn.Embedding(self.voc_size, self.pre_embeddings.shape[1])
      print 'loaded embeddings from', options.pre_trained
      if options.pre_mode == 'hw':
        self.tune = Highway(self.pre_embeddings.shape[1], options.embedding_size, False)
      elif options.pre_mode == 'lin':
        self.tune = nn.Linear(self.pre_embeddings.shape[1], options.embedding_size, False)
      elif options.pre_mode == 'gate':
        self.tune = Gated(self.pre_embeddings.shape[1], options.embedding_size, False)
    else:
      self.lut  = nn.Embedding(options.encode_voc_size, options.embedding_size)


  def forward(self, x):
    if self.pre_trained:
      pre_embed = self.pre_lut(x).detach()
      pre_size  = pre_embed.size()
      post_size = list(pre_size[:])
      post_size[-1] = -1
      res = self.tune(pre_embed.view(-1, pre_size[2])).view(post_size) * 1e-2
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

# Attention on RNN hidden states
class SelfAttention(nn.Module):


  def __init__(self, in_dim, attn_hid, dropout):
    super(SelfAttention, self).__init__()
    self.ws1 = nn.Linear(in_dim, attn_hid, bias=False)
    self.ws2 = nn.Linear(attn_hid, 1, bias=False)
    self.tanh = nn.Tanh()
    self.softmax = nn.Softmax()
    self.drop = nn.Dropout(dropout)
    self.init_weights()


  def init_weights(self, init_range=0.1):
    self.ws1.weight.data.uniform_(-init_range, init_range)
    self.ws2.weight.data.uniform_(-init_range, init_range)


  def forward(self, x):
    size = x.size()  # [bsz, len, in_dim]
    x_flat  = x.view(-1, size[2])  # [bsz*len, in_dim]
    hbar    = self.tanh(self.ws1(self.drop(x_flat)))  # [bsz*len, attn_hid]
    alphas  = self.ws2(hbar).view(size[0], size[1])  # [bsz, len]
    alphas  = self.softmax(alphas.view(-1, size[1]))  # [bsz, len]
    alphas  = alphas.view(size[0], 1, size[1])  # [bsz, 1, len]
    return torch.bmm(alphas, x).squeeze(), alphas


  def __call__(self, x):
    return self.forward(x)

# Encode batch of sentences
class Encoder(nn.Module):


  def __init__(self, options):
    super(Encoder, self).__init__()
    self.options      = options
    self.encode_type  = options.encode_type
    self.embed_dim    = options.embedding_size
    self.rep_dim      = options.hidden_size
    self.voc_size     = options.encode_voc_size
    self.combine      = options.combine_hidden
    self.word_embed   = WordEmbed(options)
    if self.encode_type == 'BoW':
      self.bow_lin      = nn.Linear(self.embed_dim, options.hidden_size)
      self.attention    = SelfAttention(self.embed_dim,
                                        options.attention_dim,
                                        options.dropout)
    elif self.encode_type in ['GRU', 'LSTM']:
      self.hidden_size  = options.hidden_size
      self.nlayers_rnn  = options.nlayers_rnn
      self.num_directions = 2 if options.bidirectional else 1
      if self.encode_type == 'GRU':
        self.rnn  = nn.GRU(input_size    = self.embed_dim,
                           hidden_size   = options.hidden_size,
                           num_layers    = options.nlayers_rnn,
                           dropout       = options.rnn_dropout,
                           bidirectional = options.bidirectional)
      else:
        self.rnn  = nn.LSTM(input_size    = self.embed_dim,
                            hidden_size   = options.hidden_size,
                            num_layers    = options.nlayers_rnn,
                            dropout       = options.rnn_dropout,
                            bidirectional = options.bidirectional)
      self.attention    = SelfAttention(self.rep_dim * self.num_directions,
                                        options.attention_dim,
                                        options.dropout)
    if options.model_file is not None:
      print 'loading model from file', options.model_file
      self.load_state_dict(torch.load(options.model_file))


  def init_params(self, x):
    for p in self.parameters():
      p.data.uniform_(-x / math.sqrt(p.data.size(-1)), x / math.sqrt(p.data.size(-1)))
    self.word_embed.init_params(x)
    # orthonormal initialization for GRUs
    if self.encode_type in ['GRU', 'LSTM']:
      for p in self.rnn.parameters():
        if p.data.size(-1) == self.rep_dim:
          if self.encode_type == 'GRU':
            m = np.concatenate([ortho_group.rvs(dim=self.rep_dim) for _ in range(3)])
          else:
            m = np.concatenate([ortho_group.rvs(dim=self.rep_dim) for _ in range(4)])
          p.data.copy_(torch.Tensor(m))
          m = None


  def forward(self, x, lengths=None, hidden=None):
    if self.encode_type == 'BoW':
      pre_y   = self.word_embed(x)
      h_for   = None
      h_back  = None
      if self.combine in ["sum", "last"]:
        y = pre_y.sum(1).squeeze()
      elif self.combine == "max":
        y = pre_y.max(1)[0].squeeze()
      elif self.combine == "avg":
        y = pre_y.mean(1).squeeze()
      elif self.combine == "attn":
        y, _ = self.attention(pre_y)
      y = self.bow_lin(y)
    elif self.encode_type in ['GRU', 'LSTM']:
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
        if self.encode_type == 'GRU':
          h_0 = Variable(sorted_reps.data.new(*h_size).zero_(),
                         requires_grad=False)
        elif self.encode_type == 'LSTM':
          h_0 =(Variable(sorted_reps.data.new(*h_size).zero_(), requires_grad=False),
                Variable(sorted_reps.data.new(*h_size).zero_(), requires_grad=False))

      else:
        h_0 = hidden.index_select(1, sorted_idx)
        # TODO LSTM case (not currently used)
      # make PackedSequence input
#      if hidden is not None:
#        print sorted_lengths.size(), type(sorted_lengths)
#      if hidden is not None:
#        print sorted_lengths.size(), type(sorted_lengths)
      rnn_input = nn.utils.rnn.pack_padded_sequence(sorted_reps,
                                                    sorted_lengths.data.tolist(),
                                                    batch_first=True)
      pre_y, h_n  = self.rnn(rnn_input, h_0)
      if self.encode_type == 'LSTM':
        h_n = h_n[1]
      # put batch dimension first, reshape and restore initial order
      pre_y, _  = nn.utils.rnn.pad_packed_sequence(pre_y, batch_first=True)
      pre_y     = pre_y.index_select(0, reverse_idx)
      if self.combine == "last":
        last_h  = h_n.transpose(0, 1).contiguous()
        last_h  = last_h.view(sorted_reps.size(0), -1)
        last_h  = last_h.index_select(0, reverse_idx)
        y = last_h
      elif self.combine == "sum":
        y = pre_y.sum(1).squeeze()
      elif self.combine == "max":
        y = pre_y.max(1)[0].squeeze()
      elif self.combine == "avg":
        y = pre_y.mean(1).squeeze()
      elif self.combine == "attn":
        y, _ = self.attention(pre_y)
      # get RNN hidden state for LM
      if self.options.bidirectional:
        h_for   = pre_y[:, :, :self.hidden_size].contiguous().view(-1, self.hidden_size)
        h_back  = pre_y[:, :, self.hidden_size:].contiguous().view(-1, self.hidden_size)
      else:
        h_for   = pre_y.view(-1, self.hidden_size)
        h_back  = None
    # combine hidden state
    return (y, h_for, h_back)


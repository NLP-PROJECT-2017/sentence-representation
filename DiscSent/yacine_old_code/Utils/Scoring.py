from Utils import *

# Task specific comparator / scoring function
# Training:   ['order', 'next', 'conj']
# Evaluation: ['msrp', 'trec', 'subj', 'snli'] #, 'sick', 'sst2']
class Comparator(nn.Module):
  
  def __init__(self, task_type, options, nconj=43):
    super(Comparator, self).__init__()
    self.task_type  = task_type
    self.ndirs      = 2 if options.bidirectional and (options.encode_type == 'GRU') else 1
    self.ndims      = options.hidden_size * self.ndirs
    if options.use_bow:
      self.ndims    += options.hidden_size
    self.nconj      = nconj
    self.do         = options.dropout
    self.comp_mode  = options.comp_mode
    self.nlayers    = options.comp_nlayers
    if options.comp_mode == 'hw':
      self.hwl        = Highway(self.ndims, self.ndims)
      self.hwr        = Highway(self.ndims, self.ndims)
      self.hwb        = Highway(4 * self.ndims, self.ndims)
      self.hwb_sup    = [Highway(self.ndims, self.ndims) for _ in range(self.nlayers)]
    elif options.comp_mode == 'lin':
      self.hwl        = nn.Linear(self.ndims, self.ndims)
      self.hwr        = nn.Linear(self.ndims, self.ndims)
      self.hwb        = nn.Linear(4 * self.ndims, self.ndims)
      self.hwb_sup    = [nn.Linear(self.ndims, self.ndims) for _ in range(self.nlayers)]
    elif options.comp_mode == 'gate':
      self.hwl        = Gated(self.ndims, self.ndims)
      self.hwr        = Gated(self.ndims, self.ndims)
      self.hwb        = Gated(4 * self.ndims, self.ndims)
      self.hwb_sup    = [Gated(self.ndims, self.ndims) for _ in range(self.nlayers)]
    # register parameters (not automatic apparently)
    for i, layer in enumerate(self.hwb_sup):
      for j, param in enumerate(layer.parameters()):
        self.register_parameter('layer_' + str(i) + '_' + str(j), param)
    if self.task_type == 'order':
      self.compare  = nn.Linear(self.ndims, self.ndims)
    elif self.task_type == 'next':
      self.leftmul  = nn.Linear(options.next_context * self.ndims, self.ndims)
      self.compare  = nn.Linear(self.ndims, self.ndims)
    elif self.task_type == 'conj':
      self.compare  = nn.Linear(self.ndims, nconj * self.ndims)
    elif self.task_type == 'msrp' or self.task_type == 'sick':
      self.compare      = nn.Linear(2 * self.ndims, 1)
    elif self.task_type == 'sst2' or self.task_type == 'subj':
      self.compare      = nn.Linear(self.ndims, 1)
    elif self.task_type == 'trec':
      self.compare      = nn.Linear(self.ndims, 6)
    elif self.task_type == 'snli':
      self.compare      = nn.Linear(self.ndims, 3)
  
  
  def forward(self, x_left, x_right=None, training=False):
    if self.task_type == 'order':
      y_left  = self.hwl(x_left)
      y_left  = self.compare(y_left)
      y_right = self.hwr(x_right)
      score   = (y_left * y_right).sum(1).view(x_left.size(0))
      return F.sigmoid(score)
    elif self.task_type == 'next':
      y_left  = self.hwl(self.leftmul(x_left.view(x_left.size(0), -1)))
      y_left  = self.compare(y_left)
      y_left  = torch.stack([y_left for _ in range(x_right.size(1))], 1)
      y_right = self.hwr(x_right.view(-1, self.ndims)).view(x_right.size(0),
                                                            x_right.size(1),
                                                            -1)
      scores  = (y_left * y_right).sum(2).view(x_right.size(0),
                                               x_right.size(1))
      return F.log_softmax(scores)
    elif self.task_type == 'conj':
      y_left  = self.hwl(x_left)
      y_left  = self.compare(y_left).view(x_left.size(0), -1, self.ndims)
      y_right = self.hwr(x_right)
      y_right = torch.stack([y_right for _ in range(self.nconj)], 1)
      scores  = (y_left * y_right).sum(2).squeeze()
      return F.log_softmax(scores)
    elif self.task_type == 'msrp':
      dif_vec = torch.abs(x_left - x_right)
      mul_vec = x_left * x_right
      pre_vec = torch.cat([dif_vec, mul_vec], 1)# , x_left, x_right], 1)
      if training:
        pre_vec = F.dropout(pre_vec, p=self.do, training=training)
      pre_score = self.compare(pre_vec).view(-1)
      return F.sigmoid(pre_score)        
    elif self.task_type == 'sick':
      pre_score_mul = self.compare_mul(x_left * x_right).view(x_left.size(0))
      pre_score_dif = self.compare_dif(torch.abs(x_left - x_right)).view(x_left.size(0))
      return F.sigmoid(pre_score_mul + pre_score_dif) * 4 + 1
    elif self.task_type == 'sst2' or self.task_type == 'subj':
      pre_vec   = x_left
      if training:
        pre_vec = F.dropout(pre_vec, p=self.do, training=training)
      score   = self.compare(pre_vec).view(x_left.size(0))
      return F.sigmoid(score)
    elif self.task_type == 'trec':
      pre_vec   = x_left
      if training:
        pre_vec = F.dropout(pre_vec, p=self.do, training=training)
      scores    = self.compare(pre_vec)
      return F.log_softmax(scores)
    elif self.task_type == 'snli':
      dif_vec = torch.abs(x_left - x_right)
      mul_vec = x_left * x_right
      pre_vec = torch.cat([dif_vec, mul_vec, x_left, x_right], 1)
      pre_vec = F.dropout(pre_vec, p=self.do, training=training)
      pre_vec = self.hwb(pre_vec)
      for layer in self.hwb_sup:
        pre_vec = layer(pre_vec)
      pre_score = self.compare(pre_vec)
      return F.log_softmax(pre_score)
    
    
  def init_params(self, x):
    for p in self.parameters():
      p.data.uniform_(-x / math.sqrt(p.data.size(-1)), x / math.sqrt(p.data.size(-1)))


# TODO: compare with attention

# Skip-thought model decoder
class LMDecoder(nn.Module):


  def __init__(self, options, word_embed=None):
    super(LMDecoder, self).__init__()
    if word_embed is None:
      self.word_embed = WordEmbed(options)
    else:
      self.word_embed = word_embed
    self.ndirs  = 2 if (options.bidirectional and (options.encode_type == 'GRU')) else 1
    self.ndims        = options.hidden_size * self.ndirs
    self.hidden_size  = options.hidden_size
    self.rnn          = nn.GRU(input_size    = self.embed_dim + self.ndims,
                               hidden_size   = options.hidden_size)
  

  def forward(self, x, seq, lengths, h_0=None):
    # sort sequences
    sorted_lengths, sorted_idx = lengths.sort(0, descending=True)
    _, reverse_idx  = sorted_idx.sort()
    sorted_seq      = seq.index_select(0, sorted_idx)
    sorted_reps     = self.word_embed(sorted_seq)
    sorted_x        = x.index_select(0, sorted_idx)
    # concatenate seq and x
    sorted_x_repl   = sorted_x.unsqueeze(1).expand(x.size(0), seq.size(1), x.size(1))
    sorted_reps_con = torch.cat([sorted_reps, sorted_x_repl], 2)
    # make initial hidden state
    if hidden is None:
      h_size = (1, sorted_reps.size(0), self.hidden_size)
      h_0 = Variable(sorted_reps.data.new(*h_size).zero_(),
                     requires_grad=False)
    else:
      h_0 = hidden.index_select(0, sorted_idx)
    # make PackedSequence input
    rnn_input     = nn.utils.rnn.pack_padded_sequence(sorted_reps_con,
                                                      (sorted_lengths - 1).data.tolist(),
                                                      batch_first=True)
    output, h_n   = self.rnn(rnn_input, h_0)
    unpack_output = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
    final_out     = unpack_output.index_select(0, reverse_idx)
    return x
  
  
  def init_params(self, x):
    for p in self.parameters():
      p.data.uniform_(-x, x)


from Utils import *
from scipy.stats import ortho_group

class ScoreExample(nn.Module):


  def __init__(self, task_type, options, nconj=9, word_embed=None):
    super(ScoreExample, self).__init__()
    self.options    = options
    self.task_type  = task_type
    self.ndirs      = 2 if options.bidirectional and (options.encode_type in ['GRU', 'LSTM']) else 1
    self.ndims      = options.hidden_size * self.ndirs
    if options.use_bow:
      self.ndims    += options.hidden_size
    self.nconj      = nconj
    self.do         = nn.Dropout(options.dropout)
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
    # register parameters
    for i, layer in enumerate(self.hwb_sup):
      for j, param in enumerate(layer.parameters()):
        self.register_parameter('layer_' + str(i) + '_' + str(j), param)
    if self.task_type == 'order':
      self.compare    = nn.Linear(self.ndims, self.ndims)
    elif self.task_type == 'next':
      self.leftmul    = nn.Linear(options.next_context * self.ndims, self.ndims)
      self.compare    = nn.Linear(self.ndims, self.ndims)
    elif self.task_type == 'conj':
      self.compare    = nn.Linear(self.ndims, nconj * self.ndims)
    elif self.task_type == 'genlm':
      self.compare_f  = nn.Linear(self.options.hidden_size, options.decode_voc_size)
      self.compare_b  = nn.Linear(self.options.hidden_size, options.decode_voc_size)
    elif self.task_type == 'skip':
      self.hidden_size  = options.hidden_size_st
      self.word_embed = word_embed
      in_dim          = self.ndims + options.embedding_size
      self.rnn_l      = nn.GRU(input_size    = in_dim,
                               hidden_size   = options.hidden_size_st,
                               dropout       = options.rnn_dropout)
      self.rnn_r      = nn.GRU(input_size    = in_dim,
                               hidden_size   = options.hidden_size_st,
                               dropout       = options.rnn_dropout)
      self.compare    = nn.Linear(options.hidden_size_st, options.decode_voc_size)

  
  def forward(self, x_left, x_right=None, training=False,
              s_left=None, s_right=None, l_left=None, l_right=None):
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
    elif self.task_type == 'disclm':
      y_left  = self.hwl(self.do(x_left))
      score   = y_left.sum(1).view(x_left.size(0))
      return F.sigmoid(score)
    elif self.task_type == 'genlm':
      y_left  = F.log_softmax(self.compare_f(self.do(x_left)))
      if x_right is not None:
        y_right = F.log_softmax(self.compare_b(self.do(x_right)))
      else:
        y_right = None
      return y_left, y_right
    elif self.task_type == 'skip':
      # make initial hidden state
      h_size  = (1, x_left.size(0), self.hidden_size)
      h_0     = Variable(x_left.data.new(*h_size).zero_(), requires_grad=False)
      ### left sentence
      sorted_lengths_l, sorted_idx_l = l_left.sort(0, descending=True)
      _, reverse_idx_l  = sorted_idx_l.sort()
      sorted_x_l        = s_left.index_select(0, sorted_idx_l)
      sorted_reps_l     = self.word_embed(sorted_x_l)
      # concatenate input x_left to word embeddings
      sorted_i_l        = x_left.index_select(0, sorted_idx_l)
      sorted_i_l        = sorted_i_l.view(x_left.size(0), 1, x_left.size(1))
      sorted_i_l        = sorted_i_l.expand(x_left.size(0), sorted_x_l.size(1), x_left.size(1))
      #~ print sorted_reps_l.size()
      #~ print sorted_i_l.size()
      #~ print '-----------'
      sorted_con_l      = torch.cat([sorted_reps_l, sorted_i_l], 2)
      # make PackedSequence input
      rnn_input_l = nn.utils.rnn.pack_padded_sequence(sorted_con_l,
                                                      sorted_lengths_l.data.tolist(),
                                                      batch_first=True)
      pre_y_l, _  = self.rnn_l(rnn_input_l, h_0)
      pre_y_l, _  = nn.utils.rnn.pad_packed_sequence(pre_y_l, batch_first=True)
      pre_y_l     = pre_y_l.index_select(0, reverse_idx_l)
      pre_y_l     = pre_y_l.view(-1, pre_y_l.size(2))
      y_l         = F.log_softmax(self.compare(pre_y_l))
      ### right sentence
      sorted_lengths_r, sorted_idx_r = l_right.sort(0, descending=True)
      _, reverse_idx_r  = sorted_idx_r.sort()
      sorted_x_r        = s_right.index_select(0, sorted_idx_r)
      sorted_reps_r     = self.word_embed(sorted_x_r)
      # concatenate input x_left to word embeddings
      sorted_i_r        = x_left.index_select(0, sorted_idx_r)
      sorted_i_r        = sorted_i_r.view(x_left.size(0), 1, x_left.size(1))
      sorted_i_r        = sorted_i_r.expand(x_left.size(0), sorted_x_r.size(1), x_left.size(1))
      sorted_con_r      = torch.cat([sorted_reps_r, sorted_i_r], 2)
      # make PackedSequence input
      rnn_input_r = nn.utils.rnn.pack_padded_sequence(sorted_con_r,
                                                      (sorted_lengths_r.data).tolist(),
                                                      batch_first=True)
      pre_y_r, _  = self.rnn_r(rnn_input_r, h_0)
      pre_y_r, _  = nn.utils.rnn.pad_packed_sequence(pre_y_r, batch_first=True)
      pre_y_r     = pre_y_r.index_select(0, reverse_idx_r)
      pre_y_r     = pre_y_r.view(-1, pre_y_r.size(2))
      y_r         = F.log_softmax(self.compare(pre_y_r))
      return (y_l, y_r)

    
  def init_params(self, x):
    for p in self.parameters():
      p.data.uniform_(-x / math.sqrt(p.data.size(-1)), x / math.sqrt(p.data.size(-1)))
    # orthonormal initialization for GRUs
    if self.task_type == 'skip':
      for p in self.rnn_l.parameters():
        if p.data.size(-1) == self.hidden_size:
          m = np.concatenate([ortho_group.rvs(dim=self.hidden_size) for _ in range(3)])
          p.data.copy_(torch.Tensor(m))
          m = None
      for p in self.rnn_r.parameters():
        if p.data.size(-1) == self.hidden_size:
          m = np.concatenate([ortho_group.rvs(dim=self.hidden_size) for _ in range(3)])
          p.data.copy_(torch.Tensor(m))
          m = None


class AllScores(nn.Module):
  
  def __init__(self, options, sent_encode):
    super(AllScores, self).__init__()
    self.options        = options
    self.order_compare  = None
    self.next_compare   = None
    self.conj_compare   = None
    self.disclm_compare = None
    self.genlm_compare  = None
    self.skip_compare   = None
    if options.order_task:
      self.order_compare  = ScoreExample('order', options).cuda()
      self.order_compare.init_params(options.init_range)
    if options.next_task:
      self.next_compare   = ScoreExample('next', options).cuda()
      self.next_compare.init_params(options.init_range)
    if options.conj_task:
      self.conj_compare   = ScoreExample('conj', options,
                                         nconj=9 if options.conj_coarse else 43).cuda()
      self.conj_compare.init_params(options.init_range)
    if options.disclm_task:
      self.disclm_compare = ScoreExample('disclm', options).cuda()
      self.disclm_compare.init_params(options.init_range)
    if options.genlm_task:
      self.genlm_compare  = ScoreExample('genlm', options).cuda()
      self.genlm_compare.init_params(options.init_range)
    if options.skip_task:
      self.skip_compare   = ScoreExample('skip', options,
                                         word_embed=sent_encode.word_embed).cuda()
      self.skip_compare.init_params(options.init_range)

  

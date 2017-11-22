from Utils import *
from TextItem import *

# Data helper for unsupervised training
class BatchedData(object):

  
  def __init__(self, options):
    self.options  = options
    #~ vocab_file    = pjoin(options.data_folder, 'SortedVocab.txt')
    #~ f = codecs.open(vocab_file, 'r', encoding='utf8')
    #~ self.vocab = [line.strip() for line in f]
    #~ f.close()
    self.vocab = pickle.load(open(options.voc_file))
    self.max_l    = options.max_length
    self.rev_voc_encode = dict([(w, i)
                                for i, w in enumerate(self.vocab[:options.encode_voc_size])])
    self.rev_voc_decode = dict([(w, i)
                                for i, w in enumerate(self.vocab[:options.decode_voc_size])])
    self.batches        = {}
    self.cur_batch_idx  = {}
    self.data           = {}
    self.name           = {}
    if options.order_task:
      # load shelve files
      self.data['order']          = shelve.open(pjoin(options.data_folder,
                                                      'OrderShelve.shlf'))
      # prepare batch variables
      self.batches['order']       = {}
      self.cur_batch_idx['order'] = {'nbatches'  : self.data['order']['len'],
                                     'train'     : 0,
                                     'valid'     : 0,
                                     'shlf_idx'  : 1}
    if options.next_task:
      # load shelve files
      self.data['next']           = shelve.open(pjoin(options.data_folder,
                                                      'NextShelve.shlf'))
      # prepare batch variables
      self.batches['next']        = {}
      self.cur_batch_idx['next']  = {'nbatches'  : self.data['next']['len'],
                                     'train'     : 0,
                                     'valid'     : 0,
                                     'shlf_idx'  : 1}
    if options.conj_task:
      # load shelve files
      self.data['conj']           = shelve.open(pjoin(options.data_folder,
                                                      'ConjShelve.shlf'))
      # prepare batch variables
      self.batches['conj']        = {}
      self.cur_batch_idx['conj']  = {'nbatches'  : self.data['conj']['len'],
                                     'train'     : 0,
                                     'valid'     : 0,
                                     'shlf_idx'  : 1}
    if options.skip_task:
      # load shelve files
      self.data['skip']           = shelve.open(pjoin(options.data_folder,
                                                      'SkipShelve.shlf'))
      # prepare batch variables
      self.batches['skip']        = {}
      self.cur_batch_idx['skip']  = {'nbatches'  : self.data['skip']['len'],
                                     'train'     : 0,
                                     'valid'     : 0,
                                     'shlf_idx'  : 1}
    if options.fastsent_task:
      # load shelve files
      self.data['fastsent']           = shelve.open(pjoin(options.data_folder,
                                                      'SkipShelve.shlf'))
      # prepare batch variables
      self.batches['fastsent']        = {}
      self.cur_batch_idx['fastsent']  = {'nbatches'  : self.data['fastsent']['len'],
                                         'train'     : 0,
                                         'valid'     : 0,
                                         'shlf_idx'  : 1}
    if options.disclm_task:
      # load shelve files
      self.name['disclm']           = 'CombinedShelve%s.shlf' % ('-'+get_time_str() if options.dynamic else '')
      self.data['disclm']           = shelve.open(pjoin(options.data_folder, self.name['disclm']))
      if options.dynamic:
        self.data['disclm']['len']  = 0
        self.data['disclm'].sync()
      # prepare batch variables
      self.batches['disclm']        = {}
      self.cur_batch_idx['disclm']  = {'nbatches'  : self.data['disclm']['len'],
                                       'train'     : 0,
                                       'valid'     : 0,
                                       'shlf_idx'  : 1}
    if options.genlm_task:
      # load shelve files
      self.data['genlm']            = shelve.open(pjoin(options.data_folder,
                                                        'RealShelve.shlf'))
      # prepare batch variables
      self.batches['genlm']         = {}
      self.cur_batch_idx['genlm']   = {'nbatches'  : self.data['genlm']['len'],
                                       'train'     : 0,
                                       'valid'     : 0,
                                       'shlf_idx'  : 1}

  # words to Variables
  def batch_to_vars(self, batch_sen, mode="encode"):
    batch_sen_trunc = [sen[:self.max_l - 1] for sen in batch_sen]
    lengths = [len(sen) + (0 if mode == "generate" else 1) for sen in batch_sen_trunc]
    max_len = max(lengths)
    if mode == "generate":
      padded_batch_sen_l = [sen[:lengths[i]] for i, sen in enumerate(batch_sen_trunc)]
    else:
      padded_batch_sen_l = [sen[:lengths[i]] + ['<PAD>'] * (max_len - lengths[i])
                            for i, sen in enumerate(batch_sen_trunc)]
    if mode == "encode" or mode == "generate":
      batch_array = np.array([[self.rev_voc_encode.get(w, 3) for w in sen]
                              for sen in padded_batch_sen_l])
    elif mode == "decode":
      batch_array = np.array([[self.rev_voc_decode.get(w, 3) for w in sen]  # use <UNK> (id: 3) for unknown words
                              for sen in padded_batch_sen_l])
    batch_var   = Variable(torch.Tensor(batch_array), requires_grad=False).long().cuda()
    batch_len   = Variable(torch.Tensor(lengths), requires_grad=False).long().cuda()
    return (batch_var, batch_len)
  

  # prepare next batch
  def next_batch(self, task, mode): # mode \in (train, valid)
    b       = self.cur_batch_idx[task][mode]
    bs      = self.options.batch_size
    # if not generated disclm in dynamic-dg task
    if self.data[task]['len'] == 0:
        return None, None, None
    # retrieve next batch
    str_id  = str(self.cur_batch_idx[task]['shlf_idx'])
    if mode == 'train':
      batch = self.data[task][str_id][b * bs:(b+1) * bs]
    elif mode == 'valid':
      batch = self.data[task]['0'][b * bs:(b+1) * bs]
    # transform into torch tensors
    if task in ["order", "next", "conj"]:
      # inputs
      if task == "next":
        batch_sen_l = [['<S>'] + sen.split() for ex in batch for sen in ex[0]]
        batch_sen_r = [['<S>'] + sen.split() for ex in batch for sen in ex[1]]
      else:
        batch_sen_l = [['<S>'] + ex[0].split() for ex in batch]
        batch_sen_r = [['<S>'] + ex[1].split() for ex in batch]
      b_xl, b_ll  = self.batch_to_vars(batch_sen_l, "encode")
      b_xr, b_lr  = self.batch_to_vars(batch_sen_r, "encode")
      # targets
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
      #batch tensors
      res =  (b_xl, b_xr, b_ll, b_lr, b_y)
    elif task == "disclm":
      # inputs
      batch_sen = [['<S>'] + ex[0].split() for ex in batch]
      b_x, b_l  = self.batch_to_vars(batch_sen, "encode")
      # targets
      b_y = Variable(torch.Tensor([ex[1] for ex in batch]), requires_grad=False).cuda()
      # batch tensors
      res = (b_x, b_l, b_y)
    elif task == "genlm":
      # inputs
      batch_sen   = [['<S>'] + ex[0].split() for ex in batch]
      b_x, b_l    = self.batch_to_vars(batch_sen, "encode")
      # targets: t+1 and t-1 for forward and backward LM
      batch_sen   = [ex[0].split() + ['</S>'] for ex in batch]
      b_yf, b_lf  = self.batch_to_vars(batch_sen, "decode")
      batch_sen   = [['<PAD>', '<S>'] + ex[0].split()[:-1] for ex in batch]
      b_yb, b_lb  = self.batch_to_vars(batch_sen, "decode")
      # batch tensors
      res = (b_x, b_l, b_yf, b_yb)
    elif task == "skip":
      # inputs
      batch_sen   = [['<S>'] + ex[0].split() for ex in batch]
      b_x, b_l    = self.batch_to_vars(batch_sen, "encode")
      # targets
      batch_sen_l = [['<S>'] + ex[1].split() for ex in batch]
      batch_sen_r = [['<S>'] + ex[2].split() for ex in batch]
      b_xl, b_ll  = self.batch_to_vars(batch_sen_l, "decode")
      b_xr, b_lr  = self.batch_to_vars(batch_sen_r, "decode")
      batch_sen_l = [ex[1].split() + ['</S>'] for ex in batch]
      batch_sen_r = [ex[2].split() + ['</S>'] for ex in batch]
      b_yl, b_ll  = self.batch_to_vars(batch_sen_l, "decode")
      b_yr, b_lr  = self.batch_to_vars(batch_sen_r, "decode")
      # batch tensors
      res = (b_x, b_xl, b_xr, b_l, b_ll, b_lr, b_yl, b_yr)
    # cycle through large batches
    self.cur_batch_idx[task][mode]  = (b + 1) % (len(self.data[task][str_id]) / bs)
    if (mode == 'train') and self.cur_batch_idx[task][mode] == 0:
      self.cur_batch_idx[task]['shlf_idx'] += 1
      if self.cur_batch_idx[task]['shlf_idx'] >= self.cur_batch_idx[task]['nbatches']:
        self.cur_batch_idx[task]['shlf_idx'] = 1
      # self.data[task].sync()
    return res


  def get_values(self, task, length):
    res     = list()
    bs      = self.options.batch_size
    while len(res) < length:
      str_id = str(self.cur_batch_idx[task]['shlf_idx'])
      b      = self.cur_batch_idx[task]['train']
      ex_len = min(length - len(res), len(self.data[task][str_id]) - b * bs)
      res.extend(self.data[task][str_id][b * bs: b * bs + ex_len])
      self.cur_batch_idx[task]['train'] = (b + ex_len // bs) % (len(self.data[task][str_id]) // bs)
      if self.cur_batch_idx[task]['train'] == 0:
        self.cur_batch_idx[task]['shlf_idx'] += 1
        if self.cur_batch_idx[task]['shlf_idx'] >= self.cur_batch_idx[task]['nbatches']:
          self.cur_batch_idx[task]['shlf_idx'] = 1
    return res

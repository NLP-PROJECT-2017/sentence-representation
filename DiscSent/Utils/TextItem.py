import codecs
import re

from random import randint
from random import shuffle
from nltk import sent_tokenize as sentences
from nltk import wordpunct_tokenize as words


########### Make TextItems
def get_ent_indices(sent):
  res       = []
  entities  = []
  for w in sent:
    if w.startswith('ent_'):
      start_id  = len(res)
      for wb in w[4:].split('_'):
        res += [wb]
      entities += [(start_id, len(res))]
    else:
      res += [w]
  return (' '.join(res), entities)


def process_wiki_line(line):
  ln = re.sub('\<a href[^\>]*\>', '<a>', line)
  ln = re.sub('\<a\>([^\<]*)\</a\>', '_ent_\\1_ent', ln)
  lntab = ln.split('_ent')
  for i, w in enumerate(lntab):
    if w.startswith('_'):
      lntab[i] = 'ent_' + re.sub(' ', '_', w[1:])
  #
  ln = ' '.join(lntab)
  sents = sentences(ln)
  return [get_ent_indices(words(sent)) for sent in sents]


class TextItem:
  
  
  def __init__(self):
    self.paragraphs = []
  
  
  def read_wiki(self, lines):
    self.text_type      = "wiki"
    self.name           = lines[0]
    self.entities       = []
    self.section_names  = []
    paragraph_title = "Introduction."
    paragraph       = []
    new_title       = ""
    for (i, line) in enumerate(lines):
      if line.startswith('='):
        new_title += " --- " + line.stip().encode('ascii', 'replace')
        if i+1 < len(lines) and not lines[i + 1].startswith('='):
          self.section_names  += [paragraph_title]
          self.paragraphs     += [[sen for sen, ents in paragraph]]
          self.entities       += [[(j, ents) for j, (sen, ents) in enumerate(paragraph)]]
          paragraph_title = new_title
          new_title       = ""
          paragraph       = []
      else:
        if i > 0 and len(line) > 0:
          paragraph += [sent for sent in process_wiki_line(line)]
    if len(new_title) > 0:
      self.section_names  += [new_title]
      self.paragraphs     += [[sen for sen, ents in paragraph]]
      self.entities       += [[(j, ents) for j, (sen, ents) in enumerate(paragraph)]]
  
  
  def read_book(self, filename):
    self.text_type  = "book"
    line_ct = 0
    in_book = False
    paragraph = ''
    # f = codecs.open(filename, 'r', encoding='utf8')
    f = codecs.open(filename, 'r', encoding='iso-8859-1')
    for line in f:
      if line.startswith('*** START') or line.startswith('***START'):
        in_book   = True
        self.name = line.strip().split('EBOOK')[-1][:-3].strip().encode('ascii', 'replace') # .decode('iso-8859-1').encode('utf8')
      elif line.startswith('*** END') or line.startswith('***END'):
        f.close()
        return
      elif in_book:
        if line.strip() == '' and len(paragraph.strip()) > 0:
          self.paragraphs += [[' '.join([word for word in words(sen)])
                             for sen in sentences(paragraph.strip())]]
          paragraph = ''
        else:
          paragraph += ' ' + line.strip().encode('ascii', 'replace') # .decode('iso-8859-1').encode('utf8')
    f.close()


def read_wiki_file(filename):
  lines = []
  f = codecs.open(filename, 'r', encoding='utf8')
  for line in f:
    if line.startswith("<doc"):
      lines = []
      t_item = TextItem()
    elif line.startswith("</doc>"):
      t_item.read_wiki(lines)
      yield t_item
    else:
      lines += [line.strip().encode('ascii', 'replace')]
  f.close()

########### Make tasks
# list of conjunctions with somewhat arbitrary grouping
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

rev_dict    = dict([(con, k) for k, ls in conjunct_dic.items() for con in ls])
conjunctions= [con  for ls in conjunct_dic.values() for con in ls]
conj_map_1  = dict([(c, i) for i, c in enumerate(conjunctions)])
conj_map_2  = dict([(c, i) for i, c in enumerate(conjunct_dic.keys())])

# returns a list of order tasks
def make_order_task(paragraph):
  length  = len(paragraph)
  if length <= 2:
    return []
  pairs = [(paragraph[i], paragraph[i+1])
           for i in range(length - 1) 
           if len(paragraph[i].split()) > 2 and len(paragraph[i+1].split()) > 2 and \
           paragraph[i].count('?') < 3 and paragraph[i+1].count('?') < 3]
  res   = []
  for pair in pairs:
    label = randint(0, 1)
    res   += [(pair[label % 2], pair[(label + 1) % 2], label)]
  return res


# returns one next task if the paragraph is long enough, None value otherwise
def make_next_task(paragraph, context_length=3, n_proposals=5):
  length  = len(paragraph)
  if length < (context_length + n_proposals):
    return []
  context   = randint(context_length, length - n_proposals)
  negatives = range(context + 1, length)
  shuffle(negatives)
  proposals = [context] + negatives[:n_proposals - 1]
  shuffle(proposals)
  res = [(paragraph[context-context_length:context],
          [paragraph[p] for p in proposals],
          proposals.index(context))]
  if len(res) > 3:
    pprint(paragraph)
    pprint(res)
    raise ValueError('Say what?')
  return res


# auxiliary for extracting conjunctions
def get_conj(sen):
  tab = sen.split()
  if len(tab) > 1 and tab[1] == ',' and rev_dict.get(tab[0], False):
    return (tab[0], ' '.join(tab[2:]))
  elif len(tab) > 2 and tab[2] == ',' and rev_dict.get(tab[0] + ' ' + tab[1], False):
    return (tab[0] + ' ' + tab[1], ' '.join(tab[3:]))
  else:
    return (False, sen)


# returns a list of conjunction tasks
def make_conj_task(paragraph):
  length  = len(paragraph)
  if length < 2:
    return []
  res = []
  stripped = [get_conj(sen) for sen in paragraph]
  for i, (conj, sen) in enumerate(stripped):
    if i > 0 and conj and stripped[i-1][1].count('?') < 3 and sen.count('?') < 3 and \
    len(stripped[i-1][1].split()) > 2 and len(sen.split()) > 2:
      res += [(stripped[i-1][1], sen, conj, rev_dict[conj])]
  return res


# returns a list of (sentence, left_sen, right_sen) triples
# for skip-thought and FastSent
def make_skip_task(paragraph):
  length  = len(paragraph)
  if length <= 3:
    return []
  else:
    return [(paragraph[i], paragraph[i-1], paragraph[i+1])
            for i in range(1, length-1)
            if paragraph[i].count('?') < 3 and paragraph[i+1].count('?') < 3 and paragraph[i-1].count('?') < 3 and \
            len(paragraph[i].split()) > 2 and len(paragraph[i+1].split()) > 2 and len(paragraph[i-1].split()) > 2]
    


# make all tasks
def make_all_tasks(text_item):
  order_tasks = []
  next_tasks  = []
  conj_tasks  = []
  skip_tasks  = []
  for paragraph in text_item.paragraphs:
    low_par = [sen.lower() for sen in paragraph]
    order_tasks += make_order_task(low_par)
    next_tasks  += make_next_task(low_par)
    conj_tasks  += make_conj_task(low_par)
    skip_tasks  += make_skip_task(low_par)
  return (order_tasks, next_tasks, conj_tasks, skip_tasks)


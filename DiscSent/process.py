'''
Read book_corpus to generate order, next, conj tasks.
'''

import argparse
import logging
from os.path import join as pjoin
from datetime import datetime
import shelve
from random import shuffle
from gensim import utils
from random import randint

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

def get_time_str():
  time_str = str(datetime.now())
  time_str = '_'.join(time_str.split())
  time_str = '_'.join(time_str.split('.'))
  time_str = ''.join(time_str.split(':'))
  return time_str

class MySentences(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            yield line.strip()

def make_order_task(paragraph):
	length  = len(paragraph)
	if length <= 2:
		return []
	pairs = [(paragraph[i], paragraph[i+1])
			for i in range(length - 1) 
			if len(paragraph[i].split()) > 2 and len(paragraph[i+1].split()) > 2]
	res = []
	for pair in pairs:
		label = randint(0, 1)
		res   += [(pair[label % 2], pair[(label + 1) % 2], label)]
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

# returns one next task if the paragraph is long enough, None value otherwise
def make_next_task(paragraph, context_length=3, n_proposals=5):
	length  = len(paragraph)
	if length < (context_length + n_proposals):
		return []
	context   = randint(context_length, length - n_proposals)
	negatives = list(range(context + 1, length))
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


def make_all_tasks(paragraph):
	order_tasks = []
	next_tasks  = []
	conj_tasks  = []
	# skip_tasks  = []

	low_par = [sen.lower() for sen in paragraph]
	order_tasks += make_order_task(low_par)
	next_tasks  += make_next_task(low_par)
	conj_tasks  += make_conj_task(low_par)
	# skip_tasks  += make_skip_task(low_par)
	return (order_tasks, next_tasks, conj_tasks)

parser = argparse.ArgumentParser()
parser.add_argument('-corpus', '--corpus', type=str, default='books_large_70m.txt')
parser.add_argument('-order', '--order', type=str, default='order_bookcorpus.shlf')
parser.add_argument('-next', '--next', type=str, default='next_bookcorpus.shlf')
parser.add_argument('-conj', '--conj', type=str, default='conj_bookcorpus.shlf')
# parser.add_argument('-log_file', '--log_file', type=str, default='')

args = parser.parse_args()
sentences = MySentences(args.corpus)
log_file = pjoin('process_' + get_time_str() + '.log')
logging.basicConfig(filename=log_file,level=logging.DEBUG)
chunksize = 100
groups = enumerate(utils.grouper(sentences, chunksize))
n_sentence = 0
order_ = []
next_ = []
conj_ = []

while True:
	try:
		sentence_no, items = next(groups)
		o, n, c = make_all_tasks(items)
		order_ += o
		next_ += n
		conj_ += c
		logging.info("%s \t  %d %d %d %d", get_time_str(), sentence_no*chunksize, len(order_), len(next_), len(conj_))
	except StopIteration:
		break

order_shelve = shelve.open(args.order, writeback=True)
order_shelve['len'] = len(order_) // 64000 + 1
shuffle(order_)
for i in range(len(order_) // 64000 + 1):
  order_shelve[str(i)] = order_[i * 64000: (i + 1) * 64000]
  order_shelve.sync()

order_shelve.close()

next_shelve = shelve.open(args.next, writeback=True)
next_shelve['len'] = len(next_) // 64000 + 1
shuffle(next_)
for i in range(len(next_) // 64000 + 1):
  next_shelve[str(i)] = next_[i * 64000: (i + 1) * 64000]
  next_shelve.sync()

next_shelve.close()

conj_shelve = shelve.open(args.conj, writeback=True)
conj_shelve['len'] = len(conj_) // 64000 + 1
shuffle(conj_)
for i in range(len(conj_) // 64000 + 1):
  conj_shelve[str(i)] = conj_[i * 64000: (i + 1) * 64000]
  conj_shelve.sync()

conj_shelve.close()


# for sentence_no, sentence in enumerate(sentences):
# 	# utils.grouper(sentences, chunksize)
# 	if sentence_no % 10000 == 0:
# 		logging.info("%s \t  %d", get_time_str(), sentence_no)

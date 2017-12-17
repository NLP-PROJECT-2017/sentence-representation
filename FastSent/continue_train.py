import os, sys, logging, argparse, pdb
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--epoch', type=int, default=1)
parser.add_argument('--autoencode', dest='autoencode', action='store_true')
parser.add_argument('--min_count',dest='min_count',type=int,default=10)
parser.add_argument('--sample',dest='sample',type=float,default=0)
parser.add_argument('--corpus', type=str, default='books_in_sentences')
parser.add_argument('--savedir', type=str, default='./')


args = parser.parse_args()

# checking for the python path used
pythonpath = os.environ['PYTHONPATH'].split(os.pathsep)
sys.path = [x for x in sys.path if not 'gensim' in x]
sys.path = pythonpath + sys.path
libs = [os.path.basename(ppath) for ppath in pythonpath]
PATH_TO_GENSIM = 'py2.7/lib/python2.7/site-packages'
PATH_TO_FASTSENT = 'sentence-representation/Fastsent'
PATH_TO_MODEL = 'out/FastSent_no_autoencoding_512_5_0'
# import skipthought and Senteval
sys.path.insert(0, PATH_TO_GENSIM)
sys.path.insert(0, PATH_TO_FASTSENT)
import gensim
import fastsent
if 'gensim' in libs:
    print 'using sentrep model not cbow'
    rep = 'sentrep'
 
from gensim.models.fastsent import FastSent

if args.autoencode:
    autoencode_flag = 1
    auto_label = 'autoencoding'
else:
    autoencode_flag = 0
    auto_label = 'no_autoencoding'

class MySentences(object):
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            yield line.split()

sentfile = args.corpus

if 'gensim' in libs:
    sentences = MySentences(sentfile)
    model = fastsent.FastSent.load(PATH_TO_MODEL)

model.build_vocab(sentences)
model.train(sentences, chunksize=1000)
model.save(args.savedir+'continue_FastSent_%s_%s_%s_%s' % (auto_label, args.dim, args.min_count, args.sample))


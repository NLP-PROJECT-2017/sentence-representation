'''
Find nearest neighbor of sentences
'''

import argparse
import logging
from os.path import join as pjoin
import numpy
import pickle
from Utils import *

def get_embed(model, sent):
	ss = ''
	for s in sent:
		if s in model:
			ss += s + ' '
	if ss == '':
		ss = '.'
	return model[ss.strip()]

def get_embeds(batched_data, encoder, batch):
	sents = map(lambda sen: ['<S>'] + [word.lower() for word in sen], batch)
	sents, lengths = batched_data.batch_to_vars(sents)
	embedding, h_for, h_back = encoder.forward(sents, lengths=lengths)
	model_embedding = embedding.data.cpu().numpy()
	return model_embedding

parser = argparse.ArgumentParser()
parser.add_argument('-sample', '--sample', type=str, default='books_sample.txt')
parser.add_argument('-query', '--query', type=str, default='books_query.txt')
parser.add_argument('-model', '--model', type=str, default='fastsent')
parser.add_argument('-model_pt', '--model_pt', type=str, default='')
parser.add_argument('-model_args', '--model_args', type=str, default='')
parser.add_argument('-topk', '--topk', type=int, default=5)
parser.add_argument("-data", "--data_folder", default='',
                      help="location of the data")
parser.add_argument("-o", "--output_folder", default='',
                      help="location of the model")

options = parser.parse_args()
log_file = pjoin('find_nearest_sents_' + options.model + '.log')

f = open(options.sample)
batch = [s.strip() for s in f.readlines()]
f.close()
logging.info("%d sample sentences loaded. first line: %s \n", len(batch), batch[0])

f = open(options.query)
queries = [s.strip() for s in f.readlines()]
f.close()
logging.info("%d queries loaded.\n", len(queries))

q_embed = []
sents_embed = []

if model == 'discsent':
	args_path = pjoin(args.output_folder, args.model_args)
	args = pickle.load(open(args_path))
	args.data_folder = options.data_folder
	# args.max_length = options.max_length
	args.max_length = 128
	encoder = Encoder(args)
	model_file = pjoin(options.output_folder, options.model_pt)
	encoder.load_state_dict(torch.load(model_file))
	batched_data  = BatchedData(args)
	
	sents_embed = get_embeds(batched_data,encoder,batch)
	q_embed = get_embeds(batched_data,encoder,queries)

if model == 'fastsent':
	fastsent_model = fastsent.FastSent.load(args.model_pt)
	for q in queries:
		sents_embed.append(get_embed(fastsent_model, q))

for i, q in enumerate(q_embed):
	logging.info("Query: %s top %d nearest:", queries[i], options.topk)
	dis = [(j, numpy.linalg.norm(sent-q)) for j, sent in enumerate(sents_embed)]
	sorted_dis = sorted(dis, key=itemgetter(1))
	for k, _ in sorted_dis[:options.topk]:
		logging.info("%s", batch[k])










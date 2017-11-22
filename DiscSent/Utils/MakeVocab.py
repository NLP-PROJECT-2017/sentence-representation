import cPickle as pickle
import torch


def make_vocab(vocab_file, embed_file, output_file):
    vocab = ['</S>', '<S>'] + map(lambda x: x.strip(), open(vocab_file).readlines())
    mapping, embed, dim = torch.load(embed_file)
    output_embed = list()
    for word in vocab:
        if word in mapping:
            output_embed.append(embed[mapping[word]].view(1, dim))
        else:
            output_embed.append(torch.zeros(1, dim))
    output_embed = torch.cat(output_embed, dim=0).cpu().numpy()
    pickle.dump((vocab, output_embed), open(output_file, 'wb'))


if __name__ == '__main__':
    make_vocab('/misc/vlgscratch4/BowmanGroup/haoyue/shelve/SortedVocab.txt',
               '/misc/vlgscratch4/BowmanGroup/haoyue/models/glove.840B.300d.pt',
               '/misc/vlgscratch4/BowmanGroup/haoyue/shelve/SortedGloVeUD.pk')

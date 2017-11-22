import shelve
from random import shuffle
from pprint import pprint
from TextItem import *

##### Wikipedia Reading
wi_shelve = shelve.open('../Data/Wikipedia/WikipediaShelveUD.shlf')
order_wi  = []
next_wi	  = []
conj_wi   = []
skip_wi   = []
print "wikipedia"
for i in range(5253025):
  try:
    wi_book     = wi_shelve[str(i)]
    o, n, c, st = make_all_tasks(wi_book)
    order_wi    += o
    next_wi     += n
    # conj_wi   += c
    conj_wi     += [t for t in c if (t[2] != 'however' or randint(0, 3) == 0)]
    skip_wi     += st
    if i % 1000000 == 0:
      print i
      wi_shelve.sync()
  except:
    print i, 'fail'

wi_shelve.close()

##### Wikipedia Writing
order_shelve = shelve.open('../Data/Wikipedia/WikipediaOrderShelve.shlf', writeback=True)
order_shelve['len'] = len(order_wi) / 64000 + 1
shuffle(order_wi)
for i in range(len(order_wi) / 64000 + 1):
  order_shelve[str(i)] = order_wi[i * 64000: (i + 1) * 64000]
  order_shelve.sync()

order_shelve.close()

next_shelve = shelve.open('../Data/Wikipedia/WikipediaNextShelve.shlf', writeback=True)
next_shelve['len'] = len(next_wi) / 64000 + 1
shuffle(next_wi)
for i in range(len(next_wi) / 64000 + 1):
  next_shelve[str(i)] = next_wi[i * 64000: (i + 1) * 64000]
  next_shelve.sync()

next_shelve.close()

conj_shelve = shelve.open('../Data/Wikipedia/WikipediaConjShelve.shlf', writeback=True)
conj_shelve['len'] = len(conj_wi) / 64000 + 1
shuffle(conj_wi)
for i in range(len(conj_wi) / 64000 + 1):
  conj_shelve[str(i)] = conj_wi[i * 64000: (i + 1) * 64000]
  conj_shelve.sync()

conj_shelve.close()

skip_shelve = shelve.open('../Data/Wikipedia/WikipediaSkipShelve.shlf', writeback=True)
skip_shelve['len'] = len(skip_wi) / 64000 + 1
shuffle(skip_wi)
for i in range(len(skip_wi) / 64000 + 1):
  skip_shelve[str(i)] = skip_wi[i * 64000: (i + 1) * 64000]
  skip_shelve.sync()

skip_shelve.close()

##### Gutenberg Reading
gu_shelve = shelve.open('../Data/Gutenberg/GutenbergShelveEN.shlf')
order_gu  = []
next_gu	  = []
conj_gu   = []
skip_gu   = []
print "gutenberg"
for i in range(53291):
  try:
    gu_book     = gu_shelve[str(i)]
    o, n, c, st = make_all_tasks(gu_book)
    order_gu    += o
    next_gu     += n
    conj_gu     += c
    skip_gu     += st
    if i % 1000 == 0:
      print i
      gu_shelve.sync()
  except:
    print i, 'fail'

gu_shelve.close()

##### Gutenberg Writing
order_shelve = shelve.open('../Data/Gutenberg/GutenbergOrderShelve.shlf', writeback=True)
order_shelve['len'] = len(order_gu) / 64000 + 1
shuffle(order_gu)
for i in range(len(order_gu) / 64000 + 1):
  order_shelve[str(i)] = order_gu[i * 64000: (i + 1) * 64000]
  order_shelve.sync()

order_shelve.close()

next_shelve = shelve.open('../Data/Gutenberg/GutenbergNextShelve.shlf', writeback=True)
next_shelve['len'] = len(next_gu) / 64000 + 1
shuffle(next_gu)
for i in range(len(next_gu) / 64000 + 1):
  next_shelve[str(i)] = next_gu[i * 64000: (i + 1) * 64000]
  next_shelve.sync()

next_shelve.close()

conj_shelve = shelve.open('../Data/Gutenberg/GutenbergConjShelve.shlf', writeback=True)
conj_shelve['len'] = len(conj_gu) / 64000 + 1
shuffle(conj_gu)
for i in range(len(conj_gu) / 64000 + 1):
  conj_shelve[str(i)] = conj_gu[i * 64000: (i + 1) * 64000]
  conj_shelve.sync()

conj_shelve.close()

skip_shelve = shelve.open('../Data/Gutenberg/GutenbergSkipShelve.shlf', writeback=True)
skip_shelve['len'] = len(skip_gu) / 64000 + 1
shuffle(skip_gu)
for i in range(len(skip_gu) / 64000 + 1):
  skip_shelve[str(i)] = skip_gu[i * 64000: (i + 1) * 64000]
  skip_shelve.sync()

skip_shelve.close()

#########
# vocab

#~ vocounts = {}

#~ for pair in order_a:
  #~ for w in pair[0].split() + pair[1].split():
    #~ vocounts[w] = vocounts.get(w, 0) + 1


#~ for pair in conj_a:
  #~ for w in pair[0].split() + pair[1].split():
    #~ vocounts[w] = vocounts.get(w, 0) + 1


#~ for pair in next_a:
  #~ for sen in pair[0] + pair[1]:
    #~ for w in sen.split():
      #~ vocounts[w] = vocounts.get(w, 0) + 1


#~ sorted_voc = sorted(vocounts.items(), key=lambda x:x[1], reverse=True)
#~ vocab = ['<UNK>'] + [x[0] for x in sorted_voc[:499999]]
#~ o = open('SortedVocabUD.txt', 'w')
#~ for w in vocab:
  #~ print >>o, w

#~ o.close()



################ Wikipedia STATS

#~ >>> len(order_wi)
#~ 51993105
#~ >>> len(skip_wi)
#~ 44987275
#~ >>> len(next_wi)
#~ 2718426
#~ >>> len(conj_wi)
#~ 832522

#~ fine_counts   = {}
#~ coarse_counts = {}
#~ for a, b, c, d in conj_gu:
  #~ fine_counts[c]    = fine_counts.get(c, 0) + 1
  #~ coarse_counts[d]  = coarse_counts.get(d, 0) + 1

#~ >>> pprint(sorted(coarse_counts.items(), key=lambda x:x[1], reverse=True))
#~ [('time', 313442),
 #~ ('addition', 224275),
 #~ ('strengthen', 138165),
 #~ ('contrast', 132967),
 #~ ('result', 131706),
 #~ ('return', 46156),
 #~ ('specific', 28074),
 #~ ('recognize', 6816),
 #~ ('compare', 5866)]

#~ >>> pprint(sorted(fine_counts.items(), key=lambda x:x[1], reverse=True))
#~ [('now', 153703),
 #~ ('then', 138167),
 #~ ('however', 109651),
 #~ ('indeed', 85892),
 #~ ('besides', 72732),
 #~ ('moreover', 57953),
 #~ ('in fact', 52273),
 #~ ('thus', 50355),
 #~ ('still', 46156),
 #~ ('therefore', 30523),
 #~ ('again', 27188),
 #~ ('finally', 25415),
 #~ ('accordingly', 19424),
 #~ ('also', 18514),
 #~ ('hence', 17955),
 #~ ('meanwhile', 15077),
 #~ ('for example', 14416),
 #~ ('that is', 12863),
 #~ ('further', 9375),
 #~ ('consequently', 9358),
 #~ ('furthermore', 9018),
 #~ ('instead', 8211),
 #~ ('anyway', 8131),
 #~ ('certainly', 5728),
 #~ ('next', 5675),
 #~ ('otherwise', 4206),
 #~ ('in addition', 4080),
 #~ ('similarly', 3949),
 #~ ('likewise', 1917),
 #~ ('subsequently', 1782),
 #~ ('rather', 1658),
 #~ ('incidentally', 1289),
 #~ ('undoubtedly', 1088),
 #~ ('henceforth', 1020),
 #~ ('thereafter', 820),
 #~ ('conversely', 697),
 #~ ('specifically', 419),
 #~ ('namely', 323),
 #~ ('in contrast', 201),
 #~ ('nonetheless', 193),
 #~ ('notably', 53),
 #~ ('contrarily', 19)]

################ Gutenberg STATS

#~ >>> len(order_gu)
#~ 155048363
#~ >>> len(skip_gu)
#~ 142637386
#~ >>> len(next_gu)
#~ 7201394
#~ >>> len(conj_gu)
#~ 1027467

#~ fine_counts   = {}
#~ coarse_counts = {}
#~ for a, b, c, d in conj_wi:
  #~ fine_counts[c]    = fine_counts.get(c, 0) + 1
  #~ coarse_counts[d]  = coarse_counts.get(d, 0) + 1

#~ >>> pprint(sorted(coarse_counts.items(), key=lambda x:x[1], reverse=True))
#~ [('addition', 235253),
 #~ ('contrast', 204896),
 #~ ('result', 126026),
 #~ ('specific', 111962),
 #~ ('time', 86460),
 #~ ('strengthen', 33959),
 #~ ('compare', 26631),
 #~ ('return', 6503),
 #~ ('recognize', 669)]
 
#~ >>> pprint(sorted(fine_counts.items(), key=lambda x:x[1], reverse=True))
#~ [('however', 142906),
 #~ ('for example', 94864),
 #~ ('in addition', 87922),
 #~ ('meanwhile', 52821),
 #~ ('thus', 44181),
 #~ ('also', 43575),
 #~ ('furthermore', 35788),
 #~ ('therefore', 35170),
 #~ ('finally', 28583),
 #~ ('instead', 26482),
 #~ ('moreover', 22398),
 #~ ('in fact', 19911),
 #~ ('similarly', 19148),
 #~ ('then', 18231),
 #~ ('consequently', 15162),
 #~ ('subsequently', 14852),
 #~ ('indeed', 14048),
 #~ ('in contrast', 10489),
 #~ ('nonetheless', 10308),
 #~ ('further', 9033),
 #~ ('hence', 8157),
 #~ ('likewise', 7483),
 #~ ('accordingly', 7164),
 #~ ('still', 6503),
 #~ ('conversely', 6489),
 #~ ('specifically', 6349),
 #~ ('that is', 6048),
 #~ ('thereafter', 5561),
 #~ ('again', 5534),
 #~ ('now', 5423),
 #~ ('rather', 4764),
 #~ ('next', 4424),
 #~ ('notably', 4093),
 #~ ('otherwise', 2954),
 #~ ('besides', 2420),
 #~ ('incidentally', 930),
 #~ ('namely', 608),
 #~ ('certainly', 505),
 #~ ('henceforth', 410),
 #~ ('anyway', 350),
 #~ ('undoubtedly', 164),
 #~ ('contrarily', 148),
 #~ ('nevertherless', 6)]


import shelve
from random import shuffle

def make_joint(st):
  guten_shelf = shelve.open('Gutenberg/' + st + 'Shelve.shlf')
  wiki_shelf  = shelve.open('Wikipedia/' + st + 'Shelve.shlf')
  joint_shelf = shelve.open('GutenWiki/' + st + 'Shelve.shlf', writeback=True)
  print st, 'guten', guten_shelf['len'], 'wiki', wiki_shelf['len'],
  n_batches = min(guten_shelf['len'], wiki_shelf['len']) - 1
  joint_shelf['len'] = 2 * n_batches
  print 'total', joint_shelf['len']
  for i in range(n_batches):
    if i % 10 == 0:
      print i
    guten_batch = guten_shelf[str(i)]
    wiki_batch  = wiki_shelf[str(i)]
    joint_a     = guten_batch[:32000] + wiki_batch[:32000]
    joint_b     = guten_batch[32000:] + wiki_batch[32000:]
    shuffle(joint_a)
    shuffle(joint_b)
    joint_shelf[str(2 * i)]     = joint_a
    joint_shelf[str(2 * i + 1)] = joint_b
    guten_shelf.sync()
    wiki_shelf.sync()
    joint_shelf.sync()
  guten_shelf.close()
  wiki_shelf.close()
  joint_shelf.close()



make_joint('Conj')
# Conj guten 16 wiki 14 total 26
# 1,664,000 examples, 3,328,000 sentences

make_joint('Next')
# Next guten 113 wiki 43 total 84
# 5,376,000 examples, 43,008,000 sentences

make_joint('Order')
# Order guten 1760 wiki 813 total 1624
# 103,936,000 examples, 207,872,000 sentences

make_joint('Real')
# Real guten 925 wiki 601 total 1200
# 76,800,000 examples, 76,800,000 sentences

make_joint('Combined')
# Combined guten 1850 wiki 1202 total 2402
# 153,728,000 examples, 153,728,000 sentences

make_joint('Skip')
# Skip guten 1175 wiki 703 total 1404
# 89,856,000 examples, 89,856,000 sentences

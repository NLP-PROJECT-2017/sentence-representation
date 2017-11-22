from glob import glob
from pprint import pprint

def read_logs(file_name):
  train_accus   = []
  valid_accus   = []
  current_epoch = 0
  arguments     = {}
  f = open(file_name)
  validated = []
  for line in f:
    if "Argument" in line:
      tab = line.strip().split()
      arguments[tab[-2][:-1]] = tab[-1]
    if "starting epoch" in line:
      current_epoch = int(line.strip().split()[-1])
    if "Loss" in line:
      if "VALIDATING" in line:
        validated += [current_epoch]
        valid_accus += [(current_epoch, float(line.strip().split()[-1]))]
      else:
        train_accus += [(current_epoch, float(line.strip().split()[-1]))]
  f.close()
  validated   = list(set(validated))
  train_accus = [(ep, x) for ep, x in train_accus if ep in validated]
  accus       = [(a[0], a[1], b[1]) for a, b in zip(train_accus, valid_accus)]
  return (arguments, accus)


# arguments, accus = read_logs('log_next_Convo_256_10_3_sent_2017-03-09_031611_828431.log')

# arguments, accus = read_logs('log_joint_GRU_128_10_10_sent_2017-03-13_013247_066345.log')

all_logs = {"next"  : {"BoW"    : [],
                       "Convo"  : [],
                       "GRU"    : [],
                       "BiGRU"  : []},
            "order" : {"BoW"    : [],
                       "Convo"  : [],
                       "GRU"    : [],
                       "BiGRU"  : []},
            "conj"  : {"BoW"    : [],
                       "Convo"  : [],
                       "GRU"    : [],
                       "BiGRU"  : []},
            "joint" : {"BoW"    : [],
                       "Convo"  : [],
                       "GRU"    : [],
                       "BiGRU"  : []}}


for task in ["next", "order", "conj", "joint"]:
  for file_name in glob("*" + task + "_BoW*"):
    all_logs[task]["BoW"] += [read_logs(file_name)]
  for file_name in glob("*" + task + "_Convo*"):
    all_logs[task]["Convo"] += [read_logs(file_name)]
  for file_name in glob("*" + task + "_GRU_256*"):
    all_logs[task]["GRU"] += [read_logs(file_name)]
  for file_name in glob("*" + task + "_GRU_128*"):
    all_logs[task]["BiGRU"] += [read_logs(file_name)]

for args, accus in all_logs["joint"]["BoW"]:
  print args['init_range'], args['learning_rate']
  pprint(accus[-3:])

for args, accus in all_logs["joint"]["Convo"]:
  print args['init_range'], args['learning_rate']
  pprint(accus[-3:])

for args, accus in all_logs["joint"]["GRU"]:
  print args['init_range'], args['learning_rate']
  pprint(accus[-3:])

for args, accus in all_logs["joint"]["BiGRU"]:
  print args['init_range'], args['learning_rate']
  pprint(accus[-3:])


args, accus = all_logs["joint"]["BiGRU"][-3]
pprint(accus[3 * i] for i in range(15))
pprint(accus[3 * i + 1] for i in range(15))
pprint(accus[3 * i + 2] for i in range(15))


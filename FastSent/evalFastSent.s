#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=20GB
#SBATCH --job-name=senteval
#SBATCH --mail-type=END
#SBATCH --mail-user=fc1315@nyu.edu
#SBATCH --output=senteval_%j.out
 
#module purge

#source /scratch/fc1315/nlp-project/py2.7/bin/activate
RUNDIR=/scratch/fc1315-share/nlp-project
module load python/intel/2.7.12 pytorch/0.2.0_1
source /scratch/fc1315-share/nlp-project/py2.7/bin/activate

#git clone https://github.com/piskvorky/gensim.git

#export PYTHONPATH="${PYTHONPATH}:$./gensim"

#cp sentence-representation/FastSent/fastsent* gensim/gensim/models/ 

#cd gensim/gensim/models
#cython -a fastsent_inner.pyx 
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I/usr/include/python2.7 -o fastsent_inner.so fastsent_inner.c
#cd ../../../


#PYTHONPATH=$PYTHONPATH:
python evalFastSent.py

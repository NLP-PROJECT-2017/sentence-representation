#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --mem=30GB
#SBATCH --job-name=FastSentDim512E10+AE
#SBATCH --mail-type=END
#SBATCH --mail-user=fc1315@nyu.edu
#SBATCH --output=fastsent_%j.out
 
#module purge

RUNDIR=/scratch/fc1315-share/nlp-project
module load python/intel/2.7.12
source /scratch/fc1315-share/nlp-project/py2.7/bin/activate
#git clone https://github.com/piskvorky/gensim.git

export PYTHONPATH="${PYTHONPATH}:$./gensim"

#cp sentence-representation/FastSent/fastsent* gensim/gensim/models/ 
#cp sentence-representation/FastSent/fastsent* py2.7/lib/python2.7/site-packages/gensim/models/

#cd gensim/gensim/models
#cython -a fastsent_inner.pyx 
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I$PYTHON27_INC -I$NUMPY_INC -o fastsent_inner.so fastsent_inner.c
#cd ../../../

#cd py2.7/lib/python2.7/site-packages/gensim/models
#cython -a fastsent_inner.pyx
#gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing -I$PYTHON27_INC -I$NUMPY_INC -o fastsent_inner.so fastsent_inner.c
#cd ../../../../../../

PYTHONPATH=$PYTHONPATH:
python sentence-representation/FastSent/trainFastSent.py --dim 512 --epoch 10 --autoencode --min_count 3  --corpus data/books_in_sentences --savedir out/

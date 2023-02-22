#!/bin/bash --login
#$ -cwd               # Run job from current directory
#$ -pe smp.pe 16      # Number of cores to use. Can be between 2 and 32.
#$ -j y
#$ -cwd
module load apps/binapps/anaconda3/2021.11
conda activate neuralprophet
export OMP_NUM_THREADS=$NSLOTS
module load apps/gcc/R/4.0.2
/mnt/hum01-home01/ambs/y06068cc/R/x86_64-pc-linux-gnu-library/4.0/irace/bin/irace --scenario="./scenario_darts.txt 

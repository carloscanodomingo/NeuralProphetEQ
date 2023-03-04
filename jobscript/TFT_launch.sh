#!/bin/bash --login
#$ -cwd               # Run job from current directory
#$ -pe smp.pe 4 # Number of cores to use. Can be between 2 and 32.
#$ -l mem512
#$ -j y
#$ -cwd
#$ -M carlos.cano@manchester.ac.uk
#$ -m ase
#      b     Mail is sent at the beginning of the job.
#      e     Mail is sent at the end of the job.
#      a     Mail is sent when the job is aborted or rescheduled.
#      s     Mail is sent when the job is suspended.
module load apps/binapps/anaconda3/2021.11
conda activate neuralprophet
export OMP_NUM_THREADS=1
module load apps/gcc/R/4.0.2
/mnt/hum01-home01/ambs/y06068cc/R/x86_64-pc-linux-gnu-library/4.0/irace/bin/irace --scenario="../scenaries/scenario_darts_TFT.txt" --parallel=4

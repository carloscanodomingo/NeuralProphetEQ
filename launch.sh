#!/bin/bash --login
#$ -cwd               # Run job from current directory
#$ -pe smp.pe 12      # Number of cores to use. Can be between 2 and 32.
module load apps/R/4.0.2
/path/to/irace --parallel 32

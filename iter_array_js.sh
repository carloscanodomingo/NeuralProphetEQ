#!/bin/bash --login
#$ -cwd
#$ -pe smp.pe 2       # Each task will use 4 cores in this example
#$ -t 1-300          # A job-array with 1000 "tasks", numbered 1...1000

# My OpenMP program will read this variable to get how many cores to use.
# $NSLOTS is automatically set to the number specified on the -pe line above.
module load apps/binapps/anaconda3/2021.11
conda activate neuralprophet
module load apps/gcc/R/4.0.2

export OMP_NUM_THREADS=$NSLOTS
./ScriptDartsFCeV.py --data_path=/mnt/hum01-home01/ambs/y06068cc/data/ --out_path=/mnt/hum01-home01/ambs/y06068cc/output/results \ 
  --forecast_type=iteration --total_index=300--historic_lenght=15 --training_lenght_days=120 --learning_rate=2 --dropout=0.4 \
  --batch_size=600 --n_epochs=300 --patience=20 --dilation_base=2 --weight_norm=1 --kernel_size=32 --num_filter=6 --verbose=0  \
  --offset_start=730 --current_index=\$((\$SGE_TASK_ID - 1))

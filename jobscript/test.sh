#!/bin/bash --login
#$ -cwd               # Run job from current directory
#$ -l mem512
#$ -j y
#$ -cwd
module load apps/binapps/anaconda3/2021.11
conda activate neuralprophet
export OMP_NUM_THREADS=1
module load apps/gcc/R/4.0.2
/mnt/hum01-home01/ambs/y06068cc/tuning/complete/ScriptDartsFCeV.py 10 8 561631777 --forecast_type=folds --total_index=12 --current_index=7   --historic_lenght=14 --training_lenght_days=66 --learning_rate=4 --dropout=0.3 --batch_size=100 --epochs=100 --n_layers=3 --internal_size=4 --use_gpu=0 --probabilistic=1 --patience=3 offset_start=730 --verbose=1 --data_path=/mnt/hum01-home01/ambs/y06068cc/data/ --out_path=/mnt/hum01-home01/ambs/y06068cc/output/results/ --model=TFT --TFT_num_attention_heads=4 --TFT_full_attention=0 --TFT_add_relative_index=0 --TFT_hidden_cont_size=3

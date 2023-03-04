#!/bin/bash --login
#$ -cwd               # Run job from current directory
#$ -l mem512
#$ -j y
#$ -cwd
#$ -M carlos.cano@manchester.ac.uk
#$ -m ase
module load apps/binapps/anaconda3/2021.11
conda activate neuralprophet
export OMP_NUM_THREADS=1
module load apps/gcc/R/4.0.2
/mnt/hum01-home01/ambs/y06068cc/tuning/TEC_constant/ScriptDartsFCeV.py 16 10 475533918 --simulation_scenario=TEC_constant --forecast_type=folds --total_index=12 --current_index=9   --historic_lenght=13 --training_lenght_days=87 --learning_rate=4 --dropout=0.4 --batch_size=100 --epochs=100 --n_layers=3 --internal_size=16 --use_gpu=0 --probabilistic=0 --patience=3 offset_start=730 --verbose=1 --data_path=/mnt/hum01-home01/ambs/y06068cc/data/ --out_path=/mnt/hum01-home01/ambs/y06068cc/output/results/ --model=TFT --TFT_num_attention_heads=3 --TFT_full_attention=1 --TFT_add_relative_index=1 --TFT_hidden_cont_size=1

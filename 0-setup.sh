#!/bin/bash
module load apps/binapps/anaconda3/2021.11
module swap tools/env/proxy tools/env/proxy2
conda create -n neuralprophet python=3.9.7
conda activate neuralprophet  
pip3 install -r requirements.txt
export DATA_PATH="/mnt/hum01-home01/ambs/y06068cc/data"

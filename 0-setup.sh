#!/bin/bash
# First move a config file you may have out of the way. If this is present it
# will force the pip install to occur outside of the conda env.
mv ~/.pydistutils.cfg ~/.pydistutils.cfg.ignore
# Now load which ever version of python you need. For example:
module load apps/binapps/anaconda3/2021.11
module swap tools/env/proxy tools/env/proxy2
conda create -n neuralprophet python=3.9.7
conda activate neuralprophet  
pip3 install -r requirements.txt

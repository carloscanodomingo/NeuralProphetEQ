#!/bin/bash
set -e
set -o pipefail

# Find our own location.
BINDIR=$(dirname "$(readlink -f "$(type -P $0 || echo $0)")")
OUTDIR="$HOME/scratch/"

# This function launches one job $1 is the job name, the other arguments is the job to submit.
qsub_job() {
    PARALLEL_ENV=smp.pe
    # We would like to use $BASHPID here, but OS X version of bash does not
    # support it.
    RUNNER=$1
    nruns=$2
    shift 2
    JOBNAME="${RUNNER}-$$"
    qsub -v PATH <<EOF
#!/bin/bash --login
#$ -t 1-$nruns
#$ -N $JOBNAME
# -pe $PARALLEL_ENV  2
# -l ivybridge
# -l mem256
# -M carlos.cano@manchester.ac.uk
# -m ase
#      b     Mail is sent at the beginning of the job.
#      e     Mail is sent at the end of the job.
#      a     Mail is sent when the job is aborted or rescheduled.
#      s     Mail is sent when the job is suspended.
#
#$ -o $OUTDIR/${JOBNAME}.stdout
#$ -j y
#$ -cwd
module load apps/binapps/anaconda3/2021.11
conda activate neuralprophet
module load apps/gcc/R/4.0.2
echo "running: ${BINDIR}/$RUNNER \$((\$SGE_TASK_ID - 1))"
${BINDIR}/$RUNNER CONFIG 23 1234567 --offset_start=1095 --historic_lenght=5 --training_lenght_days=38 --learning_rate=3 --dropout=0.2 --batch_size=600 --epochs=300 --n_layers=1 --internal_size=16 --use_gpu=0 --probabilistic=1 --patience=15 offset_start=730 --verbose=0 --data_path=/mnt/hum01-home01/ambs/y06068cc/data/ --out_path=/mnt/hum01-home01/ambs/y06068cc/output/results/ --model=RNN --RNN_model=RNN --simulation_scenario=TEC_EQ --forecast_type=iteration --total_index=$nruns --current_index=\$((\$SGE_TASK_ID - 1)) --forecast_lenght_hours=24
EOF
}


nruns=1000
LAUNCHER=qsub_job
$LAUNCHER ScriptDartsFCeV.py $nruns 

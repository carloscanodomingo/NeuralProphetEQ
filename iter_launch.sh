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
# -pe $PARALLEL_ENV $NB_PARALLEL_PROCESS 
# -l ivybridge
#$ -l mem256
#$ -M manuel.lopez-ibanez@manchester.ac.uk
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
${BINDIR}/$RUNNER --current_index=\$((\$SGE_TASK_ID - 1))
EOF
}
@click.option(
    "--forecast_type",
    default="folds",
    type=click.Choice(["folds", "iteration"]),
    help="Set event type",
)
@click.option("--current_index", default=0, help="number of the current fold")


nruns=10
LAUNCHER=qsub_job
#LAUNCHER=launch_local
$LAUNCHER "ScriptDartsFCeV.py --forecast_type=iteration --total_index=100   --historic_lenght=15 --training_lenght_days=56 --learning_rate=3 --dropout=0.0814 --batch_size=600 --n_epochs=300 --patience=9 --dilation_base=1 --weight_norm=1 --kernel_size=6 --num_filter=6 --verbose=0" $nruns

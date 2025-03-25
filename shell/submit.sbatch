#!/bin/bash
#SBATCH --partition=gpu2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=0
#SBATCH --gpus-per-task=0
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --output=%x.o%j.%N
#SBATCH --error=%x.e%j.%N
#SBATCH --job-name=matmul

# Print job details
NOW=`date +%H:%M:%S-%a-%d/%b/%Y`
echo '------------------------------------------------------'
echo 'This job is allocated on '$SLURM_JOB_CPUS_PER_NODE' cpu(s) and '$SLURM_GPUS_PER_NODE' gpu(s)'
echo 'Job is running on node(s): '
echo  $SLURM_JOB_NODELIST
echo '------------------------------------------------------'
#
# ==== End of Info part (say things) ===== #
#

cd $SLURM_SUBMIT_DIR            # here we go into the submission directory
export SLURM_NTASKS_PER_NODE=1  # need to export this, not for all clusters but Ulysses has a bug :/

# load a bunch of modules
module use /opt/contrib/mathlab/modules
module load miniconda3
source $HOME/.bashrc
module load nvhpc-hpcx/23.1

export CUDA_HOME=$NVHPC_ROOT/cuda/12.0
export NUMBAPRO_NVVM=$NVHPC_ROOT/cuda/12.0/nvvm/lib64
export NUMBAPRO_LIBDEVICE=$NVHPC_ROOT/cuda/12.0/nvvm/libdevice
echo "$CUDA_HOME"
echo "$NUMBAPRO_NVVM"
echo "$NUMBAPRO_LIBDEVICE"

conda activate matmul

# set number of threads according to available resources
export NUMBA_NUM_THREADS=1

echo "Starting at $(date +%H:%M:%S-%a-%d/%b/%Y)"
# Run the script
mpirun -n $SLURM_NTASKS --bind-to socket --map-by socket python scripts/run.py --config=experiments/config
#mpirun -n $SLURM_NTASKS --bind-to socket --map-by socket pytest
#mpirun -n $SLURM_NTASKS --bind-to socket --map-by socket --report-bindings \
#    bash -c 'kernprof -lz -o "3_20000_rank${OMPI_COMM_WORLD_RANK}.lprof" scripts/run.py --config=experiments/config'
echo "Finished at $(date +%H:%M:%S-%a-%d/%b/%Y)"

#!/bin/bash
#SBATCH --job-name tabmon_pipeline
#SBATCH -p GPU
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --array=0-6
#SBATCH --gres=gpu:1  # Request 1 GPU per job
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=0-24:00:00   
  
echo "Executing on the machine:" $(hostname)
echo "Number of nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Number of CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Processing chunk $SLURM_ARRAY_TASK_ID"

source activate tabmon

# Pass additional parameters
python inference_parallel.py chunk_files/file_chunks_$SLURM_ARRAY_TASK_ID.txt 

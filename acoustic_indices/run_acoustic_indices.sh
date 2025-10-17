#!/bin/bash
#SBATCH --job-name=acoustic_indices
#SBATCH --partition=cpu         
#SBATCH --output=slurm_output_files_2025-08-31_to_2025-09-22/slurm_output_%A_%a.out
#SBATCH --array=0-15             
#SBATCH --mem-per-cpu=32G        
#SBATCH --time=7-00:00:00    
  

echo "Executing on the machine:" $(hostname)
echo "Number of nodes allocated: $SLURM_JOB_NUM_NODES"
echo "Number of tasks: $SLURM_NTASKS"
echo "Processing chunk $SLURM_ARRAY_TASK_ID"

# Pass additional parameters
python compute_acoustic_indices.py chunk_files_2025-08-31_to_2025-09-22/file_chunks_$SLURM_ARRAY_TASK_ID.txt 

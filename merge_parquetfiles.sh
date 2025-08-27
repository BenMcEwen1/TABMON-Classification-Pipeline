#!/bin/bash
#SBATCH --job-name=merge_files
#SBATCH --partition=cpu       
#SBATCH --output=slurm_output_merge_%A.out
#SBATCH --nodes=1                
#SBATCH --mem-per-cpu=32G        
#SBATCH --time=7-00:00:00    
  
echo "Executing on the machine:" $(hostname)

# Pass additional parameters
python merge_parquetfiles.py

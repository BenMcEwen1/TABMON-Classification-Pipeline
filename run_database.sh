#!/bin/bash
#SBATCH --job-name=database
#SBATCH --partition=cpu         
#SBATCH --output=slurm_output_files_database/slurm_output_%A.out             
#SBATCH --mem-per-cpu=64G        
#SBATCH --time=1-00:00:00    
  

echo "Executing on the machine:" $(hostname)


# Pass additional parameters
fastapi dev app/main.py
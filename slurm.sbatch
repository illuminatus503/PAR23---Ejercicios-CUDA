#!/bin/bash

# Job name
#SBATCH --job-name=${JobName}

# Define the files which will contain the Standard and Error output
#SBATCH --output=M_%A.out
#SBATCH --error=M_%A.err

# Number of tasks that compose the job
#SBATCH --ntasks=1

# Advanced use
#SBATCH --cpus-per-task=20
#SBATCH --threads-per-core=2
#SBATCH --ntasks-per-core=2

# Required memory (Default 2GB)
#SBATCH --mem-per-cpu=2G

# Select one partition
#SBATCH --partition=EHP-PAR
#SBATCH --gpus-per-task=1
#SBATCH --gpus=1

# If you are using arrays, specify the number of tasks in the array
#SBATCH --array=1-1

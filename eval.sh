#!/bin/bash

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=4:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=1   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=2000M   # memory per node
#SBATCH -J "training_no0"   # job name
#SBATCH -o "training_no0"   # job output file
#SBATCH -e "training_no0"   # job error file


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/hps/software/users/birney/fanny/default/bin/python /nfs/research/birney/users/fanny/medaka/ziram_analysis/CNN_medaka/evaluation.py

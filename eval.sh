#!/bin/bash

#Submit this script with: sbatch thefilename
#For more details about each parameter, please check SLURM sbatch documentation https://slurm.schedmd.com/sbatch.html

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of tasks
#SBATCH --cpus-per-task=1   # number of CPUs Per Task i.e if your code is multi-threaded
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem=100GB   # memory per node
#SBATCH -J "eval_resnet18_SC0_cat_mod_3 "   # job name
#SBATCH -o "%x.out"   # job output file
#SBATCH -e "%x.out"   # job error file


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
/hps/software/users/birney/fanny/default/bin/python /nfs/research/birney/users/fanny/medaka/ziram_analysis/CNN_medaka/evaluation.py \
     -o /nfs/research/birney/users/fanny/medaka/ziram_analysis/outputs_Summer2024/3rep_png_CO4/resnet18_SC0_cat_mod_3 
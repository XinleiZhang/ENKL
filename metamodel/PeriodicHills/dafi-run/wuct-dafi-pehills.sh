#!/bin/bash
#SBATCH -p xahcnormal
#SBATCH -N 16
#SBATCH -n 128
#SBATCH --ntasks-per-node=8

source ~/wuchutian/envs/mpi.sh
source ~/wuchutian/envs/dafi.sh
source ~/wuchutian/envs/conda3.sh
source ~/wuchutian/envs/py310.sh
source ~/wuchutian/envs/PyFoam.sh

dafi dafi.in

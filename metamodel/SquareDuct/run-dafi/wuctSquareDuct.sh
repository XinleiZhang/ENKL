#!/bin/bash
#SBATCH -p xahcnormal
##SBATCH -N 1
#SBATCH -n 50

# source init-dafi.sh
dafi dafi.in # > log 

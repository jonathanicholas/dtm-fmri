#!/bin/bash
#
#all commands that start with SBATCH contain commands that are just used by SLURM for scheduling
#################
#set a job name
#SBATCH --job-name=sax
#################
#a file for job output, you can check job progress
#SBATCH --output=/scratch/PI/menon/projects/jnichola/2017/SAX/Jobs/repjob-%j.out
#################
# a file for errors from the job
#SBATCH --error=/scratch/PI/menon/projects/jnichola/2017/SAX/Jobs/repjob-%j.err
#################
#time you think you need; default is one hour
#in minutes in this case
#SBATCH --time=04:00:00
#################
#quality of service; think of it as job priority
#SBATCH --qos=normal
#################
#number of nodes you are requesting
#SBATCH --nodes=1
#################
#SBATCH --mem-per-cpu=100000
#################
#SBATCH -p owners
#now run normal batch commands
module load matlab/R2017a

#Run the job
matlab -nojit -nosplash -nodesktop -nodisplay -r "runSAX("$1");exit;"

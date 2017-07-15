#!/bin/bash

# Submit job for each subject.
# Make sure to change the number of subjects
for i in {1..122}
do
sbatch submit_SAX_job.sh $i
done

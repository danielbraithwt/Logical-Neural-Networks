#!/bin/sh
#
# Force Bourne Shell if not Sun Grid Engine default shell (you never know!)
#
#$ -S /bin/sh
#
cd /am/phoenix/home1/braithdani/Logical-Neural-Networks/FinalExperements
mkdir $JOB_ID
python3 RunSigmoidNet.py sigmoid-experement5.exp $JOB_ID $SGE_TASK_ID

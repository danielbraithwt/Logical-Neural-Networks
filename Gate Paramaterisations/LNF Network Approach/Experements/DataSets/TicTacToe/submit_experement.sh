#!/bin/sh
#
# Force Bourne Shell if not Sun Grid Engine default shell (you never know!)
#
#$ -S /bin/sh
#
cd "/am/phoenix/home1/braithdani/Logical-Neural-Networks/Gate Paramaterisations/LNF Network Approach/Experements/DataSets/TicTacToe/"
python3 Classifier.py $SGE_TASK_ID

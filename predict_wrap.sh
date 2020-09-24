#!/bin/sh 
module load Python/3.6.4
CUDA_VISIBLE_DEVICES=""

$HOME/bin/predict-testdata \
--epoch "$1" \
--modeldir /lustre/storeB/users/espenm/models/v52rerun \
--predictscript /lustre/storeB/users/espenm/cc-classifier/predict.py \
--labelfile /home/espenm/data/v52/testdata.txt

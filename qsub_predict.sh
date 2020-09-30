qsub -o $PWD/stats_v52_$1 -sync y -b n  -q research-el7.q -pe shmem-12 12  -l h_vmem=12G  -S /bin/bash /lustre/storeB/users/espenm/cc-classifier/predict_wrap.sh $1


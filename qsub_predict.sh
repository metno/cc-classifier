qsub -b n  -q research-el7.q -pe shmem-12 12  -l h_vmem=2G  -S /bin/bash /lustre/storeB/users/espenm/cc-classifier/predict_wrap.sh

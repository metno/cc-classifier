qlogin -q gpu-r8.q -l h=gpu-05.ppi.met.no,h_rss=16G,mem_free=16G
source /modules/rhel8/conda/install/etc/profile.d/conda.sh
conda activate TensorFlowGPU-03-2022

cd /lustre/storeB/users/espenm/cc-classifier
rm -rf checkpoints/* nohup.out /tmp/tf/*
nohup ./train.py &
sleep 2
tail -f nohup.out

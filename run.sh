rm -rf checkpoints/* nohup.out /tmp/tf/*
nohup ./train.py &
sleep 2
tail -f nohup.out

#!/bin/bash

sleep 0.5
printf "Hyper-tuning with "$1" workers "
python hypertune.py --n_workers $1 &
sleep 3

for i in `seq 1 "$1"`; do
	printf "Start worker "
	sleep 0.1
	python hypertune.py --worker &
done

wait
killall python
sleep 1
printf "DONE"
exit 1

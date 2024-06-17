#!/usr/bin/env bash

python federated_main.py --model=exq --pretrain=False --dataset=mnist --gpu=cuda:0 --epochs=5 --iid=1 --dirichlet=10.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0

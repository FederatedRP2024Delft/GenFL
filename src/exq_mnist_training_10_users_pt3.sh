#!/usr/bin/env bash

# federated baseline: with no pretraining
python federated_main.py --model=exq --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=7.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=exq --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=7.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=exq --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=7.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=exq --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=7.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=exq --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=7.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0

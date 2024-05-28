#!/usr/bin/env bash

# MNIST for 3 users
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=1 --num_users=3 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.1 --num_users=3 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.3 --num_users=3 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.5 --num_users=3 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.7 --num_users=3 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.9 --num_users=3 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=mnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=1.0 --num_users=3 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.01 --unequal=0 --num_generate=0

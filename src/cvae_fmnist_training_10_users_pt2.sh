#!/usr/bin/env bash
# local training and testing
python baseline_main.py
python baseline_main.py
python baseline_main.py
python baseline_main.py
python baseline_main.py

# federated non-iid 0.5
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.5 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.5 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.5 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.5 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.5 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0

# federated non-iid 0.8
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.8 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.8 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.8 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.8 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.8 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
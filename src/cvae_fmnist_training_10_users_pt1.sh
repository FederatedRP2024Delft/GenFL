#!/usr/bin/env bash

# baseline federated
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=1.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=1.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=1.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=1.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=1 --dirichlet=1.0 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0

# dirichlet
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.1 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.1 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.1 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.1 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.1 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0

python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.3 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.3 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.3 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.3 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0
python federated_main.py --model=cvae --dataset=fmnist --gpu=cuda:0 --epochs=20 --iid=2 --dirichlet=0.3 --num_users=10 --frac=1.0 --local_ep=2 --local_bs=32 --lr=0.001 --unequal=0 --num_generate=0

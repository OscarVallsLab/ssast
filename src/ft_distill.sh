#!/bin/bash
python train_student.py --n_epochs 50 --balance 1 --lr 0.0001 --num_tokens 384 --mlp_dim 32
python train_student.py --n_epochs 50 --balance 1 --lr 0.0001 --num_tokens 192 --mlp_dim 32
python train_student.py --n_epochs 50 --balance 1 --lr 0.0001 --num_tokens 96 --mlp_dim 32
python train_student.py --n_epochs 50 --balance 1 --lr 0.0001 --num_tokens 48 --mlp_dim 32

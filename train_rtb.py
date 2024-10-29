import os
import sys
import matplotlib.pyplot as plt
import torch
from rtb_model import RTBModel
import argparse
from distutils.util import strtobool

parser = argparse.ArgumentParser()

parser.add_argument('--n_iters', default=50000, type=int, metavar='N', help='Number of training iterations')
parser.add_argument('-bs', '--batch_size', type=int, default=64, help="Training Batch Size.")
parser.add_argument('--lr', '--learning_rate', default=5e-5, type=float, help='Initial learning rate.')
parser.add_argument('--prompt', type=str, default="A photorealistic green rabbit on purple grass.", help='Prompt for finetuning')
parser.add_argument('--reward_prompt', type=str, default="", help='Prompt for reward model (defaults to args.prompt)')
parser.add_argument('--diffusion_steps', type=int, default=100)
parser.add_argument('--wandb_track', default=False, type=strtobool, help='Whether to track with wandb.')
parser.add_argument('--prior_sample', default=False, type=strtobool, help="Whether to use off policy samples from prior")
parser.add_argument('--beta_start', default=1.0, type=float, help='Initial Inverse temperature for reward (Also used for beta if anneal=False)')
parser.add_argument('--beta_end', default=20.0, type=float, help='Final Inverse temperature for reward')
parser.add_argument('--anneal', default=False, type=strtobool, help='Whether to anneal beta (From beta_start to beta_end)')
parser.add_argument('--anneal_steps', default=15000, type=int, help="Number of steps for temperature annealing")
parser.add_argument('--loss_batch_size', type=int, default=-1, help="Batched RTB loss batch size")

args = parser.parse_args()
if args.reward_prompt == '':
    args.reward_prompt = args.prompt
if args.loss_batch_size == -1:
    args.loss_batch_size = args.batch_size

rtb_model = RTBModel(prompt=args.prompt, reward_prompt=args.reward_prompt, diffusion_steps=args.diffusion_steps, beta_start=args.beta_start, beta_end=args.beta_end)

rtb_model.finetune(shape=(args.batch_size, 16, 64, 64), n_iters = args.n_iters, wandb_track=args.wandb_track, learning_rate=args.lr, prior_sample=args.prior_sample, anneal=args.anneal, anneal_steps=args.anneal_steps, loss_batch_size=args.loss_batch_size)
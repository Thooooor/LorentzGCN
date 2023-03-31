import argparse
from models import ALL_MODELS
from optimizers import ALL_OPTIMIZERS


parser = argparse.ArgumentParser(description='LorentzGCN')
parser.add_argument("--mode", type=str, choices=["train", "valid", "test"], default="train")
# Model Parameters
parser.add_argument("--model", type=str, choices=list(ALL_MODELS.keys()), default="light_gcn")
parser.add_argument("--layer", type=int, default=3, help="layer num of GCN")
parser.add_argument("--dim", type=int, default=64, help='dimensionality of atom features')
parser.add_argument("--scale", type=float, default=1.0, help='scale of initial embedding')
parser.add_argument("--margin", type=float, default=1.0, help='margin for loss')
parser.add_argument("--c", type=float, default=1.0, help='c for hyperbolic space')
parser.add_argument("--network", type=str, default="resSumGCN", help="network type")
parser.add_argument("--k_list", default="[50]")
# Log Parameters
parser.add_argument("--log", type=bool, default=True, help="enable logging")
parser.add_argument("--log_dir", type=str, default="./logs", help="Dir for logs")
parser.add_argument("--wandb", type=bool, default=False, help="enable wandb")
parser.add_argument("--wandb_project", type=str, default="DSAA5009", help="wandb project name")
# Training Parameters
parser.add_argument("--random_seed", type=int, default=2020)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--device", type=int, default=4)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--patience", type=int, default=5)
parser.add_argument("--eval_freq", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=1024, help='mini-batch size')
parser.add_argument("--sample_size", type=int, default=-1, help="sample size, -1 to not use sampling")
# Optimizer Parameters
parser.add_argument("--optimizer", type=str, choices=list(ALL_OPTIMIZERS.keys()), default='Adam')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0.005)
parser.add_argument("--momentum", type=float, default=0.95)
# Dataset Parameters
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--dataset", type=str, default="taobao")

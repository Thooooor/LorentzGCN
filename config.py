import argparse


parser = argparse.ArgumentParser(description='LorentzGCN')
parser.add_argument("--mode", type=str, choices=["train", "valid", "test"], default="train")
# Model Parameters
parser.add_argument("--model", type=str, default="light_gcn")
parser.add_argument("--layer", type=int, default=3, help="layer num of GCN")
parser.add_argument("--dim", type=int, default=64, help='dimensionality of atom features')
parser.add_argument("--k_list", default="[10, 20, 50]")
# Log Parameters
parser.add_argument("--log", type=bool, default=True, help="enable logging")
parser.add_argument("--log_dir", type=str, default="./logs", help="Dir for logs")
parser.add_argument("--wandb", type=bool, default=False, help="enable wandb")
parser.add_argument("--wandb_project", type=str, default="DSAA5009", help="wandb project name")
# Training Parameters
parser.add_argument("--random_seed", type=int, default=2020)
parser.add_argument("--cuda", type=bool, default=True)
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--patience", type=int, default=-1)
parser.add_argument("--batch_size", type=int, default=256, help='mini-batch size')
parser.add_argument("--sample_size", type=int, default=-1, help="sample size, -1 to not use sampling")
parser.add_argument("--valid", type=int, default=5, help="")
# Optimizer Parameters
parser.add_argument("--optimizer", type=str, choices=["Adagrad", "Adam", "SGD"], default='Adam')
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--weight_decay", type=float, default=0)
parser.add_argument("--momentum", type=float, default=0.9)
# Dataset Parameters
parser.add_argument("--data_dir", type=str, default="./data")
parser.add_argument("--dataset", type=str, default="taobao")

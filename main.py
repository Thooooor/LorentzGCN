import logging
import warnings
from time import time

import torch.nn as nn
import torch.optim as optim
import wandb

from config import parser
from utils import set_up_logger, DataLoader

warnings.filterwarnings("ignore")


def main():
    # initial
    args = parser.parse_args()
    saving_path, saving_name = set_up_logger(args)

    if args.wandb is True:
        wandb.init(
            project=args.wandb_project,
            name=saving_name,
            tags=[args.model, args.dataset],
            config=args)

    # load data
    logging.info("Loading {} dataset".format(args.dataset))
    start = time()
    dataset = DataLoader(args.path)

    logging.info("Loading data costs {: .2f}s".format(time() - start))

    # build model
    logging.info("Building model")
    start = time()
    model = None
    logging.info("Building model costs {: .2f}s".format(time() - start))

    # define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam()

    # train process
    for epoch in range(args.epochs):
        train()
        evaluate()

def train():
    pass


def evaluate():
    pass


if __name__ == '__main__':
    main()

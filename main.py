import logging
import os
import warnings
from time import time

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from config import parser
from model import LightGCN
from utils import set_up_logger, Taobao, AverageRecord, Metrics

warnings.filterwarnings("ignore")


def main():
    # initial
    args = parser.parse_args()
    # convert k_list from str type to list
    args.k_list = [int(k) for k in args.k_list[1:-1].split(",")]
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

    dataset = Taobao(args.data_dir)
    train_loader = dataset.train_loader
    valid_loader = dataset.valid_loader
    test_loader = dataset.test_loader

    logging.info("Loading data costs {: .2f}s".format(time() - start))

    # build model
    logging.info("Building model")
    start = time()

    model = LightGCN(dataset.num_users, dataset.num_users, args.dim, args.layer)

    logging.info("Building model costs {: .2f}s".format(time() - start))

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # train process
    best_epoch = 0
    best_metrics = None
    counter = 0

    logging.info("Start training")
    for epoch in range(args.epochs):
        train_loss = train(train_loader, model, optimizer)
        logging.info("Epoch {} | average train loss: {:.4f}".format(epoch, train_loss))
        valid_metrics = evaluate(valid_loader, model, args.k_list, split="valid")

        if (epoch + 1) % args.eval_freq == 0:
            logging.info("Epoch {} | valid metrics: {}".format(epoch, valid_metrics))
            if not best_metrics or valid_metrics["valid NDCG@50"] > best_metrics["valid NDCG@50"]:
                best_epoch = epoch
                best_metrics = valid_metrics
                torch.save(model.cpu().state_dict(), os.path.join(saving_path, "best_model.pth"))
                logging.info("Epoch {} | save best model in {}".format(epoch, saving_path))
                model.cuda()
                counter = 0
            elif args.patience != -1:
                counter += 1
                if counter >= args.patience:
                    logging.info("Early stop at epoch {}".format(epoch))
                    break

        if args.wandb is True:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_NDCG@50": valid_metrics["valid NDCG@50"],
                "valid_Recall@50": valid_metrics["valid Recall@50"],
            })

    logging.info("Optimization Finished!")

    # load best model
    if not best_metrics:
        torch.save(model.cpu().state_dict(), os.path.join(saving_path, "best_model.pth"))
    else:
        model.load_state_dict(torch.load(os.path.join(saving_path, "best_model.pth")))
        logging.info("Load best model at epoch {} from {}".format(best_epoch, saving_path))

    model.cuda()
    model.eval()

    valid_metrics = evaluate(valid_loader, model, args.k_list, split="valid")
    logging.info("Valid metrics: {}".format(valid_metrics))
    test_metrics = evaluate(test_loader, model, args.k_list, split="test")
    logging.info("Test metrics: {}".format(test_metrics))

    if args.wandb is True:
        wandb.run.summary["epoch"] = best_epoch
        wandb.run.summary.update(valid_metrics)
        wandb.run.summary.update(test_metrics)


def train(train_loader, model, optimizer):
    """

    :param train_loader:
    :param model:
    :param optimizer:
    :return:
    """
    train_loss = AverageRecord()
    model.train()
    with tqdm(total=len(train_loader)) as bar:
        for users, pos_items, neg_items in train_loader:
            edge_index = torch.stack([users, pos_items], dim=0)
            neg_edge_index = torch.stack([users, neg_items], dim=0)
            loss = model.bpr_loss(edge_index, neg_edge_index)
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record loss and update progress bar
            train_loss.update(loss.item())
            bar.update(1)
            bar.set_postfix_str("train loss: {:.4f}".format(train_loss.avg))

    return train_loss.avg


def evaluate(data_loader, model, k_list, split="valid"):
    """

    :param k_list:
    :param split:
    :param data_loader:
    :param model:
    :return:
    """
    eval_metric = Metrics(k_list, split)
    model.eval()
    with torch.no_grad():
        for users, pos_items in data_loader:
            edge_index = torch.stack([users, pos_items], dim=0)
            pred_items = model.recommend(edge_index, k=max(k_list))
            eval_metric.update(pred_items, pos_items)

    return eval_metric


if __name__ == '__main__':
    main()

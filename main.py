import logging
import os
import warnings
from time import time

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from config import parser
from models import ALL_MODELS
from optimizers.rsgd import RiemannianSGD
from utils import set_up_logger, Taobao, AverageRecord, Metrics, set_seed, set_device

warnings.filterwarnings("ignore")


def main():
    # initial
    args = parser.parse_args()
    # convert k_list from str type to list
    args.k_list = [int(k) for k in args.k_list[1:-1].split(",")]
    saving_path, saving_name = set_up_logger(args)
    
    set_device(args.cuda, args.device)
    set_seed(args.random_seed)

    if args.wandb is True:
        wandb.init(
            project=args.wandb_project,
            name=saving_name,
            tags=[args.model, args.dataset],
            config=args)

    # load data
    logging.info("Loading {} dataset".format(args.dataset))
    start = time()

    dataset = Taobao(args.data_dir, args.batch_size)

    logging.info("Loading data costs {: .2f}s".format(time() - start))

    # build models
    logging.info("Building models")
    start = time()

    # model = LightGCN(dataset.num_users, dataset.num_items, dataset.adj_train_norm, args.dim, args.layer)
    model = ALL_MODELS[args.model](dataset.num_users, dataset.num_items, dataset.adj_train_norm)
    
    logging.info("Building models costs {: .2f}s".format(time() - start))

    # define optimizer
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    optimizer = RiemannianSGD(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)

    # train process
    best_epoch = 0
    best_metrics = None
    counter = 0

    logging.info("Start training")
    for epoch in range(args.epochs):
        train_loss = train(dataset.train_loader, model, optimizer)
        logging.info("Epoch {} | average train loss: {:.4f}".format(epoch, train_loss))
        # valid_metrics = evaluate(dataset.valid_dict, dataset.user_item_csr, model, args.k_list, split="valid")

        if (epoch + 1) % args.eval_freq == 0:
            valid_metrics = evaluate(dataset.valid_dict, dataset.user_item_csr, model, args.k_list, split="valid")
            logging.info("Epoch {} | valid metrics: {}".format(epoch, valid_metrics))
            if not best_metrics or valid_metrics["valid NDCG@50"] > best_metrics["valid NDCG@50"]:
                best_epoch = epoch
                best_metrics = valid_metrics
                torch.save(model.cpu().state_dict(), os.path.join(saving_path, "best_model.pth"))
                logging.info("Epoch {} | save best models in {}".format(epoch, saving_path))
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

    # load best models
    if not best_metrics:
        torch.save(model.cpu().state_dict(), os.path.join(saving_path, "best_model.pth"))
    else:
        model.load_state_dict(torch.load(os.path.join(saving_path, "best_model.pth")))
        logging.info("Load best models at epoch {} from {}".format(best_epoch, saving_path))

    model.cuda()
    model.eval()

    valid_metrics = evaluate(dataset.valid_dict, dataset.user_item_csr, model, args.k_list, split="valid")
    logging.info("Valid metrics: {}".format(valid_metrics))
    test_metrics = evaluate(dataset.test_dict, dataset.user_item_csr, model, args.k_list, split="test")
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
    model.cuda()
    model.train()
    with tqdm(total=len(train_loader)) as bar:
        for users, pos_items, neg_items in train_loader:
            # compute loss
            edge_index = torch.stack([users, pos_items], dim=0)
            neg_edge_index = torch.stack([users, neg_items], dim=0)
            loss = model.margin_loss(edge_index, neg_edge_index)
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record loss and update progress bar
            train_loss.update(loss.item())
            bar.update(1)
            bar.set_postfix_str("train loss: {:.4f}".format(loss.item()))

    return train_loss.avg


def evaluate(eval_dict, user_item_csr, model, k_list, split="valid"):
    """

    :param data_loader:
    :param model:
    :param k_list:
    :param split:
    :return:
    """
    eval_metric = Metrics(k_list, split)
    model.eval()
    rating_metrix = model.get_user_rating()
    eval_metric.compute_metrics(rating_metrix, eval_dict, user_item_csr)
    return eval_metric


if __name__ == '__main__':
    main()

import logging
import os
import warnings
from time import time

import torch
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm

from config import parser
from models import ALL_MODELS
from optimizers import ALL_OPTIMIZERS
from utils import set_up_logger, Taobao, AverageRecord, Metrics, set_seed, set_device
from samplers import BaseSampler, WeightedSampler

warnings.filterwarnings("ignore")


def main():
    # initial
    args = parser.parse_args()
    args.embedding_dim = args.dim
    args.num_layers = args.layers
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
    dataset = Taobao(args.data_dir)
    # sampler = WeightedSampler(dataset.num_users, dataset.num_items, args.num_negatives, dataset.train_edge_index, args.batch_size)
    sampler = BaseSampler(dataset.num_users, dataset.num_items, args.num_negatives, dataset.train_edge_index, args.batch_size)
    valid_set = dataset.valid_set
    test_set = dataset.test_set
    train_loader = DataLoader(dataset.train_set, batch_size=args.batch_size, shuffle=True)
    logging.info("Loading data costs {: .2f}s".format(time() - start))

    # build models
    logging.info("Building models")
    start = time()
    model = ALL_MODELS[args.model](dataset.num_users, dataset.num_items, dataset.adj_train_norm, args)
    logging.info("Building models costs {: .2f}s".format(time() - start))

    # define optimizer
    optimizer = ALL_OPTIMIZERS[args.optimizer](model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    
    # train process
    best_epoch = 0
    best_metrics = None
    counter = 0

    logging.info("Start training")
    for epoch in range(args.epochs):
        # train_loss = train(train_loader, model, optimizer, args.num_negatives)
        train_loader = sampler.get_data_loader()
        train_loss = train_1(train_loader, model, optimizer)
        logging.info("Epoch {} | average train loss: {:.4f}".format(epoch, train_loss))
        # valid_metrics = evaluate(valid_set, dataset.user_item_csr, model, args.k_list, split="valid")

        if (epoch + 1) % args.eval_freq == 0:
            valid_metrics = evaluate(valid_set, dataset.user_item_csr, model, args.k_list, split="valid")
            logging.info("Epoch {} | valid metrics: {}".format(epoch, valid_metrics))
            if not best_metrics or valid_metrics["valid Recall@50"] > best_metrics["valid Recall@50"]:
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

    valid_metrics = evaluate(valid_set, dataset.user_item_csr, model, args.k_list, split="valid")
    logging.info("Valid metrics: {}".format(valid_metrics))
    test_metrics = evaluate(test_set, dataset.user_item_csr, model, args.k_list, split="test")
    logging.info("Test metrics: {}".format(test_metrics))

    if args.wandb is True:
        wandb.run.summary["epoch"] = best_epoch
        wandb.run.summary.update(valid_metrics)
        wandb.run.summary.update(test_metrics)


def train_1(train_loader, model, optimizer):
    """

    :param train_loader: DataLoader
    :param model: 
    :param optimizer: 
    :return:
    """
    train_loss = AverageRecord()
    model.cuda()
    model.train()
    with tqdm(total=len(train_loader)) as bar:
        for users, pos_items, neg_item_list in train_loader:
            # compute loss
            loss = model.margin_loss_1(users, pos_items, neg_item_list)
            # compute gradient
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # record loss and update progress bar
            train_loss.update(loss.item())
            bar.update(1)
            bar.set_postfix_str("train loss: {:.4f}".format(loss.item()))

    return train_loss.avg


def train(train_loader, model, optimizer, num_negatives=1):
    """

    :param train_loader: DataLoader
    :param model: 
    :param optimizer: 
    :return:
    """
    train_loss = AverageRecord()
    model.cuda()
    model.train()
    with tqdm(total=len(train_loader)) as bar:
        for users, pos_items in train_loader:
            # compute loss
            loss = model.margin_loss(users, pos_items, num_negatives)
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

    :param eval_dict: dict of user-item pairs
    :param user_item_csr: csr_matrix of user-item interactions
    :param model: model to evaluate
    :param k_list: list of k
    :param split: valid or test
    :return:
    """
    eval_metric = Metrics(k_list, split)
    
    model.eval()
    with torch.no_grad():
        rating_metrix = model.get_user_rating()
        
    eval_metric.compute_metrics(rating_metrix, eval_dict, user_item_csr)
    
    return eval_metric


if __name__ == '__main__':
    main()

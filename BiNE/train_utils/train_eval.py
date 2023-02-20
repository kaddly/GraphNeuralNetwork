import os
import logging
from tqdm import tqdm
import time
from datetime import timedelta
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from model import BiNEModel
from utils import load_data, setup_logging, ContextsNegativesGenerator
from .loss_utils import SigmoidBCELoss
from .optimizer_utils import create_lr_scheduler


def train(args):
    setup_logging(args.run_name)
    device = args.device
    train_iter, test_iter, user_contexts, user_negatives, item_contexts, item_negatives, user_vocab, item_vocab = load_data(
        args)
    data_tool = ContextsNegativesGenerator(user_contexts, user_negatives, item_contexts, item_negatives)
    model = BiNEModel(len(user_vocab), len(item_vocab), args.embedding_size)
    loss = SigmoidBCELoss()
    optimizer = torch.optim.AdamW([{"params": model.user_net}, {"params": model.item_net}], lr=args.lr,
                                  weight_decay=args.weight_decay)

    if args.scheduler_lr:
        lr_scheduler = create_lr_scheduler(optimizer, 1, args.num_epoch)
    start_time = time.time()
    total_num = 0
    best_loss = float('inf')
    last_improve = 0
    flag = False
    for epoch in range(args.num_epoch):
        logging.info(f"Starting epoch {epoch}:")
        pBar = tqdm(train_iter)
        for u, i, w in pBar:
            u, i, w, u_cn, u_m, u_l, i_cn, i_m, i_l = [data.to(device) for data in data_tool.get_contexts_negatives_masks_labels(u, i, w)]
            model.explicit_relations()


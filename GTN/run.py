import torch


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch GTN training")
    parser.add_argument('--data_path', type=str, default='./data', help='DRIVE root')
    parser.add_argument('--dataset', type=str, help='Dataset')
    parser.add_argument('--epoch', type=int, default=40, help='Training Epochs')
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='l2 reg')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layer')
    parser.add_argument('--norm', type=str, default='true', help='normalization')
    parser.add_argument('--scheduler_lr', type=bool, default=True, help='create learning rate scheduler')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    device = torch.device(args.device)

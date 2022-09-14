import torch
from utils import load_acm
from models import GTN_Model
from train_utils import train


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch GTN training")
    parser.add_argument('--data_path', type=str, default='./data', help='DRIVE root')
    parser.add_argument('--model_dict_path', type=str, default='./saved_dict', help='Model Dict Saved Root')
    parser.add_argument('--model', type=str, default='GTN')
    parser.add_argument('--epoch', type=int, default=40, help='Training Epochs')
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='l2 reg')
    parser.add_argument('--num_hidden', type=int, default=64, help='Node dimension')
    parser.add_argument('--num_channels', type=int, default=2, help='number of channels')
    parser.add_argument('--num_layers', type=int, default=2, help='number of layer')
    parser.add_argument('--norm', type=bool, default=True, help='normalization')
    parser.add_argument('--is_current_train', type=bool, default=True, help='use current trained weight')
    parser.add_argument('--print_freq', type=int, default=20, help='print val result frequent')
    parser.add_argument('--adaptive_lr', type=bool, default=True, help='adaptive rate scheduler')
    parser.add_argument('--scheduler_lr', type=bool, default=True, help='create learning rate scheduler')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    A, node_feature, labels, train_idx, val_idx, test_idx = load_acm(args.data_path)
    num_classes = torch.max(labels).item() + 1
    net = GTN_Model(A.shape[-1], args.num_channels, node_feature.shape[1], args.num_hidden, num_classes,
                    args.num_layers, args.norm)
    train(net, [A, node_feature, labels, train_idx, val_idx], args)

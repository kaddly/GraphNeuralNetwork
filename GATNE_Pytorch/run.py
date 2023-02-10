import os
from models import GATNEModel
from utils import load_data
from train_utils import train, ValScale, test


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch GATNE training")
    # data process parameter
    parser.add_argument('--data_dir', type=str, default=os.path.join(os.path.abspath('.'), './data'),
                        help='DRIVE root')
    parser.add_argument('--dataset', type=str, default='amazon', help='Dataset name')
    parser.add_argument('--schema', type=str, default=None,
                        help='The MetaPath schema (e.g., U-I-U,I-U-I).')
    parser.add_argument('--walk_length', type=int, default=16,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num_walks', type=int, default=4,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--max_window_size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--neighbor_samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--K', type=int, default=2,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers for generating random walks. Default is 16.')

    # model parameter
    parser.add_argument('--dimensions', type=int, default=256,
                        help='Number of dimensions. Default is 256.')

    parser.add_argument('--edge_dim', type=int, default=128,
                        help='Number of edge embedding dimensions. Default is 16.')

    parser.add_argument('--att_dim', type=int, default=64,
                        help='Number of attention dimensions. Default is 32.')
    # train parameter
    parser.add_argument('--model_dict_path', type=str, default='./saved_dict', help='Model Dict Saved Root')
    parser.add_argument('--model', type=str, default='GATNE')
    parser.add_argument('--num_epoch', type=int, default=100, help='Training Epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Training Batches')
    parser.add_argument('--eval_type', type=str, default='all',
                        help='The edge type(s) for evaluation.')
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='l2 reg')
    parser.add_argument('--is_current_train', type=bool, default=False, help='use current trained weight')
    parser.add_argument('--print_freq', type=int, default=20, help='print val result frequent')
    parser.add_argument('--Max_auto_stop_epoch', type=int, default=5000)
    parser.add_argument('--scheduler_lr', type=bool, default=False, help='create learning rate scheduler')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_iter, training_data_by_type, neighbors, val_true_edge_data_by_type, val_false_edge_data_by_type, test_true_edge_data_by_type, test_false_edge_data_by_type, vocab = load_data(
        args)
    net = GATNEModel(len(vocab), args.dimensions, args.edge_dim, len(training_data_by_type), args.att_dim, None)
    val_scale = ValScale(len(vocab), list(training_data_by_type.keys()), len(training_data_by_type), neighbors, vocab)
    train(net, train_iter, val_scale, (val_true_edge_data_by_type, val_false_edge_data_by_type), args)
    test(net, (test_true_edge_data_by_type, test_false_edge_data_by_type), val_scale, args)

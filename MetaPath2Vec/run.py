from models import SkipGramModel
from utils import data_procession, generate_meta_paths, load_JData
from train_utils import train


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch MetaPath2vec training")
    parser.add_argument('--data_path', type=str, default='./data', help='DRIVE root')
    parser.add_argument('--model_dict_path', type=str, default='./saved_dict', help='Model Dict Saved Root')
    parser.add_argument('--model', type=str, default='MetaPath2vec')
    parser.add_argument('--num_epoch', type=int, default=100, help='Training Epochs')
    parser.add_argument('--num_batch', type=int, default=128, help='Training Batches')
    parser.add_argument('--meta_path', type=list, default=['user', 'item', 'user', 'item', 'user'])
    parser.add_argument('--max_window_size', type=int, default=4, help='MetaPaths max neighbor window size')
    parser.add_argument('--num_noise_words', type=int, default=4, help='MetaPaths noise words number')
    parser.add_argument("--device", default="cuda", help="training device")
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.001, help='l2 reg')
    parser.add_argument('--num_hidden', type=int, default=128, help='Node dimension')
    parser.add_argument('--is_current_train', type=bool, default=True, help='use current trained weight')
    parser.add_argument('--print_freq', type=int, default=20, help='print val result frequent')
    parser.add_argument('--Max_auto_stop_epoch', type=int, default=5000)
    parser.add_argument('--scheduler_lr', type=bool, default=True, help='create learning rate scheduler')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # generate_meta_paths(args.meta_path)
    data_iter, vocab = load_JData(batch_size=args.num_batch, max_window_size=args.max_window_size,
                                  num_noise_words=args.num_noise_words)
    net = SkipGramModel(len(vocab), args.num_hidden)
    train(net, data_iter, args)

from train_utils import train


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "BiNe"
    # model
    args.embedding_size = 128
    # train
    args.num_epoch = 500
    args.batch_size = 12
    args.weight_decay = 1e-3
    args.scheduler_lr = True
    args.alpha = 0.01
    args.beta = 0.01
    args.gamma = 0.1
    args.print_freq = 100
    args.Max_auto_stop_epoch = 1e5
    args.device = "cpu"
    args.lr = 1e-2
    # data
    args.data_set = 'wiki'
    args.train_file_name = 'case_train.dat'
    args.test_file_name = 'case_test.dat'
    args.is_digraph = False
    args.percentage = 0.15
    args.maxT = 32
    args.minT = 1
    args.max_window_size = 5
    args.K = 2
    train(args)


if __name__ == '__main__':
    launch()

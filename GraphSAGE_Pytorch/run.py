import torch
from data_utils import load_pubmed_data
from models import GraphSage

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_split, val_split = 0.3, 0.2
    batch_size, num_epochs = 8, 50
    input_dim, hidden_dim, num_neighbor_list = 500, [128, 3], [10, 10]
    assert len(hidden_dim) == len(num_neighbor_list)
    train_iter, val_iter, test_iter = load_pubmed_data('../GraphSAGE/data', batch_size, val_split, test_split, num_neighbor_list)
    for i, batch in enumerate(train_iter):
        print(batch[0][0].shape)
        print(batch[0][1].shape)
        print(len(batch[0]))
        print(batch[1])
        if i > 2:
            break
    model = GraphSage(input_dim, hidden_dim, num_neighbor_list)
    print(model)

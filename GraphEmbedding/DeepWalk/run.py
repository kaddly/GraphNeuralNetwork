from DeepWalk import DeepWalk
import networkx as nx

if __name__ == '__main__':
    G = nx.read_edgelist("../data/wiki/Wiki_edgelist.txt", create_using=nx.DiGraph())
    model = DeepWalk(G, walk_length=64, num_walks=1024, workers=4)
    model.train()
    embeddings = model.get_embeddings()
    print(embeddings)

import torch
from models import SkipGram
from utils import data_procession, generate_meta_paths, load_JData
from train_utils import train


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="pytorch MetaPath2Vec training")
    parser.add_argument()


if __name__ == '__main__':
    load_JData()

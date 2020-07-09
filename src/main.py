import argparse
import numpy as np
import tensorflow as tf
from time import time
from data_loader import load_data
from train import train

# generate seeds for tf random generators
seed = 234
np.random.seed(seed)
tf.set_random_seed(seed)

parser = argparse.ArgumentParser()

# See hyper settings in thesis

# cora
parser.add_argument('--dataset', type=str, default='cora',
                    help='which dataset to use')
parser.add_argument('--epochs', type=int, default=200,
                    help='the number of epochs')
parser.add_argument('--dim', type=int, default=32,
                    help='dimension of hidden layers')
parser.add_argument('--gcn_layer', type=int, default=5,
                    help='number of GCN layers')
parser.add_argument('--lpa_iter', type=int, default=5,
                    help='number of LPA iterations')
parser.add_argument('--l2_weight', type=float, default=1e-4,
                    help='weight of l2 regularization')
parser.add_argument('--lpa_weight', type=float, default=10,
                    help='weight of LP regularization')
parser.add_argument('--dropout', type=float, default=0.4, help='dropout rate')
parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
parser.add_argument('--vis', type=bool, default=False, help='Visualize graph')

# # citeseer
# parser.add_argument('--dataset', type=str, default='citeseer', help='which dataset to use')
# parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
# parser.add_argument('--dim', type=int, default=16, help='dimension of hidden layers')
# parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
# parser.add_argument('--lpa_iter', type=int, default=5, help='number of LPA iterations')
# parser.add_argument('--l2_weight', type=float, default=5e-4, help='weight of l2 regularization')
# parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
# parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
# parser.add_argument('--lr', type=float, default=0.2, help='learning rate')
# parser.add_argument('--vis', type=bool,default=False,help='Visualize graph')

# pubmed
# parser.add_argument('--dataset', type=str, default='pubmed', help='which dataset to use')
# parser.add_argument('--epochs', type=int, default=200, help='the number of epochs')
# parser.add_argument('--dim', type=int, default=32, help='dimension of hidden layers')
# parser.add_argument('--gcn_layer', type=int, default=2, help='number of GCN layers')
# parser.add_argument('--lpa_iter', type=int, default=1, help='number of LPA iterations')
# parser.add_argument('--l2_weight', type=float, default=2e-4, help='weight of l2 regularization')
# parser.add_argument('--lpa_weight', type=float, default=1, help='weight of LP regularization')
# parser.add_argument('--dropout', type=float, default=0.2, help='dropout rate')
# parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
# parser.add_argument('--vis', type=bool,default=False,help='Visualize graph')


t = time()
args = parser.parse_args()

if args.dataset in ['cora', 'citeseer', 'pubmed']:
    data = load_data(args.dataset)

train(args, data)
print('time used: %d s' % (time() - t))

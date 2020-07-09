import numpy as np
import tensorflow as tf
from model import GCN_LPA
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import networkx as nx
import json

def print_statistics(features, labels, adj):
    n_nodes = features[2][0]
    n_edges = (len(adj[0]) - labels.shape[0]) // 2
    n_features = features[2][1]
    n_labels = labels.shape[1]
    labeled_node_rate = 20 * n_labels / n_nodes

    n_intra_class_edge = 0
    for i, j in adj[0]:
        if i < j and np.argmax(labels[i]) == np.argmax(labels[j]):
            n_intra_class_edge += 1
    intra_class_edge_rate = n_intra_class_edge / n_edges

    print('n_nodes: %d' % n_nodes)
    print('n_edges: %d' % n_edges)
    print('n_features: %d' % n_features)
    print('n_labels: %d' % n_labels)
    print('labeled node rate: %.4f' % labeled_node_rate)
    print('intra-class edge rate: %.4f' % intra_class_edge_rate)


def visualize(labels, g, acc):
    lab = {'Case Based': 0,
           'Genetic Algorithms': 1,
           'Neural Networks': 2,
           'Probabilistic Methods': 3,
           'Reinforcement Learning': 4,
           'Rule Learning': 5,
           'Theory': 6}
    coolwarm = cm = plt.get_cmap('coolwarm')
    c_norm = colors.Normalize(vmin=0, vmax=max(labels))
    print(c_norm)
    scalar_map = cmx.ScalarMappable(norm=c_norm, cmap=coolwarm)
    print(scalar_map)

    pos = nx.spring_layout(g, seed=42)
    plt.figure(figsize=(8, 8))
    plt.axis('off')
    acc = '%.3f' % acc
    plt.title(('predict: %s' % (acc)))

    for l in lab:
        plt.scatter([0], [0], color=scalar_map.to_rgba(lab[l]), label=l)

    nx.draw_networkx(g, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
                     node_color=labels, edge_color='k',
                     arrows=False, width=0.5, style='dotted', with_labels=False)
    plt.legend()
    plt.show()


def train(args, data, batch_test=False):
    features, labels, adj, train_mask, val_mask, test_mask, graph = [
        data[i] for i in range(7)]
    print(adj)

    # uncomment the next line if you want to print statistics of the current dataset
    # print_statistics(features, labels, adj)

    model = GCN_LPA(args, features, labels, adj)

    def get_feed_dict(mask, dropout):
        feed_dict = {model.label_mask: mask, model.dropout: dropout}
        return feed_dict

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        best_val_acc = 0
        final_test_acc = 0
        for epoch in range(args.epochs):
            # train
            _, train_loss, train_acc = sess.run(
                [model.optimizer, model.loss, model.accuracy], feed_dict=get_feed_dict(train_mask, args.dropout))

            # validation, dropout must be 0
            val_loss, val_acc = sess.run(
                [model.loss, model.accuracy], feed_dict=get_feed_dict(val_mask, 0.0))

            # test
            test_loss, test_acc = sess.run(
                [model.loss, model.accuracy], feed_dict=get_feed_dict(test_mask, 0.0))

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                final_test_acc = test_acc

            if not batch_test:
                print('epoch %d   train loss: %.4f  acc: %.4f   val loss: %.4f  acc: %.4f'
                      % (epoch, train_loss, train_acc, val_loss, val_acc))

        if not batch_test:
            print('-'*30)
            print('Final Test Acc: %.4f' % final_test_acc)
            print('Best Val Acc: %.4f' % best_val_acc)
        else:
            return final_test_acc
        if args.vis or args.write_label:
            output_labels = sess.run(
                [model.predict()], feed_dict=get_feed_dict(test_mask, 0.0))
            if args.vis:
                visualize(output_labels[0], graph, final_test_acc)
            if args.write_label:
                print('-'*30)
                print('Writing predicted label...')
                with open('../data/cora/mapping.json') as f_in:
                    mapping_paper_id = json.load(f_in)
                label_id_dict = {
                                0: 'Case Based', 1: 'Genetic Algorithms',
                                2: 'Neural Networks', 3: 'Probabilistic Methods',
                                4: 'Reinforcement Learning', 5: 'Rule Learning',
                                6: 'Theory'
                                }
                predicted_label_file_name = 'gcn_lpa_skip_label.json' if args.res else 'gcn_lpa_label.json'
                paper_id_label_dict = dict()
                for i, predicted_label in enumerate(output_labels[0]):
                    paper_id_label_dict[mapping_paper_id[str(i)]] = label_id_dict[predicted_label]
                with open('../result/'+predicted_label_file_name,'w') as f_out:
                    json.dump(paper_id_label_dict, f_out, indent=4, sort_keys=True)
                print('Done')
                    

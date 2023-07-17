import argparse
import multiprocessing
from collections import defaultdict
from operator import index
import torch
import numpy as np
from six import iteritems
from sklearn.metrics import (auc, f1_score, precision_recall_curve,
                             roc_auc_score)
from tqdm import tqdm
import torch.nn.functional as func
from walk import RWGraph
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef, \
    roc_auc_score, precision_recall_curve, auc

class Vocab(object):

    def __init__(self, count, index):
        self.count = count
        self.index = index


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='/data1/botdet/GAMHN-master/mysrc/',
                        help='Input dataset path')

    parser.add_argument('--features', type=str, default=True,
                        help='Input node features')

    parser.add_argument('--walk-file', type=str, default=None,
                        help='Input random walks')

    parser.add_argument('--epoch', type=int, default=16,
                        help='Number of epoch. Default is 16.')

    parser.add_argument('--batch-size', type=int, default=2048,
                        help='Number of batch_size. Default is 2048.')

    parser.add_argument('--eval-type', type=str, default='all',
                        help='The edge type(s) for evaluation.')

    parser.add_argument('--schema', type=str, default=None,
                        help='The metapath schema (e.g., U-I-U,I-U-I).')

    parser.add_argument('--dimensions', type=int, default=100,
                        help='Number of dimensions. Default is 100.') # FIXME change this from 200 to 100

    parser.add_argument('--edge-dim', type=int, default=10,
                        help='Number of edge embedding dimensions. Default is 10.')

    parser.add_argument('--att-dim', type=int, default=20,
                        help='Number of attention dimensions. Default is 20.')

    parser.add_argument('--walk-length', type=int, default=10,
                        help='Length of walk per source. Default is 10.')

    parser.add_argument('--num-walks', type=int, default=20,
                        help='Number of walks per source. Default is 20.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--negative-samples', type=int, default=5,
                        help='Negative samples for optimization. Default is 5.')

    parser.add_argument('--neighbor-samples', type=int, default=10,
                        help='Neighbor samples for aggregation. Default is 10.')

    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping patience. Default is 5.')

    parser.add_argument('--num-workers', type=int, default=16,
                        help='Number of workers for generating random walks. Default is 16.')

    return parser.parse_args()

# 输入：[(node1,node2),(node3,node4)...] 每次输入一种边类型 对应的节点对集合
# 输出：{node1:[node2,node3...], node2:[node4,node10...]...} 与节点node_i存在特定边类型 的节点集合，已为无向图
def get_G_from_edges(edges):
    edge_dict = defaultdict(set)
    for edge in edges:
        u, v = str(edge[0]), str(edge[1])
        edge_dict[u].add(v)     # 与节点u存在某种边类型 的集合
        edge_dict[v].add(u)
    return edge_dict


def load_training_data(f_name):
    print('We are loading data from:', f_name)
    edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            if words[0] not in edge_data_by_type:
                edge_data_by_type[words[0]] = list()
            x, y = words[1], words[2]
            edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    print('Total training nodes: ' + str(len(all_nodes)))
    return edge_data_by_type


def load_testing_data(f_name):
    print('We are loading data from:', f_name)
    true_edge_data_by_type = dict()
    false_edge_data_by_type = dict()
    all_nodes = list()
    with open(f_name, 'r') as f:
        for line in f:
            words = line[:-1].split(' ')
            x, y = words[1], words[2]
            if int(words[3]) == 1:      # 两节点间存在边
                if words[0] not in true_edge_data_by_type:
                    true_edge_data_by_type[words[0]] = list()
                true_edge_data_by_type[words[0]].append((x, y))
            else:
                if words[0] not in false_edge_data_by_type:
                    false_edge_data_by_type[words[0]] = list()
                false_edge_data_by_type[words[0]].append((x, y))
            all_nodes.append(x)
            all_nodes.append(y)
    all_nodes = list(set(all_nodes))
    return true_edge_data_by_type, false_edge_data_by_type


def load_node_type(f_name):
    print('We are loading node type from:', f_name)
    node_type = {}
    with open(f_name, 'r') as f:
        for line in f:
            items = line.strip().split()
            node_type[items[0]] = items[1]
    return node_type

# 输出：feature_dic{node_i:[node1_feature],node_j:[node2_feature],...}
def load_feature_data(f_name):
    feature_dic = {}
    with open(f_name, 'r') as f:
        first = True        # flag：是否为第一行
        for line in f:
            if first:
                first = False
                continue
            items = line.strip().split()
            feature_dic[items[0]] = items[1:]
    return feature_dic


# 输入：{edge_type1:[(node1,node2),(node3,node4)...], edge_type2:[()]...} 每一种边类型 对应的节点对 集合
# 输出：[node1,node2...]   节点集合(进程池中的),edge embedding
def generate_walks(network_data, num_walks, walk_length, schema, file_name, num_workers):
    if schema is not None:      # 用于包含多个节点类型
        node_type = load_node_type(file_name + '/node_type.txt')
    else:
        node_type = None

    all_walks = []
    for layer_id, layer_name in enumerate(network_data):    # 每次循环处理一种边类型
        tmp_data = network_data[layer_name]
        # start to do the random walk on a layer

        layer_walker = RWGraph(get_G_from_edges(tmp_data), node_type, num_workers)
        print('Generating random walks for layer', layer_id)
        layer_walks = layer_walker.simulate_walks(num_walks, walk_length, schema=schema)

        all_walks.append(layer_walks)

    print('Finish generating the walks')

    return all_walks


# 输入：all_walks[ layer0[[node1,node2...], [node5,node6...]...], layer1[[node4,node7,...],[node8,...]]  ]   节点集合(进程池中的)
#      vocab{node_i:class(节点出现的次数，节点index),...}
# 输出：pairs[  (index1,index6,layer_id), (index2,index7,layer_id)... ]
def generate_pairs(all_walks, vocab, window_size, num_workers):
    pairs = []
    skip_window = window_size // 2
    for layer_id, walks in enumerate(all_walks):
        print('Generating training pairs for layer', layer_id)
        for walk in tqdm(walks):                    # walk[node1,node2,...]
            for i in range(len(walk)):              # 0~walk_length
                for j in range(1, skip_window + 1):                                                 # TODO 一个移动窗口随机提取关系？没太看懂
                    if i - j >= 0:
                        pairs.append((vocab[walk[i]].index, vocab[walk[i - j]].index, layer_id))    # 将node_i与节点node_i-j建立pair
                    if i + j < len(walk):
                        pairs.append((vocab[walk[i]].index, vocab[walk[i + j]].index, layer_id))
    return pairs


# 输入：all_walks[ layer0[[node1,node2...], [node5,node6...]...], layer1[[node4,node7,...],[node8,...]]  ]   节点集合(进程池中的)
# 输出：vocab{node_i:class(节点出现的次数，节点index),...}
#      index2word[node_i,...] 按节点出现次数降序排列，列表索引与vocab中index一致
def generate_vocab(all_walks):
    index2word = []
    raw_vocab = defaultdict(int)

    for layer_id, walks in enumerate(all_walks):
        print('Counting vocab for layer', layer_id)
        for walk in tqdm(walks):
            for word in walk:
                raw_vocab[word] += 1
    # raw_vocab:{node_i:次数, ...}  统计每个节点出现的次数
    vocab = {}
    for word, v in iteritems(raw_vocab):
        vocab[word] = Vocab(count=v, index=len(index2word))
        index2word.append(word)
    # vocab{node_i:class(节点出现的次数，节点index),...} ,index与index2word[node_i,...]中顺序一致
    index2word.sort(key=lambda word: vocab[word].count, reverse=True)   # 按出现次数排序
    for i, word in enumerate(index2word):       # 重新赋值index，保持与index2word中index一致
        vocab[word].index = i

    return vocab, index2word


def load_walks(walk_file):
    print('Loading walks')
    all_walks = []
    with open(walk_file, 'r') as f:
        for line in f:
            content = line.strip().split()
            layer_id = int(content[0])
            if layer_id >= len(all_walks):
                all_walks.append([])
            all_walks[layer_id].append(content[1:])
    return all_walks


def save_walks(walk_file, all_walks):
    with open(walk_file, 'w') as f:
        for layer_id, walks in enumerate(all_walks):
            print('Saving walks for layer', layer_id)
            for walk in tqdm(walks):
                f.write(' '.join([str(layer_id)] + [str(x) for x in walk]) + '\n')


# 输入：network_data:training_data_by_type{edge_type1:[(node1,node2),(node3,node4),...],...}
# 输出：vocab{node_i:class(节点出现的次数，节点index),...}
#      index2word[node_i,...] 按节点出现次数降序排列，列表索引与vocab中index一致
#      train_pairs[  (index1,index6,layer_id), (index2,index7,layer_id)... ]
def generate(network_data, num_walks, walk_length, schema, file_name, window_size, num_workers, walk_file):
    if walk_file is not None:       # 已有walk_file，则直接加载
        all_walks = load_walks(walk_file)
    else:                           # 若没有，则生成并保存
        all_walks = generate_walks(network_data, num_walks, walk_length, schema, file_name, num_workers)
        save_walks(file_name + 'walks.txt', all_walks)
    vocab, index2word = generate_vocab(all_walks)
    train_pairs = generate_pairs(all_walks, vocab, window_size, num_workers)

    return vocab, index2word, train_pairs


# 输出： 每个节点在每种关系上的列表 增减为 固定长度neighbor_samples，列表按node_index索引
# neighbors[ node_index1[ edge_type1[node_index, ],... ],  node_index2[ edge_type2[] ],...  ]
def generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples):
    edge_type_count = len(edge_types)
    neighbors = [[[] for __ in range(edge_type_count)] for _ in range(num_nodes)]   # neighbors[ node_index1[ edge_type1[],... ],  node_index2[ edge_type2[] ],...  ]
    for r in range(edge_type_count):
        print('Generating neighbors for layer', r)
        g = network_data[edge_types[r]]     # 存在关系edge_types[r] 的节点对集合
        for (x, y) in tqdm(g):
            ix = vocab[x].index
            iy = vocab[y].index
            neighbors[ix][r].append(iy)
            neighbors[iy][r].append(ix)
        for i in range(num_nodes):                          # 将每个节点在每种关系上的列表 增减为 固定长度neighbor_samples
            if len(neighbors[i][r]) == 0:
                neighbors[i][r] = [i] * neighbor_samples
            elif len(neighbors[i][r]) < neighbor_samples:
                neighbors[i][r].extend(
                    list(np.random.choice(neighbors[i][r], size=neighbor_samples - len(neighbors[i][r]))))
            elif len(neighbors[i][r]) > neighbor_samples:
                neighbors[i][r] = list(np.random.choice(neighbors[i][r], size=neighbor_samples))
    return neighbors


def get_score(local_model, node1, node2):
    try:
        vector1 = local_model[node1]
        vector2 = local_model[node2]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except Exception as e:
        pass


def evaluate(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    true_num = 0
    for edge in true_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(1)
            prediction_list.append(tmp_score)
            true_num += 1

    for edge in false_edges:
        tmp_score = get_score(model, str(edge[0]), str(edge[1]))
        if tmp_score is not None:
            true_list.append(0)
            prediction_list.append(tmp_score)

    sorted_pred = prediction_list[:]
    sorted_pred.sort()
    threshold = sorted_pred[-true_num]

    y_pred = np.zeros(len(prediction_list), dtype=np.int32)
    for i in range(len(prediction_list)):
        if prediction_list[i] >= threshold:
            y_pred[i] = 1

    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    ps, rs, _ = precision_recall_curve(y_true, y_scores)

    return roc_auc_score(y_true, y_scores), f1_score(y_true, y_pred), auc(rs, ps)







import math
import json
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from numpy import random
from torch.nn.parameter import Parameter
from sklearn.metrics import accuracy_score
from utils import *


# 输出一个batch的信息  源节点列表，目标节点列表，关系类型列表，每个源节点的所有邻居列表，节点特征顺序信息，用户特征，推文特征
def get_batches(pairs, neighbors, batch_size, index2word, sub_feature_dic):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size   # 一共n_batches个batch，向下取整，所以要加一个batch_size

    for idx in range(n_batches):            # 第idx个batch
        source_nodes, target_nodes, relations, neigh = [], [], [], []
        indices = []
        user_features = []
        text_features = []
        for i in range(batch_size):
            ind = idx * batch_size + i
            if ind >= len(pairs):
                break
            if not index2word[pairs[ind][0]].find('u') == -1:     # 用户
                indices.append(i)
                user_features.append(sub_feature_dic[index2word[pairs[ind][0]]])
            source_nodes.append(pairs[ind][0])       # [index_i,...]
            target_nodes.append(pairs[ind][1])       # [index_j,...]
            relations.append(pairs[ind][2])          # [layer_id,...] 其实是edge_type_i, index_i与index_j节点间边的关系集合
            neigh.append(neighbors[pairs[ind][0]])   # .append([ edge_type1[node_index1,node_index2...], edge_type2[ ],...  ])

        for i in range(batch_size):
            ind = idx * batch_size + i
            if ind >= len(pairs):
                break
            if index2word[pairs[ind][0]].find('u') == -1:     # 推文
                indices.append(i)
                text_features.append(sub_feature_dic[index2word[pairs[ind][0]]])

        # 返回一个batch_size
        yield torch.tensor(source_nodes), torch.tensor(target_nodes), torch.tensor(relations), torch.tensor(neigh), \
              torch.tensor(indices), torch.tensor(user_features, dtype=torch.float), \
              torch.tensor(text_features, dtype=torch.float)


class GATNEModel(nn.Module):
    def __init__(
        self, embedding_size, embedding_u_size, edge_type_count, dim_a, user_feature_dim, text_feature_dim
    ):
        super(GATNEModel, self).__init__()
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a

        self.edge_types = edge_types
        self.text_embed_trans = Parameter(torch.FloatTensor(text_feature_dim, embedding_size))
        self.user_embed_trans = Parameter(torch.FloatTensor(user_feature_dim, embedding_size))
        self.tweet_u_embed_trans = Parameter(torch.FloatTensor(edge_type_count, text_feature_dim, embedding_u_size))
        self.user_u_embed_trans = Parameter(torch.FloatTensor(edge_type_count, user_feature_dim, embedding_u_size))

        self.trans_weights = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, embedding_size))
        self.trans_weights_s1 = Parameter(torch.FloatTensor(edge_type_count, embedding_u_size, dim_a))
        self.trans_weights_s2 = Parameter(torch.FloatTensor(edge_type_count, dim_a, 1))

        self.reset_parameters()     # 初始化参数列表

    def reset_parameters(self):         # 参数初始化，uniform_([-1,1]均匀分布),  normal_(均值=0，标准差=std 的正态分布),
        self.text_embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.user_embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.tweet_u_embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.user_u_embed_trans.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s1.data.normal_(std=1.0 / math.sqrt(self.embedding_size))
        self.trans_weights_s2.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    # 输入：
    # train_inputs:[node_index1,node_index2...]  (1*batch_size) 源节点集合
    # train_types:[layer_id1,layer_id2...] (1*batch_size) 源节点与目标节点间关系集合
    # node_neigh:[ [edge_type1[node_index1,node_index2...], edge_type2[ ],...  ],  ] (batch_size*edge_type_count*neighbor_sample)  batch_size个源节点对应的所有关系下的节点集合
    # indices: user与text节点的顺序信息
    # user_features、text_features: 用户与推文的特征属性
    def forward(self, train_inputs, train_types, node_neigh, indices, user_features, text_features, neighbors_features):
        # 节点特征属性嵌入  (batch_size*feature_dim) * (feature_dim*embedding_size)->(batch_size*embedding_size)
        if list(user_features.size()) == [0]:
            text_node_embed = torch.mm(text_features[:, 0:text_feature_dim], self.text_embed_trans)
            node_embed = torch.index_select(text_node_embed, dim=0, index=torch.tensor(indices))  # 实现排序
        elif list(text_features.size()) == [0]:
            user_node_embed = torch.mm(user_features[:, 0:user_feature_dim], self.user_embed_trans)
            node_embed = torch.index_select(user_node_embed, dim=0, index=torch.tensor(indices))  # 实现排序
        else:
            user_node_embed = torch.mm(user_features[:, 0:user_feature_dim], self.user_embed_trans)
            text_node_embed = torch.mm(text_features[:, 0:text_feature_dim], self.text_embed_trans)
            node_embed = torch.cat([user_node_embed, text_node_embed])
            node_embed = torch.index_select(node_embed, dim=0, index=torch.tensor(indices))   # 实现排序

        # self.edge_types：包含带顺序的边类型，对不同边类型采取不同措施
        # (batch_size*edge_type_count*neighbor_sample*feature_dim) * (edge_type_count*embedding_u_size*embedding_size)
        node_embed_neighbors = []
        tweet_neighbor = None
        user_neighbor = None
        for i, edge_type in enumerate(self.edge_types):
            if edge_type == 'tweet':
                neighbor_tweet_features = neighbors_features[node_neigh][:, :, :, 0:text_feature_dim]
                tweet_neighbor = torch.einsum('bijk,akm->bijam', neighbor_tweet_features, self.tweet_u_embed_trans)
            else:
                neighbor_user_features = neighbors_features[node_neigh][:, :, :, 0:user_feature_dim]
                user_neighbor = torch.einsum('bijk,akm->bijam', neighbor_user_features, self.user_u_embed_trans)

        for i, node_edge_type in enumerate(train_types.tolist()):
            if self.edge_types[node_edge_type] == 'tweet':
                node_embed_neighbors.append(tweet_neighbor[i])
            else:
                node_embed_neighbors.append(user_neighbor[i])

        node_embed_neighbors = torch.stack(node_embed_neighbors).to(device)
        node_embed_tmp = torch.diagonal(node_embed_neighbors, dim1=1, dim2=3).permute(0, 3, 1, 2)
        node_type_embed = torch.sum(node_embed_tmp, dim=2)

        trans_w = self.trans_weights[train_types]
        trans_w_s1 = self.trans_weights_s1[train_types]
        trans_w_s2 = self.trans_weights_s2[train_types]

        attention = F.softmax(
            torch.matmul(
                torch.tanh(torch.matmul(node_type_embed, trans_w_s1)), trans_w_s2
            ).squeeze(2),
            dim=1,
        ).unsqueeze(1)
        node_type_embed = torch.matmul(attention, node_type_embed)
        node_embed = node_embed + torch.matmul(node_type_embed, trans_w).squeeze(1)

        last_node_embed = F.normalize(node_embed, dim=1)

        return last_node_embed      # 节点关于关系r的表示 (batch_size*dimensions)


class Classifier(nn.Module):
    def __init__(self, in_size, head_num, out_size=1, hidden_size=128):
        super(Classifier, self).__init__()
        self.head_num = head_num
        self.semantic_attention_layers = nn.ModuleList()
        # multi-head attention
        for i in range(head_num):
            self.semantic_attention_layers.append(
                nn.Sequential(
                    nn.Linear(in_size, hidden_size),
                    nn.Tanh(),
                    nn.Linear(hidden_size, out_size, bias=False)
                )
            )


    def forward(self, user_relation_rep):
        output = None
        for i in range(self.head_num):
            relation_weight = self.semantic_attention_layers[i](user_relation_rep).mean(0)
            norm_relation_weight = torch.softmax(relation_weight, dim=0)
            norm_relation_weight = norm_relation_weight.expand((user_relation_rep.shape[0],) + norm_relation_weight.shape)
            if output is None:
                output = (norm_relation_weight * user_relation_rep).sum(1)
            else:
                output += (norm_relation_weight * user_relation_rep).sum(1)
        # relation_weight = self.semantic_attention_layers(user_relation_rep).mean(0)
        # norm_relation_weight = torch.softmax(relation_weight, dim=0)
        # expanded_norm_relation_weight = torch.stack([norm_relation_weight for _ in range(user_relation_rep.shape[0])])
        # # norm_relation_weight.expand((user_relation_rep.shape[0],) + norm_relation_weight.shape) # this leads to error
        # output = (expanded_norm_relation_weight * user_relation_rep).sum(1)

        return output / self.head_num


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(
            torch.Tensor(
                [(math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1) for k in range(num_nodes)]
            ), dim=0,)

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


# 输入：network_data = training_data_by_type{edge_type1:[(node1,node2),(node3,node4),...],...}
#      feature_dic {node_i:[node1_feature],node_j:[node2_feature],...}
# 输出：sub_feature_dic：{node_i:[node1_feature],node_j:[node2_feature],...} 训练数据集用到的节点特征字典
def get_data(network_data, feature_dic):
    # vocab {node_i:class(节点出现的次数，节点index),...}
    # index2word [node_i,...] 按节点出现次数降序排列，列表索引与vocab中index一致
    # train_pairs [  (index1,index6,layer_id), (index2,index7,layer_id)... ] layer_id是边类型编号
    vocab, index2word, train_pairs = generate(network_data, args.num_walks, args.walk_length, args.schema, file_name,
                                              args.window_size, args.num_workers, args.walk_file)
    num_nodes = len(index2word)         # 各类节点个数

    # neighbors[ node_index1[ edge_type1[node_index, ],... ],  node_index2[ edge_type2[] ],...  ]
    # 每个节点在每种关系上的列表 增减为 固定长度neighbor_samples，NOTE 列表按node_index索引
    neighbors = generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples)

    sub_feature_dic = {}
    neighbors_features = []
    for node in index2word:
        sub_feature_dic[node] = feature_dic[node]
        neighbors_features.append(feature_dic[node])
    neighbors_features = torch.tensor(neighbors_features, dtype=torch.float).to(device)

    return index2word, train_pairs, neighbors, num_nodes, sub_feature_dic, neighbors_features


# 输入：training_data_by_type = training_data_by_type{edge_type1:[(node1,node2),(node3,node4),...],...}
#      feature_dic {node_i:[node1_feature],node_j:[node2_feature],...}
def GAMHN_model(training_data_by_type, feature_dic):

    index2word, train_pairs, neighbors, num_nodes, sub_feature_dic, neighbors_features = get_data(training_data_by_type, feature_dic)

    # import pickle
    # with open('temp_data/one_batch.pickle', 'rb') as f:
    #     index2word, train_pairs, neighbors, num_nodes, sub_feature_dic, neighbors_features = pickle.load(f)

    nsloss = NSLoss(num_nodes, num_sampled, embedding_size)
    nsloss.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), 'lr': 1e-4},
         {"params": nsloss.parameters(), 'lr': 1e-4},
         {"params": classifier.parameters(), 'lr': 1e-3}],
    )

    best_score = 0
    test_score = (0.0, 0.0, 0.0)
    best_parament_dict = {}
    best_acc = 0
    patience = 0
    for epoch in range(epochs):
        model.train()
        classifier.train()
        random.shuffle(train_pairs)
        # batch：源节点列表，目标节点列表，关系列表，每个源节点的所有邻居列表
        batches = get_batches(train_pairs, neighbors, batch_size, index2word, sub_feature_dic)

        data_iter = tqdm(
            batches,
            desc="epoch %d" % epoch,
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        loss = 0            # 每一个batch的loss
        avg_loss = 0.0      # 记录平均loss
        stop_representation_learning = True
        for i, data in enumerate(data_iter):
            optimizer.zero_grad()

            # 输入：源节点index、关系类型、源节点所有邻居(neighbors中信息)、用户与推文顺序、用户特征、推文特征
            embs = model(data[0].to(device), data[2].to(device), data[3].to(device),
                         data[4].to(device), data[5].to(device), data[6].to(device), neighbors_features)

            # if not stop_representation_learning:
            loss = nsloss(data[0].to(device), embs, data[1].to(device))
            loss.backward()
            avg_loss += loss.item()

            # NOTE 获取本batch有标签且为用户的节点，剔除没有label以及不是用户的节点
            labeled_user = list(user_label.keys())
            temp_labeled_user_index = []
            data_label = []                 # 数据的标签
            for j, ind in enumerate(data[0].tolist()):
                if index2word[ind] in labeled_user:
                    temp_labeled_user_index.append(j)
                    data_label.append(user_label[index2word[ind]])
            if len(data_label) == 0:    # 本batch中无标注的用户
                continue

            data_label = torch.tensor(data_label, dtype=torch.float).to(device)
            user_embs = embs[temp_labeled_user_index]

            predict = classifier(user_embs.detach())
            class_loss = classifier_loss(predict, data_label)
            class_loss.backward()
            optimizer.step()

            if i % 5000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))

        # 早停策略
        if not stop_representation_learning:
            final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
            for i in range(num_nodes):
                train_inputs = torch.tensor([i for _ in range(edge_type_count)]).to(device)
                train_types = torch.tensor(list(range(edge_type_count))).to(device)
                node_neigh = torch.tensor(
                    [neighbors[i] for _ in range(edge_type_count)]
                ).to(device)
                node_emb = model(train_inputs, train_types, node_neigh)
                for j in range(edge_type_count):
                    final_model[edge_types[j]][index2word[i]] = (
                        node_emb[j].cpu().detach().numpy()
                    )

            valid_aucs, valid_f1s, valid_prs = [], [], []
            test_aucs, test_f1s, test_prs = [], [], []
            for i in range(edge_type_count):
                if args.eval_type == "all" or edge_types[i] in args.eval_type.split(","):
                    tmp_auc, tmp_f1, tmp_pr = evaluate(
                        final_model[edge_types[i]],
                        valid_true_data_by_edge[edge_types[i]],
                        valid_false_data_by_edge[edge_types[i]],
                    )
                    valid_aucs.append(tmp_auc)
                    valid_f1s.append(tmp_f1)
                    valid_prs.append(tmp_pr)

                    tmp_auc, tmp_f1, tmp_pr = evaluate(
                        final_model[edge_types[i]],
                        testing_true_data_by_edge[edge_types[i]],
                        testing_false_data_by_edge[edge_types[i]],
                    )
                    test_aucs.append(tmp_auc)
                    test_f1s.append(tmp_f1)
                    test_prs.append(tmp_pr)
            print("representation learning:")
            print("\tvalid auc:", np.mean(valid_aucs))
            print("\tvalid pr:", np.mean(valid_prs))
            print("\tvalid f1:", np.mean(valid_f1s))

            average_auc = np.mean(test_aucs)
            average_f1 = np.mean(test_f1s)
            average_pr = np.mean(test_prs)

            cur_score = np.mean(valid_aucs)
            if cur_score > best_score:
                best_score = cur_score
                test_score = (average_auc, average_f1, average_pr)
                patience = 0
            else:
                patience += 1
                if patience > args.patience:
                    print("Representation Learning Early Stopping")
                    stop_representation_learning = True

        torch.save(model.state_dict(), 'temp_data/model.pt')
        torch.save(classifier.state_dict(), 'temp_data/classifier.pt')
        # classification valid
        acc, loss = classification_valid(model, classifier, valid_true_data_by_edge, feature_dic, epoch)
        if acc > best_acc:
            best_acc = acc
            best_parament_dict = {
                "model": model.state_dict(),
                "classifier": classifier.state_dict()
            }
            torch.save(model.state_dict(), 'temp_data/model.pt')
            torch.save(classifier.state_dict(), 'temp_data/classifier.pt')

    print("Overall ROC-AUC: ", test_score[0])
    print("Overall PR-AUC: ", test_score[1])
    print("Overall F1: ", test_score[2])
    print("Overall Classification Acc: ", best_acc)

    return best_parament_dict


@torch.no_grad()
def classification_valid(model, classifier, valid_true_data_by_edge, feature_dic, epoch):
    model.eval()
    classifier.eval()
    valid_index2word, valid_pairs, valid_neighbors, valid_num_nodes, valid_features_dic, valid_neighbors_features \
        = get_data(valid_true_data_by_edge, feature_dic)
    # valid_neighbors_features = torch.FloatTensor(valid_neighbors_features).to(device)

    random.shuffle(valid_pairs)
    valid_batches = get_batches(valid_pairs, valid_neighbors, batch_size, valid_index2word, valid_features_dic)  # 源节点列表，目标节点列表，关系列表，每个源节点的所有邻居列表

    valid_data_iter = tqdm(
        valid_batches,
        desc="valid epoch %d" % epoch,
        total=(len(valid_pairs) + (batch_size - 1)) // batch_size,
        bar_format="{l_bar}{r_bar}",
    )
    class_avg_loss = 0.0
    all_label = []
    all_predict = []
    for i, valid_data in enumerate(valid_data_iter):
        valid_embs = model(valid_data[0].to(device), valid_data[2].to(device), valid_data[3].to(device),
                           valid_data[4].to(device), valid_data[5].to(device), valid_data[6].to(device),
                           valid_neighbors_features)  # 源节点、关系、源节点所有邻居(neighbors中信息)

        # 获取本batch用户标签
        data_label = [user_label[valid_index2word[ind]] for ind in valid_data[0] if not valid_index2word[ind].find('u') == -1]
        data_label = torch.tensor(data_label, dtype=torch.float)

        predict = classifier(valid_embs)
        class_loss = classifier_loss(predict, data_label)

        all_label += data_label.data
        all_predict += predict.data

        if i % 5000 == 0:
            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": class_avg_loss / (i + 1),
                "loss": class_loss.item(),
            }
            valid_data_iter.write(str(post_fix))

    return accuracy_score(all_label, all_predict), class_avg_loss





if __name__ == "__main__":
    args = parse_args()
    file_name = args.input      # NOTE change the file path in utils.py
    print(args)

    # training_data_by_type = load_training_data(file_name + "/train.txt")
    read = False
    if read:
        valid_true_data_by_edge, valid_false_data_by_edge = load_testing_data(
            file_name + "/valid.txt"
        )
        testing_true_data_by_edge, testing_false_data_by_edge = load_testing_data(
            file_name + "/test.txt"
        )

    print("\nstart loading data...")
    with open(file_name + 'edge.json', 'r') as fp:
        training_data_by_type = json.load(fp)
    with open(file_name + 'node_attribute.json', 'r') as fp:
        feature_dic = json.load(fp)      # 所有node的特征属性，不等长
    with open(file_name + 'user_label.json', 'r') as fp:
        user_label = json.load(fp)

    print("\ninitialize model")
    edge_types = list(training_data_by_type.keys())
    user_feature_dim = torch.tensor(15, dtype=torch.long)       # FIXME
    text_feature_dim = torch.tensor(64, dtype=torch.long)
    edge_type_count = len(edge_types)   # 边类型数量
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    num_sampled = args.negative_samples
    dim_a = args.att_dim
    neighbor_samples = args.neighbor_samples
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = GATNEModel(embedding_size, embedding_u_size, edge_type_count, dim_a, user_feature_dim, text_feature_dim)
    classifier = Classifier(in_size=embedding_size, head_num=1)
    classifier_loss = nn.CrossEntropyLoss()

    model.to(device)
    classifier.to(device)

    print("\nstart training...")
    best_parament_dict = GAMHN_model(training_data_by_type, feature_dic)

    print("\nstart testing...")
    test_model = GATNEModel(embedding_size, embedding_u_size, edge_type_count, dim_a, user_feature_dim, text_feature_dim)
    test_classifier = Classifier(in_size=embedding_size, head_num=1)
    test_model.load_state_dict(best_parament_dict["model"])
    test_classifier.load_state_dict(best_parament_dict["classifier"])

    with open(file_name + 'test_edge.json', 'r') as fp:
        testing_true_data_by_edge = json.load(fp)
    final_acc = classification_valid(model, classifier, testing_true_data_by_edge, feature_dic, epoch=64)
    print("final accuracy: {}".format(final_acc))

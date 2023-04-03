from argparse import ArgumentParser
from dataset import SATARDataset
import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import SATAR, FollowersClassifier
from tqdm import tqdm
from utils import null_metrics, calc_metrics, is_better


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='Twibot-20')  # FIXME add [default='Twibot-20']
parser.add_argument('--max_epoch', type=int, default=8)
parser.add_argument('--n_hidden', type=int, default=128)
parser.add_argument('--n_batch', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--max_tweet_count', type=int, default=128)
parser.add_argument('--max_tweet_length', type=int, default=64)
parser.add_argument('--max_words', type=int, default=1024)
parser.add_argument('--gpu_user', type=list, default=[0, 1])
args = parser.parse_args()
dataset_size = {
    'cresci-2015': 5301,
    'Twibot-20': 229580,
    'Twibot-22': 1000000
}

dataset_name = args.dataset
assert dataset_name in ['Twibot-22', 'Twibot-20', 'cresci-2015']
# FIXME change the precessed_data path
path = './preprocess/tmp/{}'.format(dataset_name)

best_val_metrics = null_metrics()
best_state_dict = None

max_epoch = args.max_epoch
n_hidden = args.n_hidden
n_batch = args.n_batch
lr = args.lr
weight_decay = args.weight_decay
dropout = args.dropout
max_words = args.max_words
max_tweet_count = args.max_tweet_count
max_tweet_length = args.max_tweet_length


begin_time = time.time()
print('data loading...')        # SirryChen add this
data = {
    'tweets': np.load('{}/tweets.npy'.format(path), allow_pickle=True),     # 每一行对应一个用户推文中每一个词对应的编号（按顺序）
    'properties': np.load('{}/properties.npy'.format(path)),                # 包含用户特征属性的矩阵，每一列对应一类属性，每一行对应一个用户
    'neighbor_reps': np.zeros((dataset_size[dataset_name], n_hidden * 2)),
    'bot_labels': np.load('{}/bot_labels.npy'.format(path)),                # ndarray：[bot->1;human->0;other->2]
    'follower_labels': np.load('{}/follower_labels.npy'.format(path))       # ndarray[followers_count>=threshold]，1、0数组
}
print('data prepared')

word_vec = np.load('{}/vec.npy'.format(path))   # 每个词的词向量
word_vec = torch.tensor(word_vec)
words_size = len(word_vec)
blank_vec = torch.zeros((1, word_vec.shape[-1]))
word_vec = torch.cat((word_vec, blank_vec), dim=0)  # 拼接起来
num_embeddings = word_vec.shape[0]
embedding_dim = word_vec.shape[-1]
embedding_layer = nn.Embedding(num_embeddings, embedding_dim)
embedding_layer.weight.data = word_vec
embedding_layer.weight.requires_grad = False
embedding_layer.to(device)
print('loading done in {}s'.format(time.time() - begin_time))


def forward_one_batch(batch):
    return classifier(model(batch))


def forward_one_epoch(epoch):
    model.train()
    classifier.train()
    pbar = tqdm(train_loader, ncols=0)
    pbar.set_description('train {} epoch'.format(epoch))
    all_label = []
    all_pred = []
    ave_loss = 0
    cnt = 0
    for batch in pbar:
        optimizer.zero_grad()
        batch_size = batch['follower_labels'].shape[0]
        out = forward_one_batch({
            'words': embedding_layer(batch['words'].to(device)),
            'tweets': embedding_layer(batch['tweets'].to(device)),
            'neighbor_reps': batch['neighbor_reps'].to(device),
            'properties': batch['properties'].to(device)
        })
        labels = batch['follower_labels'].to(device)
        loss = loss_fn(out, labels)
        ave_loss += loss.item() * batch_size
        cnt += batch_size
        loss.backward()
        optimizer.step()
        all_label += labels.data
        all_pred += out
        pbar.set_postfix(loss='{:.5f}'.format(loss.cpu().detach().numpy()))
    ave_loss /= cnt
    all_label = torch.stack(all_label)          # 将相同维度的数据在新维度下堆叠（如2维数组堆叠为三维数组）
    all_pred = torch.stack(all_pred)
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} train loss: {:.6}'.format(epoch, ave_loss) + plog
    print(plog)
    val_metrics = validation(epoch, 'validation', val_loader)
    global best_val_metrics
    global best_state_dict
    if is_better(val_metrics, best_val_metrics):    # 通过验证集判断指标是否有提升->保存当前产生最优结果的参数
        best_val_metrics = val_metrics
        best_state_dict = model.state_dict()    # 获取模型中所有参数，包括可学习参数（weight,bias...）、不可学习参数
        torch.save(best_state_dict, 'tmp/{}/pretrain_weight.pt'.format(dataset_name))


@torch.no_grad()
def validation(epoch, name, loader):        # 利用验证集
    model.eval()        # 切换至评估模式，batchNorm、dropout层等用于优化训练的网络层被关闭，评估时不发生偏移
    classifier.eval()
    all_label = []
    all_pred = []
    ave_loss = 0
    cnt = 0
    for batch in loader:
        batch_size = batch['follower_labels'].shape[0]
        out = forward_one_batch({
            'words': embedding_layer(batch['words'].to(device)),
            'tweets': embedding_layer(batch['tweets'].to(device)),
            'neighbor_reps': batch['neighbor_reps'].to(device),
            'properties': batch['properties'].to(device)
        })
        labels = batch['follower_labels'].to(device)
        loss = loss_fn(out, labels)
        ave_loss += loss.item() * batch_size
        cnt += batch_size
        all_label += labels.data
        all_pred += out
    ave_loss /= cnt
    all_label = torch.stack(all_label)      # 将相同维度的数据在新维度下堆叠（如2维数组堆叠为三维数组）
    all_pred = torch.stack(all_pred)
    metrics, plog = calc_metrics(all_label, all_pred)
    plog = 'Epoch-{} {} loss: {:.6}'.format(epoch, name, ave_loss) + plog
    print(plog)
    return metrics


if __name__ == '__main__':
    train_set = SATARDataset(dataset_name,
                             split=['train'] if dataset_name != 'Twibot-20' else ['train', 'support'],
                             data=data,     # 包括推文信息、特征属性、关系网
                             padding_value=num_embeddings - 1,
                             max_words=max_words,
                             max_tweet_count=max_tweet_count,
                             max_tweet_length=max_tweet_length
                             )
    val_set = SATARDataset(dataset_name,
                           split=['val'],
                           data=data,
                           padding_value=num_embeddings - 1,
                           max_words=max_words,
                           max_tweet_count=max_tweet_count,
                           max_tweet_length=max_tweet_length
                           )
    train_loader = DataLoader(train_set, batch_size=n_batch, shuffle=True)      # batch_size:批大小，决定一个epoch有多少iteration，一次性读取多少数据
    val_loader = DataLoader(val_set, batch_size=n_batch, shuffle=False)         # shuffle：每个epoch读完数据后，在下一次读取时是否将数据顺序打乱

    model = SATAR(hidden_dim=n_hidden, embedding_dim=embedding_dim, dropout=dropout).to(device)

    classifier = FollowersClassifier(in_dim=n_hidden, out_dim=2).to(device)
    optimizer = torch.optim.Adam(set(model.parameters()) |      # 模型参数优化，为不同参数计算不同的自适应学习率
                                 set(classifier.parameters()),
                                 lr=lr,     # learning_rate 学习率,更新梯度时使用，控制权重更新速率
                                 weight_decay=weight_decay)     # 权重衰减，最后更新参数时，在损失函数中加一个惩罚参数
    loss_fn = nn.CrossEntropyLoss()     # 交叉熵损失函数

    # FIXME add gpu distribute
    """
    if torch.cuda.device_count() > 1:
        print('Lets use', torch.cuda.device_count(), "GPUs!")
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu_user])
        classifier = nn.parallel.DistributedDataParallel(classifier, device_ids=[args.gpu_user])
    """

    for i in range(max_epoch):
        forward_one_epoch(i)
    print('the best val acc is {}'.format(best_val_metrics['acc']))
    torch.save(best_state_dict, 'tmp/{}/pretrain_weight.pt'.format(dataset_name))

from argparse import ArgumentParser
import os
import os.path as osp
import pandas
from tqdm import tqdm
import ijson
import json
from datetime import datetime
import numpy as np
from gensim.models import word2vec
from nltk import tokenize

parser = ArgumentParser()
parser.add_argument('--dataset', type=str, default='Twibot-22')
args = parser.parse_args()

dataset = 'Twibot-20'
if not osp.exists('tmp/{}'.format(dataset)):
    os.makedirs('tmp/{}'.format(dataset))

properties_segments = ['created_at', 'description', 'entities', 'location',
                       'pinned_tweet_id', 'profile_image_url', 'protected',
                       'url', 'username', 'verified', 'withheld',
                       'public_metrics.followers_count', 'public_metrics.following_count',
                       'public_metrics.tweet_count', 'public_metrics.listed_count']


# 输入：账号创建时间
# 输出：活跃时间（从创建到2020/9/28的时间）
def calc_activate_days(created_at):
    created_at = created_at.strip()
    create_date = datetime.strptime(created_at, '%a %b %d %H:%M:%S %z %Y')
    crawl_date = datetime.strptime('2020 09 28 +0000', '%Y %m %d %z')
    delta_date = crawl_date - create_date
    return delta_date.days


# 输出：包含用户特征属性的矩阵，每一列对应一类属性，每一行对应一个用户
def get_properties():
    print('processing properties')
    with open(osp.join(path, 'node.json')) as f:
        users = ijson.items(f, 'item')      # 返回包含用户字典的集合
        properties = []
        idx = []
        for user in tqdm(users, ncols=0):
            if user['id'].find('u') == -1:     # TODO 用户名未包含u则跳过？为什么
                continue
            idx.append(user['id'])
            user_property = []
            for item in properties_segments:
                prop = user
                for key in item.split('.'):
                    prop = prop[key]
                    if prop is None:
                        continue
                if isinstance(prop, str):
                    prop = prop.strip()     # 清空字符串头部和尾部的空格
                if prop is None:            # 属性为空则输入0
                    user_property.append(0)
                elif item in ['public_metrics.followers_count', 'public_metrics.following_count',
                              'public_metrics.tweet_count', 'public_metrics.listed_count']:
                    user_property.append(int(prop))
                elif item in ['withheld', 'url', 'profile_image_url',       # 有这些属性值就输入1，离散？
                              'pinned_tweet_id', 'entities', 'location']:
                    user_property.append(1)
                elif item in ['verified', 'protected']:
                    user_property.append(int(prop == 'True'))
                elif item in ['description', 'username']:                   # TODO 用户描述、用户名的长度也算属性？
                    user_property.append(len(prop.strip()))
                elif item in ['created_at']:
                    user_property.append(calc_activate_days(prop.strip()))
            assert len(user_property) == 15                                 # 共计15个属性值
            properties.append(user_property)
        json.dump(idx, open('tmp/{}/idx.json'.format(dataset), 'w'))        # 对用户编号保存：python->json
        properties = np.array(properties)
        for i in range(properties.shape[1]):        # 每一列对应一类属性值
            if np.max(properties[:, i]) == np.min(properties[:, i]):    # 最大值等于最小值，该属性值没有区分度，删去
                continue
            mean = np.mean(properties[:, i])    # 求平均值
            std = np.std(properties[:, i])      # 求标准差
            properties[:, i] = (properties[:, i] - mean) / std  # z-score归一化
        print(properties.shape)
        np.save('tmp/{}/properties.npy'.format(dataset), properties)


# 输出：每一行对应一个用户的{follow:[用户id],friend:[用户id]}
def get_neighbors():
    edge = pandas.read_csv(osp.join(path, 'edge.csv'), chunksize=10000000)      # chunksize:每次读取的行数
    user_idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
    neighbors_index = {}    # id:{'follow':[跟随者id列表]，‘friend’:[朋友id列表]}
    for item in user_idx:
        neighbors_index[item] = {
            'follow': [],
            'friend': []
        }
    print(len(neighbors_index))
    for chunk in edge:
        for index, item in tqdm(chunk.iterrows(), ncols=0):
            source, relation, target = item['source_id'], item['relation'], item['target_id']
            if source.find('u') == 0 and target.find('u') == 0:
                if source not in user_idx or target not in user_idx:
                    continue
                neighbors_index[source][relation].append(target)
    neighbors = [neighbors_index[item] for item in user_idx]    # 去掉id信息，保留每个用户的关系字典
    print(len(neighbors))
    neighbors = np.array(neighbors, dtype=object)
    np.save('tmp/{}/neighbors.npy'.format(dataset), neighbors)


# 输出：每行对应一个用户的推文信息，得到推文语料库
def get_tweet_corpus():
    fb = open('tmp/{}/corpus.txt'.format(dataset), 'w', encoding='utf-8')
    with open(osp.join(path, 'node.json')) as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('t') == -1:      # id未包含t则跳过
                continue
            if item['text'] is None:
                continue
            fb.write(item['text'] + '\n')
    fb.close()


# 输入：数据集中推文构成的语料库
# 输出：vec.npy:每个词的词向量; key_to_index.json:每个词的编号
def get_word2vec_model():
    sentences = word2vec.Text8Corpus('tmp/{}/corpus.txt'.format(dataset))  # Text8语料库
    print('training word2vec model')
    # vector_size：输出词向量维度（隐藏层单元数）;workers：控制训练的并行数;min_count：忽视频率小于5的词
    model = word2vec.Word2Vec(sentences, vector_size=128, workers=8, min_count=5)
    vectors = model.wv.vectors                 # 词向量，保存在ndarray中
    key_to_index = model.wv.key_to_index       # 每个词的编号：1，2，3，4...
    print(vectors.shape)
    print(len(key_to_index))
    np.save('tmp/{}/vec.npy'.format(dataset), vectors)
    json.dump(key_to_index, open('tmp/{}/key_to_index.json'.format(dataset), 'w'))
    print('training done')


# 输入：所有词的编号，每个用户的推文
# 输出：每一行对应一个用户推文中每一个词对应的编号（按顺序）->tweets.npy
def get_tweets():
    edge = pandas.read_csv(osp.join(path, 'edge.csv'))
    author_idx = {}
    for index, item in tqdm(edge.iterrows(), ncols=0):
        if item['relation'] != 'post':
            continue
        author_idx[item['target_id']] = item['source_id']
    print(len(edge))
    key_to_index = json.load(open('tmp/{}/key_to_index.json'.format(dataset)))
    user_idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
    tweets_index = {}
    for user in user_idx:
        tweets_index[user] = []
    with open(osp.join(path, 'node.json')) as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('t') == -1:
                continue
            if item['text'] is None:
                continue
            words = tokenize.word_tokenize(item['text'])       # 实现分词，返回列表
            tweet = []
            for word in words:
                if word in key_to_index:
                    tweet.append(key_to_index[word])           # 将推文中的词转换为词的编号
                else:
                    tweet.append(len(key_to_index))            # 如果没有这个词，就保存为编号最大值+1
            tweets_index[author_idx[item['id']]].append(tweet)      # 每个用户的推文所包含的词的编号
    tweets = [tweets_index[item] for item in user_idx]          # 每一行对应一个用户的推文所包含的词的编号
    print(tweets[0])
    tweets = np.array(tweets, dtype=object)
    np.save('tmp/{}/tweets.npy'.format(dataset), tweets)


# 输入：用户id; label
# 输出：ndarray：[bot->1;human->0;other->2]
def get_bot_labels():
    user_idx = json.load(open('tmp/{}/idx.json'.format(dataset)))
    label_data = pandas.read_csv('{}/label.csv'.format(path))
    label_index = {}
    for index, item in tqdm(label_data.iterrows(), ncols=0):
        label_index[item['id']] = int(item['label'] == 'bot')       # 字典{用户id：bot->1;human->0}
    bot_labels = []
    for item in user_idx:
        if item in label_index:
            bot_labels.append(label_index[item])                    # 按user_ida排序的列表[bot->1;human->0;other->2]
        else:
            bot_labels.append(2)
    bot_labels = np.array(bot_labels)
    print(bot_labels.shape)
    np.save('tmp/{}/bot_labels.npy'.format(dataset), bot_labels)


# 输入：每个用户的粉丝数
# 输出：ndarray[followers_count>=threshold]
def get_follower_labels():
    follower_counts = []
    with open(osp.join(path, 'node.json')) as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('u') == -1:
                continue
            try:
                if item['public_metrics']['followers_count'] is not None:
                    follower_counts.append(item['public_metrics']['followers_count'])
                else:
                    follower_counts.append(0)
            except TypeError:
                follower_counts.append(0)
    follower_counts = np.array(follower_counts)
    print(follower_counts.shape)
    threshold = np.percentile(follower_counts, 80)      # 阈值：第80百分位数（一组n个观测值按数值大小排列,处于p%位置的值称第p百分位数）
    print(threshold)
    follower_labels = []
    with open(osp.join(path, 'node.json')) as f:
        data = ijson.items(f, 'item')
        for item in tqdm(data, ncols=0):
            if item['id'].find('u') == -1:
                continue
            try:
                label = int(item['public_metrics']['followers_count'] >= threshold)
            except TypeError:
                label = 0
            follower_labels.append(label)       # 粉丝数大于阈值->1
    follower_labels = np.array(follower_labels)
    print(follower_labels.shape)
    np.save('tmp/{}/follower_labels.npy'.format(dataset), follower_labels)


if __name__ == '__main__':
    # FIXME change path of datasets
    # path = '../../datasets/{}'.format(dataset)
    path = '/data1/botdet/datasets/{}'.format(dataset)

    get_properties()
    get_neighbors()
    get_tweet_corpus()
    get_word2vec_model()
    get_tweets()
    get_bot_labels()
    get_follower_labels()

import json
import pandas as pd
import torch
from torch.utils.data import Dataset
import random
import numpy as np


class SATARDataset(Dataset):
    def __init__(self, dataset_name, split, data, padding_value,
                 max_tweet_count=128, max_tweet_length=64, max_words=1024,
                 random_seed=20220401):
        random.seed(random_seed)
        assert dataset_name in ['Twibot-20', 'cresci-2015', 'Twibot-22']
        assert type(split) == list
        # FIXME change the path
        path = './preprocess/tmp/{}'.format(dataset_name)
        idx = json.load(open('{}/idx.json'.format(path)))
        idx = {item: index for index, item in enumerate(idx)}  # idx = {id1:1, id2:2...}
        # FIXME change the path
        split_data = pd.read_csv('/data1/botdet/datasets/Twibot-20/split.csv')
        self.idx = []
        for index, item in split_data.iterrows():
            if item['split'] in split:
                self.idx.append(idx[item['id']])  # 添加符合分类的用户id

        # FIXME 通过减少id数，减小数据集大小
        # self.smaller_dataset()

        self.data = data
        self.max_tweet_count = max_tweet_count
        self.max_tweet_length = max_tweet_length
        self.max_words = max_words
        self.padding_value = padding_value

    # 通过用户id提取每一个用户信息
    def __getitem__(self, index):
        index = self.idx[index]
        tweets = self.data['tweets'][index]         # 二维数组（i，j），第i篇推文第j个词的编号
        tweets = tweets[:self.max_tweet_count]
        tweets_cache = []
        words = []
        for tweet in tweets:
            words += tweet
            cache = tweet[:self.max_tweet_length]
            for _ in range(len(tweet), self.max_tweet_length):          # 每篇推文长度不足的地方用padding_value填充
                cache.append(self.padding_value)
            tweets_cache.append(cache)
        for _ in range(len(tweets), self.max_tweet_count):
            tweets_cache.append([self.padding_value] * self.max_tweet_length)
        tweets = torch.tensor(tweets_cache, dtype=torch.long)       # 每个用户的推文数据，二维，max_tweet_count * max_tweet_length
        words_cache = words[:self.max_words]
        for _ in range(len(words), self.max_words):
            words_cache.append(self.padding_value)
        words = torch.tensor(words_cache, dtype=torch.long)         # 每个用户推文中包含的所有词，一维，1 * max_words
        properties = torch.tensor(self.data['properties'][index], dtype=torch.float)
        neighbor_reps = torch.tensor(self.data['neighbor_reps'][index], dtype=torch.float)
        bot_labels = torch.tensor(self.data['bot_labels'][index], dtype=torch.long)
        follower_labels = torch.tensor(self.data['follower_labels'][index], dtype=torch.long)
        return {
            'words': words,
            'tweets': tweets,
            'properties': properties,
            'neighbor_reps': neighbor_reps,
            'bot_labels': bot_labels,
            'follower_labels': follower_labels,
        }

    def __len__(self):
        return len(self.idx)


import numpy as np
from argparse import ArgumentParser
import json
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--dataset', type=str,default='Twibot-20')
args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['Twibot-22', 'Twibot-20', 'cresci-2015']


n_hidden = 128

if __name__ == '__main__':
    followers_count_off = True

    path = './preprocess/tmp/{}'.format(dataset_name)

    neighbors = np.load('{}/neighbors.npy'.format(path), allow_pickle=True)   # 每一行对应一个用户的{follow:[用户id],friend:[用户id]}
    idx = json.load(open('{}/idx.json'.format(path)))
    idx = {item: index for index, item in enumerate(idx)}       # {id1:编号1,id2:编号2,...}

    if followers_count_off == True:             # # FIXME add this to ignore followers_count
        reps = np.load('{}/new_dataset/reps.npy'.format(path))
    else:
        reps = np.load('{}/reps.npy'.format(path))

    neighbor_reps = []
    for user in tqdm(neighbors, ncols=0):
        neighbor_rep = []
        for key in user:    # 对该用户的follow、friend
            neighbor = user[key]
            neighbor = [idx[item] for item in neighbor]         # 一次性获取一个用户一类neigbors的所有编号
            if len(neighbor) == 0:
                tmp = np.zeros(n_hidden)
            else:
                tmp = reps[neighbor]
                tmp = np.mean(tmp, axis=0)
            neighbor_rep.append(tmp)                            # 将与该用户有关系的人的信息拼接起来
        neighbor_rep = np.concatenate(neighbor_rep, axis=0)
        neighbor_reps.append(neighbor_rep)
    neighbor_reps = np.stack(neighbor_reps)
    print(neighbor_reps.shape)
    if followers_count_off == True:                 # # FIXME add this to ignore followers_count
        np.save('./preprocess/tmp/{}/new_dataset/neighbor_reps.npy'.format(dataset_name), neighbor_reps)
    else:
        np.save('./preprocess/tmp/{}/neighbor_reps.npy'.format(dataset_name), neighbor_reps)



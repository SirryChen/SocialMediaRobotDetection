import numpy as np
from argparse import ArgumentParser
import json
from tqdm import tqdm

parser = ArgumentParser()
parser.add_argument('--dataset', type=str)
args = parser.parse_args()

dataset_name = args.dataset
assert dataset_name in ['Twibot-22', 'Twibot-20', 'cresci-2015']


n_hidden = 128

if __name__ == '__main__':
    # FIXME change the precessed_data path
    path = 'D:/SocialMediaRobots/TwiBot-22-master/src/SATAR/preprocess/tmp/{}'.format(dataset_name)

    neighbors = np.load('{}/neighbors.npy'.format(path), allow_pickle=True)
    idx = json.load(open('{}/idx.json'.format(path)))
    idx = {item: index for index, item in enumerate(idx)}
    reps = np.load('{}/reps.npy'.format(path))
    neighbor_reps = []
    for user in tqdm(neighbors, ncols=0):
        neighbor_rep = []
        for key in user:
            neighbor = user[key]
            neighbor = [idx[item] for item in neighbor]
            if len(neighbor) == 0:
                tmp = np.zeros(n_hidden)
            else:
                tmp = reps[neighbor]
                tmp = np.mean(tmp, axis=0)
            neighbor_rep.append(tmp)
        neighbor_rep = np.concatenate(neighbor_rep, axis=0)
        neighbor_reps.append(neighbor_rep)
    neighbor_reps = np.stack(neighbor_reps)
    print(neighbor_reps.shape)
    np.save('tmp/{}/neighbor_reps.npy'.format(dataset_name), neighbor_reps)



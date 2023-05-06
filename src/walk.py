import random
import multiprocessing

from tqdm import tqdm


def walk(args):
    walk_length, start, schema = args
    # Simulate a random walk starting from start node.
    rand = random.Random()

    if schema:
        schema_items = schema.split('-')
        assert schema_items[0] == schema_items[-1]

    walk = [start]
    while len(walk) < walk_length:
        cur = walk[-1]
        candidates = []
        for node in G[cur]:
            if schema == '' or node_type[node] == schema_items[len(walk) % (len(schema_items) - 1)]:
                candidates.append(node)
        if candidates:
            walk.append(rand.choice(candidates))
        else:
            break
    return [str(node) for node in walk]


def initializer(init_G, init_node_type):
    global G
    G = init_G
    global node_type
    node_type = init_node_type


class RWGraph():
    # nx_G：{node1:[node2,node3...], node2:[node4,node10...]...} 与节点node_i存在特定边类型 的节点集合
    def __init__(self, nx_G, node_type_arr=None, num_workers=16):
        self.G = nx_G
        self.node_type = node_type_arr
        self.num_workers = num_workers

    def node_list(self, nodes, num_walks):
        for loop in range(num_walks):
            for node in nodes:
                yield node

    # 输出：当前进程池中的node_i组成的列表, 即nx_G字典中的键
    def simulate_walks(self, num_walks, walk_length, schema=None):
        all_walks = []
        nodes = list(self.G.keys())  # 所有节点node_i
        random.shuffle(nodes)  # 将顺序打乱

        if schema is None:
            with multiprocessing.Pool(self.num_workers, initializer=initializer,
                                      initargs=(self.G, self.node_type)) as pool:
                # 进程池中的imap会将 iterable 参数传入的可迭代对象分成 chunksize 份传递给不同的进程来处理
                all_walks = list(
                    pool.imap(walk, ((walk_length, node, '') for node in tqdm(self.node_list(nodes, num_walks))), chunksize=256))
        else:
            schema_list = schema.split(',')
            for schema_iter in schema_list:
                with multiprocessing.Pool(self.num_workers, initializer=initializer,
                                          initargs=(self.G, self.node_type)) as pool:
                    walks = list(pool.imap(walk, ((walk_length, node, schema_iter) for node in tqdm(self.node_list(nodes, num_walks)) if schema_iter.split('-')[0] == self.node_type[node]), chunksize=512))
                all_walks.extend(walks)

        return all_walks

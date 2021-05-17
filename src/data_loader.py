import numpy as np
import os


def load_data(args):
    n_user, n_item, train_data, eval_data, test_data = load_rating(args)
    n_entity, n_relation, n_genre, entity_adj_entity, entity_adj_relation, interest_adj_entity, interest_adj_relation = load_kg(args)
    print('data loaded.')

    return n_user, n_item, n_entity, n_relation, n_genre, train_data, eval_data, test_data, entity_adj_entity, entity_adj_relation, interest_adj_entity, interest_adj_relation


def load_rating(args):
    print('reading rating file ...')

    # reading rating file
    rating_file = '.../data/' + args.dataset + '/ratings_final'
    if os.path.exists(rating_file + '.npy'):
        rating_np = np.load(rating_file + '.npy')
    else:
        # loadtxt读取txt文件
        rating_np = np.loadtxt(rating_file + '.txt', dtype=np.int64)
        np.save(rating_file + '.npy', rating_np)

    n_user = len(set(rating_np[:, 0]))
    n_item = len(set(rating_np[:, 1]))
    train_data, eval_data, test_data = dataset_split(rating_np, args)

    return n_user, n_item, train_data, eval_data, test_data


# 数据库划分，分为测试、训练、评价数据库
def dataset_split(rating_np, args):
    print('splitting dataset ...')

    # train:eval:test = 6:2:2
    eval_ratio = 0.2
    test_ratio = 0.2
    # shape[0]就是读取矩阵第一维度的长度
    n_ratings = rating_np.shape[0]
    # eval_indices评价指标
    # choice(a, size=None, replace=True, p=None)抽样
    eval_indices = np.random.choice(list(range(n_ratings)), size=int(n_ratings * eval_ratio), replace=False)
    left = set(range(n_ratings)) - set(eval_indices)

    test_indices = np.random.choice(list(left), size=int(n_ratings * test_ratio), replace=False)
    # 训练指标
    train_indices = list(left - set(test_indices))
    if args.ratio < 1:
        train_indices = np.random.choice(list(train_indices), size=int(len(train_indices) * args.ratio), replace=False)

    train_data = rating_np[train_indices]
    eval_data = rating_np[eval_indices]
    test_data = rating_np[test_indices]

    return train_data, eval_data, test_data


def load_kg(args):
    print('reading KG file ...')

    # reading kg file
    kg_final = '.../data/' + args.dataset + '/kg_final'
    kg_final_1 = '.../data/' + args.dataset + '/kg_user_interest'
    kg_final_2 = '.../data/' + args.dataset + '/kg_genre_entity'

    kg_np = np.loadtxt(kg_final + '.txt', dtype=np.int64)
    kg_np_1 = np.loadtxt(kg_final_1 + '.txt', dtype=np.int64)
    kg_np_2 = np.loadtxt(kg_final_2 + '.txt', dtype=np.int64)

    n_entity = len(set(kg_np[:, 0]) | set(kg_np[:, 2]))
    n_relation = len(set(kg_np[:, 1]))
    print(n_relation)
    # n_genre = max(set(kg_np_1[:, 1]))
    n_genre = len(set(kg_np_2[:, 0]))
    print(n_genre)

    kg, kg_genre, kg_interest = construct_kg(kg_np, kg_np_1, kg_np_2, n_relation)

    entity_adj_entity, entity_adj_relation, interest_adj_entity, interest_adj_relation = construct_adj(args, kg, kg_genre, n_entity, n_genre)

    return n_entity, n_relation, n_genre, entity_adj_entity, entity_adj_relation, interest_adj_entity, interest_adj_relation


def construct_kg(kg_np, kg_np_1, kg_np_2, n_relation):
    print('constructing knowledge graph ...')
    kg = dict()
    kg_genre = dict()
    kg_interest = dict()

    for triple in kg_np:
        head = triple[0]
        relation = triple[1]
        tail = triple[2]
        # treat the KG as an undirected graph
        if head not in kg:
            kg[head] = []
        kg[head].append((tail, relation))
        if tail not in kg:
            kg[tail] = []
        kg[tail].append((head, relation))

    for triple in kg_np_1:
        user = triple[0]
        interest = triple[1]
        # treat the KG as an undirected graph
        if user not in kg_interest:
            kg_interest[user] = interest

    for triple in kg_np_2:
        genre = triple[0]
        entity = triple[1]
        relation = n_relation
        if genre not in kg_genre:
            kg_genre[genre] = []
        kg_genre[genre].append((entity, relation))
    return kg, kg_genre, kg_interest


def construct_adj(args, kg, kg_genre, entity_num, n_genre):
    print('constructing adjacency matrix ...')
    # each line of entity_adj_entity stores the sampled neighbor entities for a given entity
    # each line of entity_adj_relation stores the corresponding sampled neighbor relations
    # np.zeros返回来一个给定形状和类型的用0填充的数组
    entity_adj_entity = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    entity_adj_relation = np.zeros([entity_num, args.neighbor_sample_size], dtype=np.int64)
    interest_adj_entity = np.zeros([n_genre, args.neighbor_sample_size], dtype=np.int64)
    interest_adj_relation = np.zeros([n_genre, args.neighbor_sample_size], dtype=np.int64)
    # short interest entity 有偏差地选择
    entity_file0 = '.../data/movie/entity_one_hot'
    entity_present = np.loadtxt(entity_file0 + '.txt', dtype=np.int64, usecols=np.arange(0, 32))
    for entity in range(entity_num):
        neighbors = kg[entity]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            entity_neighbor = dict()
            for neighbor in range(n_neighbors):
                neighbor_score = np.dot(np.array(entity_present[neighbors[neighbor][0]]),
                                        entity_present[entity])
                entity_neighbor[neighbor] = neighbor_score
            entity_neighbor_score = sorted(entity_neighbor.items(), key=lambda x: x[1], reverse=True)

            sampled_indices = np.array([i[0] for i in entity_neighbor_score])[0:args.neighbor_sample_size]
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        entity_adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
        entity_adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

    # long interest entity 邻居实体
    # interest_file0 = 'G:/知识图谱和推荐系统/KGCN-master/KGCN-master/data/movie/user_interest'
    # interest_present = np.loadtxt(interest_file0 + '.txt', dtype=np.int64, usecols=np.arange(0, 32))
    for genre in range(n_genre):
        neighbors = kg_genre[genre]
        n_neighbors = len(neighbors)
        if n_neighbors >= args.neighbor_sample_size:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=False)
        else:
            sampled_indices = np.random.choice(list(range(n_neighbors)), size=args.neighbor_sample_size, replace=True)
        interest_adj_entity[genre] = np.array([neighbors[i][0] for i in sampled_indices])
        interest_adj_relation[genre] = np.array([neighbors[i][1] for i in sampled_indices])
    print(interest_adj_entity)
    print(interest_adj_relation)

    return entity_adj_entity, entity_adj_relation, interest_adj_entity, interest_adj_relation

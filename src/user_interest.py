import numpy as np
import tensorflow as tf
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def read_item_index_to_entity_id_file():
    file = 'G:/知识图谱和推荐系统/KGCN-master/KGCN-master/data/movie/item_index2entity_id.txt'
    i = 0
    for line in open(file, encoding='utf-8').readlines():
        item_index = line.strip().split('\t')[0]
        item_index_old2new[item_index] = i
        i += 1


def user_interest():
    entity_genres_to_one_hot, item_genres_to_one_hot = items_to_one_hot()
    user_to_interest_present = user_interest_present(item_genres_to_one_hot)
    return entity_genres_to_one_hot, user_to_interest_present


def items_to_one_hot():
    file = 'G:/知识图谱和推荐系统/KGCN-master/KGCN-master/data/movie/movies.csv'
    # rating_np = np.loadtxt(file, dtype=np.str, delimiter=",")
    genres_set = set()
    genres_cnt = 0
    genres_to_genres_index = dict()
    for line in open(file).readlines()[1:]:
        array = line.strip().split(',')
        for genres in array[2].split('|'):
            if genres not in genres_set:
                genres_present = np.array([0.0] * 32)
                genres_present[genres_cnt] = 1
                genres_to_genres_index[genres] = genres_present
                genres_set.add(genres)
                genres_cnt += 1
    print('movie genre number:', genres_cnt)

    print("movie every genre to one hot done")
    item_genres_to_one_hot = dict()
    entity_genres_to_one_hot = dict()
    entity_cnt = 0
    for line in open(file).readlines()[1:]:
        array = line.strip().split(',')
        item_genres_hot = np.array([0.0] * 32)
        if array[0] in item_index_old2new:
            for genres in array[2].split('|'):
                item_genres_hot += genres_to_genres_index[genres]
            entity_genres_to_one_hot[entity_cnt] = item_genres_hot
            item_genres_to_one_hot[int(array[0])] = item_genres_hot
            entity_cnt += 1
            print(array[0], item_genres_to_one_hot[int(array[0])])
    print('entity_cnt', entity_cnt)
    writer = open('G:/知识图谱和推荐系统/KGCN-master/KGCN-master/data/movie/entity_one_hot.txt', 'w',
                  encoding='utf-8')
    for entity in range(entity_cnt):
        for i in entity_genres_to_one_hot[entity]:
            writer.write('%d\t' % i)
        writer.write('\n')
    writer.close()
    writer = open('G:/知识图谱和推荐系统/KGCN-master/KGCN-master/data/movie/kg_genre_entity.txt', 'w',
                  encoding='utf-8')
    for entity in range(entity_cnt):
        genre_cnt = 0
        for i in entity_genres_to_one_hot[entity]:
            if i > 0:
                writer.write('%d\t%d\t\n' % (int(genre_cnt), int(entity)))
            genre_cnt += 1
    writer.close()
    print('movie genres to one hot done')
    # with tf.Session() as sess:
    # print(sess.run(item_genres_to_one_hot[1]))
    # print(sess.run(item_genres_to_one_hot[10]))
    # print(sess.run(res[1]))
    # print(np.arange(32))
    return entity_genres_to_one_hot, item_genres_to_one_hot


def user_interest_present(item_genres_to_one_hot):
    file = 'G:/知识图谱和推荐系统/KGCN-master/KGCN-master/data/movie/ratings.csv'
    print('loaded ratings.csv')
    user_to_time_set = dict()
    user_set = set()
    user_nct = 0
    for line in open(file).readlines()[1:]:
        array = line.strip().split(',')
        user = int(array[0])-1
        item = array[1]
        time = np.float64(array[3])
        if item not in item_index_old2new:  # the item is not in the final item set
            continue
        if user not in user_set:
            user_to_time_set[user] = set()
            user_set.add(user)
        user_to_time_set[user].add(time)
        user_nct += 1
    print(user_nct)
    print('user time_set done')

    user_to_interest_present = dict()
    for i in range(138159):
        user_to_interest_present[i] = [0.0] * 32
    user_nct = 0
    for line in open(file).readlines()[1:]:
        array = line.strip().split(',')
        user = int(array[0])-1
        item = array[1]
        time = np.float64(array[3])
        if item not in item_index_old2new:  # the item is not in the final item set
            continue
        if user == 138159:
            break
        time_scores = (time - min(user_to_time_set[user]) + 1) / (max(user_to_time_set[user]) -
                                                                  min(user_to_time_set[user]) + 1)
        print(time_scores)
        user_to_interest_present[user] = user_to_interest_present[user] + item_genres_to_one_hot[int(item)] * time_scores
        print('user:', user)
        # print('user_to_interest_present', user_to_interest_present[user])
        user_nct += 1
    print('user_to_interest_present', user_to_interest_present[0])
    writer = open('G:/知识图谱和推荐系统/KGCN-master/KGCN-master/data/movie/kg_user_interest.txt', 'w',
                  encoding='utf-8')
    for interest in user_to_interest_present.items():
        interest_cnt = 0
        interest_max = 0
        max_cnt = 0
        for genres in interest[1]:
            if genres > interest_max:
                interest_max = genres
                max_cnt = interest_cnt
            interest_cnt += 1
        writer.write('%d\t%d\t\n' % (interest[0], int(max_cnt)))

    # for interest in user_to_interest_present.items():
    #     for genres in interest[1]:
    #         writer.write('%d\t' % genres)
    #     writer.write('\n')
    writer.close()
    print(user_nct)
    return user_to_interest_present


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--neighbor_sample_size', type=int, default=4, help='the number of neighbors to be sampled')
    parser.add_argument('--n_iter', type=int, default=2, help='number of iterations when computing entity '
                                                              'representation')
    args = parser.parse_args()
    item_index_old2new = dict()
    read_item_index_to_entity_id_file()
    user_interest()

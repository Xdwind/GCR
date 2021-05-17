import tensorflow as tf
from KGCN.aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
# f1_score它是精确率和召回率的调和平均数,
from sklearn.metrics import f1_score, roc_auc_score


class KGCN(object):
    def __init__(self, args, n_user, n_entity, n_relation, n_genre, adj_entity, adj_relation, interest_adj_entity,
                 interest_adj_relation):
        self._parse_args(args, adj_entity, adj_relation, interest_adj_entity, interest_adj_relation)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation, n_genre)
        self._build_train()

    @staticmethod
    def get_initializer():
        # 该函数返回一个用于初始化权重的初始化程序 “Xavier”
        # 返回初始化权重矩阵
        return tf.contrib.layers.xavier_initializer()

    # parse解析，_parse_args函数解析数据，从控制台获取相关数据
    def _parse_args(self, args, adj_entity, adj_relation, interest_adj_entity, interest_adj_relation):
        # [entity_num, neighbor_sample_size]
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.interest_adj_entity = interest_adj_entity
        self.interest_adj_relation = interest_adj_relation

        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    # _build_inputs函数主要用于占位
    def _build_inputs(self):
        # placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，
        # 它只会分配必要的内存。等建立session，在会话中，
        # 运行模型的时候通过feed_dict()函数向占位符喂入数据。
        # dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
        # shape：数据形状。默认是None，就是一维值，也可以是多维（比如[2,3], [None, 3]表示列是3，行不定）
        # name：名称
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
        self.user_interest = tf.placeholder(dtype=tf.int64, shape=[None], name='user_interest')

    def _build_model(self, n_user, n_entity, n_relation, n_genre):
        # tf.get_variable() 则主要用于网络的权值设置，它可以实现权值共享
        # shape：新变量或现有变量的形状.
        # initializer：创建变量的初始化器.
        # name：新变量或现有变量的名称.
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user, self.dim], initializer=KGCN.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity, self.dim], initializer=KGCN.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation+1, self.dim], initializer=KGCN.get_initializer(), name='relation_emb_matrix')
        self.interest_emb_matrix = tf.get_variable(
            shape=[n_genre, self.dim], initializer=KGCN.get_initializer(), name='interest_emb_matrix')
        # [batch_size, dim]
        # tf.nn.embedding_lookup函数的用法主要是选取一个张量里面索引对应的元素。
        # tf.nn.embedding_lookup（tensor, id）:tensor就是输入张量，id就是张量对应的索引。

        self.user_embeddings = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_entity_neighbors(self.item_indices)
        print("entities, relations done")
        interest_entities, interest_relations = self.get_interest_neighbors(self.user_interest)
        print("interest_entities, interest_relations done")
        # [batch_size, dim]
        self.interest_embeddings, self.interest_aggregators = self.aggregate(interest_entities, interest_relations)
        # self.user_embeddings = self.interest_embeddings
        print("self.interest_embeddings, self.interest_aggregators done")
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations)
        print("self.item_embeddings, self.aggregators done")

        # [batch_size]
        self.scores = tf.reduce_sum((self.user_embeddings + self.interest_embeddings) * self.item_embeddings, axis=1)
        self.scores_normalized = tf.sigmoid(self.scores)

    def get_entity_neighbors(self, seeds):
        # TensorFlow中，想要维度增加一维，可以使用tf.expand_dims(input, dim, name=None)函数,dim表示增加维度的位置
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def get_interest_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            if i == 0:
                neighbor_entities = tf.reshape(tf.gather(self.interest_adj_entity, entities[i]), [self.batch_size, -1])
                neighbor_relations = tf.reshape(tf.gather(self.interest_adj_relation, entities[i]), [self.batch_size, -1])
                entities.append(neighbor_entities)
                relations.append(neighbor_relations)
            if i != 0:
                neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
                neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
                entities.append(neighbor_entities)
                relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations):
        aggregators = []  # store all aggregators
        entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
        relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.tanh)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    user_embeddings=self.user_embeddings)
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        # self.l2_loss = tf.nn.l2_loss(self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        for aggregator in self.interest_aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)
        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        # print("loading train")
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        return auc, f1

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)

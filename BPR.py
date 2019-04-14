#!/usr/bin/env python
# coding=UTF-8
'''
@Description: 
@Author: ZhaoSong
@LastEditors: ZhaoSong
@Date: 2019-04-10 13:38:23
@LastEditTime: 2019-04-14 16:50:56
'''
import pandas as pd 
import numpy as np
import tensorflow as tf
from collections import defaultdict
import random

def data_processor():
    header = ['user_id', 'item_id', 'rating', 'timestamp']
    df = pd.read_csv('E:ml/ml-100k/u.data', sep='\t', names = header) #()
    
    usr = df['user_id'].values #转化为inumpy数组
    user_count = np.max(usr)   #最大用户id，最多有max_u_id个用户
    itm = df['item_id'].values #电影编号
    item_count = np.max(itm)

    # 每个用户看过的电影都保存在user_ratings中
    user_ratings = defaultdict(set)
    f = open('E:ml/ml-100k/u.data','r')
    for line in f.readlines():
        u, i, _, _ = line.split("\t")
        u = int(u)
        i = int(i)  
        user_ratings[u].add(i)
   
    return user_count,item_count,user_ratings

# 随机找出用户评过分的电影,用于构造训练集和测试集
def get_test(user_count,user_ratings):
    user_test = dict()
    for u, i_list in user_ratings.items(): #dict的keys须是list
        user_test[u] = random.sample(user_ratings[u], 1)[0]
    
    del i_list
    return user_test

# 处理训练数据,获得训练用的三元组，随机用户、看过的电影、没看过的电影
def get_trainbatch(user_ratings, user_rating_test,item_count,batch_size = 512):
    t = []
    for b in range(batch_size):
        u = random.sample(user_ratings.keys(),1)[0]
        i = random.sample(user_ratings[u], 1)[0]
        while i == user_ratings_test[u]:
            i = random.sample(user_ratings[u], 1)[0]
        
        j = random.randint(1, item_count)
        while j in user_ratings[u]:
            j = random.randint(1, item_count)
        t.append([u, i, j])
    del b
    return np.asarray(t)
# 生成测试集，i是从已评分中随机抽取的集合，j是用户u没评分过的集合
def get_testbatch(user_ratings, user_rating_test, item_count):
    for u in user_ratings.keys():
        t = []
        i = user_rating_test[u]
        for j in range(1, item_count+1):
            if not(j in user_ratings[u]):
                t.append([u,i,j])
        yield np.asarray(t)
# 贝叶斯矩阵分解数据流图
def bpr_mf(user_count, item_count, hidden_dim):
    u = tf.placeholder(tf.int32,[None])
    i = tf.placeholder(tf.int32,[None])
    j = tf.placeholder(tf.int32,[None])
 
    user_emb_w = tf.Variable(tf.random_normal([user_count+1,hidden_dim],stddev = 0.35), name = 'user_emb_w') #W矩阵
    item_emb_w = tf.Variable(tf.random_normal([item_count+1,hidden_dim],stddev = 0.35), name = 'item_enb_w') #H矩阵

    u_emb = tf.nn.embedding_lookup(user_emb_w, u)
    i_emb = tf.nn.embedding_lookup(item_emb_w, i)
    j_emb = tf.nn.embedding_lookup(item_emb_w, j)
    # MF predict: u_i > u_j
    x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keepdims=True)
    
    # AUC for one user:
    # reasonable if all (u,i,j) pairs are from the same user
    # average AUC = mean( auc for each user in test set)
    mf_auc = tf.reduce_mean(tf.to_float(x>0))

    l2_norm = tf.add_n([
        tf.reduce_sum(tf.multiply(u_emb, u_emb)),
        tf.reduce_sum(tf.multiply(i_emb, i_emb)),
        tf.reduce_sum(tf.multiply(j_emb, j_emb))
    ])
    regulation_rate = 0.0001
    bprloss = regulation_rate * l2_norm - tf.reduce_mean(tf.log(tf.sigmoid(x)))
    train_op = tf.train.GradientDescentOptimizer(0.01).minimize(bprloss)

    return u,i,j,mf_auc,bprloss,train_op    

# ListRank MF
def LTR_MF(records, ratings, user_ratings, item_count):
    print()


with tf.Session() as sess:
    user_count, item_count, user_ratings = data_processor()
    user_ratings_test = get_test(user_count,user_ratings)

    u, i, j, mf_auc, bprloss, train_op = bpr_mf(user_count, item_count, 20)
    
    tf.global_variables_initializer().run()

    for epoch in range(1, 4):
        total_loss = 0
        for k in range(1, 500): #输入训练三元组
            uij = get_trainbatch(user_ratings, user_ratings_test, item_count)
            _bprloss, _train_op = sess.run([bprloss, train_op], 
                                feed_dict={u:uij[:,0], i:uij[:,1], j:uij[:,2]})
            total_loss += _bprloss
            if(k%500)==0:
                print(k,total_loss)
        
        print ("epoch: ", epoch)
        print ("bpr_loss: ", total_loss/k)
        print ("_train_op")

        user_count = 0
        _auc_sum = 0.0

        # each batch will return only one user's auc
        for t_uij in get_testbatch(user_ratings, user_ratings_test, item_count):

            _auc, _test_bprloss = sess.run([mf_auc, bprloss],
                                    feed_dict={u:t_uij[:,0], i:t_uij[:,1], j:t_uij[:,2]})
            user_count += 1
            _auc_sum += _auc
        print ("test_loss: ", _test_bprloss, "test_auc: ", _auc_sum/user_count)
        print ("")
    variable_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variable_names)
    for k,v in zip(variable_names, values):
        print("Variable: ", k)
        print("Shape: ", v.shape)
        print(v)

    sess1 = tf.Session()
    u1_dim = tf.expand_dims(values[0][0], 0)
    u1_all = tf.matmul(u1_dim, values[1],transpose_b=True)
    result_1 = sess1.run(u1_all)
    print (result_1)

    print("给用户0的推荐：")
    p = np.squeeze(result_1)
    p[np.argsort(p)[:-5]] = 0
    for index in range(len(p)):
        if p[index] != 0:
            print (index, p[index])
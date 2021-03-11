import os
import time
import pickle
import random
import numpy as np
import tensorflow as tf
import sys
from input import DataInput, DataInputTest
from model import Model

os.environ["TF_CPP_MIN_LOG_LEVEL"]='3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5'
random.seed(1234)
np.random.seed(1234)
tf.set_random_seed(1234)

predict_batch_size = 32
predict_ads_num = 100

print 'load data:'
with open('../data/dataset_without_cntxt_neg_10.pkl', 'rb') as f:
  train_set = pickle.load(f)
  dev_set = pickle.load(f)
  cate_list = pickle.load(f)
  user_count, item_count, cate_count = pickle.load(f)

print 'load test set:'
with open('../data/devset_recall_500.pkl', 'rb') as f:
  test_set = pickle.load(f)

scores = []
def _test(sess, model):
  auc_sum = 0.0
  score_arr = []
  predicted_users_num = 0
  print "test sub items"
  for _, uij in DataInputTest(test_set, predict_batch_size):
    print 'uij:', uij
    score_ = model.test(sess, uij, predict_ads_num)
    # print 'score_:', score_.shape
    score_arr.append(score_)
  return score_arr

gpu_options = tf.GPUOptions(allow_growth=True)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

  model = Model(user_count, item_count, cate_count, cate_list, predict_batch_size, predict_ads_num)
  model.restore(sess, 'save_path/cntxt_merged_neg_100_0113/ckpt')
  
  score = _test(sess, model)
  print '---'*20
  print score[:5]
  with open('../data/dev_score_recall_500_0116.pkl', 'wb') as f:
    pickle.dump(score, f, pickle.HIGHEST_PROTOCOL)
    
  
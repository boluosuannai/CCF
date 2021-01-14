from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import numpy as np
import pandas
from sklearn import metrics
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tensorflow.contrib.layers.python.layers import encoders
from process_unlabel import *
# 加载数据 去掉空数据
df_traindata = pd.read_csv('./labeled_data.csv', encoding='utf-8')
df_traindata = df_traindata.dropna()
df_unlabeldata = pd.read_csv('./unlabeled_data.csv', encoding='utf-8')
df_unlabeldata = df_unlabeldata.dropna()
unlabeldata = df_unlabeldata.content.values.tolist()
traindata = df_traindata.content.values.tolist()
lableset = df_traindata.class_label.values.tolist()
# 数据处理 去掉停用词
stopwords = pd.read_csv("./stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'], encoding='utf-8')
stopwords = stopwords['stopword'].values
def preprocess_text(content_lines, sentences, lableset):
    for i in range(len(content_lines)):
        print('正在处理id：', i)
        segs = jieba.lcut(content_lines[i])
        segs = filter(lambda x:len(x)>1, segs)
        segs = filter(lambda x:x not in stopwords, segs)
        sentences.append((" ".join(segs), lableset[i]))
sentences = []
preprocess_text(traindata, sentences, lableset)
preprocessunlabeldata(sentences, '游戏')
print('游戏标注完毕')
preprocessunlabeldata(sentences, '体育')
print('体育标注完毕')
preprocessunlabeldata(sentences, '娱乐')
print('娱乐标注完毕')
random.shuffle(sentences)
x, y = zip(*sentences)
train_data, test_data, train_target, test_target = train_test_split(x, y, random_state=1234)
learn = tf.contrib.learn
FLAGS = None

MAX_DOCUMENT_LENGTH = 2000
MIN_WORD_FREQUENCE = 1
EMBEDDING_SIZE = 500
global n_words
# 处理词汇
vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
x_train = np.array(list(vocab_processor.fit_transform(train_data)))
x_test = np.array(list(vocab_processor.transform(test_data)))
n_words = len(vocab_processor.vocabulary_)
print('Total words: %d' % n_words)
cate_dic = {'家居':1, '房产':2, '教育':3, '时尚':4, '时政':5, '科技':6, '财经':7, '游戏':8, '体育':9, '娱乐':10}
train_target = map(lambda x:cate_dic[x], train_target)
test_target = map(lambda x:cate_dic[x], test_target)
y_train = pandas.Series(train_target)
y_test = pandas.Series(test_target)
def bag_of_words_model(features, target):
	target = tf.one_hot(target, 10, 1, 0)
	features = encoders.bow_encoder(
			features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
	logits = tf.contrib.layers.fully_connected(features, 15, activation_fn=None)
	loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
	train_op = tf.contrib.layers.optimize_loss(
			loss,
			tf.contrib.framework.get_global_step(),
			optimizer='Adam',
			learning_rate=0.01)
	return ({
			'class': tf.argmax(logits, 1),
			'prob': tf.nn.softmax(logits)
	}, loss, train_op)


model_fn = bag_of_words_model
classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn))

# Train and predict
classifier.fit(x_train, y_train, steps=1000)
y_predicted = classifier.predict(x_test)['class']
score = metrics.accuracy_score(y_test, y_predicted)
print('Accuracy: {0:f}'.format(score))
print('Recall: {0:f}'.format(metrics.recall_score(y_test, y_predicted)))
print('Precision: {0:f}'.format(metrics.precision_score(y_test, y_predicted)))
print('F1: {0:f}'.format(metrics.precision_recall_fscore_support(y_test, y_predicted)))
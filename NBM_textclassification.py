import pandas as pd
import jieba
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
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
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1234)
print(len(x_train))
print(len(x_test))
vec = CountVectorizer(
    analyzer='word', # tokenise by character ngrams
    ngram_range=(1, 4),
    max_features=30000,  # keep the most common 1000 ngrams
)
vec.fit(x_train)
def get_features(x):
    vec.transform(x)
classifier = MultinomialNB()
classifier.fit(vec.transform(x_train), y_train)
print(classifier.score(vec.transform(x_test), y_test))
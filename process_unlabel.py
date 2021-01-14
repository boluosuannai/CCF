import pandas as pd
import jieba
import jieba.analyse as analyse
import random
# df_unlabeldata = pd.read_csv('unlabeled_data.csv', encoding='utf-8')
# df_unlabeldata = df_unlabeldata.dropna()
# content_unlable = df_unlabeldata.content.values.tolist()
def recalldata(category:str):
    df_labeldata = pd.read_csv('./labeled_data.csv', encoding='utf-8')
    df_labeldata = df_labeldata.dropna()
    content_lable = df_labeldata.content.values.tolist()
    label = df_labeldata.class_label.values.tolist()
    ground_trueth = {}
    for i in range(len(label)):
        ground_trueth[i] = label[i]
    train_data = {}
    for i in range(len(content_lable)):
        #tagslist = jieba.lcut(content_lable[i])
        tagslist = analyse.extract_tags(content_lable[i], topK=100, withWeight=False, allowPOS=())
        if category in tagslist and ground_trueth[i] == category:
            train_data[str(i)] = content_lable[i]
    print(category+'召回个数：', len(train_data))
    return train_data
def precisionresult(category:str, datadict:dict, groundtruthdict:dict):
    index = 0
    for key, value in datadict.items():
        if value == category:
            index += 1
    print(category, index/1000)
def precision(train_data:dict, category:str, recall, lablelist:list):
    sum = 0
    for key, value in train_data.items():
        #segs = jieba.lcut(value)
        segs = analyse.extract_tags(value, topK=100, withWeight=False, allowPOS=())
        count = 0
        if category in segs:
            for lable in lablelist:
                if lable in segs:
                    count += 1
            if count == 1:
                sum += 1
    print("准确率：", sum/recall)
# train_data = recalldata('科技')
# labellist = ['财经', '家居', '房产', '教育', '时尚', '时政', '科技']
# precision(train_data, '科技', len(train_data), labellist)

def preprocessunlabeldata(sentences:list, category:str):
    df_labeldata = pd.read_csv('./unlabeled_data.csv', encoding='utf-8')
    df_labeldata = df_labeldata.dropna()
    content_unlabel = df_labeldata.content.values.tolist()
    labellist = ['游戏', '体育', '娱乐', '财经', '家居', '房产', '教育', '时尚', '时政', '科技']
    stopwords = pd.read_csv("./stopwords.txt", index_col=False, quoting=3, sep="\t", names=['stopword'],
                            encoding='utf-8')
    stopwords = stopwords['stopword'].values
    if category == '游戏':
        gamedata = []
        for content in content_unlabel:
            tagslist = jieba.lcut(content)
            #tagslist = analyse.extract_tags(content, topK=100, withWeight=False, allowPOS=())
            count = 0
            if category in tagslist:
                for label in labellist:
                    if label in tagslist:
                        count += 1
                if count == 1:
                    gamedata.append(tagslist)
        random.shuffle(gamedata)
        print("游戏数目：", len(gamedata))
        for item in gamedata[:1000]:
            segs = filter(lambda x: len(x) > 1, item)
            segs = filter(lambda x: x not in stopwords, segs)
            sentences.append((" ".join(segs), category))
    else:
        sum = 0
        for content in content_unlabel:
            tagslist = jieba.lcut(content)
            #tagslist = analyse.extract_tags(content, topK=100, withWeight=False, allowPOS=())
            count = 0
            if category in tagslist:
                for label in labellist:
                    if label in tagslist:
                        count += 1
                if count == 1:
                    sum += 1
                    segs = filter(lambda x: len(x) > 1, tagslist)
                    segs = filter(lambda x: x not in stopwords, segs)
                    sentences.append((" ".join(segs), category))
        print(category + "个数:", sum)
    return sentences

import jieba
import re
from gensim import corpora, models, similarities
import time
import sys
# 1.文件预处理：取文件，构造测试集
def Get_file(path):
    # stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']#过滤数字m
    # stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r']
    doc = open(path, 'r', encoding='UTF-8').read()
    data = jieba.lcut(doc,cut_all=False)
    data1 = ""
    for i in data:  # 去符号
        data1 += i + " "
    texts = [word for word in data1.split()]
    return texts


# 2.去符号

def Dis_sig(str):
    # str = jieba.lcut(str)
    stopwords = [w.strip() for w in open(".\stop.txt", 'r', encoding='UTF-8').readlines()]
    result = []
    for tags in str:
        if (re.match(u"[a-zA-Z0-9\u4e00-\u9fa5]", tags)):
            if tags not in stopwords:
                # print(tags)
                result.append(tags)
        else:
            pass
    return result


# 3.计算相似值
def Com_sim(all_doc_list, doc_test_list):
   text=[all_doc_list, doc_test_list]
   dictionary = corpora.Dictionary(text)
   corpus = [dictionary.doc2bow(doc) for doc in text]
   # similarity = similarities.Similarity('-Similarity-index', corpus, num_features=len(dictionary))
   # test1 = dictionary.doc2bow(doc_test_list)
   # cosine_sim = similarity[test1][0]

   doc_test_vec = dictionary.doc2bow(doc_test_list)
   lsi = models.LsiModel(corpus)
   featurenum = len(dictionary.token2id.keys())
   similarity = similarities.SparseMatrixSimilarity(lsi[corpus], num_features=featurenum)
   cosine_sim = similarity[lsi[doc_test_vec]][0]
   return cosine_sim

def main():

    path = ".\orig.txt"  # 论文原文的文件的绝对路径（作业要求）
    path_add = ".\orig_0.8_del.txt"  # 抄袭版论文的文件的绝对路径
    test = Get_file(path )
    doc = Get_file(path_add)
    doc = Dis_sig(doc)
    test = Dis_sig(test)

    similarity = Com_sim(doc, test)
    print('查重率：%.2f' % similarity)

# 4.主函数main

if __name__ == '__main__':
    main()







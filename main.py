import jieba
import re
from gensim import corpora, models, similarities

# 1.文件预处理：取文件，构造测试集
def Get_file_contents(path):
    # stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'm', 'f', 'r']#过滤数字m
    # stop_flag = ['x', 'c', 'u', 'd', 'p', 't', 'uj', 'f', 'r']
    doc = open(path, 'r', encoding='UTF-8').read()
    data = jieba.lcut(doc)
    data1 = ""
    for i in data:  # 去符号
        data1 += i + " "
    texts = [word for word in data1.split()]
    stopwords = [line.strip() for line in open(".\stop.txt", 'r', encoding='UTF-8').readlines()]
    print(stopwords )
    outstr = []
    for word in texts:
        # if word not in stop_words and word not in stop_flag:
        if word not in stopwords :
            outstr.append(word)
        else:
            pass

    # # stopwords为停用词list
    # stopwords = [line.strip() for line in open('stop.txt', 'r', encoding='utf-8').readlines()]
    # print(stopwords)
    # outstr = ""
    # for word in texts:
    #     if word not in stopwords:
    #         outstr += word + " "
    # doc.close()outstr
    print(texts)
    return texts


# 2。去符号

def dissig(str):
    # str = jieba.lcut(str)
    result = []
    for tags in str:
        if (re.match(u"[a-zA-Z0-9\u4e00-\u9fa5]", tags)):
            result.append(tags)
        else:
            pass
    return result


# 3.计算相似值
def comsim(all_doc_list, doc_test_list):
   dictionary = corpora.Dictionary(all_doc_list)
   corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]
   # similarity = similarities.Similarity('-Similarity-index', corpus, num_features=len(dictionary))
   # test_corpus_1 = dictionary.doc2bow(doc_test_list)
   # cosine_sim = similarity[test_corpus_1]
   doc_test_vec = dictionary.doc2bow(doc_test_list)
   tfidf = models.TfidfModel(corpus)
   featurenum = len(dictionary.keys())
   similarity = similarities.SparseMatrixSimilarity(tfidf[corpus], num_features=featurenum)
   cosine_sim = similarity[tfidf[doc_test_vec]]
   return cosine_sim
# 4.主函数main

if __name__ == '__main__':
    path =".\orig.txt"  # 论文原文的文件的绝对路径（作业要求）
    path_add = ".\orig_0.8_add.txt"  # 抄袭版论文的文件的绝对路径
    path_del = ".\orig_0.8_del.txt"
    path_dis_1 = ".\orig_0.8_dis_1.txt"
    path_dis_10 = ".\orig_0.8_dis_10.txt"
    path_dis_15 = ".\orig_0.8_dis_15.txt"
    save_path = ".\save.txt"  # 输出结果绝对路径
    test = Get_file_contents(path)
    doc0 = Get_file_contents(path_add)
    doc1 = Get_file_contents(path_del)
    doc2 = Get_file_contents(path_dis_1)
    doc3 = Get_file_contents(path_dis_10)
    doc4 = Get_file_contents(path_dis_15)
    all_doc = []
    all_doc.append(doc0)
    all_doc.append(doc1)
    all_doc.append(doc2)
    all_doc.append(doc3)
    all_doc.append(doc4)

    for i in range(len(all_doc)):
        all_doc[i] = dissig(all_doc[i])

    test = dissig(test)

    similarity = comsim(all_doc, test)

    print(similarity)
    # 将相似度结果写入指定文件
    # f = open(save_path, 'w', encoding="utf-8")
    # f.write("文章相似度： %.4f" % similarity)
    # f.close()
# print('sim:')
# print(cosine_sim)
'''5。测试函数main
'''







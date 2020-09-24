import jieba
import re
from gensim import corpora, models, similarities
import time
import sys


# 1.文件预处理：读取文本文件，构造分词向量
def Get_file(doc):
    # doc = open(path, 'r', encoding='UTF-8').read()****************
    data = jieba.lcut(doc,cut_all=False)
    data1 = ""
    for i in data:  # 去符号
        data1 += i + " "
    texts = [word for word in data1.split()]
    return texts


# 2.去符号及停用字

def Dis_sig(str):
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
   dictionary = corpora.Dictionary(text)#构造语料库
   corpus = [dictionary.doc2bow(doc) for doc in text]
   doc_test_vec = dictionary.doc2bow(doc_test_list)
   lsi = models.LsiModel(corpus)#模型训练
   featurenum = len(dictionary.token2id.keys())
   similarity = similarities.SparseMatrixSimilarity(lsi[corpus], num_features=featurenum)
   cosine_sim = similarity[lsi[doc_test_vec]][0]
   return cosine_sim

def main():
    
    # 开始测量程序所需时间
    start_time = time.time()

    try:
        orig_path, add_path, answer_path = sys.argv[1:4]
    except BaseException:
        print("Error: 输入命令错误")
    else:
        # 判断命令行参数有没有错误
        try:
            orig = open(orig_path, 'r', encoding='UTF-8')
            orig_context = orig.read()
        except IOError:
            print("Error: 没有从该路径：{}找到文件/读取文件失败".format(orig_path))
            conditio_one = 0
        else:
            conditio_one = 1
            orig.close()
        # 判断抄袭文件路径等是否出错
        try:
            orig_add = open(add_path, 'r', encoding='UTF-8')
            add_context = orig_add.read()
        except IOError:
            print("Error: 没有从该路径：{}找到文件/读取文件失败".format(add_path))
            conditio_two = 0
        else:
            conditio_two = 1
            orig_add.close()
            # 判断答案文件路径等是否出错
            try:
                answer_txt = open(answer_path, 'w', encoding='UTF-8')
            except BaseException:
                print("Error: 创建文件：{}失败".format(answer_path))
                conditio_three = 0
            else:
                conditio_three = 1

            # 如果输入命令行参数没有错误则运行
            if (conditio_one & conditio_two & conditio_three):
                # path = ".\orig.txt"  # 论文原文的文件的绝对路径（作业要求）
                # path_add = ".\orig_0.8_dis_15.txt"  # 抄袭版论文的文件的绝对路径
                test = Get_file(orig_context)
                doc = Get_file(add_context)
                doc = Dis_sig(doc)
                test = Dis_sig(test)

                similarity = Com_sim(doc, test)

            # 得知程序运行所需时间
            end_time = time.time()
            time_required = end_time - start_time
            # 控制台输出，方便得知状态和答案
            print('查重率：%.2f' % similarity)
            print("输出文件到:" + answer_path)
            print('程序所耗时间：%.2f s' % (time_required))




# 4.主函数main

if __name__ == '__main__':
    main()







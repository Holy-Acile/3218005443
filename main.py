import jieba
import re
from gensim import corpora,models,similarities

doc0 = open("./orig_0.8_add.txt",encoding='UTF-8').read()
doc1 = open("./orig_0.8_del.txt",encoding='UTF-8').read()
doc2 = open("./orig_0.8_dis_1.txt",encoding='UTF-8').read()
doc3 = open("./orig_0.8_dis_10.txt",encoding='UTF-8').read()
doc4 = open("./orig_0.8_dis_15.txt",encoding='UTF-8').read()
doc_test = open("./orig.txt",encoding='UTF-8').read()


all_doc = []
all_doc.append(doc0)
all_doc.append(doc1)
all_doc.append(doc2)
all_doc.append(doc3)
all_doc.append(doc4)


all_doc_list = []
for doc in all_doc:
    doc_list = [word for word in jieba.cut(doc)]
    all_doc_list.append(doc_list)


doc_test_list = [word for word in jieba.cut(doc_test)]



for i in range(len(all_doc_list)):
    result = []
    for tags in all_doc_list[i]:
        if (re.match(u"[a-zA-Z0-9\u4e00-\u9fa5]", tags)):
            result.append(tags)
        else:
            pass
    all_doc_list[i] = result


result = []
for tags in doc_test_list:
    if (re.match(u"[a-zA-Z0-9\u4e00-\u9fa5]", tags)):
        result.append(tags)
    else:
        pass
doc_test_list = result



dictionary = corpora.Dictionary(all_doc_list)
dictionary.keys()

corpus = [dictionary.doc2bow(doc) for doc in all_doc_list]


tfidf = models.TfidfModel(corpus)

similarity = similarities.Similarity('-Similarity-index', corpus, num_features=len(dictionary))
test_corpus_1 = dictionary.doc2bow(doc_test_list)
cosine_sim = similarity[test_corpus_1]

print('sim:')
print(cosine_sim)







#encoding:utf-8
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF, IDF

from settings import spark_master, hdfs_path, mongo_host, mongo_user, mongo_pass

def spark_context(master):
    conf = SparkConf().setAppName('zhangxinyun-spark').setMaster(master)
    sc = SparkContext(conf=conf)
    return sc

def tokenize(line):
    from jieba import posseg
    terms = posseg.cut(line)
    return [term for term, flag in terms if flag in ['an', 'n', 'nz', 'vn']];

def doc_tfidf(item):
    result = []
    terms = item[0]
    tfidf_array = item[1].toArray()
    numFeatures = 1 << 20
    for term in set(terms):
        index = hash(term) % numFeatures
        tfidf = tfidf_array[index]
        if tfidf > 0.0:
            result.append((term, tfidf))
    return sorted(result, key=lambda i: i[1])

def clear_mongodb(client):
    client.zxy.terms.delete_many({})

def send_mongodb(client, doc):
    client.zxy.terms.insert_one(doc)

def main():
    # 初始化 SparkContext
    sc = spark_context(spark_master)

    # 读取文件
    data = sc.textFile(hdfs_path)

    # 分词
    documents = data.map(tokenize)
    documents.cache()

    # TF
    hashingTF = HashingTF()
    tf = hashingTF.transform(documents)

    # IDF
    idf = IDF(minDocFreq=2).fit(tf)
    
    # TFIDF
    tfidf = idf.transform(tf)

    # 链接到 MongoDB
    from pymongo import MongoClient
    mongo_client = MongoClient(mongo_host)
    mongo_client.admin.authenticate(mongo_user, mongo_pass, mechanism='SCRAM-SHA-1')
    clear_mongodb(mongo_client)

    # zip
    term_tfidf = documents.zip(tfidf).map(doc_tfidf)
    for article in term_tfidf.collect():
        data = {}
        data['terms'] = []
        for term in article:
            item = {}
            item['text'] = term[0].encode('utf-8')
            item['size'] = int(term[1] * 100)
            data['terms'].append(item)
        send_mongodb(mongo_client, data)

main()

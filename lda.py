#encoding:utf-8
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors

from settings import spark_master, hdfs_path, mongo_host, mongo_user, mongo_pass

def spark_context(master):
    conf = SparkConf().setAppName('zhangxinyun-spark').setMaster(master)
    sc = SparkContext(conf=conf)
    return sc

def tokenize(line):
    from jieba import posseg
    terms = posseg.cut(line)
    return [term for term, flag in terms if flag in ['an', 'n', 'vn', 'nz']]

def hashing_term_mapping(documents):
    return documents.flatMap(lambda document: document).distinct().map(lambda term: ((hash(term) % (2 << 10)), term))

def send_mongodb(client, doc):
    client.zxy.topics.insert_one(doc)

def main():
    # 初始化 SparkContext
    sc = spark_context(spark_master)
    
    # 加载数据
    data = sc.textFile(hdfs_path)
    
    # 计算词频
    documents = data.map(tokenize)
    hashingTF = HashingTF(2 << 10)
    tf = hashingTF.transform(documents)

    # 对文档词频进行索引
    corpus = tf.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

    # 索引和词的映射
    mapping = hashing_term_mapping(documents)
    mapping.cache()

    # 训练 LDA 模型
    ldaModel = LDA.train(corpus, k=3)

    # 链接到 MongoDB
    from pymongo import MongoClient
    mongo_client = MongoClient(mongo_host)
    mongo_client.admin.authenticate(mongo_user, mongo_pass, mechanism='SCRAM-SHA-1')

    # 打印结果
    topics = ldaModel.describeTopics(maxTermsPerTopic=10)
    for topic in range(3):
        doc = {}
        doc['name'] = "topic " + str(topic)
        doc['terms'] = []
        for i in range(10):
            term_index = topics[topic][0][i]
            for term in mapping.lookup(term_index):
                doc['terms'].append([term.encode("utf8"), topics[topic][1][i]])
        send_mongodb(mongo_client, doc)

if __name__ == '__main__':
    main()

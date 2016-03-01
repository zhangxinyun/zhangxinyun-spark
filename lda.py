#encoding:utf-8
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import LDA, LDAModel
from pyspark.mllib.linalg import Vectors

from settings import spark_master, hdfs_path

def spark_context(master):
    conf = SparkConf().setAppName('zhangxinyun-spark').setMaster(master)
    sc = SparkContext(conf=conf)
    return sc

def main():
    # 初始化 SparkContext
    sc = spark_context(spark_master)
    
    # Load and parse the data
    data = sc.textFile(hdfs_path)
    parsedData = data.map(lambda line: Vectors.dense([float(x) for x in line.strip().split(' ')]))
    # Index documents with unique IDs
    corpus = parsedData.zipWithIndex().map(lambda x: [x[1], x[0]]).cache()

    # Cluster the documents into three topics using LDA
    ldaModel = LDA.train(corpus, k=3)

    # Output topics. Each is a distribution over words (matching word count vectors)
    print("Learned topics (as distributions over vocab of " + str(ldaModel.vocabSize()) + " words):")
    topics = ldaModel.topicsMatrix()
    for topic in range(3):
        print("Topic " + str(topic) + ":")
        for word in range(0, ldaModel.vocabSize()):
            print(" " + str(topics[word][topic]))

main()

#encoding:utf-8
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF, IDF

import yaml
import jieba


def load_config():
    yaml_file = open('config/config.yml')
    config = yaml.safe_load(yaml_file)
    yaml_file.close()
    return config

def spark_context(master):
    conf = SparkConf().setAppName('zhangxinyun-spark').setMaster(master)
    sc = SparkContext(conf=conf)
    return sc

def tokenize(data):
    return data.map(lambda line: jieba.cut(line))

def main():
    # 加载配置文件
    config = load_config()

    # 初始化 SparkContext
    sc = spark_context(config['spark-master'])

    # 读取文件
    data = sc.textFile(config['data-source'])

    # 分词
    documents = tokenize(data)

    # TF
    hashingTF = HashingTF()
    tf = hashingTF.transform(documents)
    tf.cache()

    # IDF
    idf = IDF(minDocFreq=2).fit(tf)
    
    # TFIDF
    #tfidf = idf.transform(tf)

main()

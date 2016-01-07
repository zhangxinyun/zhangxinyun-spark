#encoding:utf-8
from pyspark import SparkContext, SparkConf

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

def tokenize(line):
    return jieba.cut(line)

def tf(data):
    return data.flatMap(lambda line: tokenize(line)).map(lambda term: (term, 1)).reduceByKey(lambda x,y: x + y)

def main():
    # 加载配置文件
    config = load_config()

    # 初始化 SparkContext
    sc = spark_context(config['spark-master'])

    # 读取文件
    data = sc.textFile(config['data-source'])

    # 词频计算
    term_frequence = tf(data)

    # 打印结果
    print term_frequence.collect()


main()

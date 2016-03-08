#encoding:utf-8
from pyspark import SparkContext, SparkConf
from pyspark.mllib.feature import HashingTF, IDF

from settings import spark_master, hdfs_path

def spark_context(master):
    conf = SparkConf().setAppName('zhangxinyun-spark').setMaster(master)
    sc = SparkContext(conf=conf)
    return sc

def tokenize(line):
    import jieba.posseg
    terms = posseg.cut(line)
    return [term for term, flag in terms if flag in ['an', 'n', 'nz', 'vn']];

def main():
    # 初始化 SparkContext
    sc = spark_context(spark_master)

    # 读取文件
    data = sc.textFile(hdfs_path)

    # 分词
    documents = data.map(tokenize)

    # TF
    hashingTF = HashingTF()
    tf = hashingTF.transform(documents)
    tf.cache()

    # IDF
    idf = IDF(minDocFreq=2).fit(tf)
    
    # TFIDF
    #tfidf = idf.transform(tf)

main()

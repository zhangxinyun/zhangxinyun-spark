# 毕业设计之数据分析

基于 Apache Spark (with Apache Hadoop) 的数据分析项目

## 部署环境

操作系统：Ubuntu 14.04

Java 版本：Java 8

Python 版本：Python 2.7

Spark 版本：1.4.1 +

Hadoop 版本：1.6.0 +

## 安装依赖

安装 Python 包管理器 *pip*：

	weget https://bootstrap.pypa.io/get-pip.py && python get-pip.py
	
安装 *NumPy*

	pip install numpy

## TFIDF 词频逆文档频率算法

### 安装依赖

安装 *PyYAML*

	pip install pyyaml

克隆 *jieba* 的最新代码：

	git clone https://github.com/fxsjy/jieba.git

压缩 *jieba*

	cd jieba && zip -r jieba.zip jieba	

### 运行

  spark-submit tfidf.py --py-files jieba.zip

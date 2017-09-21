# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 16:00:25 2017

@author: ligong

@description:这是用word2vec生成文章与文章之间的相似度的程序

说明：
把一个个文章id当成一个个单词，然后用word2vec把一个个文章id转换成固定位长度的向量，
之后就可以用计算文章id与文章id之间的相似度了！
"""
import math
import time
import json
from pyspark.mllib.feature import Word2Vec
from pyspark import SparkConf
from pyspark import SparkContext

def preprocess(rdd,**kw):
    """
    输入的格式是：
    user_id,item_id,score,time
    
    这里需要把score小于0的过滤掉，然后用时间排序
    reverse:True，按照时间从大到小排序；False，按照时间从小到大排序
    max_num:当点击序列大于某个值的时候，就忽略
    min_num:当点击序列小于某个值的时候，就忽略
    生成的格式是：
    item_id1,item_id2,.....
    """
    step_1 = time.time()
    reverse = kw.get('reverse',False)
    partitions = kw.get('partitions',100)
    max_num = kw.get('max_num',100)
    min_num = kw.get('min_num',1)
    def process_line(line):
        user_id,item_id,score,time = line.strip().split(',')
        status = True if score > 0 else False
        if status:
            return (user_id,([(item_id,float(score),float(time))],True))
        return (user_id,([],False))
    
    def merge(x,y):
        """
        合成
        """
        tmp = []
        if x[1] and y[1] and len(x[0])+len(y[0]) <= max_num:
            i,j=0,0
            while i < len(x[0]) and j < len(y[0]):
                if x[0][i][2] < y[0][j][2]:
                    tmp.append(x[0][i])
                    i += 1
                else:
                    tmp.append(y[0][j])
                    j += 1
            while i < len(x[0]):
                tmp.append(x[0][i])
                i += 1
            while j < len(y[0]):
                tmp.append(y[0][j])
                j += 1
            return (tmp,True)
        return ([],False)
    

    tmp_rdd = rdd.map(lambda _:process_line(_)).filter(lambda _:_[1][1]==True).reduceByKey(lambda x,y:merge(x,y),partitions).\
        filter(lambda _:_[1][1] and len(_[1][0]) > min_num)
        
    if reverse:
        tmp_rdd = tmp_rdd.map(lambda _:map(lambda x:x[0],_[1][0])[::-1])
    else:
        tmp_rdd = tmp_rdd.map(lambda _:map(lambda x:x[0],_[1][0]))
    step_2 = time.time()
    print 'Preprocess Data Finish, use:%s s!' % (step_2-step_1)
    return tmp_rdd

def word2vec(rdd,**kw):
    """
    生成向量
    vec_len:生成向量的长度
    min_count:出现最少次数
    window_size:窗口长度
    learning_rate:学习率
    """
    seed = int(time.time())
    vec_len = kw.get('vec_len',300)
    min_count = kw.get('min_count',3)
    window_size = kw.get('window_size',5)
    partitions = kw.get('partitions',5)
    lr = kw.get('learning_rate',0.025)
    
    step_1 = time.time()
    model = Word2Vec().setVectorSize(vec_len).setLearningRate(lr).setMinCount(min_count).\
        setNumPartitions(partitions).setSeed(seed).setWindowSize(window_size).fit(rdd)
        
    vectors = model.getVectors()
    step_2 = time.time()
    print 'Build Word2vec Model Using:%s s!' % (step_2-step_1)
    result = dict(vectors)
    keys = result.keys()
    for key in keys:
        result[key] = list(result[key])
    return result

def gen_similarity(vector_dict,spark_content,outfile,**kw):
    """
    获得相似度
    
    kw:运行中的参数
        {
             partition_num:分区数
             r_len:返回item最相近的若干条
        }
    """
    step_1 = time.time()
    partitions = kw.get('partitions',100)
    r_len = kw.get('r_len',20)
    #将全局字典广播（item数量不多的情形）
    #vector_str = json.dumps(vector_dict)
    b_vector_dict = spark_content.broadcast(vector_dict)
    #将字典中每一个item的矩阵分发出去
    rdd = spark_content.parallelize(vector_dict.keys(),partitions)
    
    def gen_pair(item):
        """
        计算两两相似度预处理
        """
        keys = b_vector_dict.value.keys()
        pairs = [(item,_) for _ in keys]
        return filter(lambda _:_[0] < _[1],pairs)
    
    def similarity(item):
        """
        生成相似度
        """
        k1,k2 = item[0],item[1]
        v1 = b_vector_dict.value.get(k1)
        v2 = b_vector_dict.value.get(k2)
        if v1 is None or v2 is None:
            return 0
        score = 0
        for (e1,e2) in zip(v1,v2):
            score += e1*e2
        norm_square = sum(map(lambda _:_*_,v1)) * sum(map(lambda _:_*_,v2))
        if norm_square == 0:
            return 0
        sim = float(score)/math.sqrt(norm_square)
        return [(k1,[(k2,sim)]),(k2,[(k1,sim)])]
    
    def merge(x,y):
        """
        合成结果
        """
        x.extend(y)
        return x
    def extract_result(item):
        """
        生成最后的结果
        """
        item.sort(key = lambda _:_[1],reverse =True)
        return item[:r_len]
    result = rdd.flatMap(lambda _:gen_pair(_)).flatMap(lambda _:similarity(_)).reduceByKey(lambda x,y:merge(x,y),partitions).mapValues(lambda _:extract_result(_))
    result.saveAsTextFile(outfile)
    #print result.take(10)
    step_2 = time.time()
    print 'Finish gen similarity, using :%s s!' % (step_2 - step_1)
    
if __name__ == '__main__':
    input_file = 'hdfs:/warehouse/rpt_qukan.db/algo_word2vec_data_prepare/*'
    output_file = 'hdfs:/tmp/algo/word2vec_result'
    conf = SparkConf()
    conf.setAppName('word2vec')
    conf.set("spark.executor.memory", "5g")
    conf.set("spark.cores.max","100") 
    conf.set("spark.driver.maxResultSize","5g")
    #conf.set("spark.yarn.queue"," root.lechuan.adhoc")
    sc = SparkContext(conf=conf)
    partition_num = 100
    rdd = sc.textFile(input_file,partition_num)
    tmp_rdd = preprocess(rdd)
    
    vector = word2vec(tmp_rdd)
    #print type(vector)
    #print type(vector[vector.keys()[0]])
    gen_similarity(vector,sc,output_file)

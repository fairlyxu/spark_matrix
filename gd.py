# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 14:54:35 2017

@author: ligong

@description:这是实现批量梯度下降的程序，用rdd来实现

说明：
数据最好要做一下预处理，然后再用梯度下降（标准化）
rdd的结构是
(batch_id,value)
这里的value是一个(y,x),x是个稀疏矩阵
"""
import random
import math
import traceback
from Matrix.SparseVector import SparseVector
def tobatches(rdd,batch_size=10):
    """
    将rdd转化为一个个batch
    rdd原本第一列是y，第二列是x（是一个SparseVector对象--不是spark默认的，笔者实现的那个。。。）
    """
    total_num = rdd.count()
    batch_num = total_num/batch_size
    return (rdd.map(lambda _:(random.randint(0,batch_num),(_[0],_[1]))),batch_num)
    
def squared_loss_and_derivative(y,x,weight):
    """
    平方损失及其导数
    """
    tmp = x.multiply(weight)-y
    return (0.5*tmp*tmp,x*tmp)
    

def exponential_loss_and_derivative(y,x,weight):
    """
    指数损失函数及其导数
    一般用于分类，y in [1,-1]
    """
    tmp = x.multiply(weight)
    value = math.exp(-y*tmp)
    return (value,x*(-value*y))
    
def crossentropy_loss_and_derivative(y,x,weight):
    """
    交叉熵损失函数及其导数
    """
    tmp = x.multiply(weight)
    value = 1.0/(1.0 + math.exp(-tmp))
    return (-1*(y*math.log(value)+(1-y)*math.log(1-value)),x*(value - y))

def gd(rdd,weight,loss_and_derivative,learning_rate=0.1,partitions=100):
    """
    梯度下降
    weight:权重
    rdd:数据
    loss_and_derivative:损失函数和导数
    learning_rate:学习速率
    """

    def fun(y,x,weight):
        loss,derivative = loss_and_derivative(y,x,weight)
        return (loss,derivative,1)

    def merge(x,y):
        """
        合成结果
        """
        return (x[0]+y[0],x[1]+y[1],x[2]+y[2])
    result = rdd.map(lambda _:(_[0],fun(_[1][0],_[1][1],weight))).\
        reduceByKey(lambda x,y:merge(x,y),partitions).map(lambda _:(_[1][0]/_[1][2],_[1][1]*(1.0/_[1][2]))).collect()
    if result is None or len(result) == 0:
        raise Exception('Empty Batch!')
    loss,derivative = result[0][0],result[0][1]
    #derivative.normalize()
    #print derivative[0],derivative[1]
    weight = weight - derivative*learning_rate
    return (loss,weight)
 
def bgd(rdd,weight,batch_id,loss_and_derivative,learning_rate=0.1,partitions=100):
    """
    批量梯度下降
    weight:权重
    batch_id:块号
    rdd:数据
    loss_and_derivative:损失函数和导数
    learning_rate:学习速率
    """
    
    def fun(y,x,weight):
        loss,derivative = loss_and_derivative(y,x,weight)
        return (loss,derivative,1)
    
    def merge(x,y):
        """
        合成结果
        """
        return (x[0]+y[0],x[1]+y[1],x[2]+y[2]) 
    result = rdd.filter(lambda _:_[0] == batch_id).map(lambda _:(_[0],fun(_[1][0],_[1][1],weight))).\
        reduceByKey(lambda x,y:merge(x,y),partitions).map(lambda _:(_[1][0],_[1][1]*(1.0/_[1][2]))).collect()
    if result is None or len(result) == 0:
        raise Exception('Empty Batch!')
    loss,derivative = result[0][0],result[0][1]
    #derivative.normalize() 
    weight = weight - derivative*learning_rate
    return (loss,weight)
    
if __name__ == '__main__':
    X,y = [],[] 
    for i in range(1000):
        t = SparseVector(2)
        a = random.random()
        t.set_value(0,a)
        t.set_value(1,1)
        X.append(t)
        y.append(5*a+3)
    
    weight = SparseVector(2)
    weight.set_value(0,1000)
    weight.set_value(1,1000)
    result = []
    for (i,j) in zip(y,X):
        result.append((i,j))
    import pyspark as ps
    conf = ps.SparkConf().setAppName("CF").setMaster("spark://spark31:7077")
    conf.set("spark.executor.memory", "10g")
    conf.set("spark.cores.max","100") 
    conf.set("spark.driver.maxResultSize","5g")
    #conf.set("spark.submit.pyFiles","Matrix-1.0-py2.7.egg")
    partitions = 100
    sc = ps.SparkContext(conf=conf)
    sc.addPyFile('Matrix-1.0-py2.7.egg')
    rdd = sc.parallelize(result,partitions)
    rdd,batch_num = tobatches(rdd)
    '''
    s = 0
    d = SparseVector(2) 
    for i in range(1000):
        l,dt = squared_loss_and_derivative(y[i],X[i],weight) 
        print y[i],X[i].multiply(weight)
        s += l
        d += dt
    print d,s
    '''
    for i in range(500):
        try:
            a = 1
            
            if i >= 150:
                a = 0.1
            if i >= 300:
                a = 0.05
            if i >= 400:
                a = 0.01
            loss,weight = gd(rdd,weight,squared_loss_and_derivative,a)
            print i,loss,(weight[0],weight[1])
        except:
            traceback.print_exc()

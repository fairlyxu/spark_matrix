# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 10:19:49 2017

@author: ligong

@description:这是实现奇异值分解

说明：
本程序是用lanczos算法的精度来求解k个最大的奇异值和对应的向量
"""
import math
import numpy as np
#from Matrix import MatrixInRDD,matrix_vector_multiply,transpose,multiply,Vector,orth_vector
from Matrix import *


def svds(matrix,spark_context,k=10,epsl=1e-6,partitions=100):
    """
    svd分解
    """
    assert isinstance(matrix,MatrixInRDD)
    
    def dot(obj1,obj2):
        return multiply(obj1,obj2,spark_context,partitions)
        
    #先求matrix的转置
    matrix_t = transpose(matrix)
    
    #令matrix为A，计算A^T*A,这是一个对称矩阵了
    At_A = dot(matrix_t,matrix)

    V = []
    n = At_A.cols
    #随机生成一个vector
    v = Vector(n)
    v.normal()
    V.append(v)
    alpha,beta = [0 for i in xrange(k)],[0 for i in xrange(k)]

    W_tmp = dot(At_A,V[0])
    alpha[0] = dot(W_tmp,V[0])
    W = W_tmp - V[0]*alpha[0]
    for i in range(1,k):
        beta[i] = W.norm()
        if beta[i] < epsl:
            v = orth_vector(V)
        else:
            v = W*(1.0/beta[i])
        V.append(v)
        W_tmp = dot(At_A,v)
        alpha[i] = W_tmp.multiply(v)
        W = W_tmp - v*alpha[i] - V[i-1]*beta[i]

    T = np.zeros((k,k))
    
    for i in xrange(k):
        T[i,i] = alpha[i]
        if i < k-1:
            T[i,i+1] = beta[i+1]
            T[i+1,i] = beta[i+1]

    u,s,v = np.linalg.svd(T)
    s = map(lambda _:math.sqrt(_),s)
    #生成右边的矩阵V
    right_matrix = np.row_stack(map(lambda _:_.vector,V)).T.dot(u)
    
    t = np.diag(map(lambda _:0 if _ == 0 else 1.0/_,s))
    
    tmp_matrix = numpy2MatrixInRDD(right_matrix.dot(t),spark_context,partitions)
    left_m = dot(matrix,tmp_matrix)
    right_m = numpy2MatrixInRDD(right_matrix.T,spark_context,partitions)
    return left_m,s,right_m
if __name__ == '__main__':
    import pyspark as ps
    conf = ps.SparkConf().setAppName("CF").setMaster("spark://spark31:7077")
    conf.set("spark.executor.memory", "10g")
    conf.set("spark.cores.max","100") 
    conf.set("spark.driver.maxResultSize","5g")
    sc = ps.SparkContext(conf=conf)
    #matrix_matrix_multiply_test()
    #matrix_vector_multiply_test(sc)
    #transpose_test()
    #norm_test()
    #matrix_add_test()
    r = MatrixInRDD(100,100)
    r.load(sc,'hdfs:/tmp/b.txt',MatrixInRDD.PAIR,'int',10)
    U,s,V = svds(r,sc,10,10)
    print U,s,V

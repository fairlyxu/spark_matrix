# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 14:48:18 2017

@author: ligong

@description:这是来做矩阵相关的一些值
包括：
L2
"""
from MatrixInRDD import MatrixInRDD
from BlockMatrixInRDD import BlockMatrixInRDD
from Vector import Vector
from operator import add
import math
import numpy as np

def trace(matrix,partitions=100):
    """
    获得trace
    """
    assert isinstance(matrix,MatrixInRDD)
    assert matrix.cols == matrix.rows
    matrix.change_format(MatrixInRDD.PAIR)
    r = matrix.rdd.map(lambda _:('TRACE',_[1] if _[0][1] == _[0][0] else 0)).reduceByKey(add,partitions).collect()
    return sum(map(lambda _:_[1],r))

def transpose(matrix):
    """
    转置
    """
    assert isinstance(matrix,MatrixInRDD)
    #处理block矩阵的问题
    if isinstance(matrix,BlockMatrixInRDD):
        matrix.deblocking()

    rdd = matrix.rdd
    
    #三种不同的格式处理函数
    def pair_format(kv):
        return ((kv[0][1],kv[0][0]),kv[1])
        
    def row_col_format(kv):
        return (kv[1][0],(kv[0],kv[1][1]))
    if matrix.format == MatrixInRDD.PAIR:
        trans_func = pair_format
    elif matrix.format in [MatrixInRDD.ROW,MatrixInRDD.COL]:
        trans_func = row_col_format
    else:
        raise Exception('Data Format Error!')
    
    #生成新矩阵
    new_matrix = MatrixInRDD(matrix.cols,matrix.rows)
    new_matrix.set_format(matrix.format)
    new_matrix.set_rdd(rdd.map(lambda _:trans_func(_)))
    return new_matrix
    
    
def norm(matrix,ntype='l2',partitions=100):
    """
    求norm
    """
    assert isinstance(matrix,MatrixInRDD)

    #处理block矩阵的问题
    if isinstance(matrix,BlockMatrixInRDD):
        matrix.deblocking()
    
    def denorm_value(v):
        if ntype == 'l1':
            return v
        elif ntype == 'l2':
            return math.sqrt(v)
        elif ntype.startswith('l'):
            try:
                return v**(1.0/float(ntype[1:]))
            except:
                raise Exception('Norm Type Error!')
        
    def norm_value(v):
        if ntype == 'l1':
            return abs(v)
        elif ntype == 'l2':
            return v*v
        elif ntype.startswith('l'):
            try:
                return v**float(ntype[1:])
            except:
                raise Exception('Norm Type Error!')
    
    #三种不同的格式处理函数
    def pair_format(kv):
        return norm_value(kv[1])
        
    def row_col_format(kv):
        return norm_value(kv[1][1])
        
    if matrix.format == MatrixInRDD.PAIR:
        trans_func = pair_format
    elif matrix.format in [MatrixInRDD.ROW,MatrixInRDD.COL]:
        trans_func = row_col_format
    else:
        raise Exception('Data Format Error!')
    r = matrix.rdd.map(lambda _:('NORM',trans_func(_))).reduceByKey(add,partitions).collect()
    s = 0
    return denorm_value(sum(map(lambda _:_[1],r)))
    
    
    
def orth_vector(vectors):
    """
    生成一个向量与vectors都正交
    """
    if vectors is None or len(vectors) == 0:
        raise Exception("Empty Vectors!")
    n = None
    for v in vectors:
        assert isinstance(v,Vector)
        if n == None:
            n = v.length
        else:
            assert n == v.length
    #以上是一些状态检查
    tmp = Vector(n)
    tmp.normal()
    result = Vector(n)
    result.zeros()
    for v in vectors:
       l = v.norm()
       if l == 0:
           continue
       result = result + v*(tmp.multiply(v)/(l*l))
    answer = tmp - result
    answer.normalize()
    return answer

def numpy2MatrixInRDD(matrix,spark_context,partitions=100):
    """
    numpy matrix to MatrixInRDD
    to_rdd(self,spark_context,row_id=0,partitions=100)
    """
    assert isinstance(matrix,np.ndarray)
    (row,col) = matrix.shape
    pairs = []
    for r in xrange(row):
        pairs.extend(filter(lambda x:x[1] != 0,map(lambda _:((r,_[0]),_[1]),enumerate(matrix[r]))))
    m = MatrixInRDD(row,col)
    m.rdd = spark_context.parallelize(pairs,partitions)
    m.set_format(MatrixInRDD.PAIR)
    return m

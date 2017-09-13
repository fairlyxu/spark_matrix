# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 13:39:47 2017

@author: ligong

@description:这是稀疏向量的实现

这里主要是用coo_matrix来实现
"""
import traceback
import numpy as np
from sklearn import preprocessing
from scipy.sparse import csr_matrix
import scipy.sparse as sp

MAX_LENGTH = 100000000
class SparseVector(object):
    def __init__(self,length):
        """
        初始化
        """
        global MAX_LENGTH
        if length > MAX_LENGTH or length <= 0 or not isinstance(length,int):
            raise Exception('Vector Length Error!')
        self.length = length
        self.vector = csr_matrix((1, self.length))
        
        
    def load(self,string):
        """
        这里的格式是libsvm的格式
        idx1:value1 idx2:value2 ...
        """
        data,row,col = [],[],[]
        for item in string.split(' '):
            try:
                idx,value = item.split(':')
                assert 0 <= int(idx) < self.length
                col.append(int(idx))
                data.append(float(value))
                row.append(0)
                
            except:
                traceback.print_exc()

        self.vector = csr_matrix((data, (row, col)),shape=(1, self.length))
       
    def normal(self,mu=0,sigma=1,normalized=True):
        """
        正太
        """
        tmp = np.random.normal(mu,sigma,(1,self.length))
        if normalized:
            tmp = preprocessing.normalize(tmp, norm='l2')
        row = [0 for i in xrange(self.length)]
        col = [i for i in xrange(self.length)]
        data = [tmp[0][i] for i in xrange(self.length)]
        self.vector = csr_matrix((data, (row, col)),shape=(1, self.length))

    def normalize(self):
        """
        norm l2
        """    
        self.vector = self.vector*(1.0/sp.linalg.norm(self.vector))

    def multiply(self,vector):
        """
        向量乘法
        """
        if isinstance(vector,int) or isinstance(vector,float):
            result = SparseVector(self.length)
            result.vector = vector*self.vector
            return result
        if not isinstance(vector,SparseVector):
            raise Exception('Vector Type Error!')
        if vector.length != self.length:
            raise Exception('Vector Length:(%s,%s) Not Match!' % (self.length,vector.length))
        return float((self.vector*vector.vector.T).data[0])
       
    def __mul__(self,value):
        """
        数字乘法
        """
        assert isinstance(value,int) or isinstance(value,float)
        return self.multiply(value)

    def __str__(self):
        return str(self.vector)

    def __add__(self,vector):
        """
        矩阵加法
        """
        result = SparseVector(self.length)
        if isinstance(vector,int) or isinstance(vector,float):
            result.vector = vector+self.vector
            return result
        if not isinstance(vector,SparseVector):
            raise Exception('Vector Type Error!')
        if vector.length != self.length:
            raise Exception('Vector Length:(%s,%s) Not Match!' % (self.length,vector.length))

        result = SparseVector(self.length)
        
        result = SparseVector(self.length)
        result.vector = self.vector+vector.vector
        return result

    def __sub__(self,vector):
        """
        减法
        """
        result = SparseVector(self.length)
        if isinstance(vector,int) or isinstance(vector,float):
            result.vector = self.vector - vector
            return result
        if not isinstance(vector,SparseVector):
            raise Exception('Vector Type Error!')
        if vector.length != self.length:
            raise Exception('Vector Length:(%s,%s) Not Match!' % (self.length,vector.length))

        result = SparseVector(self.length)

        result = SparseVector(self.length)
        result.vector = self.vector-vector.vector
        return result
    
    def __getitem__(self,index):
        """
        取值
        """
        return self.vector[0,index]
        
    def __len__(self):
        return self.length
    
    
    def to_rdd(self,spark_context,row_id=0,partitions=100):
        """
        转化为rdd
        """
        v = self.vector.todense()[0]
        return spark_context.parallelize(filter(lambda x:x[1] != 0,map(lambda _:((row_id,_[0]),_[1]),enumerate(v))),partitions)
        
    def __del__(self):
        del self.vector
        
    def set_value(self,index,value):
        self.vector[0,index] = value
        

 
        
if __name__ == '__main__':
    v = SparseVector(100)
    v.normal()
    v.set_value(0,100)
    print v[0]
    
 


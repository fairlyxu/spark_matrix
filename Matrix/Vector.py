# -*- coding: utf-8 -*-
"""
Created on Thu Sep 07 21:14:52 2017

@author: ligong

@description:这是Vector的实现
"""
import numpy as np
from sklearn import preprocessing
from scipy.sparse import coo_matrix
import scipy.sparse as sp

MAX_LENGTH = 100000000
class Vector(object):
    def __init__(self,length):
        """
        初始化
        """
        global MAX_LENGTH
        if length > MAX_LENGTH or length <= 0 or not isinstance(length,int):
            raise Exception('Vector Length Error!')
        self.vector = None
        self.length = length
        
    def ones(self):
        self.vector = np.ones((1,self.length))
    
    def zeros(self):
        self.vector = np.zeros((1,self.length))
        
    def normal(self,mu=0,sigma=1,normalized=True):
        """
        正太
        """
        self.vector = np.random.normal(mu,sigma,(1,self.length))
        if normalized:
            self.normalize()

    def normalize(self,norm='l2'):
        """
        正则化
        """
        self.vector = preprocessing.normalize(self.vector, norm=norm)
        
    def to_coo(self):
        """
        转化为稀疏形式
        """
        if isinstance(self.vector,coo_matrix):
            return
        row = [0 for i in xrange(self.length)]
        col = [i for i in xrange(self.length)]
        data = [self.vector[0][i] for i in xrange(self.length)]
        self.vector = coo_matrix((data, (row, col)),shape=(1, self.length))

    def multiply(self,vector):
        """
        向量乘法
        """
        if isinstance(vector,int) or isinstance(vector,float):
            result = Vector(self.length)
            result.vector = vector*self.vector
            return result
        if not isinstance(vector,Vector):
            raise Exception('Vector Type Error!')
        if vector.length != self.length:
            raise Exception('Vector Length:(%s,%s) Not Match!' % (self.length,vector.length))
        
        
        
        if isinstance(self.vector,np.ndarray) and isinstance(vector.vector,np.ndarray):
            return float(sum(sum((self.vector*vector.vector))))
        
        self.to_coo()
        vector.to_coo()
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
        result = Vector(self.length)
        if isinstance(vector,int) or isinstance(vector,float):
            result.vector = vector+self.vector
            return result
        if not isinstance(vector,Vector):
            raise Exception('Vector Type Error!')
        if vector.length != self.length:
            raise Exception('Vector Length:(%s,%s) Not Match!' % (self.length,vector.length))

        result = Vector(self.length)

        if isinstance(self.vector,np.ndarray) and isinstance(vector.vector,np.ndarray):
            result.vector = self.vector+vector.vector
            return result
        self.to_coo()
        vector.to_coo()
        result = Vector(self.length)
        result.vector = self.vector+vector.vector
        return result

    def __sub__(self,vector):
        """
        减法
        """
        result = Vector(self.length)
        if isinstance(vector,int) or isinstance(vector,float):
            result.vector = self.vector - vector
            return result
        if not isinstance(vector,Vector):
            raise Exception('Vector Type Error!')
        if vector.length != self.length:
            raise Exception('Vector Length:(%s,%s) Not Match!' % (self.length,vector.length))

        result = Vector(self.length)

        if isinstance(self.vector,np.ndarray) and isinstance(vector.vector,np.ndarray):
            result.vector = self.vector-vector.vector
            return result
        self.to_coo()
        vector.to_coo()
        result = Vector(self.length)
        result.vector = self.vector-vector.vector
        return result
    
    def __getitem__(self,index):
        """
        取值
        """
        if index >= self.length or index < 0:
            raise Exception("Index Error!")
        if isinstance(self.vector,np.ndarray):
            return self.vector[0][index]
        elif isinstance(self.vector,coo_matrix):
            return self.vector.coeffRef(0,index)
        
    def __len__(self):
        return self.length
    
    
    def to_rdd(self,spark_context,row_id=0,partitions=100):
        """
        转化为rdd
        """
        v = None
        if isinstance(self.vector,coo_matrix):
            v = self.vector.todense()[0]
        elif isinstance(self.vector,np.ndarray):
            v = self.vector[0]
        return spark_context.parallelize(filter(lambda x:x[1] != 0,map(lambda _:((row_id,_[0]),_[1]),enumerate(v))),partitions)
        
    def __del__(self):
        del self.vector
   
    def norm(self):
        if isinstance(self.vector,coo_matrix):
            return float(sp.linalg.norm(self.vector))
        elif isinstance(self.vector,np.ndarray):
            return float(np.linalg.norm(self.vector))
 
        
if __name__ == '__main__':
    v = Vector(100)
    v.ones()
    

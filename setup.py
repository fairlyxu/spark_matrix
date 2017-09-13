# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 09:29:40 2017

@author: ligong

@dewcription:这是把自定义的模块打包成egg
"""
from setuptools import setup  
  
setup(  
    name='Matrix',  
    version='1.0',  
    description='package for spark matrix',  
    packages=[  
        'Matrix'  
    ],  
    py_modules=[  
        'SparseVector'  
    ]  
)  

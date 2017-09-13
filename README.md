# spark matrix
## 代码说明
1. 代码中主要有如下的几个类：
```
MatrixInRDD:这是用spark中rdd实现的矩阵
1) 可以从文本中进行导入
    数据格式是每一行：
        item,items,....(用逗号分隔)
    item里面有三种格式
        1) row_id;col_id;value，分别代表行号，列号，值，用分号分隔
        2) 第一个item是行号，其余是col_id;value，分别代表列号和值，用分号分隔
        3) 与上一种类似，只不过第一个item代表列号，相应的其余的是行号和值，用分号分隔
    
    上述三种格式对应的编号是：
        MatrixInRDD.PAIR
        MatrixInRDD.ROW
        MatrixInRDD.COL
    
```
2) 支持格式之间的相互转换

```
  MatrixInRDD.PAIR,MatrixInRDD.ROW,MatrixInRDD.COL三种格式之两两相互转换
```

3) BlockMatrixInRDD,RowMatrixInRDD继承MatrixInRDD

4) Vector是实现的列向量，支持加法和乘法

      BlockMatrixInRDD是将矩阵进行分块，主要可以用于大规模稀疏矩阵乘法

      RowMatrixInRDD是将矩阵进行分行，主要用于矩阵与向量之间的乘法

5) 同时支持矩阵的一些操作，比如：转置，取模，求迹

## 相关算法  
1.  利用Lanczos算法实现了svd分解，可以求前K个最大的奇异值，以及对应的左右变换矩阵  
2.  添加协同过滤算法，实现item与item之间的相似度
3.  添加梯度下降，实现了梯度下降和批量梯度下降


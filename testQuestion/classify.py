#classify.py
import numpy as np
import matplotlib.pyplot as plt
import math
"""
 在二维平面上，分别定义中心点（均值）和离散程度（方差）不同，
 且符合正态分布的两个集合（5000个离散点），
 利用交叉熵和K-近邻算法进行分类，并比较两者有什么不同。
"""
X = np.arange(-2500, 2500)
Y = [np.random.normal(-100, 40) for x in X]
Z = [np.random.normal(100, 40) for x in X]
plt.plot(X,Y,'ro')
plt.plot(X,Z,'bo')
plt.show()

u_Y=-100
u_Z=100
def gen_knnclassify(y):
    if abs(y+100)>=abs(y-100):
        print("blue")
        return 1
    else:
        print("red")
        return 0

error_sum=0
for i in range(5000):
    print("红点{}被分类为：".format(i))
    if (gen_knnclassify(Y[i]))==1:
        error_sum+=1

for i in range(5000):
    print("蓝点{}被分类为：".format(i))
    if (gen_knnclassify(Z[i]))==0:
        error_sum+=1
        
print("产生了{}次错误分类".format(error_sum))


#这个二分类问题中正确答案是（1,0） 预测结果是（p,1-p）p=1-error_sum/1w 交叉熵是-(1*logp)
p=1-error_sum/10000
H=-(math.log(p))
print("该算法分类非真实概率q与真实概率p之间的交叉熵H是：{}".format(H))


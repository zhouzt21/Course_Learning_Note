# Feature Engineering (L2+L4)

## 0 Outline

* L2 Feature Engineering I:to **learn lower dimensional representations** of the raw data

  * general features: PCA

  * detecting features
  * describing features

  * learnt features

  * Features are parts or patterns of an object in an image that help to identify it;  Features include properties like corners, edges, regions of interest points, ridges, etc.

* L4 Feature Engineering II : to **analyze various features statistically** for machine vision tasks.
  * Linear Regression
    * LSE
    * GD(SGD, ……)

  * Binary Classification
    * SVM
    * Logistic regression

  * Multi-class Classification
    * Softmax regression
    * K clustering

----

## L2

### 1 General features: PCA

#### PCA thoerem

![image-20230925145307972](.\asset\PCA_T1.png)

![image-20230925145334579](.\asset\PCA_T2.png)

![image-20230925145357153](.\asset\PCA_T3.png)





![image-20230925145216163](.\asset\PCA-2.png)

### 2 Detecting features: corner detection

#### 2.1 Harris Corner Detector

![image-20230925145044102](.\asset\image-20230925145044102.png)

![image-20230925145118010](.\asset\image-20230925145118010.png)

![image-20230925145144102](.\asset\image-20230925145144102.png)

* 只能找到位置（定位），还需要特征的描述
* 对于旋转鲁棒，但是对于放缩不鲁棒（放大之后不一定是边缘），所以要加上拉普拉斯滤波

#### 2.2  Laplacian filter for scale selection

**Highest response** when the signal has

**the same characteristic scale** as the filter

### 3 Describing features

#### 3.1 SIFT

（需要看论文补充学习）

Image patch -->image gradients --> color histogram (利用颜色占比) --> Spatial histograms -->**SIFT(Scale Invariant featurer transform)**

* 图像局部特征描述子：旋转 尺度 亮度 不变

#### （1）multi-scale extrema detection 

(尺度，优于拉普拉斯滤波)

* **Difference of Gaussian(DOG)**是高斯函数的差分。它是可以通过将图像与高斯函数进行卷积得到一幅图像的低通滤波结果，即去噪过程，这里的Gaussian和高斯低通滤波器的高斯一样，是一个函数，即为正态分布函数。同时，它对高斯 拉普拉斯LoG的近似，在某一尺度上的特征检测可以通过对两个相邻高斯尺度空间的图像相减，得到DoG的响应值图像。

* 根据理论：三维图中的最大值和最小值点是角点：Detect maxima and minima of DOG in the scale space：Each point is compared to 8 neighbors in current image and 9 neighbors each in the scales above and below (26 in total)

  ![image-20230925164346395](.\asset\gaussian_scaler.png)

#### （2）keypoint localization （位置）

关键点的精确定位，候选关键点是DOG空间的局部极值点，而且这些极值点均为离散的点，精确定位极值点的一种方法是，对尺度空间DoG函数进行曲线拟合，计算其极值点，从而实现关键点的精确定位。

#### （3）orientation assignment （旋转）

* 不理解为什么？

![image-20230925173213243](.\asset\SIFT_3.png)

#### （4）keypoint descriptor （描述）

![image-20230925172757860](.\asset\SIFT_4.png)

#### 3.2 SURF

**SURF is a speeded-up (3 times faster) version of SIFT..** 

* Good at handling images with blurring and rotation
* Not good at handling viewpoint change and illumination change

![image-20230925153410456](.\asset\SURF.png)

#### 3.3 other: 

#### Haar-like Features



#### HOG (Histogram of Oriented Gradients for Human Detection)

* HOG counts **occurrences of gradient orientation** in localized portions of an image
* It’s computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy



![image-20230925155028462](.\asset\feature_summary.png)

* OGB: 经常用于机器人中

### 4 Learnt features

local feature based on learning

#### LIFT: Learned Invariant Feature Transform

(ECCV 2016)

* 用learning替代SIFT
* 类end-to-end

#### SuperPoint: Self-Supervised Interest Point 

Detection and Description (CVPR 2018)

* 基于SIFT
* FCN   VGG

#### LF-Net: Learning Local Features from Images 

(NeurIPS 2018)



#### Deep Graphical Feature Learning for the 

Feature Matching Problem (ICCV 2019)







**Pros: inheriting advantages from CNNs**

• More robust to scale, occlusion, deformation, rotation, etc.

• Pushed the limits of what was possible using traditional computer vision techniques

**Cons: inheriting disadvantages from learning methods**

• Time consuming: Need dataset for network training

• Not universal: fitting to a certain distribution, not reliable when generalizing to other scenario



## L4 

### 1 Linear Regression

线性规划问题

基本优化问题：

**LSE**

![image-20231016140202026](C:\Users\zzt\AppData\Roaming\Typora\typora-user-images\image-20231016140202026.png)

闭式解：

![image-20231016140331222](D:\My_desktop\媒体与认知\note\asset\20231016_mse_solution.png)

数值解：

* 梯度下降

![image-20231016140611426](D:\My_desktop\媒体与认知\note\asset\20231009_GD.png)

dynamic learning rate

SGD

![image-20231016140820155](C:\Users\zzt\AppData\Roaming\Typora\typora-user-images\image-20231016140820155.png)









## KNN

K 近邻 (k - Nearest N eighbo r ，简称 kNN ) 学 习是一种常 用的监 督学习 方法，

给定 测试样本 基于某种距离度量找出训练集中与其最靠近的 k 个训练样本 ，然后 基于这 k 个" 邻居 "的信息来进行预测

在分类任务中可使用**"投票法" **即选择这 k 个样本中出现 最多的类 别标记 作为预测结果;

在回归任务中时使用**"平均法"** ，即将 这 k 个样本 的实值 输出标记 的平均值作为预测结果;还可基于距离远近进行加权平均或加权投票 ，距离越近的样本权重越大.

没有显式的训练过程!事实上，它是"懒惰学习" (lazy learning ) 的 著名代表，此类学习技术在**训练阶段**仅仅是把样本**保存**起来，训练时间开 销 为零，待 收到**测试样本**后再进行**处理**;相应的，那些在**训练阶段**就对样本**进行学习**处理 的方法，称为"急切学习" (eager learning) .





## 2 Binary Classification

### SVM

![image-20231016154742589](C:\Users\zzt\AppData\Roaming\Typora\typora-user-images\image-20231016154742589.png)

![image-20231016154820130](D:\My_desktop\媒体与认知\note\asset\20231009_SVM2)

![image-20231016154849561](D:\My_desktop\媒体与认知\note\asset\20231009_SVM3)



转化为凸优化问题 

SMO（sequential minimal optimization，序列最小优化）算法是一种启发式算法。其基本思路是：如果**所有的变量的解都满足此最优问题的KKT条件**，那么这个最优问题的解就得到了，因为**KKT条件是该最优化问题有解的充要条件**。否则，我们就**选择两个变量，固定其它变量**，针对这两个变量构建一个二次规划问题，这个二次规划的问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变小。
原文链接：https://blog.csdn.net/Cyril_KI/article/details/107779454

CS229补充学习





二次规划（*Quadratic programming*）



### Logistic regression







## 3 multi-classification

### Softmax regression

logistic function (in binary) --> softmax (multi-class)

没有闭式解



上述没有ground-truth时，使用聚类的办法(umsupervised)

### K-means clustering









---

quiz: 20min

* 训练和测试的区分？
* 注意框架
* 注意二维降一维的函数操作




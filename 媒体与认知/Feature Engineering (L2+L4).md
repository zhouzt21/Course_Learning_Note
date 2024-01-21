# Feature Engineering (L2+L4)

## 0 Outline

* L2 Feature Engineering I:to **learn lower dimensional representations** of the raw data

  1. general features: PCA

  2. detecting features： corner detection

     2.1 Harris Corner Detector

     2.2 Laplacian filter for scale selection

  3. describing features： 

     3.1 SIFT

     3.2 SURF

     3.3 Others

  4. learnt features

* L4 Feature Engineering II : to **analyze various features statistically** for machine vision tasks.
  1. Linear Regression
  
     * LSE
     * GD(SGD, ……)
  
  2. Binart Classification
  
     2.1 SVM
  
     2.2 Logistic regression
  
  3. Multi-class Classification
  
     3.1 Softmax regression
  
     3.2 KNN
  
     3.3 K-means clustering

----

## L2 Feature Engineering I

### 1 General features: PCA

#### PCA thoerem

* **Unsupervised** technique for extracting variance structure from high dimensional datasets.

*  look into the **correlation** between the points
  * By finding the **eigenvalues** and **eigenvectors** of the **covariance matrix**, the eigenvectors with **the largest eigenvalues** correspond to the dimensions that have **the strongest correlation** in the dataset. (This is the **principal component**)

![image-20230925145307972](.\asset\PCA_T1.png)

![image-20230925145334579](.\asset\PCA_T2.png)

至此建立了协方差矩阵。接下来开始计算特征值和特征向量。

![image-20230925145357153](.\asset\PCA_T3.png)



PCA使用的是每一个点都可以写成均值加上各个特征向量的线性叠加。可以直接对于那些特征值较小的特征向量直接舍去来压缩数据。

PCA的应用：特征脸

**Method A:** Build a PCA subspace for each person and check which subspace can reconstruct the test image the best 

**Method B:** Build one PCA database for the whole dataset and then classify based on the weights.

![image-20230925145216163](.\asset\PCA-2.png)

* PCA is a universal feature analysis method. It works well for both 1D data like ‘voice’, 2D data like ‘image’, and even high dim data like word files and statics.
* However, PCA cannot detect features that human really cares about, such as ‘edges’, ‘bright spot’, and ‘corners

### 2 Detecting features: corner detection

#### 2.1 Harris Corner Detector

![image-20240112172309116](.\asset\harris.png)

注意是对于图像的梯度进行主成分分析。

![image-20230925145044102](.\asset\image-20230925145044102.png)

![image-20230925145118010](.\asset\image-20230925145118010.png)

上述第五步构造的式子来源于下面：（这个式子用来检测出角、边）

![image-20230925145144102](.\asset\image-20230925145144102.png)

* 只能找到位置（定位），还需要特征的描述
* 对于旋转鲁棒，但是对于放缩不鲁棒（放大之后不一定是边缘），所以要加上拉普拉斯滤波

#### 2.2  Laplacian filter for scale selection

**Highest response** when the signal has **the same characteristic scale** as the filter

### 3 Describing features

找到图片特征描述子的思路：image patch --(减少绝对大小的依赖)-->image gradient --(减少对形变的依赖)-->color histogram --(体现空间特征)-->spatial histogram--(完全抗旋转)-->SIFT(Scale Invariant featurer transform)

#### 3.1 SIFT

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

* Good at handling images with **blurring and rotation**
* **Not good** at handling **viewpoint change** and **illumination** change
* SURF approximates DOG in SIFT with **Box Filter**
  * Convolution with box filter can be easily calculated via **integral images**
  * It can be done in parallel for different scales.


![image-20230925153410456](.\asset\SURF.png)

可以通过这个表达式计算出任意一块区域内的值总和。

#### 3.3 Others

##### Haar-like Features (the first real-time face detector)

* Consider adjacent rectangular regions at a specific location in a detection window
* Sum up pixel intensities in each region and calculate the difference between these sums
* Dark region subtract white region
* **Advantage in calculation speed:**
  * Due to the use of *integral images*, a Haar-like feature of any size can be calculated in constant time.

##### HOG (Histogram of Oriented Gradients for Human Detection)

* HOG counts **occurrences of gradient orientation** in localized portions of an image
* It’s computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for improved accuracy

* is widely applied in** **pedestrian detection/human detection**

![image-20230925155028462](.\asset\feature_summary.png)

* OGB: 经常用于机器人中

### *4 Learnt features

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

#### Deep raphical Feature Learning for the 

Feature Matching Problem (ICCV 2019)



**Pros: inheriting advantages from CNNs**

• More robust to scale, occlusion, deformation, rotation, etc.

• Pushed the limits of what was possible using traditional computer vision techniques

**Cons: inheriting disadvantages from learning methods**

• Time consuming: Need dataset for network training

• Not universal: fitting to a certain distribution, not reliable when generalizing to other scenario



## L4 Feature Engineering II

### 1 Linear Regression

线性规划问题

基本优化问题：

**LSE**

![image-20231016140202026](.\asset\image-20231016140202026.png)

闭式解：

![image-20231016140331222](.\asset\20231016_mse_solution.png)

数值解：

* 梯度下降

![image-20231016140611426](.\asset\20231009_GD.png)

dynamic learning rate

SGD

![image-20231016140820155](.\asset\image-20231016140820155.png)



### 2 Binary Classification

#### 2.1 SVM

![image-20231016154742589](.\asset\image-20231016154742589.png)

![image-20231016154820130](.\asset\image-20231016154820130.png)

![image-20231016154849561](.\asset\image-20231016154849561.png)

以上则已经转化为凸优化问题。

![image-20240112225017063](.\asset\SVM_dual.png)

以上在构建损失函数的时候加上了前述约束条件，构建拉格朗日乘子。

![image-20240112224008247](.\asset\SMO.png)

（注：二次规划（*Quadratic programming*））

此时转化为两两变量迭代问题的解决

* SMO（sequential minimal optimization，序列最小优化）算法是一种启发式算法。其基本思路是：如果**所有的变量的解都满足此最优问题的KKT条件**，那么这个最优问题的解就得到了，因为**KKT条件是该最优化问题有解的充要条件**。否则，我们就**选择两个变量，固定其它变量**，针对这两个变量构建一个二次规划问题，这个二次规划的问题关于这两个变量的解应该更接近原始二次规划问题的解，因为这会使得原始二次规划问题的目标函数值变小。
  原文链接：https://blog.csdn.net/Cyril_KI/article/details/107779454
  * 基本思想：Need to solve n  values à --> construct relations between all à --> find the exact value for one à --> all n  values determined.

CS229补充学习

* 测试/预测时候的判断方法

  ![image-20240112232403733](.\asset\SVM_test.png)

* **Kernel Trick: Linear -> Non-linear**
  * SVM还可以通过引入核函数解决非线性分类问题
  * ![image-20240112232545898](.\asset\SVM_nonlinear.png)

![image-20240112232828118](.\asset\SVM_kernel.png)

#### 2.2 Logistic regression

![image-20240112233025230](.\asset\logistic_regression.png)

![image-20240112233400987](.\asset\logistic2.png)

注意向量的 似然直接相乘。

逻辑回归对于对数似然参数没有闭式解（但是线性回归有）

### 3 multi-classification

#### 3.1 Softmax regression

logistic function (in binary) --> softmax (multi-class)

![image-20240112233725817](.\asset\softmax.png)



上述没有ground-truth时，使用聚类的办法(umsupervised)

#### 3.2 KNN（K 近邻）

* 是一种常用的监督学习方法，是分类算法

* K值含义 - 对于一个样本X，要给它分类，首先从数据集中，在**X附近找离它最近的K个数据**点，将它**划分为归属于类别最多**的一类

  * 在分类任务中可使用**"投票法" **即选择这 k 个样本中出现 最多的类 别标记 作为预测结果;

  * 在回归任务中时使用**"平均法"** ，即将 这 k 个样本 的实值 输出标记 的平均值作为预测结果;还可基于距离远近进行加权平均或加权投票 ，距离越近的样本权重越大.

* 没有显式的训练过程
  * 是"懒惰学习" (lazy learning ) 的 著名代表，此类学习技术在**训练阶段**仅仅是把样本**保存**起来，训练时间开 销 为零，待 收到**测试样本**后再进行**处理**;相应的，那些在**训练阶段**就对样本**进行学习**处理 的方法，称为"急切学习" (eager learning) .

#### 3.3 K-means clustering

![image-20240112234005286](.\asset\kclustering.png)



* 是无监督学习，是聚类算法（和KNN不同；唯一相似点为：算法都包含给定一个点，在数据集中查找离它最近的点的过程。）
* 有明显的训练过程
* K值含义- K是事先设定的数字，将数据集分为K个簇，需要依靠人的先验知识



---

quiz: 20min

* 训练和测试的区分？
* 注意框架
* 注意二维降一维的函数操作




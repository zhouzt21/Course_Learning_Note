# Deep Neural Networks (L5+L6+L8)

## 0 outline

**L5 Deep Neural Network I_DNN**

1. building nn
2. training nn
3. advanced technique
4. batch normalization, ……

**L6 Deep Neural Network Il CNN(Convolutional)**

1. concept
2. architecture
3. Application
4. Important networks
5. deep learning program tutorial: pytorch

**L8 Deep Neural Network Il -RNN (Recurrent)**

1. concept +architeture
2. LSTM (long short-term memory RNNS)
3. Transformer
   3.1 Attention in DL
   3.2 Attention in Transformer
   query, key, value
   self-attention
   positional encoding
   multi-head
4. Transformer VS RNN
5. Advanced: ViT or CNN
   ViT
   Swin Transformer : Tailored ViT


---

### L5 Deep Neural Network I __ DNN

#### 1. building nn: 

1. perceptron   (weight bias activation function)
2. forward propagation
3. activation function

#### 2. trainning nn:

1. backpropagation of error (with GD)

   ![image-20231030164558677](.\asset\20231016_bp.png)
   
   * pros -- **Re-use derivatives** computed for higher layers in computing derivatives for lower layers to minimize computation
   * cons -- **Technical** view: trapped into **local minima** 

#### 3. advanced techiques

practically training DNN with BP is difficult ;  Loss functions can be difficult to optimize + Dealing with overfitting

1. adaptive **learning rate**

2. **advanced GD**: momentum  RMSProp Adam  (具体见视听导笔记优化部分)

3. **regularization**: L1/L2 , dropout,early stopping

   * **Regularization:**Prevent the model from doing too well on  training data

   ![image-20231106141403826](.\asset\20231016_regularization_l1l2.png)

   * **dropout** : During training, randomly set some activations to 0
   * **early stopping**: Stop training before having a chance to overfit

####  4. batch normalization, hyperparameters, initialization

![image-20231106141646406](.\asset\20231016_batch_normalization.png)

* **batchnorm**使得网络中每层输入数据的分布相对稳定，加速模型学习速度
* **batchnorm**使得模型对网络中的参数不那么敏感，简化调参过程，使得网络学习更加稳定
  * 权重的缩放值会被“抹去”，因此保证了输入数据分布稳定在一定范围内。
* **batchnorm**允许网络使用饱和性激活函数（例如sigmoid，tanh等），缓解梯度消失问题
  * 在不使用**batchnorm**层的时候，由于网络的深度与复杂性，很容易使得底层网络变化累积到上层网络中，导致模型的训练很容易进入到激活函数的**梯度饱和区**；通过normalize操作可以让激活函数的输入数据落在梯度非饱和区，缓解梯度消失的问题；另外通过**自适应**学习 𝛾, 𝛽 又让数据保留更多的原始信息。
*  **batchnorm**具有一定的正则化效果
  * 由于使用mini-batch的均值与方差作为对整体训练样本均值与方差的估计，尽管每一个batch中的数据都是从总体样本中抽样得到，但**不同mini-batch的均值与方差会有所不同**，为网络的学习过程中增加了随机噪音，与Dropout通过关闭神经元给网络训练带来噪音类似，在一定程度上对模型起到了正则化的效果。
  * 注意这里用来归一处理的均值和方差来自于本批次的数据，而不是全部数据。

* **A few considerations about batch norm:**
  * Often a **larger learning rate** with batch norm is used
  * Models with batch norm can train **much faster**
  * Generally **requires less regularization** (e.g. w/o. dropout)
  * Good idea in many cases

* hyperparameters   
  * **Optimization related:**
    * batch size
    * learning rate
    * momentum
    * initialization
    * batch normalization
  * **Generalization related:**
    * Dropout
  * **Both related:**
    * Architecture
    * and size of layers
    * activation

---

### L6 Deep Neural Network II __CNN(Convolutional)

#### 1. concept

* convolution: exploit spatial structures and hierachical pattern

  * shared connections / spatial filtering 

*  stack of Conv, Pool, FC Layers

  * smaller filters and deeper architectures
  * getting rid of fully concolutional layers

  * local connectivity (fully connectivity: MLP)

  * receptive field (the  region of visual space): hierachical organization

#### 2. architecture

* strided convolution: ouput size: **(N - F) / stride + 1**\
* zero padding:  **(N + 2P - F) / stride + 1**
* pooling: max / average pooling
* FC: fully convolutional

#### 3. Application

feature learning + classification

![image-20231030170214698](.\asset\20231016_cnn_architecture_1.png)



![image-20231030170251063](.\asset\20231016_cnn_architecture_2.png)

A practical issue: insufficient training data for very deep network?

Solution: **data augmentation**

![image-20231030170402023](.\asset\20231016_cnn_architecture_3.png)

* rotation: uniformly chosen random **angle** between 0° and 360°
* translation: random **translation** between -10 and 10 pixels
* rescaling: random **scaling** with scale factor between 1/1.6 and 1.6
* flipping: yes or no (bernoulli)
* shearing: random **shearing** with angle between -20° and 20° （裁剪
* stretching: random **stretching** with stretch factor between 1/1.3 and 1.3
* cropping: **crop a for random range** from the center
* color jitter: hue, saturation, brightness, contrast, …
* add noise, blur, cutout, …**

#### 4. Important  networks:

* **AlexNet(8 layers):** 
  * heavy data augmentation, SGDMomentum, L2weight decay
  * **first ReLu**显著提高了收敛速度。
  * 实现并开源了 cuda-convnet，使得在 **GPU** **上训练大规模神经网络**变得可行。
* **VGGNet(16/19 layers):**
  *  receptive field
    * 不同层次的特征之间有一个简单的比例关系

  * 适用于位置敏感的下游任务，如检测、分割等
  * more non-linearities（**Small** filters, **Deeper** networks）
    * 充分使用**3×3 的卷积核，**步长1和填充1，以保持空间维度不变
      * 三个 3x3 的卷积层（步幅为 1）堆叠具有与一个 7x7 的卷积层相同的有效感受视野—— 更小的滤波器，但更深，非线性性更强
      * 它不仅证明了深度对于分类准确性是有益的，还证明了3x3卷积核足够优秀。

    * 所有隐藏层都使用 **ReLU**

  * 空间池化由**五个最大池化层**完成，以 2×2 像素窗口执行，步幅为 2
  * 前两个**全连接（FC）层**各有 4096 个通道，第三个执行 1000 类别的 ILSVRC 分类，因此包含 1000 个通道（每个类别一个通道）。
  * 缓解过拟合
    * 图像增强：224x224 输入是从输入图片中随机裁剪的，随机水平翻转和随机 RGB 颜色偏移
    * dropout：前两个全连接层的 dropout 正则化

  * SGD：大动量，小权重衰减

* GooleNet(22 layers): 
  * 更深，更高效
  * **parallel filter** 
    * 对来自前一层的输入应用并行的滤波操作：卷积多个感受野尺寸
    * 以合并特征通道的方式将所有滤波器输出连接在一起。
    * 池化层还保留了原通道数，这意味着连接后的总通道数只能不断增长
      * 解决方案：使用 1x1 卷积来减少特征通道大小的“瓶颈”层（保留空间维度，减少通道数）（使用了 1x1 的瓶颈卷积和全局平均池化代替全连接层）
      * ![image-20240113205654470](.\asset\bottleneck.png)

  * 采用 **Inception** **模块**
    * 设计一个良好的局部网络拓扑结构（网络内的网络），然后将这些模块叠加在一起；辅助分支也进行分类的训练，向网络中传递更多的梯度
    * **Inception v1:**
      * 随机梯度下降（SGD）
      * 动量 (Momentum) 0.9    学习率每8轮下降4%
      * 通过辅助旁路分类结果传播更多梯度

    * **Inception v2:**
      * 用两个3x3卷积代替5x5卷积（减少参数量、减轻过拟合）
      * 使用了**Batch Normalization**，显示了BN的优异性能 自此之后，BN作为正则化方式被大量使用

    * **Inception v3:**
      * 将nxn卷积拆成1xn和nx1卷积的结合（节约参数、加速计算、增加非线性）

    * **Inception v4:** 结合ResNet

  * 分类层仅包含 1 个全连接层，减少参数数量 **7M** (AlexNet 62M、VGG 138M)

* ResNet:
  *  **residual net** ：Add more **direct connections** (thus enabling easier gradient flow towards lower layers)
  *  使用网络层来拟合残差映射，而不是直接尝试拟合所需的底层映射
* Deformable ConvNet 可形变卷积
  * 学习如何在卷积中对采样位置进行形变
  * 实现卷积神经网络中**对空间变换的有效建模**
    * 无需额外监督来学习空间变换
    * 在复杂的视觉任务中显著提高准确性

* GAN(generative adversarial networks): to sample from simple distribution, learn **transformation** to train distribution;  use a discriminator network to tell  ("two- player game "---descriminator & generator)
  * Discriminator**: **distinguish between real and fake images (convolution network)
  * Generator**: **fool the discriminator by generating real-looking images (unsampling network with fractionally strided convolutions)
* **CNNs Outperform MLPs on Images?**
  * CNNs: **local connectivity**, exploit spatial structures and hierarchical pattern in data
  * MLPs: **fully connectivity** (each neuron in one layer is connected to **all** neurons in the next layer, easy to overfit)
* **GAN**   (remain)
  * Goodfellow, Ian, et al. "Generative adversarial networks." *Communications of the ACM* 63.11 (2020): 139-144.

![image-20231106144106685](.\asset\20231023_GAN.png)

![image-20231106144234998](.\asset\20231023_GAN2.png)

regression problem

#### 5. deep learning program tutorial: pytorch

**Diffusion models** are currently exploding state-of-the-art beating GANs

---

### L8 Deep Neural Network III ——RNN （Recurrent）

#### 1. concept +architeture

* **Sequential** modeling: **previous hidden states affect** subsequent hidden states

* Process input sequences of **arbitrary length**

![image-20231106144540296](.\asset\20231106_RNN.png)

隐藏层不断结合以前的隐藏层和当前输入来更新，输入通过对隐藏层softmax归一化得到。
$$
loss\space fuction:\space J(\theta)=\frac{1}{T}\sum_{t=1}^T J^{(t)}(\theta)\\
=-\frac{1}{T}\sum_{t=1}^T\sum_{i=1}^N y_i^{t}\space log\widehat y_i^{(t)}
$$
![image-20231106144758775](.\asset\20231106_RNN_BP.png)

这里每一时刻的损失函数是从开始时刻一直到当前时刻输出的交叉熵损失，在使用反向传播计算的是偶的，应该将每一时刻的单个输出的交叉熵损失，对隐藏层进行求导（由于链式法则按道理应该加上这一时刻的隐藏层对于现在的隐藏层的偏导，但是因为隐层一直不变，所以这一项是1），最后相加。

* 对于RNN的评价
  * 理论上可以使用很多步之前的信息，处理任意长度的输入而同时模型的规模并不会因此而增加
  * 但是实践中会遇到梯度消失/爆炸，并不能获得多步之前的信息；同时循环计算是很慢的

![image-20231106144838688](.\asset\20231106_prosandcons.png)

#### 2. LSTM (long short-term memory RNN)

* 基本结构
  * Cell state: run straight down the entire chain, with only **minor linear** **interactions**. It’s very easy for information to just flow along it unchanged
  * The cell stores **long-term information** LSTM can **read**, **erase**, and **write** information from the cell
  * Gate: optionally let information go through cell state

* ![image-20231106145108677](.\asset\20231106_LSTM1.png)
* 遗忘门处理过去的信息，输入门处理新信息

![image-20231106145129757](.\asset\20231106_LSTM2.png)

![image-20231106145307379](.\asset\20231106_LSTM3.png)

* gate: propobility    weight 

#### 3. Transformer

##### 3.1 **Attention in DL:**

 estimate attention vector, reflecting how strongly it is **correlated** with (“attends to”) other elements

task1: translation

![image-20231106150951515](.\asset\20231106_seq2seq.png)

* 一般RNN的编码器和解码器：编码器是处理输入序列并全压缩成一个vector，表达所有位置权值的信息context vector；解码器是结合初始状态和context vector依次来得到输出
* 改进：希望不同的位置的context vector不同（虽然每个位置都需要包含之前的信息）

![image-20231106151102402](.\asset\20231106_attention.png)

* \alpha是当前i位置和t位置 的相关程度（但是是用前一时刻得到这个信息）
* 再简单使用\alpha 和隐层hi 来算当前的context vector （用的是简单的网络），这样就包含了
* 注：相当于输入一方面成为隐层的初始化，一方面是attention机制用来产生相关性的“query”的依据

![image-20231106151821057](.\asset\20231106_attention2.png)

task2: visualization of attention weights

* **CNN+RNN pipeline** **with attention**
  * Attention idea: **new context vector** at every time step 
  * Each context vector will **attend to** different image regions
  * Multiple **query vectors**, each **query** creates a new output context vector

* 以上是首次提出的attention，但是是基于RNN，还不是transformer ，接下来引入Q K V



##### 3.2 Attention in Transformer

###### query, key, value

* **可以将Attention机制看作一种软寻址（Soft Addressing）:**
  Source可以看作存储器内存储的内容，元素由地址Key和值Value组成，当前有个Key=Query的查询，目的是取出存储器中对应的Value值，通过Query和存储器内元素Key的地址进行相似性比较来寻址，之所以说是软寻址，指的不像一般寻址只从存储内容里面找出一条内容，而是可能从每个Key地址都会取出内容，取出内容的重要性根据Query和Key的相似性来决定，之后对Value进行加权求和，这样就可以获得最终输出的Value值。

  * 在机器翻译的任务中，Source中的Key和Value合二为一，指向的是同一个东西，也即输入句子中每个单词对应的语义编码

  * alignment:  correlation的程度（通过计算query和每个key的相似程度给出对应的value）可以通过向量点乘、求两者余弦相似性或者引入额外的神经网络来实现
    $$
    Attention\space(Query, Source) = \sum_{i=1}^{L_x}Softmax(Similarity(Query,Key_i))*Value_i
    $$

* 计算过程如下：![query](.\asset\query.png)
  * 注意计算过程第一阶段产生相似性计算的分值通过softmax归一化统一在0-1之间，得到的权重结果(a)对value进行加权，得到最后的attention结果

![image-20231106162638387](.\asset\20231106_attentionQKV.png)

* 上述图中展示的alignment计算就是q和k相乘这种形式相似度，后续归一化之后作为输出value的权重值
* 注意这里K 是一行是一个输入样本（而不是一列），注意观察这里的K和V都是通过输入

###### self-attention

* **key value和query**都来自于一个input vector，相比attention是改变了query的计算方式（也通过输入得到
  $$
  从矩阵角度的公式：Z=softmax(Q^T*K)*V
  $$
  

  ![image-20231106164120627](.\asset\20231106_self_attention2.png)

  * 在一层中，自注意力是global， CNN是local；（想用attention来做fully connected layer）![image-20240114165045519](.\asset\self_attention_CNN.png)

![image-20231106163451095](.\asset\20231106_self_attention.png)

###### positional encoding 

* 解决order的问题  （要知道现在处理的词在一句话中的位置，因为self-attention的运算是无向的，无法分辨）

  ![image-20231106164712619](.\asset\20231106_position_encoding.png)

  * 需要把所有的位置信息都存在了encoder里面
  * 因此，我们需要这样一种位置表示方式，满足于：
    （1）它能用来表示一个token在序列中的绝对位置
    （2）在**序列长度不同的情况**下，不同序列中token的相对位置/距离也要保持一致
    （3）可以用来表示模型在训练过程中从来没有看到过的句子长度。
  * 需要一个**有界又连续的函数，最简单的**，正弦函数sin就可以满足这一点
    * 周期性使得模型可以处理比训练时更长的输入序列**（外推性）**；缺点是不可学习，外推性还不够优秀
  * 在Transformer的论文中，比较了用positional encoding和learnable position embedding(让模型自己学位置参数）两种方法，得到的结论是两种方法对模型最终的衡量指标差别不大。不过在后面的**BERT中，已经改成用learnable position embedding的方法了**，也许是因为positional encoding在进attention层后一些优异性质消失的原因（猜想）。

###### multi-head

 在不同的channel上做attention  

* **为注意力层提供了多个“表示子空间”**。 对于多头注意力，我们不仅有一个，而且还有多组Query/Key/Value权重矩阵，这些权重矩阵集合中的每一个都是随机初始化的。 然后，在训练之后，每组用于将输入Embedding投影到不同的表示子空间中。

![image-20231106164408040](.\asset\20231106_multi_head.png)

* attention本身不是提取数据本身的特征，**而是算关联**；是从general意义上是可解释的
* q k就是Learnable的feature，attention就是建立两者的关联，然后对value做加权
* learn能做的事是本身就可以解释的；
  * 即使理论上MLP可以实现所有，但是就是有算力限制
  * 所以魔改的地方就是加上一些可解释的东西

#### 4. Transformer VS RNN

**RNNs**

* **Pros:** LSTMs work reasonably well for long sequences.

* **Cons:**
  *  Expects an ordered sequences of inputs
  * Sequential computation: subsequent hidden states can only be computed after the previous ones are done.

**Transformer**

* **Pros:**
  * Good at long sequences. Each attention looks at all inputs.
  * Can operate over unordered sets or ordered sequences with positional encodings.
  * Parallel computation: All alignment and attention scores for all inputs can be done in parallel.

* **Cons:** Require large memory: *N x M* **alignment** and **attention scalers** are calculated and stored for a single self-attention head. 

![image-20231106164254566](.\asset\20231106_transformer2.png)

* 整个结构的搭建：不是RNN，没有hidden layer的概念

![image-20231106152437052](.\asset\20231106_intuitive.png)



### 5. Advanced: ViT or CNN 

![image-20231106165157063](.\asset\20231106_advance.png)

#### ViT

首次用于视觉任务

证明视觉任务可以不适用CNN

问题：复杂度

global attention

#### Swin Transformer ：Tailored ViT

视觉比起语言更复杂的部分：

* scale

做法：

* window-based attention

不同层级之间使用不同分辨率的pattern，每一层中只使用两种pattern，交替计算防止边界不连续的情况

patch partition  + linear embedding

降低分辨率，提高attention window的感受野

rethink: swin transformer —— is CNN locality come back? 

* 重新带来了locality



**Recent renaissance of CNNs**

2022 google 重新用CNN打败了vit

convnext

* modernize a standards ConvNet 

google deepmind ： convnet match vision transformers at scale

* "compute is all you need?" : vit or cnn 不重要，capacity最重要
* local connection



**Conclusion: ViT or CNN? -- remember the core insights **

* spatial coherence

  * local connectivity
  * global long-term depency

* invariance/equivariance

  * multi scale
  * reanslation/rotation inva/equivariant operators
  * attention-based mechanism


# Deep Neural Networks (L5+L6+L8)

## 0 outline

* DNN
  * build: architecture
  * train
  * optimize
* CNN
  * concept
  * architecture
  * application
  * important networks (Alexnet ~……~  GAN)
  * pytorch
* RNN
  * build: concept +architeture
  * LSTM
  * attention
* GNN  (Graph)
  * Graph, graph notations & graph data representation
  * Building neural network for graph --- the GNN
  * Treating GNN in another aspect --- the Spectral Graph CNN
  * A glare of modern GNN architectures.


---

### L5 Deep Neural Network I __ DNN

1. building nn: 
   1. perceptron   (weight bias activation function)
   2. forward propagation
   3. activation function

2. trainning nn:
   1. backpropagation of error (with GD)

      ![image-20231030164558677](.\asset\20231016_bp.png)

3. advanced techiques

   practically training DNN with BP is difficult ;   pros: trapped into local minima

   1. adaptive learning rate
   2. advanced GD: momentum  RMSProp Adam  (具体见视听导笔记优化部分)
   3. regularization: L1/L2 , dropout,early stopping

      ![image-20231106141403826](.\asset\20231016_regularization_l1l2.png)
   4. batch mormalization, hyperparameters, initialization

      ![image-20231106141646406](.\asset\20231016_batch_normalization.png)

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



![image-20231030170402023](.\asset\20231016_cnn_architecture_3.png)

* rotation: uniformly chosen random **angle** between 0° and 360°
* translation: random **translation** between -10 and 10 pixels
* rescaling: random **scaling** with scale factor between 1/1.6 and 1.6
* flipping: yes or no (bernoulli)
* shearing: random **shearing** with angle between -20° and 20°
* stretching: random **stretching** with stretch factor between 1/1.3 and 1.3
* cropping: **crop a for random range** from the center
* color jitter: hue, saturation, brightness, contrast, …
* add noise, blur, cutout, …**

#### 4. Important  networks:

* AlexNet(8 layers): first ReLu, heavy data augmentation, SGDMomentum, L2weight decay
* VGGNet(16/19 layers): receptive field, fewer parameters, more non-linearities
* GooleNet(22 layers): deeper, high computational efficiency, parallel filter 
* ResNet: (gradient vanishing) deep residual learning --- direction connections; if optimal mapping is closer to identity, easier to find fluctuations ; if identity were optimal easy to set weight=0
* GAN(generative adversarial networks): to sample from simple distribution, learn **transformation** to train distribution;  use a discriminator network to tell  ("two- player game "---descriminator & generator)
  * Discriminator**: **distinguish between real and fake images (convolution network)
  * Generator**: **fool the discriminator by generating real-looking images (unsampling network with fractionally strided convolutions)


![image-20231106144106685](.\asset\20231023_GAN.png)

![image-20231106144234998](.\asset\20231023_GAN2.png)

regression problem

#### 5. deep learning program tutorial: pytorch



---

### L8 Deep Neural Network III ——RNN （Recurrent）

#### 1. concept +architeture

* **Sequential** modeling: **previous hidden states affect** subsequent hidden states

* Process input sequences of **arbitrary length**

![image-20231106144540296](.\asset\20231106_RNN.png)





![image-20231106144758775](.\asset\20231106_RNN_BP.png)



![image-20231106144838688](.\asset\20231106_prosandcons.png)

* vanishing: far away  --- missing long-term effect

  

#### 2. LSTM (long short-term memory RNNS)

* ![image-20231106145108677](.\asset\20231106_LSTM1.png)

![image-20231106145129757](.\asset\20231106_LSTM2.png)

![image-20231106145307379](.\asset\20231106_LSTM3.png)

* gate: propobility    weight 

#### 3. Attention

##### task1: translation

![image-20231106150951515](.\asset\20231106_seq2seq.png)

* 全压缩成一个vector表达所有位置权值的信息context vector；
* 改进：希望不同的位置的context vector不同（虽然每个位置都需要包含之前的信息）

![image-20231106151102402](.\asset\20231106_attention.png)

* \alpha是i位置和t位置（但是是用前一个位置来表达的s_{t-1}） 的相关程度
* score直接暴力用\alpha 和hi 来算 （用的是简单的网络）

![image-20231106151821057](.\asset\20231106_attention2.png)

task2: visualization of attention weights

* 以上是首次提出的attention，但是是基于RNN，还不是transformer ，接下来引入Q K V



2. query, key, value

   是基本的网络层中一层的计算操作

   * query从前面提取出信息（query input）

   * 多引入key 和 value，是从Input vector继续投影出的，在这两个上面计算

   * query 是词条，key是query同模态的东西；  key对应一个value，value是最后需要的值

     * 只是运算不同：Q和K的计算
     * alignment:  correlation的程度

     ![image-20231106162638387](.\asset\20231106_attentionQKV.png)

   * The input vectors are used for both the alignment and the 

     attention calculations

3. self-attention

   * key value都来自于一个input vector

   * 想用attention来做fully connected layer，

   * 与global convolution的区别

   * 在一层中，自注意力是global， CNN是local

     ![image-20231106164120627](.\asset\20231106_self_attention2.png)

![image-20231106163451095](.\asset\20231106_self_attention.png)

4. positional encoding 

* 解决order的问题  （要知道现在处理的词在一句话中的位置，因为self-attention的运算是无向的，无法分辨）

  ![image-20231106164712619](.\asset\20231106_position_encoding.png)

  * 因此，我们需要这样一种位置表示方式，满足于：
    （1）它能用来表示一个token在序列中的绝对位置
    （2）在序列长度不同的情况下，不同序列中token的相对位置/距离也要保持一致
    （3）可以用来表示模型在训练过程中从来没有看到过的句子长度。
  * 需要一个有界又连续的函数，最简单的，正弦函数sin就可以满足这一点
  * 把所有的位置信息都存在了encoder里面
  * 在Transformer的论文中，比较了用positional encoding和learnable position embedding(让模型自己学位置参数）两种方法，得到的结论是两种方法对模型最终的衡量指标差别不大。不过在后面的BERT中，已经改成用learnable position embedding的方法了，也许是因为positional encoding在进attention层后一些优异性质消失的原因（猜想）。Positional encoding有一些想象+实验+论证的意味，而编码的方式也不只这一种，比如把sin和cos换个位置，依然可以用来编码。关于positional encoding，我也还在持续探索中。

* multi-head: 在不同的channel上做attention  

  * **为注意力层提供了多个“表示子空间”**。 对于多头注意力，我们不仅有一个，而且还有多组Query/Key/Value权重矩阵，这些权重矩阵集合中的每一个都是随机初始化的。 然后，在训练之后，每组用于将输入Embedding投影到不同的表示子空间中。

![image-20231106164408040](.\asset\20231106_multi_head.png)

* attention本身不是提取数据本身的特征，而是算关联；是从general意义上是可解释的
* q k就是Learnable的feature，attention就是建立两者的关联，然后对value做加权
* learn能做的事是本身就可以解释的；
  * 即使理论上MLP可以实现所有，但是就是有算力限制
  * 所以魔改的地方就是加上一些可解释的东西

#### 4. Transformer

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

* **Cons:** Require large memory: *N x M* alignment and attention scalers are calculated and stored for a single self-attention head. 

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

  

---

SOTA： state-of-the-art model，并不是特指某个具体的模型，而是指在该项研究任务中，目前最好/最先进的模型。

---



## L9 GNN

1. Graph, graph notations & graph data representation

   * ![image-20231113141153016](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_graph_signal.png)

   * ![image-20231113142655280](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_graph_task.png)

   * ![image-20231113143733542](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_graph_task2.png)

2. Building neural network for graph --- the GNN;

* neighborhood-based pooling operation
* By stacking message passing GNN layers together, a node can eventually incorporate information from across the entire graph
* ![image-20231113150946536](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_GNN_update.png)

![image-20231113150838427](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_GNN_performance.png)

![image-20231113150817753](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_risk.png)

3. Treating GNN in another aspect --- the Spectral Graph CNN

* ∆f= f'(x)? − f'(x-1) = f(x+1) + f(x-1) + 2 ∗ f(x)  拉普拉斯算子
* ![image-20231113152011090](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_GNN_matrix.png)

* ![image-20231113152201101](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_GNN_matrix2.png)

![image-20231113153716063](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_laplacian_matrix.png)

* 无向图的拉普拉斯矩阵是对称阵，这时候可以将对角化
* ![image-20231113153832108](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_laplacian_matrix2.png)

![image-20231113154227447](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113-spectrum.png)

使用多项式来描述一个对角阵，则参数变成了k个。

![image-20231113154339168](D:\My_desktop\Blog备份\Course_Learning_note\媒体与认知\asset\20231113_simplified.png)

也可以使用切比雪夫多项式逼近。



4. A glare of modern GNN architectures.

* Graph Attention Networks --- GAT
  * GAT provides solutions using attention coefficient in message aggregation
  *  attention coefficient: a measurement of how relevant (important) a neighboring node is in relation to the center node
* Generative modelling & GraphVAE
  * **Generative model** for graphs: generate new graphs by sampling from a learned distribution or by completing a graph given a starting point
  * Design new drugs: novel molecular graphs with specific properties

* Graph recurrent neural network --- GRNN
  * For a time varying graph, recurrent unit could be applied

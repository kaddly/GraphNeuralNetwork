# 9. GATNE

## 9.1、背景

目前许多图嵌入应用大多都只是同构网络或者小尺度网络，而 真实世界往往大都是数以亿计的不同类型的节点和边，而且节点往往包含多种属性。即大规模多元异构属性网络。

针对大规模多元异构属性网络(attributed multiplex heterogeneous network)，即边和节点有多种类型，且节点有很多特征的网络，提出GATNE模型，学习每个节点在不同类型边下的嵌入表示，模型同时分为支持直推式的GATNE-T和归纳式的GATNE-I。
**Attributed Multiplex Heterogeneous Network的挑战：**

1. 多种类型的边：每个节点对可能有着多种关系。需要从不同的关系中学习特征再进而学习到统一的嵌入表示
2. 部分观察：真实的网络数据通常是部分被观测的，现有的大多数方法都聚焦于直推式学习，不能解决long-tail和冷启动问题
3. 大规模：真实的网络往往有亿级的节点和边，需要设计模型处理大规模数据

**文章贡献：**

1. 定义了多重异构特征网络嵌入
2. 同时支持直推式和归纳式学习嵌入，并提供了理论证明
3. 模型高效，适合处理大规模数据

## 9.2、算法原理

### 网络分类及建模

根据不同节点类型（单一节点/多种节点）、不同边类型（单一边/多种边）、节点性质（有性质/无性质），可以分为以下六种网络模型，不同网络有不同的建模方法。

![image-20221107081747514](./img/table.jpg)

而本文中关注的Attributed Multiplex Heterogeneous Network(AMHEN)：

1. Attributed：考虑节点性质，如性别、年龄、购买力等
2. Multiplex：多重边，节点之间可能有多种关系，比如两用户之间可能同为朋友、同学以及交易关系等；用户和Item之间可以同时有浏览、点击、添加到购物车等
3. Heterogeneous：异构，节点和边有多种类型，即节点类型+边类型$\gt 2$

![image-20221107081904993](./img/modules.jpg)

如上图中间，三种建模方法AMHEN(Attributed Multiplex Heterogeneous Network)、MHEN（Multiplex Heterogeneous Network）和HON（Homogeneous Network）。从右图结果中可以看出AMHEN（考虑多种类型、节点性质的异构图网络）优于MHEN（不考虑节点性质的异构网络）优于HON同构网络

![image-20221107082040744](./img/modle.jpg)

本文提出的General Attributed Multiplex Heterogeneous Network Embedding(GATNE)，希望每个节点在不同类型的边中有不同的表示

```
例如用户A在点击商品的场景下学习一种向量表示，在购买商品场景下学习另一种向量表示，而不同的场景之间并不完全独立，希望用base embedding来当做不同类型关系传递信息的桥梁，综合base embedding和每一类型边的edge embedding来进行建模。
```

模型结构如图所示，网络结构利用两部分向量Base Embedding和Edge Embedding表示，其中Base Embedding为共享向量，出现在每一种边类型中；Edge Embedding在每一种边类型中不同。GATNE-T仅利用了网络结构信息，GATNE-I同时利用了网络结构信息和节点性质。

### Transductive Model：GATNE-T

如上图所示，在GATNE-T中，每个节点在每种edge type下的embedding由两部分组成，分别是**Base** **embedding**和**Edge** **embedding**。

- base embedding：每个节点在每种edge type下共享
- edge embedding：通过相邻节点的edge embedding计算得到的

GATNE-T的核心思想就是聚合不同类型的邻居到当前节点，然后对每一种边类型的节点都生成不同的向量表示，模型整体可以分为如下4步：

1. 类似Graphsage对邻居聚合的思想，节点$v_i$对类别为$r$的第$k$阶邻居进行聚合，得到edge Embedding $u_{i,r}^{(k)};u_{i,r}^{(k)}=aggregator({u_{j,r}^{(k-1)},\forall v_i\in \mathcal{N}_{i,r}})$

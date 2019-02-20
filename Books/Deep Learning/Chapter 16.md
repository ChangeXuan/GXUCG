- 结构化概率模型使用图来描述概率分布中随机变量之间的直接相互作用，从而描述一个概率分布
- 结构化概率模型为随机变量之间直接作用提供了一个正式的建模框架，这种方式大大减少了模型的参数个数，以至于模型只需要更少的数据来进行有效的估计
- 结构化概率模型使用图来表示随机变量之间的相互作用，其中，每个结点代表一个随机变量，每一条边代表一个直接相互作用
- 图模型可被分为两类：基于有向无环图的模型和基于无向图的模型
- 有向图模型是一种结构化概率模型，也被称为信念网络或贝叶斯网络
- 无向图模型也被称为马尔可夫网络
- 当相互作用并没有本质性的指向，或者是明确的双向相互作用时，使用无向模型更适合
- 图的一个团时图中结点的一个子集，并且其中的点时全连接的
- 有向图的一个优点时，可以通过一个简单高效的过程从模型所表示的联合分布中产生样本，这个过程被称为原始采样
- 原始采样的一个缺点是其仅适用于有向图模型
- 使用结构化概率模型的主要优点是，它们能够显著降低表示概率分布，学习和推断的成本
- 结构化概率模型允许我们明确地将给定的现有知识与知识的学习或者推断分开
- 解决变量之间如何相互关联的问题是我们使用概率模型的一个主要方式
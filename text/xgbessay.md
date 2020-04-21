# ABSTRACT

提升树是一种高效且被广泛使用的机器学习方法。在本文中，我们描述了一个可扩展的端对端的提升树系统，叫做XGBoost，该系统被数据科学家广泛使用，在许多机器学习任务中取得了显著效果。针对稀疏数据，我们提出一种新的稀疏数据感知算法。我们也提出了分布式加权分位数草图来近似实现树模型的学习。更重要的是，我们陈述了缓存访问模式、数据压缩和分片的见解以构建可扩展的提升树系统。通过结合这些知识，XGBoost可以使用比现有系统少得多的资源就能够扩展数十亿的实例。

Keywords:Large-scale Mach_ine Learning

# INTRODUCTION

机器学习和数据驱动的方法在许多领域变得非常重要。智能垃圾邮件分类器通过从大量垃圾邮件数据和用户反馈中学习来如何保护我们免受垃圾邮件的侵害；广告系统学会将正确的广告放到正确的语境中；欺诈检测系统保护银行免受恶意攻击者的攻击；异常现象检测系统帮助实验物理学家发现引发新物理现象的因素。驱动这些技术成功应用的因素有两个：使用能够发现复杂数据依赖性的有效的（统计）模型，以及能从大型数据集里学习获得偏好模型的可扩展的学习系统。

在机器学习算法的实践应用中，梯度提升树算法非常卓越。提升树在很多标准分类基准上表现非常出色（我也不知道什么是分类基准，大概是有其他算法的结果做基准的任务吧）。LambdaMART是提升树算法的变种，在排序任务中也表现出了不错的效果。XGBoost除了被用作单独的预测器，还被用于实际的广告点击率的问题中。它是集成算法的一种，也经常用于Netfix等竞赛。

在本文中，我们描述了XGBoost，一种针对提升树的可扩展的机器学习系统。该系统有开源的软件包可用。该系统的影响已经在许多机器学习和数据挖掘任务重得到任何。以机器学习竞赛网站kaggle为例。2015年，kaggle的博客上发布了29个挑战获胜的解决方案，其中17个解决方案用了XGBoost。在这些解决方案中，8个只用了XGBoost来训练模型，而大多数其他解决方案将XGBoost与神经网络进行了结合。第二种常用的方法是深度神经网络，出现在了11个解决方案中。KDDCup2015也证明了该系统的成功，其中前10名的队伍都用了XGBoost。此外，获胜团队表示，集成算法的效果仅仅比XGBoost略优一点。

这些结果表明，我们的系统在各种问题中表现都非常优异。这些获胜的解决方案涉及到的问题有：商店销量预测；高能物理事件分类；网络文本分类；顾客行为预测；运动检测；广告点击率预测；恶意软件识别；产品分类；风险预测；在线课程辍学率预测；虽然数据分析和特征工程在其中发挥了重要作用，但大家都选择XGBoost算法也是一个事实，这表明了我们的系统和提升树的影响和重要性。

XGBoost成功的重要因素是它可以扩展到所有场景中。该系统在单台机器上的运行速度比现有流行的解决方案快10倍以上，并可在分布式或内存有限的环境中扩展到数十亿个示例。XGBoost的可扩展性归功于几个重要的系统和算法优化。这些创新包括：一种用于稀疏数据的树学习算法；加权分位数草图能够在近似树学习中处理样本的权重，这在理论上是合理的。并行和分布式计算使得学习速度更快，从而加快了模型的探索。更重要的是，XGBoost利用外核计算，使数据科学家能够在桌面上处理数十亿个示例。最后，更令人兴奋的是，将这些技术结合起来，利用端对端系统以最少的集群资源将其扩展到更大的数据规模。本文主要贡献如下：

1. 我们设计并构建了一个高度扩展的端对端的提升树系统
2. 我们提出了一个用于高效运算的理论上正确的加权分位数草图
3. 我们为并行树模型学习提出了一种新颖的稀疏感知算法
4. 我们提出了一种有效的缓存感知块结构用于树模型的核外学习

虽然现在存在一些并行提升树模型的研究工作，但核外计算、缓存感知和稀疏感知学习等方向还尚未有人涉略。更重要的是，结合这些方面的技术构建出的端对端的系统为实际应用提供了一种新的解决方案。这使得数据科学家和研究人员能够构建提升树算法的强大变种。除了这些主要的贡献之外，我们还提出了一个正则化学习的方法。

本文其余的部分安排如下。第二部分我们回顾了提升树，并介绍了正则化的目标。然后，我们在第三部分介绍分割节点寻找的方法，第四部分是系统设计，包括相关的实验结果，为我们提到的每个优化方法提供量化支持。相关工作放在在第五节讨论。第六部分详细的介绍了端对端的评估。最后，我们在第七部分总结这篇论文。

# TREE BOOSTING IN A NUTSHELL(容器)

我们在这一节中介绍梯度提升树算法。公式推导遵循文献中的梯度提升思想。特别地，其中的二阶方法源自Friedman等人。我们对正则项进行了微小的改进，这在实践中有所帮助。

## Regularized Learning Objective

对于给定的数据集有n个样本m个特征$D=\{(x_i,y_i)\}(∣D∣=n,x_i \in R^m,y_i \in R)$，树集成算法使用个数为K的加法模型（如图1）来预测输出。
![ Tree Ensemble Model. The final predic-
tion for a g_iven example is the sum of predictions
from each tree.](../pic/xgbfig1.png)

$$\hat{y}_i=\phi(x_i)=\sum^K_i{f_k(x_i)},f_k \in F \tag{1}$$
其中$F=\{f(x)=w_{q(x)}\}(q:R^m→T,w \in R^T)$是回归树（也叫做CART）的空间。$q$表示将样本映射到叶节点的树的结构。$T$是每棵树叶子的数量。每个$F_k$​对应了独立的树结构$q$和叶权值$w$。与决策树不同，每棵回归树的每个叶子上包含连续的连续值打分，我们用$w_i$​表示第$i$个叶子的打分。对于一个给定的例子，我们将使用树中的决策规则（由$q$给出）将其分类到叶子节点中，并通过对相应叶子中的分数求和来计算最终预测（由$w$给出）。为了学习模型中使用的函数集合，我们最小化下面的正则化的项。
$$L(\phi)=\sum_il(\hat{y}_i,y_i)+\sum_k\Omega(f_k)\quad  where \quad \Omega(f)=\gamma T+\frac{1}{2}λ∥w∥^2 \tag{2}$$

这里$L$是一个可微的凸损失函数，它表示预测$y_i$​和目标$y$之间的差值。第二项$\Omega$是惩罚项，惩罚模型的复杂度（即回归树模型）。附加正则化项会有助于使最终学习到的权值更加平滑，避免过拟合。直观地说，带有正则化的目标函数倾向于选择简单的预测模型。类似的正则化技术已被用于正则化贪心森林算法（RGF）模型中。我们的目标函数和相应的学习算法比RGF更简单，更容易实现并行化。当正则化参数被设置为零时，目标函数退化为传统的梯度提升树。

## Gradient Tree Boosting

公式（2）中的树集成模型中包含函数作为参数的情况，不能使用欧氏空间中的传统优化方法来优化。替代的方法是模型以累加的方式训练。形式上，$\hat{y}_i^t$是第$t$次迭代中第$i$个实例的预测，我们把$f_t$​加到最小化目标中。

$$L^{(t)}=\sum^n_i{l(y_i,\hat{y}_i^{t-1}+f_t(x_i))}+\Omega(f_t)$$

也就是说我们根据公式（2）贪婪地将$f_t$​加到了目标函数中，这对我们模型提升最大（因为是沿梯度下降的）。一般情况下，二阶近似（也就是泰勒二阶展开近似）可以用于快速优化目标函数。（因为有二阶信息，所以优化起来比一阶速度快。例如，牛顿法就比传统的梯度下降快）
$$L^{(t)}≃\sum^n_i{[l(y_i,\hat{y}^{t-1})+g_if_t(x_i)+\frac{1}{2} h_i f^2_t(x_i)]}+\Omega(f_t)$$
（这步是上一个公式经过泰勒二阶展开得到的，具体推导步骤可以去看我转载的一篇XGBoost的博客，那篇公式推导比较全）

其中$g_i=∂_{\hat{y}^{t-1}} l(y_i,\hat{y}^{t-1})，h_i=∂^2_{\hat{y}^{t-1}}l(y_i,\hat{y}^{t-1})$，分别为损失函数一阶和二阶的梯度值。在第$t$步迭代中，我们可以去掉常数项以简化目标函数。
$$L˜(t)=\sum^n_i{[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]}+\Omega(f_t)\tag{3}$$
（意思是针对第$t$步来说，之前的 $t-1$ 步已成定局，也就是常数项了，所以针对这一步优化来说可以去掉）

![](../pic/xgbfig2.png)

定义$I_j=\{i∣q(x_i)=j\}$为叶子节点$j$里的样本，我们可以通过扩展$\Omega$来重写公式（3）：
$$\begin{aligned}
\tilde{\mathcal{L}}^{(t)} &=\sum_{i=1}^{n}\left[g_{i} f_{t}\left(\mathbf{x}_{i}\right)+\frac{1}{2} h_{i} f_{t}^{2}\left(\mathbf{x}_{i}\right)\right]+\gamma T+\frac{1}{2} \lambda \sum_{j=1}^{T} w_{j}^{2} \\
&=\sum_{j=1}^{T}\left[\left(\sum_{i \in I_{j}} g_{i}\right) w_{j}+\frac{1}{2}\left(\sum_{i \in I_{j}} h_{i}+\lambda\right) w_{j}^{2}\right]+\gamma T \tag{4}
\end{aligned}$$

对于一个固定结构$q(x)$，我们可以计算叶节点$j$的最优权$w_j^*$：
$$w_{j}^{*}=-\frac{\sum_{i \in I_{j}} g_{i}}{\sum_{i \in I_{j}} h_{i}+\lambda} \tag{5}$$

并通过下式计算相应的最优值：
$$\tilde{\mathcal{L}}^{(t)}(q)=-\frac{1}{2} \sum_{j=1}^{T} \frac{\left(\sum_{i \in I_{j}} g_{i}\right)^{2}}{\sum_{i \in I_{j}} h_{i}+\lambda}+\gamma T \tag{6}$$

公式（6）可以作为一个评估方程去评价一棵树的结构$q$。这个打分就像评估决策树的杂质分数，不同的是它是为了更广泛的目标函数导出的。图2示出了如何计算这个分数。

通常来说不可能枚举出所有的树结构$q$，而是用贪心算法，从一个叶子开始分裂，反复给树添加分支。假设$IL$和$IR$​是分裂后左右节点中包含的样本集合。使$I＝IL∪IR$​，通过下式分裂后会使损失函数降低。
$$\mathcal{L}_{s p l i t}=\frac{1}{2}\left[\frac{\left(\sum_{i \in I_{L}} g_{i}\right)^{2}}{\sum_{i \in I_{L}} h_{i}+\lambda}+\frac{\left(\sum_{i \in I_{R}} g_{i}\right)^{2}}{\sum_{i \in I_{R}} h_{i}+\lambda}-\frac{\left(\sum_{i \in I} g_{i}\right)^{2}}{\sum_{i \in I} h_{i}+\lambda}\right]-\gamma \tag{7}$$

这个公式用于评价候选分裂节点的好坏。
![Figure 2: Structure Score Calculation. We only
need to sum up the gradient and second order gra-
dient statistics on each leaf, then apply the scoring
formula to get the quality score.](../pic/xgbfig2.png)

## Shrinkage and Column Subsampling

除了在2.1节中使用的正则化项，我们还使用了两种技术来进一步防止过拟合。第一种技术是Friedman引入的收缩。在每一次提升树训练迭代后，在前面乘一个因子$η$来收缩其权重（也就是我们说的学习率，或者叫步长）。与随机优化中的学习率类似，收缩减少了每棵树的影响，并为将来的树模型留出了改进模型的空间。第二种技术上列（特征）子采样。这个技术用于随机森林中，在商业软件TreeNet4中实现，用于梯度增强，但未在现有的开源包中实现。根据用户反馈，使用列子采样可以比传统的行子采样（也支持）更能防止过度采样。列子采样还能加速稍后描述的并行算法。

# SPLIT FINDING ALGORITHMS
## Basic Exact Greedy Algorithm

树模型学习过程中的一个关键问题是找到最佳分裂节点，如公式（7）所示。为了做到这一点，一个分裂查找算法枚举出了所有特征上的所有可能的分裂节点，我们称之为贪婪算法。大多数现有的单机版本的提升树已经实现了，如scikit-learn、R中的GBM以及XGBoost的单机版本。在Alg.1中给出贪婪算法的详细描述。算法要求列举出所有特征的所有可能的分割点。为了提高效率，算法必须先将特征取值排序，并按顺序访问数据，然后根据公式（7）计算出当前分割点的梯度统计量。
![Algorithm 1: Exact Greedy Algorithm for Split Finding](../pic/xgbal1.png)

## Approximate Algorithm 近似算法

贪婪算法是非常有效的，因为它贪婪地枚举除了所有可能的分裂点。然而，**当数据不能完全读入内存时，这样做就不会很有效率**。同样的问题也出现在分布式环境中。为了有效支持这两种环境中的提升树，我们需要一种近似算法。

我们总结了一个近似框架，类似于在过去的文献中提到的想法「参考文献17、2、22」，如Alg.2描述。总结来说，该算法首先**根据特征分布的百分位数提出可能的候选分裂点**（具体的准则在3.3中给出）。然后算法将**连续特征值映射到候选分割点分割出的箱子中**。计算出每个箱子中数据的统计量（这里的统计量指的是公式（7）中的$g$和$h$），然后**根据统计量找到最佳的分割点**。
![Algorithm 2: Approximate Algorithm for Split Finding](../pic/xgbal2.png)

该算法有两种变体，区别为分裂点的准则何时给出。全局选择在树构造的初始阶段要求给出所有候选分裂点，并且在树的所有层中使用相同的分裂节点用于分裂。局部选择在分裂后重新给出分裂候选节点。全局方法比局部方法需要更少的步骤。然而，**通常在全局选择中需要更多的候选点**，因为在每次分裂后候选节点没有被更新。**局部选择在分裂后更新候选节点，并且可能更适合于深度更深的树**。图3给出了基于希格斯玻色子数据集的不同算法的比较。我们发现，**本地变种确实需要更少的候选节点。当给出足够的候选节点，全局变种可以达到与本地变种一样的准确率**。

（这段确实难翻译，而且第一次看也一脸懵逼，主要的锅是作者写作顺序有问题。看完3.3这里就懂了。也有可能作者默认这个很简单？？？其中局部选择步骤多的意思就是每分裂一次都需要更新3.3中对应的min(x)和max(x)，相比全局选择来说候选点间隔更细）

大多数现有的分布式树模型学习的近似算法也遵循这一框架。值得注意的是，**直接构造梯度统计量的近似直方图也是可行的。也可以使用分箱策略来代替分位数划分**。**分位数策略的优点是可分配和可重计算**，我们将在下一节中详细说明。从图3中，我们还发现，当设置合理的近似水平，分位数策略可以得到与贪心算法相同的精度。

![](../pic/xgbfig3.png)

我们的系统有效地支持单机版的贪心算法，同时也支持近似算法的本地变种和全球变种的所有设置。用户可以根据需求自由选择。

## Weighted Quantile Sketch 加权分位数草图

近似算法中很重要的一步是列出候选的分割点。通常特征的百分位数作为候选分割点的分布会比较均匀。具体来说，设$D_k={(x_{1k},h_1),(x_{2k},h_2)⋯(x_{nk},h_n)}$表示样本的第k个特征的取值和其**二阶梯度统计量**。我们可以定义一个排序方程
$$r_{k}(z)=\frac{1}{\sum_{(x, h) \in \mathcal{D}_{k}} h} \sum_{(x, h) \in \mathcal{D}_{k}, x<z} h \tag{8}$$

上式表示样本中第k个特征的取值小于z的比例（直译过来确实是这样，不过公式表达的是取值小于z的二阶梯度统计量的比例）。我们的目标是找到候选的分割节点$\{s_{k1},s_{k2},⋯s_{kl}\}$。
$$\left|r_{k}\left(s_{k, j}\right)-r_{k}\left(s_{k, j+1}\right)\right|<\epsilon, \quad s_{k 1}=\min _{i} \mathbf{x}_{i k}, s_{k l}=\max _{i} \mathbf{x}_{i k} \tag{9}$$
这里$\epsilon$是一个近似因子（别被名字吓倒了，其实就是衡量两者的差距）。直观的说，大概有$1/\epsilon$个分割点（这应该好理解吧，如果从0-1之间分割，分割点之间差距小于0.2，那么就是大概有5个分割点）。这里每一个数据点用$h_i$来代表权重。我们来看看为什么$h_i$能代表权重，我们可以把公式（3）重写为：
$$L˜(t)=\sum^n_i{[g_if_t(x_i)+\frac{1}{2}h_if_t^2(x_i)]}+\Omega(f_t)\tag{3}$$
$$\sum^n_i{\frac{1}{2}h_i(f_t(x_i)−g_i/h_i)^2}+\Omega(f_t)+constant$$
这实际上是权值为$h_i$​，标签为$g_i/h_i$

$g_i​/h_i​$的加权平方损失。对于大数据集来说，找到满足标准的候选分割点是非常不容易的。**当每个实例具有相等的权重时，一个现存的叫分位数草图的算法解决了这个问题**。然而，对于加权的数据集没有现存的分位数草图算法。因此，大部分现存的近似算法要么对可能失败的数据的随机子集进行排序，要么使用没有理论保证的启发式算法。

为了解决这个问题，我们**引入了一种新的分布式加权分位数草图算法**，该算法可以处理加权数据，并且可以从理论上证明。通常的做法是提出一种支持合并和修建操作的数据结构，每个操作都是可以被证明保持一定准确度的。附录中给出了算法的详细描述以及证明。

## Sparsity-aware Split Finding

在许多实际问题中，输入$X$稀疏是常见的。稀疏有多个可能的原因导致：
1) 数据中存在缺失值；
2) 有些统计数值常常为0；
3) 特征工程的结果，如独热编码。

算法对数据中稀疏模式的感知能力是非常重要的。为了做到这一点，我们建议在每个树节点中添加一个默认的方向，如图4所示。当稀疏矩阵x中的值丢失时，实例被分类为默认的方向。

![](../pic/xgbfig4.png)

在每个分支中有两种默认方向。最优的默认方向是从数据中学习出来的。具体算法在Alg.3显示。关键步骤是只访问非缺失的数据$I_k$​。算法将不存在的值视为缺失值并学习默认方向去处理它（这里的不存在的值应该说的是不符合特征意义或者不合理的值）。当非存在的值对应于用户特定说明的值时，可以将枚举结果限制为一致的方案来应用这个算法。(bucket,桶)（最后一句不太好翻译，原文为：The same algorithm can also be applied when the non-presence corresponds to a user specified value by limiting the enumeration only to consistent solutions。意思应该是多次枚举限制到同一方向（存疑））

![Algorithm 3: Sparsity-aware Split Finding](../pic/xgbal3.png)

据我们所知，大多数现有的树学习算法要么只对密集数据进行优化，要么需要特定的步骤来处理特殊情况，例如类别编码。XGBoost以统一的方式处理所有稀疏模式。更重要的是，我们的方法利用稀疏性，使得计算的复杂度与输入中的非缺失数据的数量成线性关系。图5显示了稀疏感知算法和一个常规算法在数据集Allstate-10K（此数据集在第6部分描述）上的比较。我们发现稀疏感知算法的运行速度比常规版本快50倍。这证实了稀疏感知算法的重要性。
![Figure 5: Impact of the sparsity aware algorithm
on Allstate-10K. The dataset is sparse mainly due
to one-hot encoding. The sparsity aware algorithm
is more than 50 times faster than the naive version
that does not take sparsity into consideration.](../pic/xgbfig5.png)

# SYSTEM DESIGN

## Column Block for Parallel Learning

树学习中最耗时的部分是数据排序。为了减少排序的成本，我们提出将数据存储在内存单元中，称之为block。每个block中的数据每列根据特征取值排序，并以压缩列（compressed colum,CSC）格式储存。这种输入数据布局只需要在训练前计算一次，可以在后续迭代中重复使用。

在贪婪算法中，我们将整个数据集存储在单个block中，并通过对预排序的数据进行线性扫描来实现分割点搜索。我们集体对所有叶子进行分割查找，这样只需扫描一次block就可以得到所有叶子节点处所有候选分裂节点的统计信息。图6显示了如何将数据集转换成相应格式并使用block结构找到最优分割。

![](../pic/xgbfig6.png)

当使用近似算法时，block结构也非常有用。在这种情况下，可以使用多个block，每个block对应于数据集中的不同的行子集。不同的block可以分布在机器上，或者在非核心设置中存储在磁盘上。使用排序结构，分位数查找步骤在完成排序的列上就变成了线性扫描。这对于在每个分支中频繁更新候选分割点的本地优先算法非常有价值。直方图聚合中的二分搜索也变成了线性时间合并样式算法。

收集每列统计信息这一步骤可以实现并行化，这也给我们提供了一种寻找分割点的并行算法。还有一点重要的是，列的block结构同样支持列子采样，因为从block结构中选择列子集是非常容易的。

时间复杂度分析 设d为树的最大深度，K为树的总数。对于贪婪算法，原始稀疏感知算法的时间复杂度为$O(Kd∥x∥_0logn)$。这里我们使用$∥x∥0$来表示训练数据中的非缺失条目的数量。另一方面，在block结构上仅消耗$O(Kd∥x∥0+∥x∥0logn)$。这里$O(∥x∥0logn)$是可以摊销的一次性预处理成本。该分析表明block结构可以节省额外的$logn$的复杂度，这在$n$很大时很重要。对于近似算法，基于二分搜索的原始算法的时间复杂度为$O(Kd∥x∥0logq)$。这里$q$是数据集中候选分割节点的数量。虽然$q$通常介于32和100之间，但对数因子仍会引入开销。使用block结构，我们可以将时间减少到$O(Kd∥x∥0+∥x∥0logB)$，其中$B$是每个块中的最大行数。我们再次可以在计算中保存额外的$logq$因子。
（只要懂$O(∥x∥0logn)$是一次排序的开销其他就比较好懂了，常用的排序算法的复杂度一般都为$O(nlogn)$）
## Cache-aware Access

虽然block结构有助于优化分割点查找的时间复杂度，但是算法需要通过行索引间接提取梯度统计量，因为这些值是按特征的顺序访问的，这是一种非连续的内存访问（意思就是按值排序以后指针就乱了）。 分割点枚举的简单实现在累积和非连续内存提取之间引入了即时读/写依赖性（参见图8）。 当梯度统计信息不适合CPU缓存进而发生缓存未命中时，这会减慢分割点查找的速度。
![](../pic/xgbfig7.png)

![](../pic/xgbfig8.png)

**对于贪心算法，我们可以通过缓存感知预取算法来缓解这个问题**。 具体来说，我们在每个线程中分配一个内部缓冲区，获取梯度统计信息并存入，然后以小批量方式执行累积。 预取的操作将直接读/写依赖关系更改为更长的依赖关系，有助于数据行数较大时减少运行开销。 图7给出了Higgs和Allstate数据集上缓存感知与非缓存感知算法的比较。 我们发现，当数据集很大时，实现缓存感知的贪婪算法的运行速度是朴素版本的两倍。

对于**近似算法，我们通过选择正确的block尺寸来解决问题**。 我们将block尺寸定义为block中包含的最大样本数，因为这反映了梯度统计量的高速缓存存储成本。 选择过小的block会导致每个线程的工作量很小，并行计算的效率很低。 另一方面，过大的block会导致高速缓存未命中现象，因为梯度统计信息不适合CPU高速缓存。良好的block尺寸平衡了这两个因素。 我们在两个数据集上比较了block大小的各种选择，结果如图9所示。该结果验证了我们的讨论，并表明每个块选择$2^{16}$个样本可以平衡缓存资源利用和并行化效率。

![](../pic/xgbfig9.png)

## Blocks for Out-of-core Computation

我们系统的一个目标是充分利用机器的资源来实现可扩展的学习。 除处理器和内存外，利用磁盘空间处理不适合主内存的数据也很重要。为了实现核外计算，我们将数据分成多个块并将每个块存储在磁盘上。在计算过程中，使用独立的线程将块预取到主存储器缓冲区是非常重要的，因为计算可以因此在磁盘读取的情况下进行。但是，这并不能完全解决问题，因为磁盘读取会占用了大量计算时间。减少开销并增加磁盘IO的吞吐量非常重要。 我们主要使用两种技术来改进核外计算。

Block Compression 我们使用的第一种技术是块压缩。该块从列方向压缩，并在加载到主存储器时通过独立的线程进行解压。这可以利用解压过程中的一些计算与磁盘读取成本进行交换。我们使用通用的压缩算法来压缩特征值。对于行索引，我们通过块的起始索引开始减去行索引，并使用16位整型来存储每个偏移量。这要求每个块有$2^{16}$个样本，这也被证实是一个好的设置（好的设置指的是$2^{16}$这个数字的设置）。在我们测试的大多数数据集中，我们实现了大约26％到29％的压缩率。

Block Sharding 块分区 第二种技术是以另一种方式将数据分成多个磁盘。为每个磁盘分配一个实现预取的线程，并将数据提取到内存缓冲区中。然后，训练线程交替地从每个缓冲区读取数据。当有多个磁盘可用时，这有助于提高磁盘读取的吞吐量。

# RELATED WORKS

我们的系统通过函数的加法模型实现了梯度提升。梯度提升树已成功用于分类，排序，结构化预测以及其他领域。XGBoost采用正则化模型来防止过度拟合，类似于正则化贪心森林，但简化了并行化的目标和算法。列采样是一种从随机森林借鉴来技术，简单且有效。虽然稀疏感知学习在其他类型的模型（如线性模型）中是必不可少的，但很少有关这方面在树模型学习中的研究。本文提出的算法是第一个可以处理各种稀疏模式的统一方法。

现存有很多树模型并行学习的研究。大多数算法都属于本文所述的近似框架。值得注意的是，还可以按列对数据进行分区并应用贪婪算法。我们的框架也支持这一点，并且可以使用诸如缓存感知预防之类的技术来使这类算法受益。虽然大多数现有的工作都集中在并行化的算法方面，但我们的工作在两个未经探索的方面得到了成果：核外计算和缓存感知学习。这让我们对联合优化系统和算法的有了深刻的理解，并构建了一个端到端系统，可以在非常有限的计算资源下处理大规模问题。在表1中，我们还总结了我们的系统与现存开源系统的对比。
![](../pic/xgbtb1.png)
分位数摘要（无权重）是数据库社区中的经典问题。然而，近似提升树算法揭示了一个更普遍的问题——在加权数据上找分位数。据我们所知，本文提出的加权分位数草图是第一个解决该问题的方法。 加权分位数摘要也不是专门针对树模型学习的，可以在将来服务于数据科学和机器学习中的其他应用。

# END TO END EVALUATIONS

## System Implementation

我们以开源软件包的形式实现了XGBoost。该软件包是可移植和可重复使用的。它支持各种加权分类和各种阶的目标函数，以及用户定义的目标函数。它对流行的语言都提供支持，例如python，R，Julia，并且与语言特定的数据科学库（如scikit-learn）自然集成。分布式版本构建在rabit库上，用于allreduce。XGBoost的可移植性使其可用于许多生态系统，而不仅仅是绑定在特定平台。分布式XGBoost可以轻松运行在Hadoop，MPI Sun Grid引擎上。最近，我们还在jvm 大数据栈（如Flink和Spark）上实现了分布式XGBoost。分布式版本也已集成到阿里巴巴的天池云平台中。我们相信未来会有更多的整合。

## Dataset and Setup

我们在实验中使用了四个数据集。表2给出了这些数据集的摘要信息。在一些实验中，由于基线较慢，我们使用随机选择的数据子集，或者演示算法在不同大小的数据集下的性能。在这些情况下，我们使用后缀来表示大小。例如，Allstate-10K表示具有10K实例的Allstate数据集的子集。
![](../pic/xgbtb2.png)

我们使用的第一个数据集是Allstate保险索赔数据集。任务是根据不同的风险因素预测保险索赔的可能性和成本。在实验中，我们将任务简化为仅预测保险索赔的概率。此数据集用于评估在3.4节中提到的稀疏感知算法。此数据中的大多数稀疏特征都是独热编码。我们随机选择10M样本作为训练集，并将其余部分用作验证集。

第二个数据集是高能物理学的希格斯玻色子数据集。该数据是使用蒙特卡洛仿真模拟物理现象生成的。它包含21个运动学特征，由加速器中的粒子探测器测量得到。还包含七个额外的粒子派生物理量。 任务是分类是否物理现象与希格斯玻色子相对应。我们随机选择10M实例作为训练集，并将其余部分用作验证集。

第三个数据集是Yahoo! learning to rank比赛数据集，这是learning to rank算法最常用的基准之一。 数据集包含20K网页搜索查询结果，每个查询对应于一个有大约22个文档的列表。任务是根据查询的相关性对文档进行排名。我们在实验中使用官方的训练测试集分割标准。

最后一个数据集是criteo百万级别的点击日志数据集。我们使用此数据集来评估系统在核外和分布式环境中的扩展性。该数据包含13个数值特征和26个ID特征，其中有用户ID，项目ID和广告商信息ID等。由于树模型更擅长处理连续特征，前十天我们通过计算平均的CTR和ID特征的统计信息对数据预处理，接下来的十天用相应的统计信息替换ID特征，处理完成后就可以作为训练集。预处理后的训练集包含1.7billion个样本，每个样本具有67个特征（13个数值特征，26个平均CTR特征和26个统计特征）。整个数据集的LibSVM格式超过1TB。

我们将前三个数据集用于单机并行环境中，将最后一个数据集用于分布式和核外计算的环境。所有单机实验均在戴尔PowerEdge R420上进行，配备两个八核Intel Xeon（E5-2470）（2.3GHz）和64GB内存。如果未指定，则所有实验使用机器中的所有可用核心运行。分布式和核外实验的机器配置将在相应的部分中描述。在所有实验中，我们统一设置提升树的最大深度等于8，学习率等于0.1，除非明确指定否则不进行列子采样。当我们将最大深度设置为其他时，我们可以得到相似的结果。

## Classification

在本节中，我们在Higgs-1M数据集上通过对比其他两种常用的基于贪心算法的提升树，评估基于贪心算法的XGBoost的性能。由于scikit-learn只能处理非稀疏输入，我们选择密集Higgs数据集进行比较。我们在1M的数据子集上运行scikit-learn版本的XGBoost，这样可以在合理的时间内跑完。在比较中，R的GBM使用贪心算法，只扩展树的一个分支，这使它更快但可能导致准确性不足，而scikit-learn和XGBoost都生成完整的树。结果在表3中显示，XGBoost和scikit-learn都比R的GBM表现出更好的性能，而XGBoost的运行速度比scikit-learn快10倍。在实验中，我们还发现列子采样后的结果略差于使用所有特征训练的结果。这可能是因为此数据集中的重要特征很少，贪心算法的精确结果会更好。
![](../pic/xgbtb3.png)

## Learning to Rank

我们接下来评估XGBoost在learning to rank问题上的表现。我们与pGBRT进行比较，pGBRT是以前此类任务中表现最好的系统。XGBoost使用贪心算法，而pGBRT仅支持近似算法。结果显示在表4和图10中。我们发现XGBoost运行速度更快。有趣的是，列采样不仅可以缩短运行时间，还能提高准确性。原因可能是由于子采样有助于防止过拟合，这是许多用户观察出来的。
![](../pic/xgbfig10.png)

## Out-of-core Experiment

我们还在核外环境中使用criteo数据评估了我们的系统。我们在一台AWS c3.8xlarge机器上进行了实验（32个vcores，两个320 GB SSD，60 GB RAM）。 结果显示在图11中。我们可以发现压缩将计算速度提高了三倍，并且分成两个磁盘进一步加速了2倍。对于此类实验，非常重要的一点是使大数据集来排空系统文件缓存以实现真正的核外环境。这也是我们所做的。当系统用完文件缓存时，我们可以观察到转折点。要注意的是，最终方法中的转折点不是那么明显。这得益于更大的磁盘吞吐量和更好的计算资源利用率。我们的最终方法能够在一台机器上处理17亿个样本。
## Distributed Experiment

最后，我们在分布式环境中评估系统。我们在EC2上使用m3.2xlarge机器建立了一个YARN集群，这是集群的非常常见。每台机器包含8个虚拟内核，30GB内存和两个80GB SSD本地磁盘。数据集存储在AWS S3而不是HDFS上，以避免购买持久存储。
![](../pic/xgbfig11.png)

我们首先将我们的系统与两个生产力级别的分布式系统进行比较：Spark MLLib和H2O。我们使用32 m3.2xlarge机器并测试不同输入尺寸的系统的性能。两个基线系统都是内存分析框架，需要将数据存储在RAM中，而XGBoost可以在内存不足时切换到核外设置。结果如图12所示。我们可以发现XGBoost的运行速度比基线系统快。更重要的是，它能够利用核外计算的优势，在给定有限的计算资源的情况下平稳扩展到所有17亿个样本。基线系统仅能够使用给定资源处理数据的子集。该实验显示了将所有系统的改进结合在一起优势。我们还通过改变机器数量来评估XGBoost的缩放属性。结果显示在图13中。随着我们添加更多机器，我们可以发现XGBoost的性能呈线性变化。重要的是，XGBoost只需要四台机器即可处理17亿个数据。这表明系统有可能处理更大的数据。

![](../pic/xgbfig12.png)
![](../pic/xgbfig13.png)

# CONCLUSION

在本文中，我们叙述了在构建XGBoost过程中学到的经验（XGBoost是一个可扩展的提升树系统，被数据科学家广泛使用，并在很多问题上有很好的表现）。 我们提出了一种用于处理稀疏数据的新型稀疏感知算法和用于近似学习的理论上合理的加权分位数草图算法。我们的经验表明，缓存访问模式，数据压缩和分片是构建可扩展的端到端系统以实现提升树的基本要素。这些经验也可以应用于其他机器学习系统。通过结合这些经验，XGBoost能够使用最少量的资源解决大规模的实际问题。
---
版权声明：本文为CSDN博主「了不起的赵队」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/zhaojc1995/article/details/89238051
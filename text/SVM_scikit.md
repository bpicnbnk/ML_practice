scikit-learn SVM算法库封装了**libsvm 和 liblinear 的实现**，仅仅重写了算法了接口部分。

[链接](https://www.cnblogs.com/pinard/p/6117515.html)
# scikit-learn SVM算法库使用概述

scikit-learn中SVM的算法库分为两类，一类是分类的算法库，包括SVC， NuSVC，和LinearSVC 3个类。另一类是回归算法库，包括SVR， NuSVR，和LinearSVR 3个类。相关的类都包裹在sklearn.svm模块之中。

对于SVC， NuSVC，和LinearSVC 3个分类的类，SVC和 NuSVC差不多，区别仅仅在于对损失的度量方式不同，而LinearSVC从名字就可以看出，他是线性分类，也就是不支持各种低维到高维的核函数，仅仅支持线性核函数，对线性不可分的数据不能使用。

同样的，对于SVR， NuSVR，和LinearSVR 3个回归的类， SVR和NuSVR差不多，区别也仅仅在于对损失的度量方式不同。LinearSVR是线性回归，只能使用线性核函数。

我们使用这些类的时候，如果有经验知道**数据是线性可以拟合的，那么使用LinearSVC去分类 或者LinearSVR去回归**，它们不需要我们去慢慢的调参去选择各种核函数以及对应参数，速度也快。如果我们对数据分布没有什么经验，**一般使用SVC去分类或者SVR去回归，这就需要我们选择核函数以及对核函数调参了**。

什么特殊场景需要使用**NuSVC分类 和 NuSVR回归**呢？如果我们**对训练集训练的错误率或者说支持向量的百分比有要求**的时候，可以选择NuSVC分类 和 NuSVR 。它们有一个参数来控制这个百分比。

# 回顾SVM分类算法和回归算法

## SVM分类
对于SVM分类算法，其原始形式是：
$$min\;\; \frac{1}{2}||w||_2^2 +C\sum\limits_{i=1}^{m}\xi_i $$ 
$$ s.t.  \;\; y_i(w \bullet \phi(x_i) + b)  \geq 1 - \xi_i \;\;(i =1,2,...m)$$ 
$$\xi_i \geq 0 \;\;(i =1,2,...m)$$

其中m为样本个数，我们的样本为$(x_1,y_1),(x_2,y_2),...,(x_m,y_m)$。$w,b$是我们的分离超平面的$w \bullet \phi(x_i) + b = 0$系数, $\xi_i $为第i个样本的松弛系数， C为惩罚系数。$\phi(x_i)$为低维到高维的映射函数。

通过拉格朗日函数以及对偶化后的形式为：
$$ \underbrace{ min }_{\alpha}  \frac{1}{2}\sum\limits_{i=1,j=1}^{m}\alpha_i\alpha_jy_iy_jK(x_i,x_j) - \sum\limits_{i=1}^{m}\alpha_i $$ 
$$ s.t. \; \sum\limits_{i=1}^{m}\alpha_iy_i = 0 $$ 
$$0 \leq \alpha_i \leq C$$

其中和原始形式不同的$\alpha$为拉格朗日系数向量。$K(x_i,x_j) $为我们要使用的核函数。

## SVM回归

对于SVM回归算法，其原始形式是：
$$min\;\; \frac{1}{2}||w||_2^2 + C\sum\limits_{i=1}^{m}(\xi_i^{\lor}+ \xi_i^{\land}) $$ 
$$s.t. \;\;\; -\epsilon - \xi_i^{\lor} \leq y_i - w \bullet \phi(x_i ) -b \leq \epsilon + \xi_i^{\land}$$ 
$$\xi_i^{\lor} \geq 0, \;\; \xi_i^{\land} \geq 0 \;(i = 1,2,..., m)$$

其中m为样本个数，我们的样本为$(x_1,y_1),(x_2,y_2),...,(x_m,y_m)$。$w,b$是我们的回归超平面的$w \bullet x_i + b = 0$系数, $\xi_i^{\lor}， \xi_i^{\land}$为第i个样本的松弛系数， C为惩罚系数，$\epsilon$为损失边界，到超平面距离小于$\epsilon$的训练集的点没有损失。$\phi(x_i)$为低维到高维的映射函数。

通过拉格朗日函数以及对偶化后的形式为：
$$ \underbrace{ min}_{\alpha^{\lor}, \alpha^{\land}}\; \frac{1}{2}\sum\limits_{i=1,j=1}^{m}(\alpha_i^{\land} - \alpha_i^{\lor})(\alpha_j^{\land} - \alpha_j^{\lor})K(x_i,x_j) - \sum\limits_{i=1}^{m}(\epsilon-y_i)\alpha_i^{\land}+ (\epsilon+y_i)\alpha_i^{\lor}  $$ 
$$ s.t. \; \sum\limits_{i=1}^{m}(\alpha_i^{\land} - \alpha_i^{\lor}) = 0 $$ 
$$ 0 < \alpha_i^{\lor} < C \; (i =1,2,...m)$$ 
$$ 0 < \alpha_i^{\land} < C \; (i =1,2,...m)$$

其中和原始形式不同的$\alpha^{\lor}， \alpha^{\land}$为拉格朗日系数向量。$K(x_i,x_j) $为我们要使用的核函数。
# SVM核函数概述

在scikit-learn中，内置的核函数一共有4种，当然如果你认为线性核函数不算核函数的话，那就只有三种。

1）线性核函数（Linear Kernel）表达式为：$K(x, z) = x \bullet z $，就是普通的内积，LinearSVC 和 LinearSVR 只能使用它。

1)  多项式核函数（Polynomial Kernel）是线性不可分SVM常用的核函数之一，表达式为：$K(x, z) = (\gamma x \bullet z  + r)^d$ ，其中，$\gamma, r, d$都需要自己调参定义,比较麻烦。

3）高斯核函数（Gaussian Kernel），在SVM中也称为径向基核函数（Radial Basis Function,RBF），它是libsvm默认的核函数，当然也是scikit-learn默认的核函数。表达式为：$K(x, z) = exp(-\gamma||x-z||^2)$， 其中，$\gamma$大于0，需要自己调参定义。

4）Sigmoid核函数（Sigmoid Kernel）也是线性不可分SVM常用的核函数之一，表达式为：$K(x, z) = tanh(\gamma x \bullet z  + r)$， 其中，$\gamma, r$都需要自己调参定义。

一般情况下，对非线性数据使用默认的高斯核函数会有比较好的效果，如果你不是SVM调参高手的话，建议使用高斯核来做数据分析。
# SVM算法库其他调参要点

上面已经对scikit-learn中类库的参数做了总结，这里对其他的调参要点做一个小结。

1）一般推荐在做训练之前对数据进行归一化，当然测试集中的数据也需要归一化。。

2）在特征数非常多的情况下，或者样本数远小于特征数的时候，使用线性核，效果已经很好，并且只需要选择惩罚系数C即可。

3）在选择核函数时，如果线性拟合不好，一般推荐使用默认的高斯核'rbf'。这时我们主要需要对惩罚系数C和核函数参数进行艰苦的调参，通过多轮的交叉验证选择合适的惩罚系数C和核函数参数。

4）理论上高斯核不会比线性核差，但是这个理论却建立在要花费更多的时间来调参上。所以实际上能用线性核解决问题我们尽量使用线性核。
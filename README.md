#### GAE

**广义优势估计（Generalized Advantage Estimation）**是一种对于动作的优劣评判方法，它不同于**n-step A2C**中所用的优势估计，而是采取了整条轨迹上未来部分的所有**reward**，但是又不同于一般的**REINFORCE**，它也用到了所有未来部分的状态价值函数

首先我们定义$\delta_k$，意为第**k**步的**td error**：
$$
\delta_k=\gamma V(s_{k+1})+r(k)-V(s_k)
$$
接下来对轨迹进行拓展，具体来说，若终止态$s_\perp=s_n$，则对任何$m \geq n$，有：
$$
r(m)=0\\
V(s_m)=0
$$
这样子对任意$0 \leq k < \infty$，$\delta_k$都有定义

下面定义$\hat{A}_t^n$，它意为从第**t**步开始对后续**n**项$\delta$的加权和：
$$
\hat{A}_t^n=\sum_{k=t}^{t+n-1} \gamma^{k-t} \delta_k
$$
对其进行化简，有：
$$
\begin{align}
\hat{A}_t^n&=\sum_{k=t}^{t+n-1} \gamma^{k-t} \delta_k\\
&=\sum_{k=t}^{t+n-1} \gamma^{k-t} [\gamma V(s_{k+1})+r(k)-V(s_k)]\\
&=\sum_{k=t}^{t+n-1} \gamma^{k-t+1} V(s_{k+1})-\sum_{k=t}^{t+n-1} \gamma^{k-t} V(s_k)+\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\\
&=\left[\gamma^n V(s_{t+n})+\sum_{k=t}^{t+n-2} \gamma^{k-t+1} V(s_{k+1})\right]-\left[\sum_{k=t+1}^{t+n-1} \gamma^{k-t} V(s_k)+V(s_t)\right]+\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\\
&=\sum_{k=t}^{t+n-2} \gamma^{k-t+1} V(s_{k+1})-\sum_{k=t+1}^{t+n-1} \gamma^{k-t} V(s_k)+\gamma^n V(s_{t+n})-V(s_t)+\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\\
&=\underbrace{\sum_{k=t}^{t+n-2} \gamma^{k-t+1} V(s_{k+1})-\sum_{k=t}^{t+n-2} \gamma^{k-t+1} V(s_{k+1})}_{0}+\gamma^n V(s_{t+n})-V(s_t)+\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\\
&=\gamma^n V(s_{t+n})-V(s_t)+\sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\tag{1}
\end{align}
$$
根据**n-step A2C**所用的公式（[证明](https://github.com/Septemberlemon/A2C?tab=readme-ov-file#多步a2c)），我们知道：
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\gamma^n V(s_{t+n}) - V(s_t) + \sum_{k=t}^{t+n-1} \gamma^{k-t} r(k)\right]\nabla_\theta \ln \pi_\theta(a_t|s_t)\right]
$$
将**公式（1）**代入其中得：
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^n \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]
$$
可以对每个**n**分配权重$w_n$，限定$\sum_{n=1}^\infty w_n = 1$，则：
$$
\begin{align}
&\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\sum_{n=1}^\infty w_n \hat{A_t^n}\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
=&\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\sum_{n=1}^\infty w_n \hat{A_t^n}\gamma^t \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
=&\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{n=1}^\infty \sum_{t=0}^{T-1} w_n \hat{A_t^n}\gamma^t \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
=&\sum_{n=1}^\infty \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1} w_n \hat{A_t^n}\gamma^t \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
=&\sum_{n=1}^\infty w_n \mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\hat{A_t^n}\gamma^t \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
=&\sum_{n=1}^\infty w_n \nabla_\theta J(\theta)\\
=&\nabla_\theta J(\theta)\sum_{n=1}^\infty w_n\\
=&\nabla_\theta J(\theta)\tag{2}
\end{align}
$$
取权重为几何级数：
$$
w_n=(1-\lambda)\lambda^{n-1}
$$
其中$0 \leq \lambda \leq 1$，则当$0<\lambda<1$时，有：
$$
\frac{w_{n+1}}{w_n}=\lambda
$$
并满足：
$$
\begin{align}
\sum_{n=1}^\infty w_n&=\sum_{n=1}^\infty (1-\lambda) \lambda^{n-1}\\
&=(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1}\\
&=(1-\lambda)\lim_{n \to \infty}\frac{1-\lambda^n}{1-\lambda}\\
&=(1-\lambda)\cdot \frac{1}{1-\lambda}\\
&=1
\end{align}
$$
代入**公式（2）**得：
$$
\begin{align}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\sum_{n=1}^\infty w_n \hat{A_t^n}\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\sum_{n=1}^\infty (1-\lambda)\lambda^{n-1} \hat{A_t^n}\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} \hat{A_t^n}\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\tag{3}
\end{align}
$$
而：
$$
\begin{align}
\sum_{n=1}^\infty \lambda^{n-1} \hat{A}_t^n&=\sum_{n=1}^\infty \lambda^{n-1} \sum_{k=t}^{t+n-1} \gamma^{k-t} \delta_k\\
&=\delta_t+\lambda(\delta_t+\gamma \delta_{t+1})+\lambda^2(\delta_t+\gamma \delta_{t+1}+\gamma^2 \delta_{t+2})+\cdots\\
&=\sum_{n=0}^\infty \lambda^n \delta_t+\sum_{n=1}^\infty \lambda^n \gamma \delta_{t+1}+\sum_{n=2}^\infty \lambda^n \gamma^2 \delta_{t+2}+\cdots\\
&=\sum_{n=0}^\infty \lambda^n \delta_t+\sum_{n=0}^\infty \lambda^{n+1} \gamma \delta_{t+1}+\sum_{n=0}^\infty \lambda^{n+2} \gamma^2 \delta_{t+2}+\cdots\\
&=\sum_{n=0}^\infty \lambda^n \delta_t+\lambda \sum_{n=0}^\infty \lambda^n \gamma \delta_{t+1}+\lambda^2\sum_{n=0}^\infty \lambda^n \gamma^2 \delta_{t+2}+\cdots\\
&=\frac{1}{1-\lambda}\delta_t+\frac{\lambda}{1-\lambda}\gamma \delta_{t+1}+\frac{\lambda^2}{1-\lambda}\gamma^2 \delta_{t+2}+\cdots\\
&=\sum_{n=0}^\infty \frac{\lambda^n}{1-\lambda}\gamma^n \delta_{t+n}
\end{align}
$$
代入**公式（3）**得：
$$
\begin{align}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[(1-\lambda)\sum_{n=1}^\infty \lambda^{n-1} \hat{A_t^n}\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[(1-\lambda)\sum_{n=0}^\infty \frac{\lambda^n}{1-\lambda}\gamma^n \delta_{t+n}\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\sum_{n=0}^\infty \lambda^n\gamma^n \delta_{t+n}\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\sum_{n=0}^\infty (\lambda\gamma)^n \delta_{t+n}\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\tag{4}
\end{align}
$$
这就是**GAE**的具体做法

不难验证，当$\lambda=0$的时候，**GAE**就是**单步AC**，当$\lambda=1$的时候，**GAE**就是**REINFROCE with baseline**，我们知道后者偏差小，但是方差大，而前者方差小，但是偏差大

对**公式（4）**进行进一步的展开：
$$
\begin{align}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\sum_{n=0}^\infty (\lambda\gamma)^n \delta_{t+n}\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \left[\sum_{n=0}^\infty (\lambda\gamma)^n [\gamma V(s_{t+n+1})+r(t+n)-V(s_{t+n})]\right] \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
\end{align}
$$
当$0<\lambda<1$时，其中：
$$
\begin{align}
&\sum_{n=0}^\infty (\lambda\gamma)^n [\gamma V(s_{t+n+1})+r(t+n)-V(s_{t+n})]\\
=&\sum_{n=0}^\infty (\lambda\gamma)^n \gamma V(s_{t+n+1})+\sum_{n=0}^\infty (\lambda\gamma)^n r(t+n)-\sum_{n=0}^\infty (\lambda\gamma)^n V(s_{t+n})\\
=&\sum_{n=1}^\infty (\lambda\gamma)^{n-1} \gamma V(s_{t+n})-\sum_{n=0}^\infty (\lambda\gamma)^n V(s_{t+n})+\sum_{n=0}^\infty (\lambda\gamma)^n r(t+n)\\
=&\sum_{n=1}^\infty (\lambda\gamma)^{n-1} \gamma V(s_{t+n})-\sum_{n=1}^\infty (\lambda\gamma)^n V(s_{t+n})+\sum_{n=1}^\infty (\lambda\gamma)^n r(t+n)-V(s_t)+r(t)\\
=&\sum_{n=1}^\infty \frac{(\lambda\gamma)^n}{\lambda} V(s_{t+n})-\sum_{n=1}^\infty (\lambda\gamma)^n V(s_{t+n})+\sum_{n=1}^\infty (\lambda\gamma)^n r(t+n)-V(s_t)+r(t)\\
=&\sum_{n=1}^\infty (\lambda\gamma)^n [(\frac{1}{\lambda}-1) V(s_{t+n})+r(t+n)]-V(s_t)+r(t)\\
\end{align}
$$
从上式可以看出，$\lambda$越大，$V$所占的比例就越小，偏差越小，方差越大；反之$V$所占的比例就越大，偏差越大，方差越小

对于**公式（4）**，其一般记为：
$$
\nabla_\theta J(\theta)=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\tag{5}
$$

***

#### IS

**重要性采样（Importance Sampling）**旨在利用旧网络采来的轨迹进行新网络上的学习，进而大幅提升样本利用率，具体来说，**PPO**会先用当前策略采集一批轨迹，然后在这批轨迹上进行多次学习，在第二次及之后的学习时，当前网络已经不同于采样时的网络，梯度自然也不一样

由**公式（5）**有：
$$
\begin{align}
\nabla_\theta J(\theta)&=\mathbb{E}_{\tau \sim \pi_\theta} \left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\sum_\tau P(\tau)\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\sum_\tau \left[P(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)\right]\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\sum_\tau \left[P(s_0) \prod_{t=0}^{T-1} \pi_\theta(a_t|s_t)P(s_{t+1}|s_t,a_t)\frac{\pi_{old}(a_t|s_t)}{\pi_{old}(a_t|s_t)}\right]\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\sum_\tau \left[P(s_0) \prod_{t=0}^{T-1} \pi_{old}(a_t|s_t)P(s_{t+1}|s_t,a_t)\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\right]\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\sum_\tau \left[P(s_0) \prod_{t=0}^{T-1} \pi_{old}(a_t|s_t)P(s_{t+1}|s_t,a_t)\right]\left[\prod_{t=0}^{T-1}\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\right]\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\sum_\tau P_{old}(\tau)\left[\prod_{t=0}^{T-1}\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\right]\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_{old}}\left[\prod_{t=0}^{T-1}\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\right]\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\right]\\
&=\mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\left[\prod_{t=0}^{T-1}\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\right]\right]\\
&=\mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\left[\prod_{k=0}^{T-1}\frac{\pi_\theta(a_k|s_k)}{\pi_{old}(a_k|s_k)}\right]\right]\\
&=\mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \nabla_\theta \ln \pi_\theta(a_t|s_t)\left[\prod_{k=0}^{t-1}\frac{\pi_\theta(a_k|s_k)}{\pi_{old}(a_k|s_k)}\right]\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\left[\prod_{k=t+1}^{T-1}\frac{\pi_\theta(a_k|s_k)}{\pi_{old}(a_k|s_k)}\right]\right]\\
&=\mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_\theta(a_t|s_t)}\left[\prod_{k=0}^{t-1}\frac{\pi_\theta(a_k|s_k)}{\pi_{old}(a_k|s_k)}\right]\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\left[\prod_{k=t+1}^{T-1}\frac{\pi_\theta(a_k|s_k)}{\pi_{old}(a_k|s_k)}\right]\right]\\
&=\mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\left[\prod_{k=0}^{t-1}\frac{\pi_\theta(a_k|s_k)}{\pi_{old}(a_k|s_k)}\right]\left[\prod_{k=t+1}^{T-1}\frac{\pi_\theta(a_k|s_k)}{\pi_{old}(a_k|s_k)}\right]\right]\\
\end{align}
$$
我们假设后面两部分连乘为**1**，即可得：
$$
\begin{align}
\nabla_\theta J(\theta)&\approx \mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \frac{\nabla_\theta \pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\right]\\
&=\nabla_\theta \left(\mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t \hat{A}_t^{GAE} \frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}\right]\right)\\
\end{align}
$$
其中的$\frac{\pi_\theta(a_t|s_t)}{\pi_{old}(a_t|s_t)}$被称作**ratio**，记作$r_t(\theta)$，则有：
$$
\nabla_\theta J(\theta) \approx \nabla_\theta \left(\mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t r_t(\theta)\hat{A}_t^{GAE}\right]\right)
$$
括号内的部分被称作**代理目标函数**，一般记作$L^{CPI}(\theta)$，其中的CPI是**保守策略迭代（Conservative Policy Iteration）**的缩写：
$$
L^{CPI}(\theta)=\mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t r_t(\theta)\hat{A}_t^{GAE}\right]
$$
我们期望对**代理目标函数**求梯度能近似作为原目标函数的梯度

前面我们进行了连乘项的近似，这实际上需要先保证新策略的更新幅度不大，**PPO**的做法是进行**截断（CLIP）**，具体来说，它要求**ratio**处在一个给定的范围内，通常是：
$$
[1-\epsilon,1+\epsilon]
$$
当**ratio**处于范围之外时进行截断，这意味着梯度为**0**，将不再进行更新，这就一定程度上保证了新策略的更新幅度维持较小，除此之外，**PPO**还会进行一个额外的**min**操作：
$$
L^{CLIP}(\theta)=\mathbb{E}_{\tau \sim \pi_{old}}\left[\sum_{t=0}^{T-1}\gamma^t \min (r_t(\theta)\hat{A}_t^{GAE} ,\operatorname{clip}(r_t (\theta),1-\epsilon,1+\epsilon)\hat{A}_t^{GAE})\right]
$$
**min**和**clip**两个操作的组合产生的效果可以进行分类讨论，总的来说：

当$\hat{A}_t^{GAE}>0$时，则增大选择此动作的概率，但若**ratio**过大则不进行更新，这防止了此动作的概率过大

当$\hat{A}_t^{GAE}<0$时，则减小选择此动作的概率，但若**ratio**过小则不进行更新，这保证了此动作的概率不会被压缩到过小

同样的，前面的折扣部分往往省略

***

#### 算法实现

一般来说**PPO**会拿当前策略采取一定量的样本，然后用样本进行学习，这两步合称一个**iteration**，**PPO**总共训练若干个**iteration**。

在采样阶段，需要根据当前策略计算好**GAE**的值供学习时使用，另外，还需要计算**critic**的目标值（这一般被称作**return**），具体值为**GAE**加上$V(s_t)$，这也将在学习阶段被使用

在学习阶段，会重复进行多个**epoch**，在每个**epoch**内，一批一批从样本中取子集进行学习，直到遍历完整个样本集。对于取出的一批样本，根据采样阶段算得得**GAE**和**return**，分别进行**actor**和**critic**的学习

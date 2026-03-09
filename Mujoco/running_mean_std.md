$$
\begin{align}
\mu_{n+1}&=\frac{n\mu_n+x}{n+1}\\
&=\frac{(n+1)\mu_n+x-\mu_n}{n+1}\\
&=\mu_n+\frac{x-\mu_n}{n+1}
\end{align}
$$

设$\delta=x-\mu_n$，$M2_n=\sum_{i=1}^n (x_i-\mu_n)^2$，则有：
$$
\begin{align}
M2_{n+1}&=(x-\mu_{n+1})^2+\sum_{i=1}^n (x_i-\mu_{n+1})^2\\
&=(x-\mu_{n+1})^2+\sum_{i=1}^n [(x_i-\mu_n)-(\mu_{n+1}-\mu_n)]^2\\
&=(x-\mu_{n+1})^2+\sum_{i=1}^n(x_i-\mu_n)^2+\sum_{i=1}^n (\mu_{n+1}-\mu_n)^2-\sum_{i=1}^n 2(x_i-\mu_n)(\mu_{n+1}-\mu_n)\\
&=(x-\mu_{n+1})^2+M2_n+n(\mu_{n+1}-\mu_n)^2-2(\mu_{n+1}-\mu_n)\sum_{i=1}^n (x_i-\mu_n)\\
&=(x-\mu_{n+1})^2+M2_n+n(\mu_{n+1}-\mu_n)^2\\
&=M2_n+(x-\mu_n-\frac{x-\mu_n}{n+1})^2+n(\frac{x-\mu_n}{n+1})^2\\
&=M2_n+(\frac{n\delta}{n+1})^2+n(\frac{\delta}{n+1})^2\\
&=M2_n+\frac{(n^2+n)\delta^2}{(n+1)^2}\\
&=M2_n+\frac{n\delta^2}{n+1}
\end{align}
$$
记$\delta_2=x-\mu_{n+1}$，则：
$$
\begin{align}
\delta_2&=x-\mu_n-\frac{x-\mu_n}{n+1}\\
&=\frac{n\delta}{n+1}
\end{align}
$$
代入前面有：
$$
\begin{align}
M2_{n+1}&=M2_n+\frac{n\delta^2}{n+1}\\
&=M2_n+\delta \delta_2
\end{align}
$$
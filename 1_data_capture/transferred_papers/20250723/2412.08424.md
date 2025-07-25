# From Logistic Regression to the Perceptron Algorithm: Exploring Gradient Descent with Large Step Sizes

Alexander Tyurin

AIRI, Moscow, Russia Skoltech, Moscow, Russia alexandertiurin $@$ gmail.com

# Abstract

We focus on the classification problem with a separable dataset, one of the most important and classical problems from machine learning. The standard approach to this task is logistic regression with gradient descent $\scriptstyle ( \mathrm { L R + G D } )$ . Recent studies have observed that $\mathrm { L R + G D }$ can find a solution with arbitrarily large step sizes, defying conventional optimization theory. Our work investigates this phenomenon and makes three interconnected key observations about $\mathrm { L R + G D }$ with large step sizes. First, we find a remarkably simple explanation of why $\mathrm { L R + G D }$ with large step sizes solves the classification problem: $\mathrm { L R + G D }$ reduces to a batch version of the celebrated perceptron algorithm when the step size tends to infinity. Second, we observe that larger step sizes lead $\mathrm { L R + G D }$ to higher logistic losses when it tends to the perceptron algorithm, but larger step sizes also lead to faster convergence to a solution for the classification problem, meaning that logistic loss is an unreliable metric of the proximity to a solution. Surprisingly, high loss values can actually indicate faster convergence. Third, since the convergence rate in terms of loss function values of $\mathrm { L R + G D }$ is unreliable, we examine the iteration complexity required by $\mathrm { L R + G D }$ with large step sizes to solve the classification problem and prove that this complexity is suboptimal. To address this, we propose a new method, Normalized $\mathrm { L R + G D }$ —based on the connection between $\mathrm { L R + G D }$ and the perceptron algorithm—with much better theoretical guarantees.

# 1 Introduction

We consider the classical classification problem from machine learning with a dataset $\{ ( a _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ and two classes, where $a _ { i } ~ \in ~ \mathbb { R } ^ { d }$ and $y _ { i } \in \{ - 1 , 1 \}$ for all $i \ \in \ [ n ] \ : =$ $\{ 1 , \ldots , n \}$ . The goal of the classification problem is to

This is the supervised learning problem that finds a linear model (hyperplane) that separates the dataset. In general, this problem is infeasible, and one can easily find an example when the dataset is not linearly separable. We focus on the setup where the data is separable, which is formalized by the assumption:

# Assumption 1.1.

$$
\mu : = \operatorname* { m a x } _ { \| \theta \| = 1 } \operatorname* { m i n } _ { i \in [ n ] } y _ { i } a _ { i } ^ { \top } \theta > 0 .
$$

This condition ensures that for some $\boldsymbol { \theta } \in \mathbb { R } ^ { d }$ , the dataset can be perfectly classified. The quantity $\mu$ is a margin (Novikoff 1962; Duda, Hart, and G.Stork 2001), which characterizes the distance between the two classes. This assumption is practical in modern machine learning problems (Soudry et al. 2018; Ji and Telgarsky 2018). Albeit it is mostly attributed to large-scale nonlinear models (Brown et al. 2020), where the number of parameters is huge, the analysis of the methods in the linear case is equally important as it serves as a foundation for the nonlinear case. Let us define $R : = \operatorname* { m a x } _ { i \in [ n ] } \| a _ { i } \|$ .

There is a huge number of ways (Bishop and Nasrabadi 2006) how one can solve the problem, including support vector machines (SVMs) (Cortes and Vapnik 1995), logistic regression, and the perceptron algorithm (Novikoff 1962). This work focuses on the latter two, starting with logistic regression, which can be formalized by the following optimization problem:

$$
f ( \theta ) : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \log ( 1 + \exp ( - y _ { i } a _ { i } ^ { \top } \theta ) )  \operatorname* { m i n } _ { \theta \in \mathbb { R } ^ { d } } .
$$

This optimization problem does not have a finite minimum when the data is separable. Indeed, if $\theta$ separates the dataset, then $y _ { i } a _ { i } ^ { \top } \theta > 0$ for all $i \in [ n ]$ and $f ( c \cdot \theta ) \to 0$ , when $c \to \infty$ .

# Gradient Descent

The logistic regression problem (2) can be solved with gradient descent (GD) (Nesterov 2018), stochastic gradient descent (Robbins and Monro 1951), L-BFGS (Liu and Nocedal 1989), and variance-reduced methods (e.g., SAG, SVRG) (Schmidt, Le Roux, and Bach 2017; Johnson and Zhang 2013). We consider the GD method, arguably one of the simplest and most well-understood methods:1

$$
\theta _ { t + 1 } = \theta _ { t } - \gamma \nabla f ( \theta _ { t } ) ,
$$

where $\theta _ { 0 }$ is a starting point, $\gamma > 0$ is a step size, and $\nabla f ( \theta _ { t } )$ is the gradient of (2) at the point $\theta _ { t }$ .

What do we know about GD in the context of logistic regression $\mathbf { ( L R { + } G D ) }$ ? Surprisingly, despite the huge popularity of GD and logistic regression, we still lack a comprehensive understanding.

# Previous Work

Classical convex and nonconvex optimization theory. Let us recall the classical result for GD: it is well-known that if $\gamma \ < \ ^ { 2 } / L$ , and a function $f$ is $L -$ smooth and lower bounded, which is true for (2), then2 $\begin{array} { r } { f ( \theta _ { T } ) - \operatorname* { i n f } _ { \theta \in \mathbb { R } ^ { d } } f ( \theta ) = \widetilde { \mathcal { O } } \left( { 1 } / { \gamma T } \right) } \end{array}$ (Ji and Telgarsky 2018) for convex problems, or $\begin{array} { r } { \operatorname* { m i n } _ { t \in [ T ] } \| \nabla f ( \theta _ { t } ) \| ^ { 2 } \leq \mathcal { O } \left( { 1 } / { \gamma T } \right) } \end{array}$ for nonconvex problems. At the same time, if $\gamma > { ^ 2 } / { _ L }$ , then one can find a $L$ –smooth function such that GD diverges (see (Cohen et al. 2021, Sec.2)). Under $L$ –smothness, the value $^ 2 / L$ is special because it divides GD into the convergence and divergence regimes.

The edge of stability $\mathbf { ( E o S ) }$ and large step sizes. Nonetheless, in practice, it was many times observed (e.g., (Lewkowycz et al. 2020; Cohen et al. 2021)) that when a step size is large, $\gamma > ^ { 2 } / L$ , GD not only not diverges, but non-monotonically with oscillation converges on the task (2). This phenomenon was coined as the edge of stability (Cohen et al. 2021). This means that there is something special about the practical machine learning problems.

The mathematical aspects of the large step size regime have attracted significant attention within the research community, which analyzes the phenomenon through the sharpness of loss functions (the largest eigenvalue of Hessians) (Kreisler et al. 2023), small dimension problems (Zhu et al. 2022; Chen and Bruna 2022; Ahn et al. 2024), bifurcation theory (Song and Yun 2023), sharpness behavior in networks with normalization (Lyu, Li, and Arora 2022), 2-layer linear diagonal networks (Even et al. 2023), non-separable data (Ji and Telgarsky 2018; Meng et al. 2024), self-stabilization (Damian, Nichani, and Lee 2023; Ahn, Zhang, and Sra 2022; Ma et al. 2022; Wang, Li, and Li 2022). The papers by Wu, Braverman, and Lee (2024); Wu et al. (2024) are the closest to our research since they also analyze GD and logistic regression $\mathrm { ( L R + G D ) }$ ). They demonstrate that GD can converge with an arbitrary step size $\gamma > 0$ .

In particular, the results from ( $\mathrm { w } _ { \mathrm { u } }$ , Braverman, and Lee 2024) show that for any fixed $\begin{array} { r l r } { \gamma } & { { } > } & { 0 , \ f ( \theta _ { t } ) } \end{array}$ is approximately less than or equal to $\mathrm { p o l y } ( e ^ { \gamma } ) / { } _ { t }$ (plus additional terms that depend on other parameters). Wu et al. (2024) refined the dependence on $\gamma$ and demonstrated that GD with a large step size initially operates in the nonstable regime, where $\mathsf { \bar { f } } ( \theta _ { t } ) = \widetilde { \mathcal { O } } ( ( 1 + \gamma ^ { 2 } ) / \gamma t )$ . After approximately $\widetilde { \Theta } \big ( \operatorname* { m a x } \{ n , \gamma \} / \mu ^ { 2 } \big )$ iterati es, GD transitions to the stable regimee, where $\dot { f } ( \theta _ { t } ) = \widetilde { \mathcal { O } } ( 1 / \gamma t )$ . By tuning and taking the fixed step size $\gamma = \Theta ( T )$ , they get the accelerated rate $f ( \theta _ { T } ) = \widetilde { \mathcal { O } } ( 1 / T ^ { 2 } )$ .

# 2 Contributions

This paper delves deeper into understanding the dynamics of the non-stable regime, where the loss chaotically oscillates due to large step sizes. We explore logistic regression with gradient descent $\mathrm { ( L R + G D ) }$ and find that 1) $\mathrm { L R + G D }$ reduces to a batch version of the perceptron algorithm (Batch Perceptron), 2) the fastest convergence of $\mathrm { L R + G D }$ to a solution of (1) is achieved when step sizes and loss values are large, and 3) $\mathrm { L R + G D }$ is a suboptimal method, does not scale with the number of data points, and can be improved. Let us clarify:

1) We begin with a key observation that the iterates of $\mathrm { L R + G D }$ , when divided by the step size $\gamma$ , converge to the iterates of a batch version (Batch Perceptron) of the celebrated perceptron algorithm (Block 1962; Novikoff 1962) when $\gamma  \infty$ . In other words, $\mathrm { L R + G D }$ reduces to Batch Perceptron. The proof of this fact is straightforward and occupies less than half a page. This is an advantage of our paper because the typical proofs on this topic are technical and non-intuitive. When combined with the classical convergence results (Novikoff 1962), it offers a clear intuition and explanation for why the method solves (1) with large step sizes in the non-stable regime—a detail that, to our knowledge, has been previously overlooked and nonproven in the literature. (see Section 3)

![](images/db608c5f664ac0653c4c8dba2a5290daaded52d925644000425a7d67943613fc.jpg)  
Figure 1: Illustration on a subset of CIFAR-10 dataset (Krizhevsky, Hinton et al. 2009) with 5 000 samples and two classes. We run $\mathrm { L R + G D }$ with various step sizes. Note that there is no randomness involved in the process. Oscillation is a natural behavior of $\mathrm { L R + G D }$ with separable data and large step sizes.

When we test this fact, Theorem 3.2, with numerical experiments (see Figure 1), we observe that larger step sizes result in higher loss values (Figure 1a) before the moment when $\mathrm { L R + G D }$ attains Accuracy $= 1 . 0$ . Additionally, both loss and accuracy oscillate more (Figure 1a and 1b). And despite that, $\mathrm { L R + G D }$ solves (1) faster with large step sizes. Indeed, notice that $\mathrm { L R + G D }$ has the fastest convergence to Accuracy $= 1 . 0$ with the step sizes $\gamma \in \{ 1 . 0 , 1 0 . 0 , 1 0 0 . 0 \}$ , but at the same time, it has the highest loss values with these steps (more experiments in the next sections and appendix).

2) We investigate this phenomenon further and show that the logistic loss and the norm of gradients are unreliable metrics. We argue that the fact that a loss value $f ( \theta _ { t } )$ is small does not necessarily indicate that $\theta _ { t }$ is close to solving (1). Surprisingly, the opposite can be true. Our experiments and theorem show that high loss values may indicate fast convergence. (see Section 4)

3) This finding implies that when analyzing and developing methods for solving (1), we have to look at the number of iterations required by methods to solve (1) rather than relying solely on loss and gradient values. Therefore, we looked at the iteration complexity $n R ^ { 2 } / \mu ^ { 2 }$ of $\mathrm { L R + G D }$ with $\gamma  \infty$ and noticed that it is suboptimal with respect to $n$ since the iteration complexity $R ^ { 2 } { \big / } \mu ^ { 2 }$ can be attained by the classical (non-batch) perceptron algorithm (Perceptron). Moreover, we prove a lower bound, showing that the dependence on $n$ cannot be avoided in $\mathrm { L R + G D }$ with $\gamma  \infty$ . Provably, $\mathrm { L R + G D }$ is a suboptimal method with large step sizes. Finally, we slightly modify $\mathrm { L R + G D }$ and develop a new method, Normalized LR $+ \mathrm { G D }$ , basing on the connection between $\mathrm { L R + G D }$ and Batch Perceptron. This new method provably improves the iteration rate of $\mathrm { L R + G D }$ to $\scriptstyle R ^ { 2 } / \mu ^ { 2 }$ when $\gamma \to \infty$ . The new iteration rate to solve (1) is $n$ times better.

# 3 Reduction to the Batch Perceptron Algorithm

Before we state our first result, let us recall a batch version of the perceptron algorithm (Novikoff 1962; Duda, Hart, and G.Stork 2001):

Take the first step ${ \hat { \theta } } _ { 1 } = { \hat { \theta } } _ { 0 } + { \frac { 1 } { 2 n } } \sum _ { i = 1 } ^ { n } y _ { i } a _ { i } .$ For all $t \geq 1$ , find the set $S _ { t } : = \{ i \in [ n ] : y _ { i } { a _ { i } } ^ { \top } \hat { \theta } _ { t } \leq 0 \}$ , and take the step $\hat { \theta } _ { t + 1 } = \hat { \theta } _ { t } + \frac { 1 } { n } \sum _ { i \in S _ { t } } y _ { i } a _ { i }$ while $\vert S _ { t } \vert \neq 0$ , (Batch Perceptron) where $\hat { \theta } _ { 0 }$ is a starting point. This method finds all misclassified samples and uses them to find the next iterate $\widehat { \theta } _ { t + 1 }$ of the algorithm. Note that the classical version of the perceptron algorithm does the step only with one misclassified sample, as presented in (Perceptron).

We will require the following technical assumption in Theorem 3.2:

Assumption 3.1. (Non-Degenerate Dataset) For all $j \in [ n ]$ , the hyperplane $\begin{array} { r } { \{ x \ \in \ \mathbb { R } ^ { n ^ { - } } : \ \sum _ { i = 1 } ^ { n } x _ { i } \left. y _ { i } a _ { i } , y _ { j } a _ { j } \right. \ = \ 0 \} } \end{array}$ does not intersect the point $( 0 . 5 + k _ { 1 } , \ldots , 0 . 5 + k _ { n } )$ for all $k _ { 1 } , . . . , k _ { n } \in \mathbb { N } _ { 0 }$ .

This is a very weak assumption that cuts off pathological datasets since the chances that any hyperplane will intersect the countable set are zero in practice. Indeed, assume that $\begin{array} { r } { \{ x \in \mathbb { R } ^ { n } : \sum _ { i = 1 } ^ { n } x _ { i } \left. y _ { i } a _ { i } , y _ { j } a _ { j } \right. = 0 \} } \end{array}$ intersects some point $\left( 0 . 5 + k _ { 1 } , \ldots , 0 . 5 + k _ { n } \right)$ . For any arbitrarily small $\sigma > 0$ , let us take i.i.d. normal noises $\xi _ { 1 } , \ldots , \xi _ { n } \sim \mathcal { N } ( 0 , \sigma )$ . Then the probability that a slightly perturbed hyperplane $\{ x \in \mathbb { R } ^ { n } \ :$ $\begin{array} { r } { \sum _ { i = 1 } ^ { n } x _ { i } ( \dot { \langle { y _ { i } } { a _ { i } } , { y _ { j } } { a _ { j } } \rangle } + \dot { \xi } _ { i } { \bar { ) } } = 0 \} } \end{array}$ intersects any point $( 0 . 5 +$ $k _ { 1 } , \ldots , 0 . 5 + k _ { n } )$ is zero. We are ready to state and prove the first result:

Theorem 3.2. Let Assumption 1.1 hold. For $\gamma \  \ \infty$ and3 $\theta _ { 0 } = 0$ , the logistic regression with gradient descent $( L R { + } G D )$ reduces to the batch perceptron algorithm (Batch Perceptron), i.e., $\theta _ { t } / \gamma \to \hat { \theta } _ { t }$ for all $t \geq 0$ , with $\hat { \theta } _ { 0 } = 0$ if the dataset satisfies Assumption 3.1 (almost all datasets).

Proof. Clearly, we have

$$
\nabla f ( \theta ) = - \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( 1 + \exp ( y _ { i } a _ { i } ^ { \top } \theta ) ) ^ { - 1 } y _ { i } a _ { i }
$$

and $\theta _ { 1 } = \theta _ { 0 } - \gamma \nabla f ( \theta _ { 0 } )$ . Thus $\textstyle \theta _ { 1 } / \gamma \to { \hat { \theta } } _ { 1 } = { \frac { 1 } { 2 n } } \sum _ { i = 1 } ^ { n } y _ { i } a _ { i }$ and $\theta _ { 0 } / \gamma \to \hat { \theta } _ { 0 } = 0$ when $\gamma \to \infty$ . We now use mathematical induction, and assume that $\theta _ { t } / \gamma$ converges to $\widehat { \theta } _ { t }$ when $\gamma \to \infty$ . Using simple algebra, we get

$$
\begin{array} { c } { \displaystyle \frac { \theta _ { t + 1 } } { \gamma } = \frac { \theta _ { t } + \gamma \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 1 } { 1 + \exp ( y _ { i } a _ { i } ^ { \top } \theta _ { t } ) } y _ { i } a _ { i } } { \gamma } } \\ { = \displaystyle \frac { \theta _ { t } } { \gamma } + \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { 1 } { 1 + \exp ( \gamma \cdot y _ { i } a _ { i } ^ { \top } \frac { \theta _ { t } } { \gamma } ) } y _ { i } a _ { i } . } \end{array}
$$

For all $t \geq 1$ , notice that $\begin{array} { r } { \hat { \theta } _ { t } \ = \ \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( 0 . 5 + k _ { i } ) y _ { i } a _ { i } } \end{array}$ in Batch Perceptron for some $k _ { 1 } , \ldots , k _ { n } \ \in \ \mathbb { N } _ { 0 }$ . Using Assumption 3.1, we have4 $y _ { i } a _ { i } ^ { \top } \hat { \theta } _ { t } \ne 0$ for all $i \in [ n ]$ . Since $\theta _ { t } / \gamma ~  ~ \hat { \theta } _ { t }$ when $\gamma \  \ \infty$ , we get $\mathrm { s i g n } ( y _ { i } a _ { i } ^ { \top } \theta _ { t } / \gamma ) ~ =$ $\mathrm { s i g n } ( y _ { i } a _ { i } ^ { \top } \hat { \theta } _ { t } ) \neq 0$ for all $i \in [ n ]$ and $\gamma$ large enough. Therefore $( 1 + \exp ( \gamma \cdot y _ { i } a _ { i } ^ { \top } \frac { \theta _ { t } } { \gamma } ) ) ^ { - 1 }  1$ if $y _ { i } a _ { i } ^ { \top } \hat { \theta } _ { t } < 0$ , and $( 1 + \exp ( \gamma \cdot y _ { i } a _ { i } ^ { \top } \frac { \theta _ { t } } { \gamma } ) ) ^ { - 1 } \to 0$ if $y _ { i } a _ { i } ^ { \top } \hat { \theta } _ { t } > 0$ when $\gamma \to \infty$ for all $i \in [ n ]$ , meaning $\begin{array} { r } { \frac { \theta _ { t + 1 } } { \gamma } \stackrel { \gamma \to \infty } { = } \hat { \theta } _ { t } + \frac { 1 } { n } \sum _ { i \in S _ { t } } y _ { i } a _ { i } } \end{array}$ . We have showed that $\theta _ { t + 1 } / \gamma \to \hat { \theta } _ { t + 1 }$ . □

Thus, indeed, $\mathrm { L R + G D }$ reduces to Batch Perceptron when $\gamma \to \infty$ . It is left to recall the following classical result, which we prove in Section B for completeness.

Theorem 3.3. [(Novikoff 1962; Duda, Hart, and G.Stork 2001)] Let Assumption 1.1 hold. The batch perceptron algorithm (Batch Perceptron) solves (1) after at most

$$
\frac { n R ^ { 2 } } { \mu ^ { 2 } }
$$

![](images/55f7bdbb1205801d37922954f49ad98a87cfc98ff1ee04133eae1b60d3e54213.jpg)

Figure 3: We show that $\mathrm { L R + G D }$ with large step sizes aligns with Batch Perceptron on FashionMNIST.

iterations $i f { \hat { \theta } } _ { 0 } = 0$ .

Theorem 3.3 and Theorem 3.2 explain why $\mathrm { L R + G D }$ solves (1) with $\gamma \to \infty$ . Notice that the convergence rate (4) does not degenerate when $\gamma \to \infty$ . Crucially, we provide the convergence guarantees for the task (1), not for the task (2). The latter is merely a proxy problem. In practice, what matters is how fast we find a separator rather than how fast the loss converges to zero, and in fact, we will see in Section 4 that the logistic loss is an unreliable metric.

Remark: A scaled version of (2), $\begin{array} { r } { { \frac { 1 } { t \times n } } \sum _ { i = 1 } ^ { n } \log { \big ( } 1 + \exp ( - t \times y _ { i } a _ { i } ^ { \top } \theta ) { \big ) } } \end{array}$ , reduces to the perceptron loss when $t ~  ~ \infty$ . Thus, there can potential connection between $\mathrm { L R + G D }$ with large step sizes and this fact.

Numerical experiments. We now numerically verify the obtained results by comparing the performance of two algorithms that solve (1): logistic regression with gradient descent $\scriptstyle ( \mathrm { L R + G D } )$ and the perceptron algorithm (Batch Perceptron). Batch Perceptron has no hyperparameters, while $\mathrm { L R + G D }$ requires the step size $\gamma$ . We evaluate these algorithms on four datasets: CIFAR-10 (Krizhevsky, Hinton et al. 2009), FashionMNIST (Xiao, Rasul, and Vollgraf 2017), EuroSAT (Helber et al. 2019), and MNIST (LeCun, Cortes, and Burges 2010), selecting two classes and 5 000 samples from each dataset (see details in Section E). For $\mathrm { L R + G D }$ , we vary the step size from 0.001 to 100. Figures 2, 3, 8, and 10 present the results side by side. The results indicate that $\mathrm { L R + G D }$ with a small step size has monotonic and stable convergence curves. However, as the step size increases, the plots become unstable and chaotic. The behavior of $\mathrm { L R + G D }$ with large step sizes aligns closely with that of Batch Perceptron across all datasets almost exactly, which supports our theory. And in the limit of $\gamma \to \infty$ , converges to Batch Perceptron. We also run experiments with 1 000 and 10 000 samples in Section E for additional support.

# 4 Logistic Loss and the Norm of Gradients are Unreliable Metrics

Looking closer at the results of the experiments on datasets (Figures 4, 5, 9, and 11), we notice that the large step size not only leads to faster convergence rates but also to larger function values (before the moment when Accuracy $= 1 . 0$ ). Is this a coincidence, or is there some pattern? We can prove the following simple theorem that explains the phenomenon:

Theorem 4.1. Assume that $\theta _ { 1 } = 0$ . There exists a separable dataset (Assumption 1.1) such that

1. $f ( \theta _ { 1 } ) \to \infty$ and $f ( \theta _ { 2 } ) \to 0$ when $\gamma \to \infty$ , 2. $\theta _ { 2 } / \gamma$ is a solution of (1) when $\gamma \to \infty$ , 3. $\| \nabla f ( \theta _ { 1 } ) \|  \sqrt { 2 } / 2$ and $\| \nabla f ( \theta _ { 2 } ) \| \to 0$ when $\gamma \to \infty$ , where $\theta _ { 1 }$ and $\theta _ { 2 }$ are the first and second iterates of $L R { + } G D$ .

Proof. We take the dataset with one sample $( 1 , - 1 ) ^ { \top }$ assigned to the class 1 and one sample $( - 1 , - 4 ) ^ { \top }$ assigned to the class $- 1$ . Using $\mathrm { ( L R + G D ) }$ ) and (3), we have $\begin{array} { r l r } { \theta _ { 1 } } & { { } = } & { \gamma ( \frac { 2 } { 4 } , \frac { 3 } { 4 } ) ^ { \top } } \end{array}$ . Thus $\begin{array} { r l r } { f ( \theta _ { 1 } ) } & { = } & { \frac { 1 } { 2 } \left( \log \left( 1 + \exp ( \frac { \gamma } { 4 } ) \right) + \log \left( 1 + \exp ( - \frac { 7 \gamma } { 2 } ) \right) \right) , } \end{array}$ meaning $f ( \theta _ { 1 } ) \to \infty$ when $\gamma \to \infty$ . On the other hand, a direct calculation yield

$$
\begin{array} { r l } & { \frac { \theta _ { 2 } } { \gamma } = ( \frac { 2 } { 4 } , \frac { 3 } { 4 } ) ^ { \top } + \frac { 1 } { 2 } ( ( 1 + \exp ( - \frac { \gamma } { 4 } ) ) ^ { - 1 } ( 1 , - 1 ) ^ { \top }  } \\ & { \qquad + ( 1 + \exp ( \frac { 7 \gamma } { 2 } ) ) ^ { - 1 } ( 1 , 4 ) ^ { \top } )  ( 1 , \frac { 1 } { 4 } ) ^ { \top } } \end{array}
$$

when $\gamma \to \infty$ . The point $( 1 , \frac { 1 } { 4 } ) ^ { \top }$ is a solution of (1), and $f ( \theta _ { 2 } ) \to 0$ when $\gamma \to \infty$ . The last statement of the theorem can be verified using (3). □

1.0   
T LR+GD sp10 LR+GD SEp 100 LR+GDsp 0 LR+GD Step: 0.1 LR+GD Step: 0.1 LR+GD Step: 0.1 LR+GD Step: 0.01 10-6 LR+GD Step: 0.01 10-5 LR+GD Step: 0.01 LR+GD Step:0.001 LR+GD Step:0.001 $\angle R + G D$ Step: 0.001 0.7 # of iterations # of iterations # of iterations Figure 4: Accuracy, function values, and the norm of gradients of the logistic loss (2) on CIFAR-10 during the runs of $\mathrm { L R + G D }$ .   
国 SOTTPTP L+GDSep 1000 WWL LR+GDStp 100 LR+GDSsp1 LR+GD Step: 1.0 uu LR+GD Step: 1.0 LR+GD Step: 1.0 10-2 LR+GD Step: 0.01 LR+GD Step: 0.01 10-4 LR+GD Step: 0.01 LR+GD Step: 0.001 10-4 LR+GD Step: 0.001 $\mathsf { L R } + \mathsf { G D }$ Step: 0.001 0.90 # of iterations # of iterations # of iterations

Even though the first value $f ( \theta _ { 1 } )$ of the loss indicates divergence, the algorithm solves (1) after two steps when $\gamma  \infty !$ In this example, it is clear that the high value of $f ( \theta _ { 1 } )$ does not reflect the fact that the algorithm will solve the problem in the next step. The experiments from Figures 4 and 5 (see also Section E) support this theorem. That also applies to the norm of gradients. The ratio between $\| \nabla f ( \theta _ { 1 } ) \|$ and $\| \nabla f ( \theta _ { 2 } ) \|$ can be arbitrarily large for large $\gamma$ . In Figures 4 and 5 (see also Section E), the norm of gradients are chaotic and large until the moment when $\mathrm { L R + G D }$ finds a solution of (1).

# 5 $\mathbf { L R + G D }$ is a Suboptimal Method

In the previous section, we explain that logistic loss and the norm of gradient do not provide sufficient information about our proximity to solving (1). Recall that GD is a method of choice because, for instance, it is an optimal method in the nonconvex setting (Carmon et al. 2020) and has the optimal convergence rate by the norm of gradients. In the case of the task (1) and $\mathrm { L R + G D }$ , this is no longer true. Indeed, let us now consider the iteration rate $n R ^ { 2 } / \mu ^ { \bar { 2 } }$ from Theorem 3.3 by $\mathrm { L R + G D }$ when $\gamma \to \infty$ . The iteration rate is suboptimal since it linearly depends on $n$ , and can be improved by the classical (non-batch) perceptron algorithm (Novikoff 1962). The following lower bound proves that the dependence is unavoidable for Batch Perceptron.

Theorem 5.1. There exists a separable dataset (Assumption 1.1) with $\mu = \Theta ( 1 )$ and $R = \Theta ( 1 )$ such that Batch Perceptron $\scriptstyle ( L R + G D$ when $\gamma  \infty ,$ ) requires at least $\Omega ( n )$ iterations to solve (1) $i f { \hat { \theta } } _ { 0 } = 0$ and $n \geq 1 0$ .

Proof. We take the dataset with one sample $( 0 . 5 , - 1 ) ^ { \top }$ assigned to the class 1 and $n - 1$ samples $( - 0 . 5 , - 1 ) ^ { \top }$ assigned to the class $- 1$ . We start at the point $\widehat { \theta } _ { 0 } = ( 0 , 0 ) ^ { \top }$ . Then $\begin{array} { r } { \hat { \theta } _ { 1 } = \hat { \theta } _ { 0 } + \frac { 1 } { 2 n } \sum _ { i = 1 } ^ { n } y _ { i } a _ { i } = ( 0 . 2 5 , \frac { n - 2 } { 2 n } ) ^ { \top } } \end{array}$ . Only the sample from the class $- 1$ is misclassified at $\hat { \theta } _ { 1 }$ and belongs to $S _ { 1 }$ . Therefore $\begin{array} { r l r } { \hat { \theta } _ { 2 } } & { { } = } & { ( 0 . 2 5 , \frac { n - 2 } { 2 n } ) ^ { \top } + \frac { 1 } { n } ( 0 . 5 , - 1 ) ^ { \top } = } \end{array}$ $( 0 . 2 5 ( 1 + \textstyle { \frac { 2 } { n } } ) , \textstyle { \frac { n - 4 } { 2 n } } ) ^ { \top }$ . Again, only the sample from the class $- 1$ belongs to $S _ { 2 }$ . Thus $\begin{array} { r l r } { \hat { \theta } _ { 3 } } & { { } = } & { ( 0 . 2 5 ( 1 + \frac { 2 } { n } ) , \frac { n - 4 } { 2 n } ) ^ { \top } + } \end{array}$ $\begin{array} { r } { \frac { 1 } { n } ( 0 . 5 , - 1 ) ^ { \top } = ( 0 . 2 5 ( 1 + \frac { 4 } { n } ) , \frac { n - 6 } { 2 n } ) ^ { \top } } \end{array}$ . This will happen further with $\hat { \theta } _ { 4 } , \ldots , \hat { \theta } _ { k }$ until either the last coordinate becomes negative (the samples from the class 1 will be misclassified), i.e., n2−n2k < 0, or the sample from the class −1 stops being misclassified, i.e., $0 . 5 \times 0 . 2 5 ( 1 + { \frac { 2 k - 2 } { n } } ) + - 1 \times { \frac { n - 2 k } { 2 n } } > 0$ . Both conditions require $k$ to be greater or equal to $\Omega ( n )$ .

We have proved that $\mathrm { L R + G D }$ with $\gamma \to \infty$ is a suboptimal method. At the same time, it is well-known that we can improve the rate using the classical versions of the perceptron algorithm that yield better iteration rates:

Theorem 5.2 ((Duda, Hart, and G.Stork 2001),Theorem 5.1). The classical perceptron algorithm (Novikoff 1962), defined as

For all $t \geq 0$ , find the set $S _ { t } : = \{ i \in [ n ] : y _ { i } { a _ { i } } ^ { \top } \hat { \theta } _ { t } \leq 0 \}$ , choose $j \in S _ { t }$ and take the step $\widehat { \theta } _ { t + 1 } = \widehat { \theta } _ { t } + y _ { j } a _ { j }$ ,

(Perceptron)

solves (1) after at most $\textstyle { \frac { R ^ { 2 } } { \mu ^ { 2 } } }$ iterations $i f { \hat { \theta } } _ { 0 } = 0$ .

We can also consider a practical variant with proper normalization. Using a different normalization factor, $1 / \vert S _ { t } \vert$ instead of $^ 1 / n$ , we can provide better guarantees:

Theorem 5.3. [Proof in Section $D { \big / }$ The batch perceptron algorithm with a proper normalization, defined as For all $t \geq 0$ , find the set $S _ { t } : = \{ i \in [ n ] : y _ { i } { a _ { i } } ^ { \top } \hat { \theta } _ { t } \leq 0 \}$ , and take the step $\hat { \theta } _ { t + 1 } = \hat { \theta } _ { t } + \frac { 1 } { | S _ { t } | } \sum _ { i \in S _ { t } } y _ { i } a _ { i }$ while $\vert S _ { t } \vert \neq 0$ , (Normalized Batch Perceptron)

solves (1) after at most

$$
\frac { R ^ { 2 } } { \mu ^ { 2 } }
$$

![](images/ea1a09b595a6cde2d3269c24601e85796e02eaed96ac8b50de1f8adc5fd09cf4.jpg)

Figure 7: Comparison of perceptron algorithms on imbalanced data (see details in Section E). On two of three datasets Normalized Batch Perceptron converges to Accuracy $= 1 . 0$ faster than Batch Perceptron.

# iterations $i f { \hat { \theta } } _ { 0 } = 0$ .

One can see that Perceptron and Normalized Batch Perceptron have $n$ times better convergence rates than Batch Perceptron. The only difference between Normalized Batch Perceptron and Batch Perceptron is the proper normalization, which is crucial to get a better iteration rate.

Numerical experiments. In Figure 6, we compare Normalized Batch Perceptron and Batch Perceptron numerically and observe that Batch Perceptron converges slightly better in practice despite worse theoretical guarantees. However, in a setup with imbalanced data, described in Section E, we observe that Normalized Batch Perceptron finds a solution faster than Batch Perceptron (Figure 7). One research question is to uncover the reasons behind this. A potential highlevel explanation is that Batch Perceptron performs well “on average”, but not robust to inbalanced data.

# 6 A New Method, Normalized $\mathbf { L R + G D } _ { \mathrm { ~ : ~ } }$ , Yields a Faster Iteration Rate

Since Normalized Batch Perceptron converges faster than Batch Perceptron by $n$ times, it raises the question of whether it is possible to modify $\mathrm { L R + G D }$ and obtain a better rate. The answer is affirmative. We propose the following method:

$$
\theta _ { t + 1 } = \theta _ { t } - \gamma \beta _ { t } \nabla f ( \theta _ { t } ) , \qquad \mathrm { ( N o r m a l i z e d L R + G D ) }
$$

where $\nabla f ( \theta _ { t } )$ is the gradient of (2) at the point $\theta _ { t }$ , and

$$
\begin{array} { r } { \beta _ { t } = \left( \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( 1 + \exp ( y _ { i } a _ { i } ^ { \top } \theta _ { t } ) ) ^ { - 1 } \right) ^ { - 1 } . } \end{array}
$$

We reverse-engineered this method from Normalized Batch Perceptron, observing that $\begin{array} { r } { \nabla f ( \theta _ { t } )  \frac { 1 } { | S _ { t } | } \sum _ { i \in S _ { t } } y _ { i } a _ { i } } \end{array}$ (see Theorem 3.2) and $\beta _ { t } \to | S _ { t } |$ when $\gamma \to \infty$ . There are many ways how one can interpret it. For instance, it can be seen as $\mathrm { L R + G D }$ but with adaptive step sizes. Note that this method is specialized for the problem (2) because $\beta _ { t }$ requires the features $\{ a _ { i } \} _ { i = 1 } ^ { n }$ and labels $\{ y _ { i } \} _ { i = 1 } ^ { n }$ . We can prove that this method solves (1) faster than $_ \mathrm { L R + G D }$ :

Theorem 6.1. Let Assumption 1.1 hold. Normalized $L R { + } G D$ solves (1) after at most

$$
\frac { R ^ { 2 } } { \mu ^ { 2 } } + \frac { 2 \log ( 2 n - 1 ) } { \gamma \mu ^ { 2 } }
$$

iterations if $\theta _ { 0 } = 0$ .

The theorem suggests that we should increase the step size and let $\gamma \to \infty$ .

Numerical experiments. This theoretical insight is supported by the experiments from Table 1, where the best convergence rate is achieved with large step sizes. In Table 1, we compare the methods and observe that Normalized $\mathrm { L R + G D }$ is more robust to imbalanced datasets.

# 7 Conclusion

In this work, we analyze the classification problem (1) through the logistic regression (2). Our key takeaways are

1. Logistic regression and GD with large step sizes reduces to the celebrated (batch) perceptron algorithm, which can explain why $\mathrm { L R + G D }$ solves (1) even when $\gamma \to \infty$ .

2. We can not fully trust function and gradient values when optimizing logistic regression problems. The same caution applies to theoretical works. If a theoretical method has a good convergence rate based on function value residuals or gradient norms $( f ( \theta _ { t } ) - f ^ { * } \leq . . .$ or $\left. \nabla f ( \theta _ { t } ) \right. ^ { 2 } \leq$ . ), it does not necessarily mean that this method will perform well in practical machine learning tasks. In fact, the opposite may be true.

<html><body><table><tr><td>Dataset</td><td>Method</td><td>#of Iterations</td></tr><tr><td>“Worst-case” Dataset from Theorem 5.1</td><td>Normalized LR+GD Step:100.0 Normalized LR+GD Step: 10.0 Normalized LR+GD Step: 1.0 Normalized LR+GD Step: 0.1 LR+GD Step: 100.0</td><td>2 2 16 157 308</td></tr><tr><td>Imbalanced Fashion MNIST (Sec.E)</td><td>Normalized LR+GD Step:100.0 Normalized LR+GD Step: 10.0 Normalized LR+GD Step:1.0 Normalized LR+GD Step: 0.1 LR+GD Step: 100.0</td><td>311 349 353 504 538</td></tr><tr><td>Balanced Fashion MNIST (Sec.E)</td><td>LR+GD Step: 10.0 LR+GD Step: 100.0 Normalized LR+GD Step:100.0 Normalized LR+GD Step: 1.0 Normalized LR+GD Step:10.0</td><td>2778 2928 3160 3351 3359</td></tr></table></body></html>

3. While it is well-known that GD is an optimal method for nonconvex problems when measuring convergence by the norm of gradients (Carmon et al. 2020), and (almost5) optimal for convex problems when measuring convergence by function values (Nesterov 2018), the iteration rate of GD to solve (1) with logistic regression is suboptimal and does not scale with # of of data points $n$ in the worst case. This can be improved with the new Normalized $\mathrm { L R + G D }$ method.

# 8 Future Work

Nonlinear models and neural networks. The natural question is whether extending this work to nonlinear models and neural networks is possible. In the general case, we have to find a vector $\theta \in \mathbb { R } ^ { m }$ such that $y _ { i } g ( a _ { i } ; \theta ) > 0 \quad \forall i \in [ n ] ,$ where $g : \mathbb { R } ^ { d } \times \mathbb { R } ^ { m }  \mathbb { R }$ is a nonlinear mapping. The mathematical aspects in this section are not strict but rather serve as a foundation for future research since the general case with nonlinear models is very challenging and unexplored. All theorems from the previous sections heavily utilize the fact that $g$ is a linear model. If $g$ is a neural network, then it is well-known that logistic regression will diverge for large step sizes. Let us look at the gradient step with the logistic loss:

$$
\theta _ { t + 1 } = \theta _ { t } + \frac { 1 } { n } \sum _ { i = 1 } ^ { n } ( 1 + \exp ( y _ { i } g ( a _ { i } ; \theta _ { t } ) ) ) ^ { - 1 } \nabla _ { \theta } g ( a _ { i } ; \theta _ { t } ) .
$$

For the linear model, in the proof of Theorem 3.2, we show that $( 1 + \exp ( y _ { i } g ( a _ { i } ; \theta _ { t } ) ) ) ^ { \bar { - } 1 } \to \mathbf { 1 } [ y _ { i } g ( a _ { i } ; \theta _ { t } ) < 0 ]$ when $\gamma \to \infty$ . For the nonlinear models, it is not clear if it is true. Nevertheless, if we assume that $( 1 + \exp ( y _ { i } g ( a _ { i } ; \theta _ { t } ) ) ) ^ { - 1 } \approx$

$\mathbf { 1 } [ y _ { i } g ( a _ { i } ; \theta _ { t } ) < 0 ]$ , then

$$
\begin{array} { l } { \displaystyle \theta _ { t + 1 } \approx \theta _ { t } + \frac { 1 } { n } \sum _ { i \in S _ { t } } \nabla _ { \theta } g ( a _ { i } ; \theta _ { t } ) , } \\ { \displaystyle S _ { t } : = \{ i \in [ n ] : y _ { i } g ( a _ { i } ; \theta _ { t } ) \leq 0 \} } \end{array}
$$

which can be seen as a generalized perceptron algorithm. As far as we know, the analysis and convergence rates for this method have not been explored well. We believe these questions are important research endeavors.

Implicit bias. One of the main features of $\mathrm { L R + G D }$ is implicit bias. For small step sizes, $\gamma ~ < ~ 2 / L$ , Soudry et al. (2018); Ji and Telgarsky (2018) showed that the iterates of $\mathrm { L R + G D }$ not only solve (1), but also have a stronger property: $\theta _ { t } ~ \to ~ \theta _ { * }$ when $t  \infty$ , where $\theta _ { * } ~ =$ arg $\begin{array} { r } { \operatorname* { m a x } _ { \| \theta \| = 1 } \operatorname* { m i n } _ { i \in [ n ] } y _ { i } a _ { i } ^ { \top } \theta } \end{array}$ is the max-margin/SVM solution. From our observation, for $\gamma \to \infty$ , $\mathrm { L R + G D }$ reduces to Batch Perceptron, which generally does not return the max-margin solution. Therefore, to ensure the implicit bias property, one has to choose $\gamma < \infty$ , and intuitively, the larger $\gamma$ , the slower the convergence to the max-margin solution, but faster convergence to a solution of (1) according to the experiments and Theorem 6.1.

Theoretical guarantees of optimization methods. Typically, when researchers develop new methods, they compare them with previous methods using convergence rates by (loss) function values or by the norm of gradients. We believe that this work raises an important concern about this methodology in the context of machine learning tasks. While this work only analyzes the logistic loss with a linear model, the problem can be even more dramatic with more complex losses and models.
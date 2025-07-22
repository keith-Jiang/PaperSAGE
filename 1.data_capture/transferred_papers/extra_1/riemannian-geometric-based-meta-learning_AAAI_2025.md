# Riemannian Geometric-based Meta Learning

JuneYoung Park1,2\*, YuMi Lee3\*, Tae-Joon $\mathbf { K i m } ^ { 2 \dagger }$ , Jang-Hwan Choi3†

1Opt-AI Inc. 2Ajou University School of Medicine 3Ewha Womans University jyoung.park $@$ opt-ai.kr, eumi@ewha.ac.kr, tjkim $2 3 @$ ajou.ac.kr, choij@ewha.ac.kr

# Abstract

Meta-learning, or “learning to learn,” aims to enable models to quickly adapt to new tasks with minimal data. While traditional methods like Model-Agnostic Meta-Learning (MAML) optimize parameters in Euclidean space, they often struggle to capture complex learning dynamics, particularly in few-shot learning scenarios. To address this limitation, we propose Stiefel-MAML, which integrates Riemannian geometry by optimizing within the Stiefel manifold, a space that naturally enforces orthogonality constraints. By leveraging the geometric structure of the Stiefel manifold, we improve parameter expressiveness and enable more efficient optimization through Riemannian gradient calculations and retraction operations. We also introduce a novel kernel-based loss function defined on the Stiefel manifold, further enhancing the model’s ability to explore the parameter space. Experimental results on benchmark datasets—including Omniglot, MiniImageNet, FC-100, and CUB—demonstrate that StiefelMAML consistently outperforms traditional MAML, achieving superior performance across various few-shot learning tasks. Our findings highlight the potential of Riemannian geometry to enhance meta-learning, paving the way for future research on optimizing over different geometric structures.

# Introduction

Meta-learning is a framework designed to enhance the ability of models to rapidly adapt to new tasks by leveraging prior knowledge gained from diverse learning experiences (Vilalta and Drissi 2002; Vanschoren 2018). This approach is particularly effective in fields where data acquisition is limited, such as healthcare, natural language processing, robotics, recommendation systems, and finance (Rafiei et al. 2023; Hospedales et al. 2021; Vettoruzzo et al. 2024). By enabling models to generalize well from only a few examples, meta-learning offers a promising alternative to traditional deep learning methods, which typically require large amounts of data to perform well on new tasks.

One of the most widely used algorithms in meta-learning is Model-Agnostic Meta-Learning (MAML) (Finn, Abbeel, and Levine 2017), which focuses on finding model initialization parameters that allow for fast adaptation to new tasks with minimal updates. MAML operates through an inner loop of task-specific learning and an outer loop that updates the model based on its performance across multiple tasks. While effective, MAML’s reliance on parameter optimization in Euclidean space can limit its ability to capture the complex learning dynamics that arise as models scale in size and complexity (Rajeswaran et al. 2019; Antoniou, Edwards, and Storkey 2019).

In this paper, we propose Stiefel-MAML, which performs loss computations within a Riemannian manifold to more effectively reflect the characteristics of the loss surface in meta-learning, thereby enabling rapid adaptation. Riemannian manifold operations can better capture the geometric properties of data and models compared to Euclidean space (Boumal et al. 2014; Amari 1998). For instance, in Euclidean space, maintaining the orthogonality of a rotation matrix requires post-optimization procedures like Gram-Schmidt orthogonalization or projection operations to form a symplectic matrix, which can distort the optimization path (Mishra et al. 2014). However, when optimization is performed on a Riemannian manifold, the geometric properties of the manifold automatically satisfy these constraints, allowing the optimization path to reflect the true geometric structure without distortion. Additionally, in tasks like low-rank matrix completion, Riemannian optimization techniques can more effectively incorporate geometric properties (Vandereycken 2013).

Research by Liu and Boumal (Liu and Boumal 2020) further suggests that Riemannian optimization is more efficient and accurate in handling high-dimensional problems compared to methods used in Euclidean space. For example, applying Riemannian geometry to high-dimensional data analysis tasks such as minimum balanced cut, non-negative PCA, and K-means clustering improves both computational speed and accuracy. Similarly, Smith (Smith 2014) demonstrates that Newton’s method and conjugate gradient methods on Riemannian manifolds achieve faster and more efficient convergence compared to their Euclidean counterparts.

As illustrated in Figure 1, Stiefel-MAML introduces Riemannian geometry into the optimization loop, which allows the model to more effectively capture the geometric properties of the parameter space. By shifting the optimization process onto the Stiefel manifold—a space defined by orthogonal matrices—Stiefel-MAML naturally preserves the underlying geometric constraints throughout the learning process. This inherent preservation of structure, unlike previous studies that applied different manifolds to each task, facilitates effective optimization, even when operating on fixedmanifolds $( M )$ . The workflow of Stiefel-MAML follows two main loops. The inner loop performs Riemannian gradient descent on task-specific data, updating the model parameters while preserving the orthogonality constraints imposed by the Stiefel manifold. This is achieved through the use of a retraction operation that projects the updated parameters back onto the manifold after each gradient step. In the outer loop, the meta-parameters are updated by computing the meta-gradient over all tasks. This approach allows the model to adapt quickly to new tasks, capturing the curvature of the parameter space and preventing the optimization path from being distorted.

![](images/e0f7bc437abe161db734598dd66c0b2d0d4b433d3843adda46c31fe28ffb9354.jpg)  
Figure 1: Diagram of the Stiefel-MAML algorithm, which the adaptation of the traditional MAML method within a Riemannian manifold, enabling more efficient navigation and optimization on fixed-manifold $( M )$ .

Through experiments conducted on four benchmark datasets—Omniglot, Mini-ImageNet, FC-100, and CUB—we demonstrate that Stiefel-MAML consistently outperforms conventional MAML in terms of accuracy across all tasks in few-shot learning scenarios. The results indicate that leveraging Riemannian manifolds enables the model to better capture the geometric characteristics of the parameter space, guided by the curvature of the manifold, facilitating faster and more effective adaptation to new tasks.

In summary, the key contributions of this study are as follows:

• We propose Stiefel-MAML, a novel meta-learning algorithm that leverages Riemannian geometry to improve adaptation in few-shot learning tasks by optimizing model parameters on the Stiefel manifold. • We introduce a new kernel-based loss function defined on the Stiefel manifold, which enhances optimization efficiency and task adaptability by capturing the geometric structure of the parameter space. • We validate the effectiveness of Stiefel-MAML through extensive experiments on multiple benchmark datasets, demonstrating significant improvements in both accuracy and generalization over existing meta-learning al

gorithms.

• We provide a comprehensive analysis of the computational cost associated with Riemannian optimization, showing that the performance gains are achieved with only a marginal increase in computational overhead.

By introducing the Stiefel manifold into the meta-learning framework, this study offers a new perspective on optimizing learning processes in non-Euclidean spaces, demonstrating that geometric constraints can be effectively leveraged to enhance model adaptability and performance.

# Related Works

# Black-box Adaptation

Black box adaptation in meta-learning refers to a method where the model learns based solely on the input-output relationships without analyzing the internal learning processes. This approach enhances the model’s ability to generalize across different tasks, allowing it to quickly adapt to new tasks without the need for explicit access to the model’s internal state or gradient information. This contrasts to models such as Recurrent Neural Networks (RNNs) or Long ShortTerm Memory networks (LSTMs), which make predictions based on understanding how updates are processed when new inputs are received (Sherstinsky 2020). Such learning methods are analogous to the human ability to classify new tasks accurately even with limited examples, making them particularly valuable in fields where data acquisition is challenging.

MAML (Finn, Abbeel, and Levine 2017) is a type of optimization-based meta-learning that aims to learn generalized initial parameters enabling the model to quickly adapt to new tasks. It has been demonstrated that even without specific task information, MAML can adapt rapidly to new tasks, showcasing strong performance in few-shot learning scenarios and proving the model’s adaptability. However, these learning methods do not explicitly address uncertainty. Bayesian meta-learning was developed to address uncertainty directly, thereby improving model optimization based on prior research. BMAML (Yoon et al. 2018) uses a Bayesian approach to quantify Epistemic Uncertainty and learns a posterior distribution over parameters. PMAML (Finn, Xu, and Levine 2018) represents parameters as probability distributions through Variational Inference and learns a posterior distribution. By incorporating uncertainty into the learning process, these methods improve generalization performance and maximize adaptability to new tasks, thereby enhancing the overall performance of meta-learning.

Despite the advancements in improving generalization and optimization in meta-learning, challenges remain, particularly in capturing complex learning dynamics as models scale. Therefore, this paper proposes addressing these challenges through the use of Riemannian geometry on Stiefel manifolds.

# Metric-based Meta-learning in Riemannian Manifold and Hyperbolic

Metric-based Meta-Learning is a method that facilitates rapid learning of new tasks by relying on the distances between data points. This approach operates by embedding data points from similar classes close to each other in the embedded space, while data from different classes are placed farther apart (Koch et al. 2015; Satorras and Estrach 2018; Chen et al. 2020). Representative examples include Prototypical Networks, Matching Networks, and Relation Networks (Sung et al. 2018), which enable effective classification of new classes even with a limited amount of training data. These networks typically operate within Euclidean space, where their training processes have been conducted. To overcome this limitation, recent studies have employed embeddings based on Riemannian geometry, which better reflect more intricate geometric structures.

For instance, Gao et al. introduced a data modeling approach using angle-based sectional curvature and manifolds (Gao et al. 2022). They proposed a system of curvature generation and updating, which allows for the initialization of task-specific curvatures and their dynamic adjustment during the training process. This approach effectively captures the intrinsic geometry of the data, shortens the optimization trajectory, and achieves high performance with fewer optimization steps. Additionally, Li et al. (Li et al. 2023) proposed Geometry Flow, a metric learning method based on Riemannian geometry. This method defines the path, or “Flow,” that data traverses, learning the geometric transformations that occur as the data moves along the manifold. The Flow is modeled through Riemannian geometry, and this method has shown superior performance, particularly for high-dimensional manifold-structured data. Lastly, First-Order Riemannian Meta-Learning (FORML) (Tabealhojeh et al. 2024) is a first-order meta-learning method based on Riemannian geometry designed for learning on Stiefel manifolds. While traditional Hessian-free methods have been used in Euclidean space to avoid the complex computation of Hessians in second-order optimization problems (Martens et al. 2010; Absil, Mahony, and Sepulchre 2008; Boumal et al. 2014), FORML extends this approach to Stiefel manifolds, making it well-suited for meta-learning.

# Justification for Utilizing the Stiefel Manifold in Meta-Learning

In this study, we choose the Stiefel manifold for metalearning due to its ability to strictly maintain orthogonality while efficiently handling matrices of various dimensions (Massart and Abrol 2023; Huang, Wu, and Van Gool 2018). This property provides several advantages over other Riemannian manifolds. For example, the Grassmann manifold, which focuses on entire subspaces rather than individual basis vectors, is less effective for learning class-specific weight vectors (Huang, Wu, and Van Gool 2018). Similarly, while the Special Orthogonal Group enforces orthogonality, it is restricted to square matrices, limiting its usefulness for the non-square weight matrices commonly encountered in deep learning (Li et al. 2019). In contrast, the Stiefel manifold preserves orthogonality across a wide range of matrix shapes, reducing parameter correlations and mitigating overfitting—both of which are critical for rapid adaptation in meta-learning scenarios.

Additionally, optimization on the Stiefel manifold is inherently more efficient in terms of computation and memory usage compared to traditional Euclidean approaches. This increased efficiency provides a more direct and practical way to handle non-smooth optimization challenges, which in previous studies often required complex computational strategies (Tabealhojeh et al. 2024; Wang, Ma, and Xue 2022; Hu et al. 2024). By leveraging these strengths, the Stiefel manifold facilitates more effective and scalable solutions for meta-learning tasks.

# Method Conventional Meta-Learning

The core objective of few-shot meta-learning is to develop a model capable of rapidly adapting to new tasks using only a small amount of data and brief training (Gharoun et al. 2024; Jamal and Qi 2019). To achieve this, the learning model undergoes a meta-learning phase composed of various tasks. Through this process, the model becomes able to swiftly adapt to new tasks with just a few examples or attempts. In meta-learning, individual tasks are treated as single training instances. This framework is designed to enable the model to adapt to a variety of tasks represented by a distribution denoted as $p ( T )$ .

In a $K$ -shot learning environment, the goal is to train the model to quickly adapt to a new task $T _ { i }$ , randomly selected from $p ( T )$ , using only a small amount of information. Specifically, the model must learn using only $K$ examples and the evaluation criterion $\mathcal { L } _ { T _ { i } }$ for that task. The meta-learning process proceeds as follows: First, a task $T _ { i }$ is randomly chosen from $p ( T )$ , and $K$ training examples are extracted. The model learns using these $K$ examples and the evaluation criterion $\mathcal { L } _ { T _ { i } }$ . After learning, new test samples are drawn from $T _ { i }$ to evaluate the model’s performance.

Based on the test results, the parameters of model $f$ are updated, with the test errors from each $T _ { i }$ serving as the learning signal for the overall meta-learning process. Once meta-learning is complete, we assess the model’s generalization ability by selecting a new, unseen task from $p ( T )$ . The model is given only $K$ examples from this new task to learn from, and after training, its performance is measured to evaluate the effectiveness of the meta-learning. It is crucial to note that the tasks used for meta-testing are different from those used during meta-learning. This ensures that we are truly testing the model’s ability to adapt quickly to new tasks.

# Stiefel-MAML

The Stiefel-MAML algorithm is a novel approach developed to overcome the limitations of the existing MAML algorithm. The overall workflow of the algorithm can be found in Algorithm 1. MAML is a widely used technique in meta-learning, aimed at learning model initialization parameters that can quickly adapt to a variety of tasks (Yoon

Require: Task distribution $p ( \mathcal { T } )$ , learning rates $\alpha , \beta$ , number of inner loop steps $K$

Ensure: Model parameters $\theta$

1: Initialize $\theta$ on Stiefel manifold $S t ( n , p )$   
2: for each iteration do   
3: Sample batch of tasks $\{ \mathcal { T } _ { i } \} \sim p ( \mathcal { T } )$   
4: for each task $\mathcal { T } _ { i }$ do   
5: Initialize task-specific parameters $\theta _ { i } = \theta$   
6: for $k = 1$ to $K$ do   
7: Inner loop: Riemannian gradient descent on   
$S t ( n , p )$   
8: Compute loss ${ \mathcal { L } } _ { T _ { i } } ( \theta _ { i } )$   
9: Compute Riemannian gradient grad ${ \mathcal { L } } _ { { \mathcal { T } } _ { i } } ( \theta _ { i } )$   
10: Update parameters:

$$
\theta _ { i } \gets R _ { \theta _ { i } } ( - \alpha \cdot \mathrm { g r a d } \mathcal { L } _ { \mathcal { T } _ { i } } ( \theta _ { i } ) )
$$

11: end for

# 12: end for

13: Compute meta-gradient:

$$
\nabla _ { \theta } \sum _ { \mathcal { T } _ { i } } \mathcal { L } _ { \mathcal { T } _ { i } } ( R _ { \theta _ { i } } ( - \alpha \cdot \mathrm { g r a d } \mathcal { L } _ { \mathcal { T } _ { i } } ( \theta _ { i } ) ) )
$$

14: Update meta parameters $\theta$ :

$$
\theta  R _ { \theta } ( - \beta \cdot \nabla _ { \theta } \sum _ { \mathcal { T } _ { i } } \mathcal { L } _ { \mathcal { T } _ { i } } ( \theta ) )
$$

# 15: end for

et al. 2018). However, MAML performs optimization in Euclidean space, which is not ideal for problems involving orthogonality constraints. In such cases, performance may degrade if the parameters fail to maintain these constraints during the optimization process. To address this issue, StiefelMAML enhances MAML by utilizing optimization on the Stiefel manifold. The Stiefel manifold, denoted as $S t ( n , p )$ , represents the space of $n \times p$ orthogonal matrices. Mathematically, it is defined as:

$$
S t ( n , p ) = X \in \mathbb { R } ^ { n \times p } : X ^ { T } X = I _ { p }
$$

where $X ^ { T }$ is the transpose of matrix $X$ , and $I _ { p }$ is the $\textit { p } \times \textit { p }$ identity matrix. This manifold is ideal for problems requiring orthogonality constraints, as it naturally enforces these constraints throughout the optimization process. Unlike MAML, which relies on Euclidean optimization, Stiefel-MAML employs Riemannian optimization on the Stiefel manifold, better handling problems with orthogonality constraints.

The learning process of Stiefel-MAML involves two main loops: an inner loop that finds optimal parameters for each task and an outer loop that optimizes parameters across all tasks. In the inner loop, the model parameters $\theta$ are iteratively updated for each task, providing a common starting point for all tasks. When a task $T _ { i }$ is sampled from the task distribution $p ( T )$ , it comes with its own dataset and learning objective, prompting the model to adjust its parameters to meet these task-specific requirements.

In the inner loop, parameter updates are performed using a Riemannian gradient descent on the Stiefel manifold. This ensures that the parameters are optimized while maintaining orthogonality constraints. Specifically, the parameter $\theta _ { i }$ is updated to minimize the loss on the dataset for each task $T _ { i }$ . The optimization in the inner loop is carried out over $K$ steps of Riemannian gradient descent, represented by:

$$
\theta _ { k + 1 } = R _ { \theta _ { k } } \bigl ( - \alpha \cdot \mathrm { g r a d } \mathcal { L } _ { T } ( \theta _ { k } ) \bigr )
$$

Here, $\alpha$ represents the learning rate, $R _ { \theta }$ denotes the retraction operation, and grad ${ \mathcal { L } } _ { T } ( \theta )$ signifies the Riemannian gradient. The retraction operation, performed through QR decomposition, maps the parameter updates from the tangent space back onto the Stiefel manifold, ensuring the preservation of the manifold’s structure. The calculation of the Riemannian gradient on the Stiefel manifold involves projecting the gradient from Euclidean space onto the tangent space. The Riemannian gradient is given by the equation:

$$
\mathrm { g r a d } { \mathcal { L } } ( X ) { = } ( I { - } X X ^ { T } ) \nabla { \mathcal { L } } ( X ) { + } X s k e w ( X ^ { T } \nabla { \mathcal { L } } ( X ) ) 
$$

In this equation, $I$ represents the identity matrix, $\nabla { \mathcal { L } } ( X )$ denotes the Euclidean gradient, and $s k e w ( A )$ refers to the skew-symmetric part of matrix A. This process ensures stable optimization even in problems that require orthogonality constraints.

The outer loop of Stiefel-MAML aims to minimize the average loss across all tasks. Parameter updates in this loop are performed by calculating the meta-gradient, expressed as:

$$
\theta  R _ { \theta } ( - \beta \cdot \mathrm { g r a d } \mathcal { L } ( \theta ) )
$$

Here $\beta$ represents the meta-learning rate, and grad $\mathcal { L } ( \boldsymbol { \theta } )$ denotes the meta-gradient. The outer loop’s optimization is directed towards minimizing the average loss across various tasks, playing a crucial role in the meta-learning process.

Another important element in Stiefel-MAML is the kernel-based loss function. This loss function utilizes the geodesic distance on the Stiefel manifold to measure the similarity between two points.

Theorem 1 Let $S t ( n , p )$ be the Stiefel manifold of $n \times p$ orthonormal matrices. We define a positive definite kernel function $K : S t ( n , p ) \times S t ( n , p )  \mathbb { R } ^ { + }$ and a corresponding loss function $\mathcal { L } : S t ( n , p ) \times S t ( n , p )  \mathbb { R } ^ { + }$ is defined as: $\mathcal { L } ( X , Y ) = 1 - K ( X , Y )$

This kernel-based loss function measures the similarity between two points on the Stiefel manifold and is designed to decrease as similarity increases. This enables effective optimization in problems with orthogonality constraints. Specifically, this loss function plays a crucial role in naturally encoding orthogonality constraints by reflecting the geometric structure of the Stiefel manifold. The ultimate goal of Stiefel-MAML is to learn initial parameters that enable rapid adaptation to new tasks in problems with orthogonality constraints. This approach demonstrates higher generalization performance than the existing MAML algorithm, particularly excelling in problems where orthogonality constraints are important.

Table 1: The results of 1-shot (1) and 5-shot (5) learning for 3-way, 5-way, and 10-way tasks across four datasets (Omniglot, Mini-ImageNet, FC-100, and CUB). The table compares the accuracy (ACC) and $9 5 \%$ confidence intervals (CI) of MAML, FirstOrder(FO)-MAML, and Stiefel(S)-MAML, demonstrating the consistent performance improvement of S-MAML over the other methods. The 5-way experimental results on the Omniglot and Mini-ImageNet datasets for MAML have been extracted from Finn et al. (Finn, Abbeel, and Levine 2017).   

<html><body><table><tr><td>Dataset</td><td colspan="3">Omniglot</td><td colspan="3">Mini-ImageNet</td><td colspan="3">FC-100</td><td colspan="3">CUB</td></tr><tr><td>Num-Way</td><td>3</td><td>5</td><td>10</td><td>3</td><td>5</td><td>10</td><td>3</td><td>5</td><td>10</td><td>3</td><td>5</td><td>10</td></tr><tr><td>MAML (1) ACC</td><td>99.59</td><td>98.70</td><td>94.80</td><td>64.32</td><td>48.70</td><td></td><td>46.32</td><td>31.28</td><td>17.85</td><td>70.21</td><td>57.33</td><td>40.98</td></tr><tr><td>CI</td><td>±0.31</td><td>±0.40</td><td>±0.56</td><td>±1.65</td><td>±0.92</td><td></td><td>±1.45</td><td>±0.97</td><td>±0.66</td><td>±0.94</td><td>±1.15</td><td>±0.97</td></tr><tr><td>FO-MAML(1) ACC</td><td>98.13</td><td>96.82</td><td>94.67</td><td>61.09</td><td>46.77</td><td></td><td>43.21</td><td>30.25</td><td>16.95</td><td>70.24</td><td>56.86</td><td>40.55</td></tr><tr><td>CI</td><td>±0.55</td><td>±0.59</td><td>±0.48</td><td>±2.02</td><td>±1.48</td><td></td><td>±1.36</td><td>±0.83</td><td>±0.56</td><td>±1.66</td><td>±1.52</td><td>±1.84</td></tr><tr><td>S-MAML(1) ACC</td><td>99.67</td><td>98.25</td><td>98.02</td><td>63.78</td><td>50.49</td><td></td><td>49.07</td><td>34.81</td><td>20.33</td><td>73.11</td><td>60.15</td><td>43.08</td></tr><tr><td>CI</td><td>±0.30</td><td>±0.36</td><td>±0.33</td><td>±1.89</td><td>±1.41</td><td></td><td>±1.56</td><td>±1.00</td><td>±1.11</td><td>±1.21</td><td>±0.88</td><td>±0.90</td></tr><tr><td>MAML (5) ACC</td><td>99.79</td><td>99.90</td><td>98.64</td><td>76.31</td><td>63.11</td><td></td><td>56.12</td><td>42.12</td><td>26.46</td><td>83.73</td><td>74.71</td><td>59.65</td></tr><tr><td>CI</td><td>±0.25</td><td>±0.10</td><td>±0.18</td><td>±1.58</td><td>±0.46</td><td></td><td>±1.54</td><td>±1.07</td><td>±0.58</td><td>±1.37</td><td>±1.59</td><td>±1.08</td></tr><tr><td>FO-MAML (5) ACC</td><td>99.47</td><td>99.16</td><td>97.87</td><td>75.24</td><td>61.53</td><td></td><td>57.00</td><td>42.12</td><td>25.78</td><td>84.04</td><td>74.85</td><td>58.87</td></tr><tr><td>CI</td><td>±0.35</td><td>±0.18</td><td>±0.23</td><td>±1.78</td><td>±1.34</td><td></td><td>±1.46</td><td>±1.14</td><td>±0.70</td><td>±1.71</td><td>±1.34</td><td>±1.95</td></tr><tr><td>S-MAML (5) ACC</td><td>99.92</td><td>99.69</td><td>98.73</td><td>78.71</td><td>66.39</td><td></td><td>63.63</td><td>48.85</td><td>33.81</td><td>85.09</td><td>75.35</td><td>61.27</td></tr><tr><td>CI</td><td>±0.13</td><td>±0.10</td><td>±0.21</td><td>±1.37</td><td>±1.25</td><td></td><td>±1.41</td><td>±1.07</td><td>±0.73</td><td>±0.92</td><td>±1.14</td><td>±1.72</td></tr></table></body></html>

Through optimization on the Stiefel manifold, which can naturally encode orthogonality constraints, Stiefel-MAML enhances the efficiency of meta-learning and provides a model capable of swiftly adapting to various tasks.

# Experimental Setup

# Datasets

We conducted a series of experiments to evaluate the effectiveness of our proposed framework. These experiments primarily focused on few-shot classification, scenario analysis, and an ablation study. The datasets used for these evaluations include Omniglot, Mini-ImageNet, FC-100, and CUB, all of which are widely recognized in the field of metalearning. By employing these datasets, we comprehensively assessed the performance of our method across a broad range of tasks.

# Environments

We implemented our methodology using Python 3.8, PyTorch 1.8.1, and the torchmeta library (Deleu et al. 2019), ensuring a consistent experimental environment. The experiments were conducted using an NVIDIA A6000 (48G) GPU.

# Hyperparameters

Following the approach of Finn et al. (2017), we sampled 60,000 episodes for our experiments. We adopted the 4- convolution architecture described by Vinyals et al. (2016) to implement our parameter update method. The learning rates were set to 0.01 for the inner loop and 0.001 for the outer loop. Additionally, we matched the number of gradient steps in the inner loop to those used in the experiments by Finn et al. (2017).

# Cross-domain Few-shot Learning

The goal of meta-learning is not only to achieve high performance within the trained distribution but also to generalize well to out-of-distribution datasets. To test this, we conducted two cross-domain few-shot learning studies to evaluate whether performance could be maintained. The first scenario involved training on a general dataset and then conducting meta-testing on a specific dataset $\boldsymbol { G }  \boldsymbol { S }$ ). The second scenario involved training on a specific dataset and then conducting meta-testing on a general dataset $[ S  G ]$ ). For these experiments, we used the Mini-ImageNet dataset as the general dataset and the CUB dataset (Wah et al. 2011) as the specific dataset.

# Experimental Results

# Few-Shot Classification

Experiments were conducted on Stiefel-MAML, MAML, and FirstOrderMAML across four different datasets. We conducted 1-shot and 5-shot learning for 3-way, 5-way, and 10-way tasks, reporting both accuracy and $9 5 \%$ confidence intervals (CI), as detailed in Table 1. The results demonstrate that Stiefel-MAML consistently outperforms the other methods in most cases.

Compared to MAML, Stiefel-MAML showed an average improvement of: 1) $+ 0 . 4 8 \%$ on the Omniglot dataset; 2) $+ 1 . 7 3 \%$ on Mini-ImageNet; 3) $+ 5 . 0 5 \%$ on FC-100; 4) $+ 1 . 9 1 \%$ on the CUB dataset. Additionally, the following average improvements were observed for each shot: 1) $+ 1 . 8 8 \%$ in 1-shot, and 2) $+ 2 . 8 1 \%$ in 5-shot. Thus, our method has demonstrated its ability to effectively adapt to varying datasets and tasks while maintaining high accuracy.

An effect size analysis using Cohen’s d further supports Stiefel-MAML’s superiority, showing large effect sizes in both the 1-shot $\mathrm { ( d = 0 . 8 3 ) }$ ) and 5-shot $\mathrm { \check { d } } = 0 . 9 1$ ) scenarios. Additionally, the $5 . 0 5 \%$ improvement on the FC-100 dataset $( \mathtt { p } < 0 . 0 2 )$ demonstrates a statistically significant enhancement.

Table 2: The results of cross-domain few-shot learning, comparing each model (G: General, S: Specific). These results illustrate the flexibility of the algorithm, demonstrating how it remains adaptable and effective regardless of the dataset characteristics.   

<html><body><table><tr><td>Scenario</td><td>G→S</td><td>S→G</td></tr><tr><td>Train→Test</td><td>Mini→CUB</td><td>CUB→Mini</td></tr><tr><td>Num-Shot</td><td>1 5</td><td>1 5</td></tr><tr><td>MAML(5)ACC CI</td><td>47.81 63.35 ±1.34 ±1.23</td><td>28.81 39.50 ±0.72 ±0.74</td></tr><tr><td>FO-MAML (5) ACC CI</td><td>45.72 61.13 ±1.44 ±1.37</td><td>26.95 38.75</td></tr><tr><td>S-MAML (5) ACC CI</td><td>47.97 65.61 ±1.61 ±1.30</td><td>±1.07 ±1.15 30.94 40.13 ±1.71 ±1.53</td></tr></table></body></html>

# Cross-domain Few-shot Classification

The traditional machine learning paradigm assumes that training and testing data follow the same statistical patterns. However, in reality, unexpected distribution shifts can occur, leading to a decline in model performance when encountering unfamiliar data. This is known as the Out-OfDistribution (OOD) problem (Liu et al. 2021). Given the diverse and unpredictable nature of real-world problems, it is impossible to cover all possible scenarios with just the training data. Therefore, algorithms with strong OOD generalization capabilities can maintain performance even in unforeseen situations, making them effective in real-world environments.

Our proposed Stiefel-MAML algorithm effectively addresses the OOD generalization problem that traditional machine learning models often struggle with. Although existing models perform well only on test data similar to the training data, our algorithm maintains robust performance even when faced with unexpected distribution shifts. This means that Stiefel-MAML more effectively achieves the core goal of meta-learning, which is to “learn how to learn.” In the face of diverse and unpredictable real-world challenges, our algorithm demonstrates excellent adaptability, even in situations that it has not encountered during training. To validate this generalization ability, we introduced a new evaluation method called a “scenario study.” This approach assesses the algorithm’s domain generalization capability by using data from distributions not seen during the meta-testing phase.

Specifically, we designed two contrasting scenarios: one in which the model is trained on a general dataset and tested on a specific domain dataset, and the other where the reverse is done. We used various way and shot settings in these scenarios to test the flexibility of the algorithm. As shown in Table 2, we conducted evaluations using the Mini-ImageNet general dataset and the CUB specific domain dataset, focusing on 5-way tasks to validate the performance of our approach. In these results, our Stiefel-MAML outperformed the original MAML algorithm, demonstrating its robustness and high adaptability. Stiefel-MAML proves its ability to learn and generalize effectively even in new environments not encountered during training, making it a more suitable solution for complex and ever-changing real-world problems.

![](images/0e3f475860c1125d736830abfaf157b48fd1d6e8c1f70c242f084d3f7847432f.jpg)  
Figure 2: Gradient norm results for each task, which the comparative performance between Stiefel(S)-MAML and MAML. Higher gradient norms observed in S-MAML indicate steeper movements along the loss surface, suggesting more effective task adaptation.; x-axis: Adaptation steps; yaxis: Gradient norms

# Gradient Norm Comparison with Conventional Meta-Learning

In this experiment, we compare the magnitudes of the gradient norm between our proposed algorithm, Stiefel-MAML, and the MAML algorithm. The results of the experiment are shown in Figure 2. First, we evaluate the meta-learned model after performing several adaptation steps on unseen tasks during meta-testing.

The results show that our method consistently exhibits higher gradient norm values than the conventional MAML. This suggests that our algorithm is making steep movements along the loss surface to adapt to the tasks, whereas the conventional algorithms may be approaching local optima, making further improvements challenging. As evidenced by the results in Table 1, Stiefel-MAML consistently achieves higher accuracy than the existing algorithms, indicating that our algorithm is more effectively navigating towards the optimum. Furthermore, these results suggest that Riemannian manifolds allow for faster adaptation than Euclidean space.

# Computational Cost

Previous research on neural network algorithms leveraging Riemannian geometry has predominantly utilized symmetric positive definite matrices (SPDs) (Zhao et al. 2023), necessitating complex operations such as matrix exponentials and logarithms. Although these methods effectively capture the characteristics of the loss surface, they exhibit inefficiencies as a result of the need for specific designs tailored to each new manifold. To address this issue, residual networks have been proposed to generalize the geometry of manifolds (Katsman et al. 2024). However, even these approaches rely on Riemannian exponential maps, resulting in substantial computational demands.

Table 3: Comparison of computational time per iteration, comparing MAML and Stiefel(S)-MAML on the Omniglot dataset across different Num-Way settings. Despite the operations being carried out in Riemannian space, S-MAML does not show a significant increase in computational time compared to MAML.   

<html><body><table><tr><td>Dataset</td><td colspan="3">Omniglot</td></tr><tr><td>Num-Way</td><td>3</td><td>5</td><td>10</td></tr><tr><td>MAML(1) sec/iter</td><td>1.27</td><td>1.10</td><td>1.56</td></tr><tr><td>S-MAML(1) sec/iter</td><td>1.52</td><td>1.64</td><td>1.99</td></tr><tr><td>MAML (5) sec/iter</td><td>1.44</td><td>1.51</td><td>3.02</td></tr><tr><td>S-MAML(5) sec/iter</td><td>1.87</td><td>2.40</td><td>3.87</td></tr></table></body></html>

In contrast, our method efficiently captures the properties of Riemannian manifolds without resorting to exponential operations. When evaluated in the Omniglot dataset across different shots, our approach requires slightly more time compared to MAML, but the extent of this increase is not severe (as shown in Table 3). Consequently, Stiefel-MAML has proven to be an effective model that ensures high accuracy while demanding less computational capacity. This shows that our approach strikes a balance between computational efficiency and accuracy, making it a viable alternative in scenarios with limited computing resources.

# Feasibility of Application in Complex Architectures

Simple CNNs and ResNet differ significantly in their structural characteristics, such as network depth and residual connections, which can lead to variations in learning dynamics. Thus, verifying whether the meta-learning strategy of MAML remains effective in more complex architectures is a critical aspect of evaluation.

To address this, we conducted experiments to assess the performance of Stiefel-MAML compared to MAML on complex architectures like ResNet, rather than simpler CNNs. Using the Omniglot dataset, we evaluated both methods with ResNet50 as the backbone. The results, presented in Table 4, demonstrate that Stiefel-MAML outperforms MAML in the majority of tasks. These findings confirm that the proposed algorithm is not only effective on CNNs but also performs robustly on more complex models like ResNet, highlighting its high applicability across diverse architectures.

Table 4: Validation of Stiefel(S)-MAML applicability on ResNet-50 under various Num-Way settings. Unlike simpler CNN architectures, the increased complexity of ResNet-50 does not degrade S-MAML performance, thereby illustrating its robustness.   

<html><body><table><tr><td>Model</td><td colspan="3">ResNet-50</td></tr><tr><td>Num-Way</td><td>3</td><td>5</td><td>10</td></tr><tr><td>MAML(1) ACC CI</td><td>93.33 ±0.83</td><td>90.44 ±0.74</td><td>72.22 ±1.41 72.44</td></tr><tr><td>S-MAML(1) ACC CI MAML(5)ACC</td><td>94.44 ±0.40 89.26</td><td>93.33 ±0.44 74.22</td><td>±0.31 64.00</td></tr><tr><td>CI S-MAML (5)ACC CI</td><td>±0.36 86.30 ±0.42</td><td>±0.71 76.22 ±0.67</td><td>±0.61 65.44 ±0.69</td></tr></table></body></html>

# Conclusion

In this paper, we introduced Stiefel-MAML, an innovative algorithm that transitions parameter updates from Euclidean to non-Euclidean space, leveraging Riemannian geometry in the context of meta-learning. Our approach builds on the premise that non-Euclidean spaces, which inherently incorporate curvature information, can facilitate faster convergence and improve learning dynamics compared to traditional Euclidean spaces. Through extensive experiments, we demonstrated that Stiefel-MAML outperforms conventional algorithms in few-shot learning scenarios, showcasing superior adaptability across diverse datasets in various scenarios. Notably, these improvements are achieved without significant increases in computational cost, ensuring practical efficiency.

While prior research on non-Euclidean spaces has often focused on task-specific optimizations or employed varying manifolds for different tasks, we adopted a fixed-manifold approach using the Stiefel manifold. This contrasts with traditional meta-learning methods, which adapt to specific tasks by altering the underlying manifold. Our findings emphasize the potential of exploring parameter spaces beyond Euclidean geometry, prompting reflection on two fundamental questions: Why confine ourselves to Euclidean space? And must data always be represented as vectors?

By applying this algorithm to meta-learning, we demonstrated its capacity to address the challenges of data scarcity. Meta-learning, with its emphasis on learning and testing with limited data, represents a form of “data lightening” that is particularly valuable in domains where data is scarce. Our findings underscore the need for further research in these promising directions, expanding the possibilities for metalearning in real-world applications constrained by limited data.
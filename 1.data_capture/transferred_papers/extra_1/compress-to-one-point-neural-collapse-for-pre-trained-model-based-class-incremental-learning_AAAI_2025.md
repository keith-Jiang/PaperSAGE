# Compress to One Point: Neural Collapse for Pre-Trained Model-Based Class-Incremental Learning

Kun Wei, Zhe Xu, Cheng Deng

School of Electronic Engineering, Xidian University, Xian 710071, China weikunsk, zhexu.xd, chdeng.xd @gmail.com

# Abstract

Class-Incremental Learning (CIL) requires an artificial intelligence system to learn different tasks without class overlaps continually. To achieve CIL, some methods introduce the PreTrained Model (PTM) and leverage the generalized feature representation of PTM to learn downstream incremental tasks continually. However, the generalized feature representations of PTM are not adaptive and discriminative for these various incremental classes, which may be out of distribution for the pre-trained dataset. In addition, since the incremental classes cannot be learned at once, the class relationship cannot be constructed optimally, leading to undiscriminating feature representation for understream tasks. Thus, we propose a novel Pre-Trained Model-based Class-Incremental Learning (PTMCIL) method to explore the potential of PTM and obtain optimal class relationships. Inspired by Neural Collapse theory, we introduce the frozen Equiangular Tight Frame classifier to construct optimal classifier structure for all seen classes, guiding the feature representation adaptation for downstream continual tasks. Specifically, Task-Related Adaptation is proposed to modulate the generalized feature representation to bridge the gap between the pre-trained dataset and various downstream datasets. Then, the Feature Compression Module is introduced to compress various features to the specific classifier weights, constructing the feature transfer pattern and satisfying the characteristic of Neural Collapse. Optimal Structural Alignment is designed to supervise the feature compression process, assisting in achieving optimal class relationships across different tasks. Sufficient experiments on seven datasets prove the effectiveness of our method.

# Introduction

The complex streaming data (Guo et al. 2024)(Tao et al. 2021) in real-world situations requires artificial intelligence systems (Tao et al. 2022) to update themselves to continually adapt to novel tasks (Chen et al. 2021)(Dong et al. 2022)(Wei et al. 2022). Since the change of model parameters in the adaptation process, feature representation shifts will appear in the different incremental tasks and lead to the performance drop for previous tasks. In the incremental steps, the feature representation of the same data changes seriously, which is denoted as Catastrophic Forgetting (Robins 1995)(McCloskey and Cohen 1989) and limits artificial intelligence systems applied to real-world situations. Thus, Incremental Learning (Dong et al. 2024)(Kirkpatrick et al. 2017)(Wei et al. 2020) is a learning pattern to continually learn novel tasks without retraining previous tasks, improving model efficiency and performance. Benefitting from the generalization ability of Pre-Trained Models (PTM) trained on large-scale datasets, class-incremental learning methods can leverage the generalized feature representation to adapt to incremental tasks (Chen et al. 2022)(Wei et al. 2024)(Wang et al. 2022a), avoiding feature shifts and obtaining impressive performance compared with traditional class-incremental methods (Wei et al. 2021b)(Yan, Xie, and He 2021).

![](images/bae75938b02ed80da9d3c4b46ef39446ec27298c6819994ceda19d16394795b3.jpg)  
Figure 1: A sketch comparison between prior methods and our method. Since the classes of different steps cannot be learned simultaneously and the objective function may be unsuitable, the classifier structure is not optimal for all seen classes. Our method assigns the optimal classifier structure before the model optimization and freezes the classifier weight, guiding the feature representation adaptation.

However, the generalized feature representation of PTM is indiscriminative for these continual downstream tasks, limiting incremental performance. Many PTM-CIL methods (Wang et al. 2022c)(Wang et al. 2022b) adapt the generalized feature representation to streaming out-of-distribution tasks (Chen et al. 2023). These methods can be divided into prompt-based methods, representation-based methods, and model-mixture methods. Prompt-based (Wang et al. 2022c)(Wang et al. 2022b) methods leverage the stored prompts to update the PTM lightweight, where limited learnable parameters of prompts constrain the performance of feature transfer. Representation-based methods (Zhou et al. 2023a)(Zhou et al. 2023a) employ the adapter to change the generalized representation, where the continual update of the adapter still faces Catastrophic Forgetting. Model-mixture methods (Zhou et al. 2024)(Wang et al. 2023) aim to merge the different task models to obtain the generalized feature representation for the whole incremental learning. In addition, since the classes of different tasks do not overlap, these methods cannot construct and optimize the optimal class relationships of different tasks, leading to undiscriminating feature representations for downstream incremental classes. Thus, the key to tackling the PTM-CIL problem is to leverage the generalized feature representation to adapt to downstream tasks, and construct discriminative feature representation.

Inspired by Neural Collapse (Papyan, Han, and Donoho 2020), we propose a novel PTM-CIL method, which assigns the optimal classifier structure to guide the feature representation adaptation for downstream tasks. Neural Collapse elucidates an elegant phenomenon about the last-layer features and the classifier, and the following characteristics can concisely summarize it:

• Variability Collapse $( \mathcal { N C } _ { 1 } )$ : Within-class variability of features collapses to zero. • Convergence to Simplex ETF $( \mathcal { N C } _ { 2 } )$ : Class-mean features converge to a simplex Equiangular Tight Frame (ETF), achieving equal lengths, equal pair-wise angles, and maximal distance in the feature space. • Self-Duality $( \mathcal { N C } _ { 3 } )$ : Linear classifiers converge to classmean features, up to a global rescaling.

The requirement for Neural Collapse is consistent with the PTM-CIL problem shown as Figure 1, establishing the robust relationship between the last-layer feature representation and optimal classifier structure. Specifically, benefitting from the feature replay strategy, we introduce the Equiangular Tight Frame classifier to construct the optimal structural classifier for all seen classes. First, we design Task-Related Adaptation to modulate the feature representation for each transformer block of PTM, bridging the domain gap between the pre-trained dataset and various downstream datasets, which are out of distribution for the pre-trained dataset. Then, the Feature Compression Module is introduced to compress various and undiscriminating features to the assigned classifier weights, which satisfies $\mathcal { N C } _ { 1 }$ and $\mathcal { N C } _ { 3 }$ . Finally, Optimal Structural Alignment is designed to supervise the feature compression process to satisfy $\mathcal { N C } _ { 2 }$ , guiding the feature representation adaptation.

In summary, the contributions are as follows:

• To the best of our knowledge, we are the first to introduce Neural Collapse theory to Pre-Trained Model feature adaptation, providing a novel paradigm to leverage

Pre-Trained Model to tackle downstream tasks1.

• We construct an Optimal Structural Classifier and align the feature representation with the assigned optimal classifier structure, guaranteeing the generalized feature representation can be explored adequately for downstream tasks.   
• Sufficient experiments on seven datasets in different class-incremental settings prove the effectiveness of our method.

# Related Work

# Pre-Trained Model-Based Class-Incremental Learning

Class-incremental learning (Ashok, Joseph, and Balasubramanian 2022)(Yu et al. 2020)(Kirkpatrick et al. 2017)(Wei et al. 2021a) is a learning pattern where the model learns different tasks with non-overlapped classes in a continual manner. In order to address the challenges of class-incremental learning, a significant number of researchers make use of the Pre-Trained Model (PTM) as a means to support and enhance the Class-Incremental Learning process. These methods can be divided into three parts: prompt-based methods (Wang et al. 2022c) (Wang et al. 2022b), representationbased methods (Zhou et al. 2023a) (Zhou et al. 2023a), and model mixture-based methods (Zhou et al. 2023a) (Zhou et al. 2023a). Prompt-based methods utilize prompt-tuning strategies to bring about a minor alteration in the feature representation. In such methods, limited learnable parameters restrict the model’s capacity to adapt to long-sequence incremental tasks. Representation-based methods aim to utilize the generalized representation for constructing the classifier. However, they achieve unsatisfactory performance when dealing with out-of-distribution downstream datasets. Model mixturebased methods involve the design of a collection of models during the learning process. By employing model merging and model ensemble strategies, they seek to represent all tasks. Nevertheless, this approach leads to an increase in memory cost due to the growing number of model parameters. In addition, PTM-CIL methods can be developed in many other application fields, such as continual learning with pretrained large language models ( LLM ) (Zhang et al. 2024), learning with restricted computational resources (Prabhu et al. 2023) and multi-modal recognition (Zhou et al. 2023b). How to integrate Incremental Learning and various Pre-Trained Models is the future research direction.

# Neural Collapse

Neural Collapse (Papyan, Han, and Donoho 2020) explains an elegant phenomenon about the last-layer features and the classifier. Neural Collapse theory offers a mathematically sophisticated and refined characterization of the learned representations or features within deep learning-based classification models. It holds a remarkable degree of independence, being unaffected by various factors such as network architectures, dataset characteristics, and optimization algorithms.

This theory thus presents a highly generalized and fundamental understanding of the nature of learned features in the context of deep learning classification. Based on Neural Collapse, many methods (Fang et al. 2021; Graf et al. 2021) leverage the last-layer optimization to achieve balanced training. In addition, Neural Collapse theory is also employed to Transfer Learning (Galanti, Gyo¨rgy, and Hutter 2022, 2021), Incremental Learning (Yang et al. 2023), and architecture designs (Chan et al. 2022), bringing impressive performance improvement.

Unlike other PTM-CIL methods, we construct an optimal classifier structure for all seen classes to guide the feature adaptation for downstream tasks. In addition, the Feature Compress Module and Optimal Structural Alignment are designed to guarantee the feature adaptation satisfies the Neural Collapse theory, obtaining discriminative class relationships for downstream tasks. Sufficient experiments prove the effectiveness of our method for PTM-related adaptation tasks.

# Preliminaries

In this section, we introduce the background of PTM-CIL and Neural Collapse, which is the basic theory for our method.

# Pre-Trained Model-based Class-Incremental Learning

Class-incremental learning is the learning pattern where a model continually learns new tasks with novel classes. The whole incremental learning process can be divided into $T$ tasks, and the datasets for these tasks are denoted as $D =$ $\{ D _ { 1 } , D _ { 2 } , \ldots , D _ { T } \}$ , where $D _ { t } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { t } }$ is the t−th training step with $n _ { t }$ samples. In addition, the class sets of different tasks $C = \{ C _ { i } , \ldots , C _ { T } \}$ are not overlapped, where the number of all seen classes is $K _ { t }$ . The goal of CIL is to learn a unified classifier for all seen classes. Following traditional PTM-CIL methods, we denote a pre-trained model is available as the initialization for feature exactor $f \left( \cdot \right)$ and the linear classifier as $h \left( \cdot \right)$ .

# Equiangular Tight Frame Classifier

Neural Collapse refers to a phenomenon on balanced data, which reveals a geometric structure formed by the last-layer feature and classifier. A simplex Equiangular Tight Frame (ETF), refers to a matrix composed of $K$ vectors in $\mathcal { R } ^ { d }$ and satisfies:

$$
E = \sqrt { \frac { K } { K - 1 } } U \left( I _ { K } - \frac { 1 } { K } \boldsymbol { 1 } _ { K } \boldsymbol { 1 } _ { K } ^ { T } \right) ,
$$

where $E = [ e _ { 1 } , \dots , e _ { K } ] \in \mathbb { R } ^ { d \times k }$ , $U \in \mathbb { R } ^ { d \times k }$ allows a rotation and satisfies $U ^ { T } U = I _ { K }$ , $I _ { K }$ is the identity matrix, and $1 _ { K }$ is an all-ones vectors.

All column vectors in $E$ have the same $l _ { 2 }$ norm and any pair has an inner produce of $- \frac { 1 } { K - 1 }$ :

$$
e _ { k _ { 1 } } ^ { T } e _ { k _ { 2 } } = \frac { K } { K - 1 } \delta _ { k _ { 1 } , k _ { 2 } } - \frac { 1 } { K - 1 } , \forall k _ { 1 } , k _ { 2 } \in [ 1 , K ] ,
$$

where $\delta _ { k _ { 1 } , k _ { 2 } } = 1$ when $k _ { 1 } = k _ { 2 }$ and 0 otherwise.

Since Neural Collapse describes an optimal geometric structure of the last-layer feature and classifier, we pre-assign such optimality by fixing a learnable classifier as the structure. We adopt an ETF classifier that initializes a classifier as a simplex ETF and fixes it during training. Based on the total number of seen classes in different incremental steps, we reassign the classifier weights to achieve optimal structural alignment for the current step. We hope ETF can guide the generalized feature representation transfer to the optimal classifier structure.

# Methodology

Observing Neural Collapse can construct optimal classifier structure and guide the feature representation transfer, we aim to tackle the PTM-CIL problem with ETF classifier. Hence, we first introduce a Task-Related adapter to bridge the domain gap between the pre-trained dataset and downstream tasks that are out of distribution. Then, we introduce the Feature Compression Module to compress the feature to the assigned classifier weights, decreasing within-class variability and aligning with the specific simplex Equiangular Tight Frame. Finally, Optimal Structural Alignment is proposed to supervise the feature compression process. The frame of our method is shown as Figure 2.

# Task-Related Adaptation

Since the domain gap between the pre-trained dataset and the downstream task datasets, the generalized feature representation of the pre-trained model would be undiscriminating for downstream incremental tasks. Thus, we introduce lightweight adapter tuning in the base step to bridge the domain gap. Denote there are $L$ transformer blocks in the Pre-Trained backbone $f \left( \cdot \right)$ , each containing a self-attention module and an MLP (Multi-Layer Perceptron) layer. To preserve the generalization ability of the Pre-Trained model, we learn an adapter module as a side branch for the MLP to modulate the downstream task information into the generalized feature representation in this transformer block. Specifically, an adapter is a bottleneck module that contains a downprojection layer $W _ { d o w n } \in \mathbb { R } ^ { d \times r }$ , a non-linear activation function $\sigma$ , and an up-projection layer $W _ { d o w n } \in \mathbb { R } ^ { r \times d }$ . It adjusts the outputs of the MLP as:

$$
m _ { o } = \sigma \left( m _ { i } W _ { d o w n } \right) W _ { u p } + M L P \left( m _ { i } \right) ,
$$

where $m _ { i }$ and $m _ { o }$ represent the input and output of MLP, respectively. We designate the set of adapters among all $L$ transformer blocks as $A$ , and we refer to the adapted embedding function with adapter $A$ as $f \left( x ; A \right)$ . During the Task-Related Adaptation process, we keep the pre-trained weights fixed and only focus on optimizing the parameters of the adapters. In the remaining learning steps, both the weights of the Pre-Trained Model and those of the adapters are frozen. This is done to maintain the generalized feature representation, which is crucial for representing the unseen classes.

# Feature Compression Module

Since the ETF classifier is introduced as the classifier to guide feature representation optimization, we need to guide feature representation transfer to the specific classifier weight, following the Neural Collapse characteristics. Thus, we introduce the Feature Compression Module (FCM) to compress the various features. After the feature extractor, we add Feature Compression Module $P \left( \cdot \right)$ to transform the generalized feature representation to adjust the pre-assign ETF class prototypes. Feature Compression Module $P \left( \cdot \right)$ is constructed by a two-layer MLP and BatchNorm layer. In the Incremental learning steps, $P \left( \cdot \right)$ will be fine-tuned to adapt to the pre-assign class prototype continually:

![](images/aec1ee8674a20a5d6084dbf8a8925c2c0fcbb1fd1ae706cb3c45b0047ce3049e.jpg)  
Figure 2: Our method’s framework comprises a Pre-Trained Backbone, a Task-Related adapter, a Feature Compression Module, and an ETF classifier. The ETF classifier represents the optimal classifier structure for guiding feature representation adaptation. During all Incremental learning processes, the Pre-Trained Backbone is frozen to maintain the generalization of feature representation. In the base step, the Task-Related Adapter is utilized to adjust the feature representation of the Pre-Trained Backbone to suit downstream tasks. Thanks to the Feature Replay strategy, the Feature Compression Module can compress the features of all seen classes into the assigned classifier weights. Optimal Structural Alignment is incorporated to supervise the feature compression process, ensuring the Neural Collapse characteristic.

$$
z _ { i } = f ( x _ { i } ) , \quad \hat { z } _ { i } = P ( z _ { i } ) ,
$$

where $z _ { i }$ and $\hat { z } _ { i }$ are the extracted and compressed features respectively.

To alleviate representation overfitting for ETF classifier, we introduce the mixup strategy to increase the feature diversity, which can be denoted as:

$$
\tilde { z } _ { i } = \lambda \hat { z } _ { i } + ( 1 - \lambda ) \hat { z } _ { j } , \quad \tilde { y } _ { i } = \lambda y _ { i } + ( 1 - \lambda ) y _ { j } ,
$$

where $\lambda$ is the random hyper-parameter to balance the samples $x _ { i }$ and $x _ { j }$ . Then, we leverage the cross-entropy to optimize the learnable parameters $\bar { P _ { \bf \Lambda } } ( \cdot )$ , denoted as:

$$
\mathcal { L } _ { c e } \left( \tilde { z } _ { i } , \tilde { y } _ { i } \right) = - l o g \frac { e x p \left( \tilde { z } _ { i } / \tau \right) } { \sum _ { j = 1 } ^ { K } e x p \left( \tilde { z } _ { j } / \tau \right) } ,
$$

where $\tau$ is the temperature parameters.

# Optimal Structural Alignment

Since both the Pre-Trained Model and the adapter are frozen in the incremental steps, we need to supervise the feature compression process, guaranteeing the feature transfer satisfies the Nerual Collapse characteristic. In addition, we also want to capture the class relationships among classes from different tasks. Thus, we design two feature replay strategies to introduce the features of previous tasks in the current training step.

• Full Memory (FM) : construct the memory $M _ { F M }$ to store the features $z$ of all samples; • Class Memory(CM): construct the memory $M _ { C M }$ to collect mean $\mu$ and covariance $\Sigma$ of all classes.

Owing to the utilization of the Pre-Trained Model, which yields well-distributed and generalized representations, each class exhibits a tendency to be single-peaked and can thus be intuitively modeled as a Gaussian $\bar { \mathcal { N } _ { \mathrm { } } } ( \mu _ { c } , \Sigma _ { c } )$ . We employ the re-parameter strategy to replay the features of prior tasks. By means of the features generated from the CM, we are able to acquire a substantial amount of simulation features for each class to achieve Optimal Structural Alignment. In all experiments, the replay number $\rho$ is set to 256.

Benefitting from two feature replay strategies, we can reassign the weight of the ETF classifier to preserve the optimal classifier structure following the seen class number. To compress the various features, we need to align the feature and the assigned corresponding classifier weight $e _ { i }$ . Since the capacity gap between random re-assigned classifier weight and the feature representation is large, the optimal classifier alignment becomes challenging. The Batchnorm layer within the Feature Compression Module is capable of minimizing collapse (Tian, Krishnan, and Isola 2019). It causes far fewer singular values to shrink towards zero, thereby fulfilling the requirement of $\mathcal { N } \mathcal { C } _ { 1 }$ .

Table 1: Average and last performance comparison on seven datasets with ViT-B/16-IN21K as the backbone. ‘CF’ stands for ‘CIFAR100,’‘IN-R/A’ stands for ‘ImageNet-R/A,’ ‘ON’ stands for ‘ObjectNet’, ‘OB’ stands for ‘OmniBenchmark’ and ’VB’ stands for ’VTAB’. The best performance is shown in bold. The second performance is shown with underline.   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="2">CF BO Inc5</td><td colspan="2">CUB BO Inc10</td><td colspan="2">IN-R BO Inc5</td><td colspan="2">IN-A BO Inc20</td><td colspan="2">ON B0 Inc10</td><td colspan="2">OB B0 Inc30</td><td colspan="2">VB B0 Inc10</td></tr><tr><td>A</td><td>At</td><td>A</td><td>At</td><td>A</td><td>At</td><td>A</td><td>At</td><td>A</td><td>At</td><td>A</td><td>At</td><td>A</td><td>At</td></tr><tr><td>Finetune</td><td>38.91</td><td>20.17</td><td>26.08</td><td>13.96</td><td>21.61</td><td>10.79</td><td>24.28</td><td>14.51</td><td>19.14</td><td>8.73</td><td>23.61</td><td>10.57</td><td>34.95</td><td>21.25</td></tr><tr><td>Finetune Adapter</td><td>60.51</td><td>49.32</td><td>66.84</td><td>52.99</td><td>47.59</td><td>40.28</td><td>45.41</td><td>41.10</td><td>50.22</td><td>35.95</td><td>62.32</td><td>50.53</td><td>48.91</td><td>45.12</td></tr><tr><td>LwF</td><td>46.29</td><td>41.07</td><td>48.97</td><td>32.03</td><td>39.93</td><td>26.47</td><td>37.75</td><td>26.84</td><td>33.01</td><td>20.65</td><td>47.14</td><td>33.95</td><td>40.48</td><td>27.54</td></tr><tr><td>SDC</td><td>68.21</td><td>63.05</td><td>70.62</td><td>66.37</td><td>52.17</td><td>49.20</td><td>29.11</td><td>26.63</td><td>39.04</td><td>29.06</td><td>60.94</td><td>50.28</td><td>45.06</td><td>22.50</td></tr><tr><td>L2P</td><td>85.94</td><td>79.93</td><td>67.05</td><td>56.25</td><td>66.53</td><td>59.22</td><td>49.39</td><td>41.71</td><td>63.78</td><td>52.19</td><td>73.36</td><td>64.69</td><td>77.11</td><td>77.10</td></tr><tr><td>DualPrompt</td><td>87.87</td><td>81.15</td><td>77.47</td><td>66.54</td><td>63.31</td><td>55.22</td><td>53.71</td><td>41.67</td><td>59.27</td><td>49.33</td><td>73.92</td><td>65.52</td><td>83.36</td><td>81.23</td></tr><tr><td>CODA-Prompt</td><td>89.11</td><td>81.96</td><td>84.00</td><td>73.37</td><td>64.42</td><td>55.08</td><td>53.54</td><td>42.73</td><td>66.07</td><td>53.29</td><td>77.03</td><td>68.09</td><td>83.90</td><td>83.02</td></tr><tr><td>SimpleCIL</td><td>87.57</td><td>81.26</td><td>92.20</td><td>86.73</td><td>62.58</td><td>54.55</td><td>59.77</td><td>48.91</td><td>65.45</td><td>53.59</td><td>79.34</td><td>73.15</td><td>85.99</td><td>84.38</td></tr><tr><td>ADAM+Finetune</td><td>87.67</td><td>81.27</td><td>91.82</td><td>86.39</td><td>70.51</td><td>62.42</td><td>61.01</td><td>49.57</td><td>61.41</td><td>48.34</td><td>73.02</td><td>65.03</td><td>87.47</td><td>80.44</td></tr><tr><td>ADAM+VPT-S</td><td>90.43</td><td>84.57</td><td>92.02</td><td>86.51</td><td>66.63</td><td>58.32</td><td>58.39</td><td>47.20</td><td>64.54</td><td>52.53</td><td>79.63</td><td>73.68</td><td>87.15</td><td>85.36</td></tr><tr><td>ADAM+VPT-D</td><td>88.46</td><td>82.17</td><td>91.02</td><td>84.99</td><td>68.79</td><td>60.48</td><td>58.48</td><td>48.52</td><td>67.83</td><td>54.65</td><td>81.05</td><td>74.47</td><td>86.59</td><td>83.06</td></tr><tr><td>ADAM+ SSF</td><td>87.78</td><td>81.98</td><td>91.72</td><td>86.13</td><td>68.94</td><td>60.60</td><td>61.30</td><td>50.03</td><td>69.15</td><td>56.64</td><td>80.53</td><td>74.00</td><td>85.66</td><td>81.92</td></tr><tr><td>ADAM+Adapter</td><td>90.65</td><td>85.15</td><td>92.21</td><td>86.73</td><td>72.35</td><td>64.33</td><td>60.47</td><td>49.37</td><td>67.18</td><td>55.24</td><td>80.75</td><td>74.37</td><td>85.95</td><td>84.35</td></tr><tr><td>EASE</td><td>91.51 93.51</td><td>85.80</td><td>92.23</td><td>86.81</td><td>78.31</td><td>70.58</td><td>65.34</td><td>55.04</td><td>70.84</td><td>57.86</td><td>81.11</td><td>74.85</td><td>93.61</td><td>93.55</td></tr><tr><td>RanPAC</td><td></td><td>89.30</td><td>93.13</td><td>89.40</td><td>75.74</td><td>68.75</td><td>64.16</td><td>52.86</td><td>71.67</td><td>60.08</td><td>85.95</td><td>79.55</td><td>92.56</td><td>91.83</td></tr><tr><td>Ours/with FM</td><td>94.41</td><td>90.49</td><td>93.52</td><td>89.14</td><td>81.64</td><td>76.22</td><td>67.43</td><td>59.51</td><td>75.23</td><td>63.94</td><td>87.23</td><td>81.67</td><td>95.42</td><td>90.60</td></tr><tr><td>Ours/with CM</td><td>94.01</td><td>89.71</td><td>93.51</td><td>88.51</td><td>79.75</td><td>73.43</td><td>65.63</td><td>55.63</td><td>75.02</td><td>63.64</td><td>86.53</td><td>80.22</td><td>96.02</td><td>92.09</td></tr></table></body></html>

Consequently, we investigate the application of a soft maximum loss with the aim of achieving Optimal Structural Alignment. In this way, the loss, denoted as Optimal Classifier Loss, can be adjusted to compensate for poorly aligned features. We use the simple LogSum function to achieve alignment between feature representation and the optimal classifier weight, denoted as:

$$
\mathcal { L } _ { o c l } \left( Z _ { i } , Z _ { k } ; W _ { p } \right) = l o g \sum _ { i } \left| z _ { i } W _ { p } - e _ { k } \right| _ { i } ^ { \alpha } ,
$$

where $\alpha$ is a smoothing factor, $\boldsymbol { e } _ { k }$ is the classifier weight of corresponding class and $W _ { p }$ is the weight of feature Compression module $P$ .

The final objective function for the whole incremental model is denoted as:

$$
\mathcal { L } = \xi * \mathcal { L } _ { c e } + \mathcal { L } _ { o c l } ,
$$

where $\xi$ is the hyper-parameter and denoted as 10.

# Experiments

This section tests our method on seven PTM-CIL benchmark datasets and compares it with state-of-the-art methods. We also perform qualitative and quantitative ablation studies to validate the effectiveness of our method.

# Implementation Details

Dataset: Following the evaluation setting of EASE (Zhou et al. 2024), we select CIFAR100 (Krizhevsky, Hinton et al.

2009), CUB (Wah et al. 2011), ImageNet-R (Hendrycks et al. 2021a), ImageNet-A (Hendrycks et al. 2021b), ObjectNet (Barbu et al. 2019), Omnibenchmark (Zhang et al. 2022), and VTAB (Zhai et al. 2019) datasets to evaluate the performance. These datasets contain typical CIL benchmarks and out-of-distribution datasets, which have a large domain gap with ImageNet. There are 50 classes in VTAB, 100 classes in CIFAR100, 200 classes in CUB, ImageNet-R, ImageNet-A, ObjectNet, and 300 classes in OmniBenchmark.

Dataset split: Following EASE (Zhou et al. 2024), we denote the class split as ’B- $m$ Inc- $\mathbf { \nabla } _ { n } \mathbf { \dot { \Omega } }$ ’, where $m$ indicates the number of classes in the first stage and $n$ represents the number of classes in every incremental learning step. For a fair comparison, we randomly shuffle class orders with random seed 1993 before the data split and keep the training and testing set as EASE (Zhou et al. 2024).

Comparison methods: We select state-of-the-art PTMCIL methods for comparison, L2P (Wang et al. 2022c), DualPrompt (Wang et al. 2022b), CODA-Prompt (Smith et al. 2023), SimpleCIL (Zhou et al. 2023a), ADAM (Zhou et al. 2023a) and EASE (Zhou et al. 2024). In addition, we also compare our method with typical CIL methods with PTM, LwF (Li and Hoiem 2017), SDC (Yu et al. 2020), RanPAC (McDonnell et al. 2024).

Training details: We run experiments on NVIDIA RTX A6000 and report the results of other methods as EASE (Zhou et al. 2024). Following the previous setting, we consider two representative models, ViT-B/16-IN21K pre-trained on ImageNet21K and ViT-B/16-IN1K pre-trained on ImageNet1K. In our method, we train the model using an SGD optimizer for 35 epochs in the first learning step and an Adam optimizer for 100 epochs in other incremental steps, with a batch size of 256 and a 0.01 learning rate. The Pytorch deep learning structure supports all our experiments.

Evaluation metric: Following the benchmark (Zhai et al. 2019), we use $\mathbf { \mathcal { A } } _ { t }$ to evaluate the model’s accuracy after the b-th step. In addition, we introduce average incremental accuracy $\begin{array} { r } { \bar { \mathcal { A } } = \frac { 1 } { t } \sum _ { i = 1 } ^ { t } \mathcal { A } _ { t } } \end{array}$ to evaluate the whole incremental learning process. Through the two evaluation metrics, we can roundly evaluate the incremental performance of all PTMCIL methods.

![](images/a7da3891805bd7ee014821768b4cfdbf8698fe4687002952c90a96bf6a8b060b.jpg)  
Figure 3: Performance curve of different methods under different settings. All methods are initialized with ViT-B/16-IN1K.

Table 2: The ablation study of different modules on seven datasets.   

<html><body><table><tr><td>FT</td><td>TRA</td><td>ETF</td><td>FC</td><td>OSA</td><td>CF BO Inc5</td><td>CUB BO Inc10</td><td>IN-R BO Inc5</td><td>IN-A BO Inc20</td><td>ON BO Inc10</td><td>OB BO Inc30</td><td>VB BO Inc10</td></tr><tr><td>√</td><td>×</td><td>×</td><td>×</td><td>×</td><td>38.91</td><td>26.08</td><td>21.61</td><td>24.28</td><td>19.14</td><td>23.61</td><td>34.95</td></tr><tr><td>×</td><td>×</td><td>×</td><td>×</td><td>×</td><td>90.22</td><td>93.42</td><td>74.43</td><td>61.21</td><td>69.40</td><td>85.81</td><td>94.81</td></tr><tr><td>×</td><td>√</td><td>×</td><td>×</td><td>×</td><td>90.62</td><td>93.42</td><td>74.43</td><td>61.21</td><td>69.40</td><td>85.81</td><td>94.81</td></tr><tr><td>×</td><td>√</td><td>√</td><td>×</td><td>×</td><td>91.32</td><td>93.63</td><td>75.11</td><td>62.53</td><td>69.84</td><td>85.93</td><td>95.21</td></tr><tr><td>×</td><td>√</td><td>√</td><td></td><td>×</td><td>93.43</td><td>93.42</td><td>77.73</td><td>64.64</td><td>73.53</td><td>86.61</td><td>94.93</td></tr><tr><td>×</td><td>√</td><td>×</td><td>√</td><td>√</td><td>91.26</td><td>93.57</td><td>75.43</td><td>62.55</td><td>70.18</td><td>86.23</td><td>95.22</td></tr><tr><td>×</td><td>√</td><td>√</td><td>√</td><td>√</td><td>94.41</td><td>93.52</td><td>81.64</td><td>67.43</td><td>75.23</td><td>87.23</td><td>95.42</td></tr></table></body></html>

# Benchmark Comparison

In this section, we compare our method with other state-ofthe-art methods on seven benchmark datasets and different backbone weights. Table 1 presents the comparison of different methods with the ViT-B/16-IN21K pre-trained model. We can note our method with FM obtains the best average incremental accuracy performance $\bar { A }$ on seven popular benchmarks, except the $\boldsymbol { \mathcal { A } } _ { t }$ on VTAB dataset. Our method with FM obtains $0 . 9 0 \%$ , $0 . 3 9 \%$ , $3 . 3 3 \%$ , $2 . 0 9 \%$ , $2 . 6 8 \%$ , $1 . 2 8 \%$ and $1 . 8 1 \%$ average incremental accuracy performance improvement on CIFAR100, CUB, ImageNet-R, ImageNet-A, OmniBenchmark, ObjectNet and VTAB datasets, respectively. Our method with CM also obtains impressive performance with $0 . 5 0 \%$ , $1 . 2 7 \%$ , $1 . 4 4 \%$ , $0 . 2 9 \%$ , $2 . 4 7 \%$ , $0 . 5 8 \%$ and $2 . 4 1 \%$ performance improvement on CIFAR100, CUB, ImageNetR, ImageNet-A, OmniBenchmark, ObjectNet and VTAB datasets, respectively. Our method with FM needs memory to store the extracted features of all samples, while our method with CM is without memory PTM-CIL methods and achieves the second-best performance. In addition, our methods also have the best performance on $\mathbf { \mathcal { A } } _ { t }$ evaluation metric, except the VTAB dataset, where EASE performance fluctuates in the learning steps and is unreasonable.

We also report the incremental performance trend of different methods in Figure 3 with Vit-B/16-IN1K. We can note that our methods still obtain the best performance on CIFAR100, ImageNet-A, ImageNet-R, ObjectNet, and Omnibenchmark datasets with impressive improvements in each learning step. In addition, we obtain the second best performance on the VTAB dataset, which is a small dataset with limited classes. The experiments indicate that our method can be applied to different Pre-Trained Models and experiment settings, further proving its effectiveness. Furthermore, our method obtains the best performance in the base step, surpassing other methods to a large extent. This phenomenon proves that our method has the potential to be a novel paradigm to leverage the Pre-Trained Model to tackle downstream tasks.

![](images/16827eb5661c624e3f6f72277fcbc60c2db4e0256c97582216f9806fcb24cc78.jpg)  
Figure 4: t-SNE visualizations of different modules, presenting the effectiveness of different modules.

![](images/57e4eebc0b96223114c584cd5fc73febbd1f720252136d1d91905c464b0ca415.jpg)  
Figure 5: The parameter sensitivity analysis of the number of replayed feature.

# Ablation Study

To evaluate the performance of our different modules further, we introduce visual qualitative results to prove the feature representation transformation performance, shown in Figure 4. The samples are the first ten classes of the CIFAR100 dataset, where the pre-trained model is ViT-B/16-IN21K and the setting is B0 Inc5. We can note that different classes in t-SNE visualization results of the Pre-Trained Model are not separated clearly, proving the generalized feature representation of the Pre-Trained Model is not suitable and discriminative for downstream tasks. In addition, by gradually applying Task-Related adapters and Feature Compression Modules, different classes are clearly separated. Obviously, Task-Related Adapter can bridge the domain gap between pre-trained dataset and various downstream datasets. Based on the Task-Related adapter, our method is more compact and discriminative, proving that the Feature Compression Module can achieve Neural Collapse and feature representation adaptation.

To evaluate the different modules of our method quantitatively, we conduct several ablation experiments shown in Table 2, where pre-trained model ViT-B/16-IN21K is continually learning on seven different datasets. We denote Finutune Basebone, Task-Related Adaptation, ETF classifier, Feature

Compression Module, and Optimal Structural Alignment as ’FT’, TRA’, ’ETF’, ’FCM’, and ’OSA’. The base model is the pre-trained model ViT-B/16-IN21K, which finetunes in a continual learning process. We observe that our method obtains better performance when applied gradually to different modules, which proves the effectiveness of different modules. We present the experimental results of our method with fixed random classifier weights. Without the guidance of the ETF, the performance of our method noticeably decreases, demonstrating that non-optimal structural alignment will lead to poor feature representation. Optimal Structural Alignment is essential for aligning the feature representation between pre-trained datasets and downstream datasets.

In addition, we also discuss the number of replayed features of our-CM on CIFAR100 dataset in B0 Inc5 setting shown as Figuer 5, where the number is set to 32, 64, 128, 192, 256, 384 and 512. The performance of our method improves with the increase of the number. When the number is bigger than 256, the performance of our method grows slowly, approaching the upper-performance limit. Hence, we set the number of replayed feature as 256 to obtain the trade-off between the performance and computing cost.

# Conclusion

In this paper, we propose a novel Class-Incremental PreTrained method, which constructs an optimal classifier structure to guide the generalized feature adaptation. Task-Related Adaptation is introduced to bridge the domain gap between pre-trained datasets and downstream datasets. In addition, the Feature Compression Module is designed to compress various feature representations to corresponding class prototypes, satisfying $\mathcal { N } \mathcal { C } _ { 1 }$ , $\mathcal { N C } _ { 2 }$ and $\mathcal { N C } _ { 3 }$ characteristic. Optimal Structural Alignment is proposed to supervise the Feature Compression process. Sufficient comparative experiments and ablation experiments prove the effectiveness of our method. In addition, our method has the potential to be a novel paradigm to leverage PTM to tackle downstream tasks.

# Acknowledgments

Our work is supported in part by National Natural Science Foundation of China (62132016, 62406238), and Natural Science Basic Research Program of Shaanxi (2020JC-23).
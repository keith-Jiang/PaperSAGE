# Spiking Point Transformer for Point Cloud Classification

Peixi $\mathbf { W _ { u } } ^ { 1 * }$ , Bosong Chai3\*, Hebei $\mathbf { L i } ^ { 1 }$ , Menghua Zheng4, Yansong Peng1, Zeyu Wang3, Xuan Nie5, Yueyi Zhang1†, Xiaoyan $\mathbf { S u n } ^ { 1 , 2 \dag }$

1MoE Key Laboratory of Brain-inspired Intelligent Perception and Cognition, University of Science and Technology of China 2Institute of Artificial Intelligence, Hefei Comprehensive National Science Center 3College of Computer Science and Technology, Zhejiang University 4Tsingmao Intelligence 5School of Software, Northwestern Polytechnical University wupeixi@mail.ustc.edu.cn, {zhyuey, sunxiaoyan}@ustc.edu.cn

# Abstract

Spiking Neural Networks (SNNs) offer an attractive and energy-efficient alternative to conventional Artificial Neural Networks (ANNs) due to their sparse binary activation. When SNN meets Transformer, it shows great potential in 2D image processing. However, their application for 3D point cloud remains underexplored. To this end, we present Spiking Point Transformer (SPT), the first transformer-based SNN framework for point cloud classification. Specifically, we first design Queue-Driven Sampling Direct Encoding for point cloud to reduce computational costs while retaining the most effective support points at each time step. We introduce the Hybrid Dynamics Integrate-and-Fire Neuron (HD-IF), designed to simulate selective neuron activation and reduce over-reliance on specific artificial neurons. SPT attains state-of-the-art results on three benchmark datasets that span both real-world and synthetic datasets in the SNN domain. Meanwhile, the theoretical energy consumption of SPT is at least $6 . 4 \times$ less than its ANN counterpart.

# Code — https://github.com/PeppaWu/SPT

# Introduction

Bio-inspired Spiking Neural Networks (SNNs) are regarded as the third generation of neural networks (Maass 1997). In SNNs, spiking neurons transmit information through sparse binary spikes, where a binary value of 0 denotes neural quiescence and a binary value of 1 denotes a spiking event. Neurons communicate via sparse spike signals, with only a subset of spiking neurons being activated to perform sparse synaptic accumulation (AC), while the rest remain idle. Their high biological plausibility, sparse spikedriven communication (Roy, Jaiswal, and Panda 2019), and low power consumption on neuromorphic hardware (Pei et al. 2019) make them a promising alternative to traditional AI for achieving low-power, efficient computational intelligence (Schuman et al. 2022).

Drawing on the success of Vision Transformers (Dosovitskiy et al. 2020), researchers have combined SNNs with Transformers, achieving significant performance improvements on the ImageNet benchmark (Shi, Hao, and Yu 2024;

Zhou et al. 2024; Yao et al. 2024) and in various application scenarios (Yu et al. 2024; Ouyang and Jiang 2024). A question is naturally raised: can transformer-based SNNs be adapted to the 3D domain while maintaining their energy efficiency and fully leveraging the ability of transformers? To this end, we present Spiking Point Transformer (SPT), the first spiking neural network based on transformer architecture for deep learning on point cloud.

The successful application of transformer-based traditional artificial neural networks (ANNs) in the 3D point cloud domain has been widely demonstrated (Zhao et al. 2021; Park et al. 2022; Wu et al. 2022, 2024b). Since point clouds are collections embedded in 3D space, the core selfattention operator in Transformer networks is in essence a set operator which is invariant to the permutation and number of input elements, making it highly suitable for processing point cloud data. Considering the computational costs, point cloud transformers cannot perform global attention. The Point Transformer series (Zhao et al. 2021; Wu et al. 2022) calculates local self-attention within the $\mathbf { k }$ -nearest neighbors (KNN) neighborhood. In order to integrate this self-attention operation with SNNs, we follow the design of spiking self-attention (Yao et al. 2024; Li et al. 2024) and employ a spiking local self-attention mechanism to model sparse point cloud using spike Query, Key, and Value. By using AC operations instead of numerous multiply accumulate (MAC) operations, we significantly reduce the energy consumption of self-attention computations for 3D point cloud.

Training point cloud networks requires more expensive memory and computational costs than images because point cloud data requires more dimensions to describe itself. Researchers have proposed various optimization strategies, including sparse convolutions (Choy, Gwak, and Savarese 2019), optimization during the data processing phase (Hu et al. 2020), and local feature extraction (Ma et al. 2022). If the existing direct encoding methods used by transformerbased SNNs (Zhou et al. 2024; Yao et al. 2024) for 2D static images or used by SNNs for 3D point clouds (Ren et al. 2024; Wu et al. 2024a) are directly applied to the Transformer structure for point cloud, the training of SNNs with multiple time steps will result in a sharp increase in computational costs. Point cloud data is high-dimensional but has low information density. The current direct encoding methods for point clouds means we need to repeat T times along the temporal dimension. A clear approach is to consider whether we can split the point set across $\mathrm { \Delta T }$ time steps instead. To this end, we propose Queue-Driven Sampling Direct Encoding (Q-SDE), an improved direct encoding method for point cloud. Our method efficiently covers the original point cloud information through First-in, Firstout (FIFO) sampling mechanism while maintaining certain key supporting points unchanged.

Many studies (Niiyama, Fujimoto, and Imai 2023; Sakai 2020) have shown that during brain development, neurons undergo a use it or lose it process, where neural circuits are remodeled to prune excessive or incorrect neurons. Inspired by this, we fuse different neural dynamic models to simulate neuronal pruning and selective activation of neurons in biological brains through divide-and-conquer and gating mechanisms, which is referred to as Hybrid Dynamics Integrateand-Fire Neuron (HD-IF) and placed in some critical position within the network. Our main contributions can be summarized as follows:

• We build a Spiking Point Transformer (SPT), which is the first transformer-based SNN framework for point cloud classification that significantly reduces energy consumption.   
• We design Queue-Driven Sampling Direct Encoding (QSDE), an improved SNN direct encoding method for point cloud that slightly enhances accuracy while significantly reducing memory usage.   
• We propose a Hybrid Dynamics Integrate-and-Fire Neuron (HD-IF) to effectively integrate multiple neural dynamic mechanisms and simulate the selective activation of biological neurons.   
• The performance on two benchmark datasets ModelNet40 (Wu et al. 2015) and ScanObjectNN (Uy et al. 2019) demonstrates the effectiveness of our method and achieves a new state-of-the-art in the SNN domain.

# Related Work

# Spiking Neural Networks and Transformers

There are typically three ways to address the challenge of the non-differentiable spike function: (1) Spike-timingdependent plasticity (STDP) schemes (Bi and Poo 1998). (2) converting trained ANNs into equivalent SNNs using neuron equivalence, i.e., ANN-to-SNN conversion schemes (Hu et al. 2023; Wang et al. 2023). (3) Training SNNs directly (Guo et al. 2023) using surrogate gradients. STDP is a biology-inspired method but is limited to small-scale datasets. Spiking neurons are the core components of SNNs, with common types including Integrate-and-Fire (IF) (Bulsara et al. 1996) and Leaky Integrate-and-Fire (LIF) (Gerstner and Kistler 2002). IF neurons can be seen as ideal integrators, maintaining a constant voltage in the absence of spike input. LIF neurons build on IF neurons by adding a voltage decay mechanism, which more closely approximates the dynamic behavior of biological neurons. In addition to IF and LIF neurons, Exponential Integrate-andFire (EIF) (Brette and Gerstner 2005) and Parametric Leaky Integrate-and-Fire (PLIF) (Fang et al. 2021b) neurons are also commonly used models. These neurons better simulate the dynamic characteristics of biological neurons.

Various studies have explored Transformer-based SNNs that fully leverage the unique advantages of SNNs (Kai et al. 2024). Spikformer (Zhou et al. 2023b) firstly converts all components of ViT (Dosovitskiy et al. 2020) into spike-form. Spike-driven Transformer (Yao et al. 2024) advances further by introducing the spike-driven paradigm into Transformers. Spikingformer (Zhou et al. 2023a) proposes a hardware-friendly spike-driven residual learning architecture. In this work, we extend the Transformer-based SNNs from 2D images to 3D point clouds while employing efficient direct training methods.

# Deep Learning on Point Cloud

Deep neural network architectures for understanding point cloud data can be broadly classified into projectionbased (Lang et al. 2019; Chen et al. 2017), voxelbased (Song et al. 2017), and point-based methods (Ma et al. 2022; Zhao et al. 2019). Projection-based methods project 3D point clouds onto 2D image planes, using a 2D CNN-based backbone for feature extraction. Voxel-based methods convert point clouds into voxel grids and apply 3D convolutions. Pioneering point-based methods like PointNet use max pooling for permutation invariance and global information extraction (Qi et al. 2017a), while Point$\mathrm { N e t + + }$ introduces hierarchical feature learning (Qi et al. 2017b). Recently, point-based methods have shifted towards Transformer-based architectures (Zhao et al. 2021; Park et al. 2022; Wu et al. 2022, 2024b). The self-attention mechanism of the point transformer, insensitive to input order and size, is applied to each point’s local neighborhood, crucial for processing point clouds.

Wu et al. construct a point-to-spike residual classification network by stacking 3D spiking residual blocks and combining spiking neurons with conventional point convolutions (Wu et al. 2024a). Spiking PointNet, the first SNN framework for point clouds, proposes a trained-less but learning-more paradigm based on PointNet (Ren et al. 2024). It adopts direct encoding of point clouds, repeating over time steps, making it hard to train point clouds with large time steps. Due to these limitations, further accuracy improvement is challenging. To address this, we propose a transformer-based SNN framework and design Q-SDE, significantly saving computational costs, enabling training in multiple time steps, and achieving higher accuracy.

# Method

In this paper, we propose a Spiking Point Transformer (SPT) for 3D point cloud classification, integrating the spiking paradigm into Point Transformer. First, we perform QueueDriven Sampling Direct Encoding (Q-SDE) on the point cloud. Then, we preliminarily encode the membrane potential with an MLP Module and a Spiking Point Transformer Block (SPTB). Next, further encoding is done through $L$ Spiking Point Encoder Modules, mainly including Spiking Transition Down Block (STDB) for downsampling and SPTB for feature interaction. Finally, membrane potential is sent to Classification Head to output the prediction.

![](images/371ebbab0753a41bdd46c3084734f3399dc649d93a503229ab2f5de10cb2cc73.jpg)  
Figure 1: The overview of Spiking Point Transformer (SPT), which consists of Queue-Driven Sampling Direct Encoding (QSDE), MLP Module for adaptive learning, Spiking Point Encoder Module for feature interaction and Classification Head.

# Queue-Driven Sampling Direct Encoding

Most of the high-performance SNN studies (Zhou et al. 2024; Yao et al. 2024; Ren et al. 2024) are based on direct encoding. Direct encoding is to repeat the input $T$ times along the time dimension, which incurs expensive computational costs. We design an encoding method suitable for point clouds, which is an improved direct encoding called Queue-Driven Sampling Direct Encoding (Q-SDE). Q-SDE uses a first-in, first-out queue-driven sampling method to retain the most effective support points of the original points at different time steps, while reducing computational costs.

The original point queue $P$ has a shape of $( N , C _ { 0 } )$ . We initialize the encoded multi-time-step point matrix $P _ { e }$ with a shape of $( T , N _ { s } , C _ { 0 } )$ . $T$ represents the number of time steps, $N _ { s }$ represents the number of sampled points per time step, and $C _ { 0 }$ represents the number of feature dimensions per point.

As shown in Figure 1, through furthest point sampling (FPS), $N _ { s }$ points are extracted from $P$ and stored in the first time step of $P _ { e }$ . The sampled points at first time step contain the object’s key contours but lacks the $N - N _ { s }$ points which are unsampled, which are crucial for recognizing difficult objects. Subsequent time step sampling should efficiently cover the unsampled points.

The specific approach is to dequeue the first $N _ { p }$ points referred to as discarded points from $P$ , then use FPS to select $N _ { p }$ points called sampling points from the unsampled points, and concatenate these points with the first $N - N _ { p }$ points of

# Algorithm 1: Queue-Driven Sampling Direct Encoding

1: Input: Point queue $P$ , Sample number $N _ { s }$ , Timestep $T$   
2: Output: Encoded point matrix $P _ { e }$   
3: $N _ { p } = \lfloor ( N - N _ { s } ) / \left( T - 1 \right) \rfloor \ \triangleright$ Initialize $N _ { p }$ , points   
dequeued per timestep   
4: $P _ { e } [ 0 ] \  \ \mathrm { F P S } ( P , N _ { s } ) \ \triangleright \ \mathrm { S e t } \ P _ { e } [ 0 ]$ , denotes the first   
timestep point cloud   
5: for $i = 1 , 2 , 3 , \dots , T - 1$ do   
6: $D$ Remaining Point Check   
7: if $P \setminus P _ { e } [ i - 1 ]$ is empty then   
8: $P _ { e } [ i ]  P _ { e } [ i - 1 ]$ ▷ Coverage   
9: $D$ Queue-driven Sample   
10: else   
11: $\begin{array} { r l r } & { \tilde { S }  \{ P _ { e } [ i - 1 ] [ j ] \mid j \geq N _ { p } \} } & { \triangleright \mathrm { S u b s e t ~ } } \\ & { F  \mathrm { F P S } ( P \setminus P _ { e } [ i - 1 ] , N _ { p } ) } & { \triangleright \mathrm { S a m p l e ~ } } \\ & { P _ { e } [ i ]  S \cup F } & { \triangleright \mathrm { M e r g e ~ } } \\ & { P  P \setminus \{ P _ { e } [ i - 1 ] [ j ] \mid j < N _ { p } \} \triangleright \mathrm { U p d a t e ~ } } & \end{array}$   
12:   
13:   
14:   
15: end if   
16: end for

$P$ . The resulting point cloud data is stored in the next time step of $P _ { e }$ . This process of dequeuing and concatenation is repeated $T - 1$ times.

$N _ { p }$ represents the number of points to be dequeued at each time step. When $T > 1$ , to ensure that the number of remaining points in $P$ at the final time step is not less than $N _ { s }$ , while minimizing the number of unused points, the following constraints must be satisfied:

$$
N _ { p } = \left\lfloor { \frac { N - N _ { s } } { T - 1 } } \right\rfloor , T > 1
$$

When $T = 1$ , the first time step of $P _ { e }$ is also the only time step that stores all points in $P$ . Together, the main steps of Q-SDE are summaried in Algorithm 1.

# Spiking Point Encoder Module

As shown in Figure 1, Spiking Point Encoder Module is the main component of the whole architecture, which contains the Spiking Transition Down Block (STDB) and Spiking Point Transformer Block (SPTB).

Spiking Transition Down Block. STDB is employed for spatial downsampling of point clouds to expand the spatial receptive field. Specifically, it involves obtaining a new spatial point cloud $P _ { l }$ and its corresponding membrane potential features $U _ { l }$ through FPS. We then utilize K-nearest neighbors (KNN) sampling to extract the features of the nearest points for each point in the new point cloud and project these features into a higher-dimensional space after spiking neuron firing. Finally, by using LocalMaxPooling (LAP), we aggregate the local features $F$ from the neighborhood of spatial point cloud $P _ { l }$ onto the membrane potential features $U _ { l } ^ { \prime }$ . STDB can be expressed as:

$$
\begin{array} { r l } & { F _ { l - 1 } = \{ P _ { l - 1 } , U _ { l - 1 } \} } \\ & { F _ { l } = \mathrm { F P S } ( F _ { l - 1 } , N _ { l } ) } \\ & { F = \mathrm { K N N } ( F _ { l } , F _ { l - 1 } , N _ { k } ) } \\ & { U _ { l } ^ { \prime } = \mathrm { L A P } ( \mathrm { M L P } ( S \mathcal { N } ( F ) ) ) } \end{array}
$$

where $N _ { l }$ is the number of points in the $l .$ -th layer, $N _ { k }$ is the number of sampled points in the neighborhood. $\mathcal { S N } ( \cdot )$ represents the spiking neuron. $\mathrm { K N N } ( \mathcal { A } , B , N _ { k } )$ denotes sampling the $N _ { k }$ nearest points from point set $\boldsymbol { B }$ to point set $\mathcal { A }$ through KNN.

$P _ { l } , U _ { l }$ are features in $\mathbb { R } ^ { T \times N _ { l } \times 3 }$ and $\mathbb { R } ^ { T \times N _ { l } \times C _ { l } }$ respectively, representing the position information and membrane potential feature information of the point cloud in the $l$ -th layer. $F$ represents the KNN neighborhood membrane potential feature of $F _ { l } . \ F _ { l }$ represents the union of $P _ { l }$ and $U _ { l }$ , which belongs to RT ×Nl×(3+Cl).

Spiking Point Transformer Block. SPTB further encodes the membrane potential feature $U _ { l } ^ { \prime }$ , and conducts extensive information interaction at a more advanced semantic level, so that the feature carried by each point can better represent the local points, thereby achieving better shape classification.

The specific implementation of SPTB, as shown in Figure 1, begins with the preliminary encoding of the spike signals $S _ { l } ^ { \prime }$ input by HD-IF. Then, by using KNN sampling, the $N _ { k }$ point neighborhood features of $P _ { l }$ are indexed, and these features are encoded to obtain spike Query and Value. Moreover, the input spike $S _ { l } ^ { \prime \prime }$ is further encoded to obtain the spike Key. The learnable relative position encoding is performed on $P _ { l }$ and its neighborhood. They are aggragated according to the methodology proposed by Point Transformer (Zhao et al. 2021). Finally, output encoding is performed and membrane potential interaction is conducted through residual connection. SPTB can be written as follows:

![](images/abb912dccb1e7793cd1c360e456f5847cb19b32d6b71f42922bbcbab30acc9be.jpg)  
Figure 2: (a) The main structure of HD-IF integrating neuronal membrane potential and firing. (b) The membrane potential of different neurons with 0.4 input and 0.5 threshold.

$$
\begin{array} { r l } & { S _ { l } ^ { \prime \prime } = S \mathcal { N } ( \mathrm { M L P } ( S _ { l } ^ { \prime } ) ) } \\ & { K = S \mathcal { N } ( \mathrm { M L P } ( S _ { l } ^ { \prime \prime } ) ) } \\ & { Q , V = S \mathcal { N } ( \mathrm { M L P } ( \mathrm { K N N } ( S _ { l } ^ { \prime \prime } , N _ { k } ) ) ) } \\ & { \delta \ = S \mathcal { N } ( \mathrm { M L P } ( \mathrm { K N N } ( P _ { l } , N _ { k } ) - P _ { l } ) ) } \\ & { U _ { l } ^ { \prime \prime } = \displaystyle \sum _ { \chi } \rho \left( \gamma \left( \beta \left( Q , K \right) + \delta \right) \right) \odot \left( V + \delta \right) } \\ & { U _ { l } = \mathrm { M L P } ( S \mathcal { N } ( U _ { l } ^ { \prime \prime } ) ) + U _ { l } ^ { \prime } } \end{array}
$$

where $\delta$ represents relative position encoding. $\mathcal { X }$ represents the $N _ { k }$ point neighborhood. $\beta$ is a relation function (e.g., subtraction), $\rho$ is a normalization function, and $\gamma$ is a mapping function (e.g., MLP with $\mathcal { S N }$ ) that produces attention vectors for feature aggregation. $\mathrm { K N N } ( \mathcal { A } , N _ { k } )$ denotes sampling the $N _ { k }$ nearest points from point set $\mathcal { A }$ to itself.

# Hybrid Dynamics Integrate-and-Fire Neuron

The spiking neuron model is simplified from the biological neuron model. In this paper, we uniformly adopt the LIF for $\mathcal { S N }$ function. Meanwhile, we design HD-IF which integrate different neuronal dynamic models, including LIF (Gerstner and Kistler 2002), IF (Bulsara et al. 1996), EIF (Brette and Gerstner 2005), and PLIF (Fang et al. 2021b) and place it before each SPTB.

We begin by briefly revisiting their dynamic characteristics. Figure 2(b) shows that the IF neuron acts as an ideal integrator, with membrane potential changing through input accumulation. The LIF neuron is IF neuron with leakage, where the membrane potential gradually approaches the input with input and returns to the resting state without input. The EIF neuron is a nonlinear LIF model. It adds an exponential term to the LIF model to simulate the sudden jump in potential near the firing threshold. The PLIF neuron adds a learnable membrane time constant $\tau$ , dynamically adjusted by the parameter $w$ via Sigmoid $( w )$ function. The detailed equations for each neuron can be found in the Appendix.A.

Then, we introduce a novel HD-IF neuron, which aims to promote competition among different neurons by selectively activating suitable neurons and fusing their dynamic characteristics to generate membrane potential spikes. This hybrid design effectively reduces over-reliance on specific artificial neurons and enhances the robustness of SNNs.

The HD-IF neuron is embedded before each SPTB to optimize the dynamic behavior of the spiking neural network. Specifically, the HD-IF neuron processes the membrane potential $U _ { l } ^ { \prime }$ of STDB and outputs the spike $S _ { l } ^ { \prime }$ , as shown in Figure 2(a). First, the temporal dimension and feature dimension of the membrane potential $U _ { l } ^ { \prime }$ is combined to create an input feature with spatial and temporal dual features. Then, a gate network calculates weights for membrane potential generated by various neurons at different spatial points. During training, the model adjusts neuron responses through dense propagation and weighted summation. During inference, the Top-2 neural models are selected to reduce computational complexity and improve efficiency. Finally, the Heaviside function fires the mixed membrane potential to produce the spike sequence $S _ { l } ^ { \prime }$ .

# Experiments

# Experimental Settings

Datasets. We evaluate the performance of 3D point cloud classification on the synthetic dataset ModelNet40 (Wu et al. 2015) and the real dataset ScanObjectNN (Uy et al. 2019). ModelNet40 contains 40 different object categories, each of which contains approximately 12,311 CAD models across 40 different categories. The training set contains 9,843 instances, and the testing set contains 2,468 instances. ModelNet10 is a subset of ModelNet40. The training set contains 3,991 instances, and the testing set contains 908 instances. ScanObjectNN is constructed from real-world scans, characterized by varying degrees of data missing and noise contamination. The entire dataset consists of 3D objects from 15 categories, with 11,416 samples as a training set and 2,882 samples as a testing set.

Implementation Details. We implement the Spiking Point Transformer in PyTorch 1.13 (Paszke et al. 2019) on 2 $\times$ RTX 3090Ti GPUs. SPT is developed using the SpikingJelly framework1 (Fang et al. 2023) based on PyTorch. We use the AdamW optimizer with momentum and weight decay set to 0.9 and 0.0001, respectively. The initial learning rate is set to 0.001 and is decreased by a factor of 0.3 every 50 epochs.The number of input point cloud points $N$ is set to 1024. For all our SNN models, we set $V _ { t h }$ as 0.5 for fair comparison with Spiking Pointnet (Ren et al. 2024). The remaining hyperparameters are consistent with those used in the Point Transformer (Zhao et al. 2021). We conducted iterative training on the entire dataset for 200 epochs.

# Experimental Results

In this experiment, we evaluate our model’s performance using two metrics: overall accuracy (OA) and mean class accuracy (mAcc). These metrics provide a comprehensive assessment of our model on the test set.

![](images/9b30159fe7a9fc59c38836d29f57e1352823fdc4d13015d7168750b7f299f5df.jpg)

Figure 3: Visualization of support points and points at each time step. Support points repeated across most time steps capture the essence of the object shape. Blue points are the enqueue points while red points are the dequeue points.   
Table 1: Ablation study of time step on ModelNet10/40 and ScanObjectNN.   

<html><body><table><tr><td>Time Step</td><td>ModelNet10 OA(%)</td><td>ModelNet40 OA(%)</td><td>ScanObjectNN OA(%)</td></tr><tr><td>1</td><td>94.35</td><td>90.87</td><td>76.33</td></tr><tr><td>2</td><td>94.29</td><td>91.13</td><td>77.03</td></tr><tr><td>3</td><td>94.54</td><td>91.38</td><td>77.51</td></tr><tr><td>4</td><td>94.76</td><td>91.43</td><td>78.03</td></tr></table></body></html>

ModelNet10/40 Dataset. From Table 2, we can see that our SPT model shows superior performance on both ModelNet10 and ModelNet40 datasets. In the SNN domain, the SPT model achieves the highest accuracy, surpassing the SNN baselines. Specifically, on ModelNet40, SPT attains $9 1 . 4 3 \%$ OA and $8 9 . 3 9 \%$ mAcc, reflecting a $0 . 8 3 \%$ and $0 . 1 9 \%$ improvement over P2SResLNet-B respectively. On ModelNet10, SPT significantly outperforms Spiking Pointnet, with $9 4 . 7 6 \%$ OA and $9 3 . 6 9 \%$ mAcc, reflecting a $1 . 4 5 \%$ improvement in OA. In the ANN domain, while the SPT model’s accuracy on ModelNet40 is slightly lower than Point Transformer, it even surpasses the ANN baseline on ModelNet10, with $9 4 . 7 6 \%$ OA and $9 3 . 6 9 \ \%$ mAcc, relecting $0 . 4 8 \%$ improvement in OA.

ScanObjectNN Dataset. From Table 2, we can see that our SPT model still achieves the state-of-the-art performance in the SNN domain. Specifically, the SPT model attains $78 . 0 3 \%$ OA without voting, reflecting a $3 . 5 7 \%$ improvement over P2SResLNet-B, and $8 2 . 2 3 \%$ OA with voting, reflecting a $1 . 0 3 \%$ improvement over P2SResLNet-B. In the ANN domain, the SPT model’s accuracy is slightly lower compared to Point Transformer without voting. Considering the theoretical energy consumption, our model provides a proper balance between classification accuracy and spike-based biological characteristics.

<html><body><table><tr><td rowspan="2">Methods</td><td rowspan="2">Type</td><td rowspan="2">Time Step</td><td colspan="2">ModelNet10</td><td colspan="2">ModelNet40</td><td colspan="2">ScanObjectNN</td></tr><tr><td>OA(%)</td><td>mAcc(%)</td><td>OA(%)</td><td>mAcc(%)</td><td>OA(%)</td><td>mAcc(%)</td></tr><tr><td>Pinintet+</td><td>ANN</td><td></td><td>92.98</td><td></td><td>92.20</td><td>86.00</td><td>7.20</td><td>63.40</td></tr><tr><td></td><td></td><td></td><td></td><td>--</td><td></td><td></td><td></td><td></td></tr><tr><td>Point Transformer*</td><td>ANN</td><td></td><td>94.28</td><td>94.01</td><td>91.73</td><td>89.56</td><td>81.32</td><td>80.34</td></tr><tr><td>PointMLP</td><td>ANN</td><td></td><td></td><td>-</td><td>94.10</td><td>91.50</td><td>85.40</td><td>83.90</td></tr><tr><td>KPConv-SNN</td><td>ANN2SNN</td><td>40</td><td>--</td><td>-</td><td>70.50</td><td>67.60</td><td>43.90</td><td>38.70</td></tr><tr><td>Spiking Poinetnet</td><td>SNN</td><td>4</td><td>93.31</td><td></td><td>88.61</td><td></td><td>64.04*</td><td>60.14*</td></tr><tr><td>P2SResLNet-B</td><td>SNN</td><td>1</td><td>-</td><td></td><td>90.60</td><td>89.20</td><td>74.46*/81.20</td><td>72.58*/79.40</td></tr><tr><td>SPT(Q-SDE512)</td><td>SNN</td><td>4</td><td>94.66</td><td>93.54</td><td>91.43</td><td>89.39</td><td>76.51 /80.02</td><td>74.53 / 78.12</td></tr><tr><td>SPT(Q-SDE768)</td><td>SNN</td><td>4</td><td>94.76</td><td>93.69</td><td>91.22</td><td>88.45</td><td>78.03 / 82.23</td><td>75.87 / 80.12</td></tr></table></body></html>

Table 2: Performance comparison with the baseline methods. The best results in the SNN domain are presented in bold, with \* indicating self-reproduced results and $0$ indicating results based on test voting.

# Ablation Study

Ablation on Time Step. In our ablation study on time step, we observe a significant difference compared to previous models like Spiking PointNet and P2SResLNet-B. These models typically show a trend that longer time steps bring either reduced or stable accuracy. However, as illustrated in Table 1, our model basically improves accuracy with longer time steps, consistent with findings in 2D image classification (Fang et al. 2021a).

Unlike 2D image, 3D point cloud is highly sparse. For direct encoding method, longer time steps may mean more redundancy rather than more useful information. As shown in Figure 3, our model improves this by modifying direct encoding so that each time step contains only a subset of the initial point cloud $P$ . The point cloud at each time step may look similar which maintains the repetitiveness of direct encoding, but there is a difference of $N _ { p }$ points between them which exploits the dynamic characteristics of neurons to leverage longer time steps effectively.

However, excessively long time steps are impractical due to expensive memory and computational cost (Wu et al. 2024a). Therefore, we set the maximum time step to 4 in our ablation study. Table 3 shows that the optimal accuracy at each time step. We can see that OA improves with longer time steps, reaching a peak of $9 1 . 4 3 \%$ at 4 time steps on the ModelNet40 dataset and $78 . 0 3 \%$ on the ScanObjectNN dataset.

Ablation on Encoding Method. We first conduct ablation experiments on different input encoding methods on the ModelNet40 dataset, including direct encoding, RandomSDE (randomly sampling $\lfloor N / T \rfloor$ points per time step), and our proposed $\mathrm { Q - S D E } ( N _ { s } )$ . Here, $N _ { s }$ represents the number of sampled points per time step, typically set to 256, 512, 768 or 1024. In our ablation study, these encoding methods are evaluated based on the performance and efficiency.

Moreover, too many support points increase encoding redundancy, failing to leverage the inherent sparsity of point clouds while introducing unnecessary points and even noise. This impacts the SNN model’s performance over longer time steps, causing slightly lower accuracy for Q-SDE1024 than

Table 3: Ablation study of encoding method performance on ModelNet40.   

<html><body><table><tr><td rowspan="2">Methods</td><td colspan="2">T=2</td><td colspan="2">T=4</td></tr><tr><td>OA(%)</td><td>mAcc(%)</td><td>OA(%)</td><td>mAcc(%)</td></tr><tr><td>Direct Encoding</td><td>91.12</td><td>88.72</td><td>91.17</td><td>88.38</td></tr><tr><td>Random-SDE</td><td>90.14</td><td>87.61</td><td>89.94</td><td>87.24</td></tr><tr><td>Q-SDE1024</td><td>91.07</td><td>88.58</td><td>91.08</td><td>87.98</td></tr><tr><td>Q-SDE768</td><td>91.13</td><td>88.93</td><td>91.22</td><td>88.45</td></tr><tr><td>Q-SDE512</td><td>90.87</td><td>87.97</td><td>91.43</td><td>89.39</td></tr><tr><td>Q-SDE256</td><td></td><td>-</td><td>90.89</td><td>88.35</td></tr></table></body></html>

Table 4: Ablation study of encoding method efficiency on ModelNet40.   

<html><body><table><tr><td>Methods</td><td colspan="2">Training</td><td colspan="2">Inference</td></tr><tr><td>(T=4)</td><td>Runtime</td><td>Memory</td><td>Runtime</td><td>Memory</td></tr><tr><td>Direct Encoding</td><td>478ms</td><td>15.3G</td><td>234ms</td><td>9.3G</td></tr><tr><td>Q-SDE1024</td><td>431ms</td><td>15.2G</td><td>227ms</td><td>9.5G</td></tr><tr><td>Q-SDE768</td><td>385ms</td><td>12.5G</td><td>201ms</td><td>7.3G</td></tr><tr><td>Q-SDE512</td><td>326ms</td><td>9.7G</td><td>191ms</td><td>5.2G</td></tr><tr><td>Q-SDE256</td><td>273ms</td><td>6.9G</td><td>164ms</td><td>3.0G</td></tr></table></body></html>

Q-SDE768 at 2 time steps and for both Q-SDE768 and QSDE1024 than Q-SDE512 at 4 time steps.

Performance. In our ablation study on different encoding methods, we compare the performance of the SPT model using common time steps of 2 and 4. From Table 3, we can see that at 2 time steps, Q-SDE768 and direct encoding exhibit comparable overall accuracy. However, at 4 time steps, Q-SDE512 surpasses direct encoding by $0 . 2 6 \%$ in overall accuracy. In contrast, Random-SDE performs notably worse than direct encoding, further validating the effectiveness of Q-SDE.

Nevertheless, the overall accuracy of Q-SDE does not monotonically increase with fewer sampled points. Table 3 shows that Q-SDE512 has lower accuracy than Q-SDE768 at 2 time steps, and Q-SDE256 has lower accuracy than QSDE512 at 4 time steps. This indicates that each time step should include a certain degree of repetition to ensure the core object shape is represented across most time steps. This core shape representation is called as support points. As shown in Figure 3, highly sparse support points capture the essence of an object’s shape.

Efficiency. We evaluate encoding method efficiency based on two metrics: runtime and memory consumption.The ablation experiments use a setting of 4 time steps and a batch size of 4. Efficiency metrics are measured on a single RTX 3090Ti, excluding the initial iteration to ensure steady-state measurements.

The results presented in Table 4 clearly show that using fewer sampled points significantly reduces both runtime and memory consumption both during training and inference with the SPT model, which is consistent with our expectations. Compared to direct encoding, Q-SDE exhibits substantial advantages in optimizing runtime and memory consumption. During inference, encoding methods such as QSDE512 achieve a notable balance between model efficiency and inference accuracy, as corroborated by Table 1. This further underscores that the Q-SDE encoding method effectively reduces redundancy and computational costs, making point cloud sampling at each time step more efficient and effective.

Ablation on HD-IF. Table 6 presents the results of the ablation study of HD-IF conducted on the ModelNet40 dataset. The experiment compares the overall accuracy of different encoding methods with various spiking neuron models at 4 time steps, aiming to demonstrate the universal superiority of HD-IF over other single neuron(e.g., IF, LIF, EIF, and PLIF).

From Table 6, we can see that incorporating HD-IF before each SPTB significantly enhances the overall accuracy across all encoding methods. Specifically, compared to replacing HD-IF with IF, for Q-SDE256, the accuracy increases from $9 0 . 5 3 \%$ to $9 0 . 8 9 \%$ . For Q-SDE512, the accuracy increases from $9 0 . 9 9 \%$ to $9 1 . 4 3 \%$ , and for Q-SDE768, the accuracy increases from $9 1 . 0 9 \%$ to $9 1 . 2 2 \%$ . Other single neurons replacing HD-IF also show various degrees of accuracy change, with some achieving minor improvements. However, HD-IF consistently attains the highest accuracy across all encoding methods, further demonstrating its effectiveness in enhancing model performance by leveraging the dynamic firing characteristics of different neurons. As shown in Figure 4, HD-IF can adapt to diverse data scenarios during inference by selectively activating different neurons to process information efficiently.

# Energy Efficiency

In this section, we investigate energy efficiency of our SPT model on the ModelNet40 dataset. In the ANN domain, the dot product operation, or MAC operation, involves both addition and multiplication operations. However, the SNN leverages the multiplication-addition transformation advantage, eliminating the need for multiplication operations in all layers except the first $\mathrm { C o n v + B N }$ layer. According to the research (Horowitz 2014), a 32-bit floating-point consumes 4.6pJ for a MAC operation and $0 . 9 \mathrm { p J }$ for an AC operation.

![](images/7b42d76d8806ade96ed4560261bb1418ed8219b5960629a2c3d613d42547b172.jpg)  
Figure 4: Visualization of selectively activated neurons on different datasets.The solid line shows the most frequently Top-1 activated neurons while the dashed line shows the most frequently Top-2 activated neurons.

Table 5: Power of ANN (Point Transformer) and SPT.   

<html><body><table><tr><td>TimeStep</td><td>OA(%)</td><td>AC(GB)</td><td>MAC(GB)</td><td>Power(mJ)</td></tr><tr><td>ANN</td><td>91.73</td><td>0.0</td><td>18.42</td><td>84.7</td></tr><tr><td>1</td><td>90.87</td><td>3.10</td><td>0.044</td><td>3.0</td></tr><tr><td>4</td><td>91.43</td><td>13.85</td><td>0.179</td><td>13.3</td></tr></table></body></html>

Table 6: Ablation study of HD-IF on ModelNet40.   

<html><body><table><tr><td>Neurons (T=4)</td><td>Q-SDE256 OA(%)</td><td>Q-SDE512 OA(%)</td><td>Q-SDE768 OA(%)</td></tr><tr><td>IF</td><td>90.53</td><td>90.99</td><td>91.09</td></tr><tr><td>LIF</td><td>90.34</td><td>91.08</td><td>91.07</td></tr><tr><td>EIF</td><td>90.25</td><td>91.15</td><td>91.08</td></tr><tr><td>PLIF</td><td>90.78</td><td>91.28</td><td>91.13</td></tr><tr><td>HD-IF</td><td>90.89</td><td>91.43</td><td>91.22</td></tr></table></body></html>

Based on our SPT model, we calculate the energy consumption and present the results in Table 5. The specific method of energy consumption calculation is provided in Appendix.B. Our SPT shows remarkable energy efficiency, requiring only $3 . 0 \mathrm { m J }$ of energy per forward pass at 1 time step with a firing rate of $1 7 . 9 \%$ , reflecting a 28.2-fold reduction compared to conventional ANNs. Furthermore, when we conduct inference at 4 time steps, the performance reaches $9 1 . 4 3 \%$ , while the energy consumption is merely about 6.4 times less than that of its ANN counterpart.

# Conclusion

In this paper, we present the Spiking Point Transformer (SPT) which combines the low energy consumption of SNN and the excellent accuracy of Transformer for 3D point cloud classification. The results show that SPT achieves overall accuracies of $9 4 . 7 6 \%$ , $9 1 . 4 3 \%$ , and $78 . 0 3 \%$ on the ModelNet10, ModelNet40, and ScanObjectNN datasets, respectively, making it the state-of-the-art in the SNN domain. We hope that our work can inspire the application of SNNs in other tasks, such as 3D semantic segmentation and object detection, and also promote the design of next-generation neuromorphic chips for point cloud processing.

# Acknowledgments

This work was in part supported by the National Natural Science Foundation of China under grants 62472399 and 62021001.
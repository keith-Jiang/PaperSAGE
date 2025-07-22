# Beyond Skip Connection: Pooling and Unpooling Design for Elimination Singularities

Chengkun $\mathbf { S u n } ^ { 1 }$ , Jinqian $\mathbf { P a n } ^ { 1 }$ , Zhuoli $\mathbf { J i n } ^ { 3 }$ , Russell Stevens Terry2, Jiang Bian1, Jie $\mathbf { X } \mathbf { u } ^ { 1 }$

1Department of Health Outcomes and Biomedical Informatics, University of Florida, Gainesville, FL 32611, USA 2Department of Urology, University of Florida, Gainesville, FL 32611, USA 3Department of Statistics and Applied Probability, University of California Santa Barbara, CA 93106-3110, USA sun.chengkun,jinqianpan,bianjiang,xujie $@$ ufl.edu, zhuoli jin $@$ pstat.ucsb.edu, russell.terry $@$ urology.ufl.edu

# Abstract

Training deep Convolutional Neural Networks (CNNs) presents unique challenges, including the pervasive issue of elimination singularities—consistent deactivation of nodes leading to degenerate manifolds within the loss landscape. These singularities impede efficient learning by disrupting feature propagation. To mitigate this, we introduce Pool Skip, an architectural enhancement that strategically combines a Max Pooling, a Max Unpooling, a $3 \times 3$ convolution, and a skip connection. This configuration helps stabilize the training process and maintain feature integrity across layers. We also propose the Weight Inertia hypothesis, which underpins the development of Pool Skip, providing theoretical insights into mitigating degradation caused by elimination singularities through dimensional and affine compensation. We evaluate our method on a variety of benchmarks, focusing on both 2D natural and 3D medical imaging applications, including tasks such as classification and segmentation. Our findings highlight Pool Skip’s effectiveness in facilitating more robust CNN training and improving model performance.

Code — https://github.com/sunck1/PoolSkip

# Introduction

Convolutional Neural Networks (CNNs) are pivotal in advancing the field of deep learning, especially in image processing (Jiao and Zhao 2019; Razzak, Naz, and Zaib 2018; Miao et al. 2019, 2018). However, as these networks increase in depth to enhance learning capacity, they often encounter a notable degradation in performance (He et al. 2016a). This degradation manifests as a saturation point in accuracy improvements, followed by a decline, a phenomenon primarily driven by optimization challenges including vanishing gradients (He et al. 2016a). The introduction of Residual Networks (ResNets) with skip connections marked a significant advancement in mitigating these issues by preserving gradient flow during deep network training (He et al. 2016a; Orhan and Pitkow 2017).

Despite these advancements, very deep networks, such as ResNets with upwards of 1,000 layers, still face pronounced degradation issues (He et al. 2016a). A critical aspect of this problem is the elimination singularity (ES)—stages in the training process where neurons consistently deactivate, producing zero outputs and creating ineffective nodes within the network (Orhan and Pitkow 2017; Qiao et al. 2019). This condition not only disrupts effective gradient flow but also significantly compromises the network’s learning capability. ES often results from zero inputs or zero-weight configurations in convolution layers, which are frequently observed due to the tendency of training processes to drive weights towards zero, contributing to excessively sparse weight matrices (Orhan and Pitkow 2017; Huang et al. 2020). Additionally, the widely used Rectified Linear Unit (ReLU) activation function exacerbates these issues by zeroing out all negative inputs (Qiao et al. 2019; Lu et al. 2019). This phenomenon, known as Dying ReLU, causes neurons to remain inactive across different data points, effectively silencing them and further complicating the training of deep networks (Qiao et al. 2019; Lu et al. 2019).

To address these persistent challenges, we developed Pool Skip, a novel architectural module that strategically incorporates Max Pooling, Max Unpooling, and a $3 \times 3$ convolution linked by a skip connection. This design is specifically engineered to counteract elimination singularities by enhancing neuron activity and preserving the integrity of feature transmission across network layers. Our approach not only aims to stabilize the learning process but also to enhance the robustness of feature extraction and representation in deep networks. The key contributions of our work are summarized below:

• We propose the Weight Inertia hypothesis to explain how persistent zero-weight conditions can induce network degradation. Based on this theory, we developed the Pool Skip module, which is positioned between convolutional layers and the ReLU function to help mitigate the risks associated with elimination singularities. We also provide mathematical proofs that demonstrate how Pool Skip’s affine compensation and dimensional Compensation optimize gradient fluctuations during the backpropagation process, thus addressing the degradation problem at a fundamental level. • We evaluated the proposed Pool Skip module across various deep learning models and datasets, including wellknown natural image classification benchmarks (e.g., CIFAR-10 and CIFAR-100), segmentation tasks (e.g., Pascal VOC and Cityscapes), and medical imaging challenges (e.g., BTCV, AMOS). Our findings validate the effectiveness of Pool Skip in reducing elimination singularities and demonstrate its capacity to enhance both the generalization and performance of models.

# Related Work

# Pooling Operations in CNNs

Max Pooling (Ranzato et al. 2007), a staple in CNN architectures, segments convolutional output into typically nonoverlapping patches, outputting the maximum value from each to reduce feature map size (Dumoulin and Visin 2016; Gholamalinezhad and Khosravi 2020). This not only yields robustness against local transformations but also leverages the benefits of sparse coding (Boureau, Ponce, and LeCun 2010; Boureau et al. 2011; Ranzato et al. 2007). Its efficacy is well-documented in prominent CNN architectures like VGG (Simonyan and Zisserman 2014), YOLO (Redmon et al. 2016), and UNet (Ronneberger, Fischer, and Brox 2015), and is essential in numerous attention mechanisms, such as CBAM (Woo et al. 2018), for highlighting salient regions. Despite these advantages, Ruderman et al. (Ruderman et al. 2018) indicate that networks can maintain deformation stability without pooling, primarily through the smoothness of learned filters. Furthermore, Springenberg et al. (Springenberg et al. 2014) suggest that convolutional strides could replace Max Pooling, as evidenced in architectures like nnUNet (Isensee et al. 2021).

Max Unpooling, designed to reverse Max Pooling effects by restoring maximum values to their original locations and padding zeros elsewhere, complements this by allowing CNNs to learn mid and high-level features (Zeiler and Fergus 2014; Zeiler, Taylor, and Fergus 2011). However, the traditional “encoder and decoder” architecture, foundational to many modern CNNs like UNet (Ronneberger, Fischer, and Brox 2015), rarely adopts Max Unpooling due to concerns that zero filling can disrupt semantic consistency in smooth areas (Liu et al. 2023).

Our work reevaluates the conventional combination of Max Pooling and Max Unpooling, arguing that its effective utilization can still substantially benefit CNNs by focusing on significant features. Moreover, the finite number of layers in common encoder and decoder architectures limits the use of Max Pooling in deeply nested CNNs, posing challenges for deep-level information recognition.

# Skip Connection and Batch Normalization

The concept of ES was first proposed by Wei et al. (Wei et al. 2008), to describe the issue of zero weights in the output of convolutional layers, a phenomenon commonly referred to as ”weight vanishing” (Wei et al. 2008). This issue is particularly concerning because these zero weights do not contribute to the model’s calculations, leading to inefficiencies in learning processes. ES is deeply associated with slow learning dynamics and unusual correlations between generalization and training errors and presents a significant challenge in training deep neural networks effectively (Amari, Park, and Ozeki 2006).

To mitigate ES, two primary strategies have emerged: normalization, specifically Batch Normalization (BN) (Ioffe and Szegedy 2015)), and skip connections (He et al. 2016a,b). BN helps maintain a stable distribution of activation values throughout training, while skip connections effectively increase network depth by preventing the elimination of singularities, ensuring that even with zero incoming or outgoing weights, certain layers maintain unit activation (Ioffe and Szegedy 2015; He et al. 2016a). This allows for the generation of non-zero features, making previously non-identifiable neurons identifiable, thereby addressing ES challenges (Orhan and Pitkow 2017). Despite these advancements, the degradation problem persists in extremely deep networks, even with the implementation of skip connections (He et al. 2016a). Further analysis by He et al. (He et al. 2016b) on various ResNet-1001 components—such as constant scaling, exclusive gating, shortcut-only gating, conv shortcut, and dropout shortcut—reveals that the degradation issue in the original ResNet block not only remains but is also exacerbated.

# Pool Skip Mechanism

In this section, we introduce the Pool Skip mechanism, beginning with the Weight Inertia hypothesis that motivates its development. We then provide theoretical insights into how this hypothesis helps mitigate degradation caused by elimination singularities through dimensional and affine compensation.

# Weight Inertia Hypothesis

In the context of back-propagation (Rumelhart, Hinton, and Williams 1986), the process is defined by several essential components. $L$ denotes the loss function, $W$ represents the weights, $Y$ refers to the output feature maps, $X$ to the input feature maps, while $c _ { i n }$ and $c _ { o u t }$ indicate the input and output channels, respectively. The $*$ is used for the convolution operation. The operation of back-propagation is captured as follows:

$$
\begin{array} { r l r } {  { \frac { \partial L } { \partial W _ { c _ { i n } , c _ { o u t } } } = \frac { \partial L } { \partial Y _ { c o u t } } \cdot \frac { \partial Y _ { c o u t } } { \partial W _ { c _ { i n } , c _ { o u t } } } } } \\ & { } & { = \frac { \partial L } { \partial Y _ { c _ { o u t } } } \cdot \frac { \partial \mathrm { R e } \mathrm { L U } ( X _ { c _ { i n } } \ast W _ { c _ { i n } , c _ { o u t } } ) } { \partial W _ { c _ { i n } , c _ { o u t } } } , } \\ & { } & { \frac { \partial L } { \partial X _ { c _ { i n } } } = \frac { \partial L } { \partial Y _ { c _ { o u t } } } \cdot \frac { \partial Y _ { c _ { o u t } } } { \partial X _ { c _ { i n } } } } \\ & { } & { = \frac { \partial L } { \partial Y _ { c _ { o u t } } } \cdot \frac { \partial \mathrm { R e } \mathrm { L U } ( X _ { c _ { i n } } \ast W _ { c _ { i n } , c _ { o u t } } ) } { \partial X _ { c _ { i n } } } . } \end{array}
$$

The activation function employed in this context is ReLU. During the convolution process, when the output $X _ { c _ { i n } } *$ $W _ { c _ { i n } , c _ { o u t } }$ is less than or equal to zero, ReLU sets both derivatives, ∂W ∂L and ∂X∂L to zero, in accordance with its operational rules. Or if the weights themselves $( W _ { c _ { i n } , c _ { o u t } } )$ are zero, it would still result in zero gradients.

According to the standard gradient descent update rule, the weights are adjusted by subtracting a portion of the gradient from the current weight values: $\begin{array} { r l } { \hat { W } _ { c _ { i n } , c _ { o u t } } } & { { } = } \end{array}$ $\begin{array} { r } { W _ { c _ { i n } , c _ { o u t } } \ - \ \eta \frac { \partial L } { \partial W _ { c _ { i n } , c _ { o u t } } } } \end{array}$ , where $\eta$ represents the learning rate. When training utilizes a fixed input space (a consistent set of training samples) and employs a diminishing learning rate towards the end of the training cycle, the updates to the weights in both the current and previous layers become minimal. This minimal update results in inputs $X _ { c _ { i n } }$ and output $X _ { c _ { i n } } * W _ { c _ { i n } , c _ { o u t } }$ at the current layer becoming inert. Consequently, the outputs $Y _ { c _ { i n } }$ of subsequent layers also remain unchanged. The worst case is a continuous negative or zero output.

![](images/52f2e7a9221b5cc077bd17274e90a92e0eed3a1b4e426be4b224703d60da7737.jpg)  
Figure 1: Schematic representation of the computational process of Pool Skip.

This stagnation, which we term Weight Inertia, results from sparse weights (i.e. ES) and consistent non-positive outputs (i.e Dying ReLU), particularly prevalent in what we refer to as the degradation layer. This layer is marked by a continuous inability to update zero weights, leading to a limited number of effective weights and exacerbating the degradation problem. This forms a self-reinforcing cycle: as weights fail to update, the network’s ability to learn and adapt diminishes, deepening the degradation. To break this cycle, controlling the negative outputs, specifically $X _ { c _ { i n } } * W _ { c _ { i n } , c _ { o u t } }$ , is crucial. By effectively managing these outputs, it is possible to interrupt the cycle, prevent further degradation, and enhance the network’s overall learning capabilities.

Motivated by the weight inertia hypothesis, we design Pool Skip, which consists of a Max Pooling layer, a Max Unpooling layer, and a $3 \times 3$ convolutional layer, tightly interconnected with skip connections spanning from the beginning to the end of the module. Figure 1 illustrates the computational process of Pool Skip. Initially, the Max Pooling layer prioritizes important features, facilitating the extraction of key information crucial for subsequent processing. Subsequently, the Max Unpooling layer ensures that the feature size remains consistent, preserving the gradient propagation process established by max-pooling. This characteristic allows Pool Skip to be seamlessly integrated into convolutional kernels at any position within the network architecture. Moreover, by selectively zeroing out non-key positions, Pool Skip effectively controls the magnitude of the compensatory effect, further enhancing its utility in stabilizing the training process.

# Affine and Dimensional Compensation

![](images/7366d1784688021eebdc3c67eb977ce890b0f514c57fba4e4dd0c10283404312.jpg)  
Figure 2: A simple example of dimensional and affine and compensation.

As discussed earlier in the weight inertia hypothesis, the output of a neural network is often predominantly determined by a linear combination of a few specific and influential input elements, despite the potential presence of numerous input elements. This selective influence suggests that modifying this fixed linear combination—either by activating input elements through changes in input dimensions or by adjusting the coefficients of the linear combination—can significantly impact the output. For convenience, we refer to these two adjustment mechanisms as dimensional compensation and affine compensation. A simple example illustrating this process is provided in Fig. 2. Initially, with one dimension $x _ { 1 }$ , the region where $x _ { 1 } < 0$ is shown in orange on the left side of Fig. 2(A). Introducing a second dimension $x _ { 2 }$ changes this to $x _ { 1 } + x _ { 2 } \ < \ 0$ , shifting the orange area to the right side of Fig. 2(A). This demonstrates dimensional compensation. When the coefficient in front of $x _ { 1 }$ and $x _ { 2 }$ changes from 1 to $^ { - 1 }$ , the shift in the orange area in Fig. 2(B) represents affine compensation. These compensations alter the negative region of the input space, thereby disrupting Weight Inertia.

Next, we will theoretically explain how Pool Skip introduces dimensional and affine compensation, subsequently affecting the output results. As depicted in Figure 1, we begin by establishing the input configuration based on the computational process of Pool Skip:

In the convolutional computation of a single layer, the output $y _ { i , j }$ is derived from a linear combination of the input $x _ { i , j }$ before the Pool Skip (Goodfellow, Bengio, and Courville

$\overline { { 1 . \ : X _ { H \times W } = \{ x _ { i , j } \} _ { H \times W } } } .$ input matrix;   
$2 . \ W _ { M \times M } = \{ w _ { i , j } \} _ { M \times M }$ : the convolutional kernel before Pool Skip. Assume $W$ is $M \times M$ kernel, and $M$ is an odd number;   
3. $~ . ~ { \cal Y } _ { ( H - M + 1 ) \times ( W - M + 1 ) } = \{ y _ { i , j } \} _ { ( H - M + 1 ) \times ( W - M + 1 ) } :$ the output of first convolutional computation;   
$4 . e \colon \mathbf { M a x }$ Pooling size which satisfies $e | H , e | W , e | H -$ $M + 1$ and $e | W - M + 1$ ;   
5. $A _ { c \times d }$ : the matrix obtained from max-pooling on $Y$ . $c = ( H - M + 1 ) / e$ and $d = ( W - M { + } 1 ) / e$ ;   
6. $\tilde { W } _ { 3 \times 3 } ~ = ~ \{ \tilde { w } _ { i , j } \} _ { 3 \times 3 }$ : the convolutional kernel in the Pool Skip;   
7 $\mathrm { ~ : ~ } O _ { H - M + 1 , W - H + 1 } = \{ o _ { i , j } \} _ { H - M + 1 , W - H + 1 }$ : the output matrix.

2016):

$$
y _ { i , j } = \sum _ { m = 0 } ^ { M - 1 } \sum _ { n = 0 } ^ { M - 1 } w _ { m , n } \times x _ { i + m , j + n } .
$$

It’s important to note that the convolution kernel is not flipped in this context. Based on the Max-Pooling, we can divide the $Y$ from previous computation by size $e \times e$ into $c \times d$ blocks. For each block of $Y$ , indexed as $Y ^ { ( u , v ) }$ , we have:

$$
Y ^ { ( u , v ) } = \left( \begin{array} { c c c } { { y ^ { ( u e , v e ) } } } & { { \cdot \cdot \cdot } } & { { y ^ { ( u e , ( v + 1 ) e - 1 ) } } } \\ { { \vdots } } & { { \ddots } } & { { \vdots } } \\ { { y ^ { ( ( u + 1 ) e - 1 , v e ) } } } & { { \cdot \cdot \cdot } } & { { y ^ { ( ( u + 1 ) e - 1 , ( v + 1 ) e - 1 ) } } } \end{array} \right)
$$

where $u \in \{ 0 , 1 , \cdot \cdot \cdot , c - 1 \} , v \in \{ 0 , 1 , \cdot \cdot \cdot , d - 1 \}$ . For each block $Y ^ { ( u , v ) }$ , the maximum element is

$$
\tilde { y } ^ { ( u , v ) } = \operatorname* { m a x } _ { i ^ { ( u , v ) } , j ^ { ( u , v ) } \in \{ 0 , 1 , \cdots , e - 1 \} } Y _ { i ^ { ( u , v ) } , j ^ { ( u , v ) } } ^ { ( u , v ) } ,
$$

with the corresponding $\tilde { i _ { a } } ^ { ( u , v ) } , \tilde { j _ { a } } ^ { ( u , v ) }$ as

$$
( \tilde { i _ { a } } ^ { ( u , v ) } , \tilde { j _ { a } } ^ { ( u , v ) } ) = \underset { i ^ { ( u , v ) } , j ^ { ( u , v ) } \in \{ 0 , 1 , \cdots , e - 1 \} } { \arg \operatorname* { m a x } } Y _ { i ^ { ( u , v ) } , j ^ { ( u , v ) } } ^ { ( u , v ) } .
$$

Therefore, we can write $A$ as $A = \{ \tilde { y } ^ { ( u , v ) } \} _ { c \times d }$ . And then via Max Unpooling and padding operations, we could get:

$$
y _ { i , j } ^ { \prime } = \left\{ \begin{array} { l } { \tilde { y } _ { ( i _ { a } ^ { \sim } ( u , v ) } ^ { ( u , v ) } , \mathrm { i f } e m o d ( i - 1 ) = \tilde { i _ { a } } ^ { ( u , v ) } } \\ { \mathrm { a n d } e m o d ( j - 1 ) = \tilde { j _ { a } } ^ { ( u , v ) } \mathrm { i n b l o c k } Y ^ { ( u , v ) } } \\ { 0 , \mathrm { o . w . } } \end{array} \right.
$$

where $1 \leq i \leq H - M + 1$ and $1 \leq j \leq W - M + 1$ . After passing the $3 { \times } 3$ convolutions, the result of the convolutional computation is shown as:

$$
y _ { o u t , i , j } = \sum _ { s = 0 } ^ { 2 } \sum _ { t = 0 } ^ { 2 } \tilde { w } _ { s , t } \times y _ { i + s , j + t } ^ { \prime } .
$$

After introducing the Pool Skip, the output $o _ { i , j }$ is calculated by Eq. (2), where we denote set:

$$
\begin{array} { r l } & { { K _ { i } } = \{ ( m , s ) : u e + \tilde { i _ { a } } ^ { ( u , v ) } + m } \\ & { + s \in ( [ i , i + M ] \cap [ u e + \tilde { i _ { a } } ^ { ( u , v ) } , u e + \tilde { i _ { a } } ^ { ( u , v ) } + M + 2 ] ) \} , } \\ & { { L _ { j } } = \{ ( n , t ) : v e + \tilde { j _ { a } } ^ { ( u , v ) } + n } \\ & { + t \in ( [ j , j + M ] \cap [ v e + \tilde { j _ { a } } ^ { ( u , v ) } , v e + \tilde { j _ { a } } ^ { ( u , v ) } + M + 2 ] ) \} . } \end{array}
$$

Note that when $( m , s ) \in K _ { i }$ , $u e + \tilde { i _ { a } } ^ { ( u , v ) } + m + s = i + m ,$ ， and when $( n , t ) \in L _ { j }$ , $v e + \tilde { j _ { a } } ^ { ( u , v ) } + n + t = j + n$ .

According to Eq. (2) (see detailed derivation in supplementary material), the original linear combination obtained two types of compensation. When the maximum value obtained after passing through the Pool Skip consists of the original linear combination elements $x$ (from the input feature of the convolutional kernel before the Pool Skip), part of the $x$ coefficients changed from $w _ { m , n }$ to $( 1 + \tilde { w } _ { s , t } ) \times$ $w _ { m , n }$ , representing affine compensation. This change correlates closely with the weights of the convolutional kernel in Pool Skip. On the other hand, the remaining maximum values, which cannot be added to the original linear combination of $x$ , expand the output dimensions (i.e. adding xue+i˜a(u,v)+m+s,ve+j˜a(u,v)+n+t), constituting dimensional compensation. This compensation also remains closely tied to the weights of the convolutional kernels in the Pool Skip.

These adjustments not only alter the contribution of input elements but also affect the negative range space of the output. This effectively breaks the constraints imposed by weight inertia, promoting diversity in output results and updating zero weights. Additionally, by adjusting the size of the Max Pooling and Max Unpooling kernels, we can control the number of maximum values, directly influencing the strength of the compensatory effects. Specifically, when the size of pooling kernels is 1, indicating only one convolution in the skip connection, every output element receives compensation. After receiving the compensatory effect, the original negative value range changes, allowing the original linear combination to output a non-zero effective value after ReLU. This activation enables neurons in the next layer to be activated during forward propagation and ensures that convolutional kernels with zero weights before ReLU receive gradient updates during backpropagation, thus alleviating the ES problem.

# Experiments

We integrated the proposed Pool Skip into various deep networks and conducted comprehensive evaluations across common image tasks, including classification as well as natural image and medical image segmentation, utilizing diverse datasets for robust validation. All models were equipped with BN and ReLU or ReLU variations following the convolutions, ensuring a standardized architecture for comparison. Furthermore, all models were trained using a single NVIDIA A100 GPU with 80G of memory to maintain consistency in computational resources.

Final Output: oi,j = yi,j + yout,i,j

$$
\begin{array} { r l r } {  { ( 1 + \overbrace { w } _ { s , t } \ \sum _ { i = 1 } ^ { \sum } \sum _ { \substack { ( ( n , t ) ) = 1 } } [ ( 1 + \tilde { w } _ { s , t } ) \times w _ { m , n } \times x _ { i - 1 + m , j - 1 + n } ] +  } } \\ & { } & { = \{ \begin{array} { l l } { \displaystyle 1 _ { K _ { i } ( ( m , s ) ) = 1 } \sum _ { i = 1 } \ ( w _ { m , n } ) = 1 } & { \displaystyle 1 } \\ { \qquad \sum _ { \substack { ( w , n ) = 1 } } \sum _ { i = 1 + m , j - 1 + n } + \tilde { w } _ { s , t } \times w _ { m , n } \times x _ { w \in + \tilde { \nu } _ { a } ( \nu , \nu ) + m + s , \nu \in + \tilde { f } _ { a } ( \nu , \nu ) + n + t } \rangle } \\ { \displaystyle 1 _ { K _ { i } ( ( m , s ) ) \neq 1 } \mathbb { E } _ { I _ { 3 } ( [ n , t ) ) \neq 1 } } & { \displaystyle  ( 1 , \tilde { w } _ { 1 } ) \sum _ { \substack { ( i , n ) = 1 } } ) \mathrm { a n d } \ e m o d ( j - 1 ) = \tilde { f } _ { a } ^ { ( ( n , \nu ) } \ \mathrm { i n b b l o c k } \ Y ^ { ( u , \nu ) } , } \\ { \displaystyle  \sum _ { m = 0 } ^ { 1 - 1 } \sum _ { m , n } ^ { - 1 } w _ { m , n } \times x _ { i - 1 + m , j - 1 + n } , } & { \mathrm { o . w . } } \end{array}  } \\ & { } & { \quad \mathrm { f o r a l l ~ } i \in \{ 1 , 2 , \dots , H - M + 1 \} \mathrm { ~ a n d ~ } j \in \{ 1 , 2 , \dots , W - M + 1 \} . } \end{array}
$$

# Image Classification

Datasets For the classification task, we utilized the CIFAR datasets (Krizhevsky, Hinton et al. 2009). CIFAR-10 comprises 60,000 color images categorized into 10 classes. CIFAR-100 consists of 60,000 images divided into 100 classes, with each class containing 600 images. The images are colored and share the same dimensions of $3 2 \times 3 2$ pixels.

Comparison Methods We evaluated the effectiveness of the Pool Skip across various CNN architectures, including MobileNet (Howard et al. 2017), GoogLeNet (Szegedy et al. 2015), VGG16 (Simonyan and Zisserman 2014), ResNet18 (He et al. 2016a), and ResNet34 (He et al. 2016a). To ensure robustness, each model was trained with 5 diverse seeds on the official training dataset, and the average and standard deviation of the Top-1 error from the final epoch were calculated on the official test dataset. Moreover, to assess the impact of the Pool Skip, it was implemented in each convolutional layer rather than solely in the first layer. All the training settings followed Devries and Taylor’s work (DeVries and Taylor 2017).

Experimental Results Our experimental findings, detailed in Table 1, showcase the performance enhancements observed across the CIFAR10 and CIFAR100 datasets. Notably, we observed moderate improvements ranging from 0.5 to 5.44 on CIFAR100 and from 0.03 to 2.74 on CIFAR10 in networks with fewer layers. However, the magnitude of improvement varied depending on the architecture of the network. Of particular significance was the notable enhancement observed for MobileNet upon the integration of the Pool Skip.

# Natural Image Segmentation

Datasets For this task, we utilized Cityscapes (Cordts et al. 2016) and PASCAL Visual Object Classes (VOC) Challenge (Pascal VOC) (Everingham et al. 2010) datasets. Cityscapes offers a comprehensive insight into complex urban street scenes, comprising a diverse collection of stereo video sequences captured across streets in 50 different cities. Pascal VOC provides publicly accessible images and annotations, along with standardized evaluation software. For segmentation tasks, each test image requires predicting the object class of each pixel, with “background” designated if the pixel does not belong to any of the twenty specified classes.

Table 1: Top-1 error rates $\mathrm { \mathbf { M e a n } } \pm \mathrm { \mathbf { S t d } } )$ for image classification on CIFAR 100 and CIFAR 10 datasets.   

<html><body><table><tr><td rowspan="2">Model</td><td>CIFAR100</td><td>CIFAR10</td></tr><tr><td>Top-1error(%)↓</td><td>Top-1 error(%)↓</td></tr><tr><td>MobileNet</td><td>33.75±0.24</td><td>9.21 ± 0.19</td></tr><tr><td>+ours</td><td>28.31 ± 0.23 -5.44</td><td>6.47 ± 0.20 -2.74</td></tr><tr><td>GoogleNet</td><td>22.95 ±0.24</td><td>5.35 ± 0.19</td></tr><tr><td>+ours</td><td>22.36± 0.32 -0.59</td><td>5.19±0.14 -0.16</td></tr><tr><td>VGG16</td><td>27.84 ±0.38</td><td>6.24 ±0.18</td></tr><tr><td>+ours</td><td>27.23 ± 0.21-0.61</td><td>5.90± 0.24 -0.34</td></tr><tr><td>ResNet18</td><td>24.06 ±0.18</td><td>5.17 ± 0.15</td></tr><tr><td>+ours</td><td>23.32 ±0.14 -0.74</td><td>5.10±0.14 -0.07</td></tr><tr><td>ResNet34</td><td>22.69 ±0.18</td><td>4.89 ±0.07</td></tr><tr><td>+ours</td><td>22.19 ± 0.22 -0.50</td><td>4.86± 0.06-0.03</td></tr></table></body></html>

Comparison Methods We evaluated the effectiveness of the Pool Skip on DeepLabv $^ { 3 + }$ models (Chen et al. 2017, 2018), utilizing ResNet101 (He et al. 2016a) and MobileNet-v2 (Sandler et al. 2018) backbones. The Pool Skip was exclusively employed in the convolution of the head block, the first convolution of the classifier, and all the atrous deconvolutions to validate its compatibility with atrous convolution. The models were trained using the official training data and default settings in (Chen et al. 2017, 2018), and the Intersection over Union (IoU) of the finalepoch model was recorded on the official validation data. Five seeds were selected to calculate the mean and standard deviation of the results.

Experimental Results As illustrated in Table 2, our experiments demonstrated a modest improvement in mIoU ranging from $0 . 1 6 \%$ to $0 . 5 3 \%$ for DeepLabv $^ { 3 + }$ models, considering the incorporation of only five layers of the Pool Skip (one in the Deeplab head, three in the atrous deconvolutional layers, and one in the classifier). This indicates the compatibility of the Pool Skip with atrous deconvolutions.

# Medical Image Segmentation

Datasets We used abdominal multi-organ benchmarks for medical image segmentation, i.e., AMOS (Ji et al.

Table 2: The mIoU (Me $\mathrm { a n } \pm \mathrm { S t d } )$ results for natural image segmentation on the Cityscapes and Pascal VOC datasets. “DLP” denotes “DeepLabv $3 \mathrm { + } ^ { \dag }$ .   

<html><body><table><tr><td rowspan="2">Model</td><td>Cityscapes</td><td>Pascal VOC</td></tr><tr><td>mIoU (%) ↑</td><td>mIoU(%) ↑</td></tr><tr><td rowspan="2">DLP_MobileNet +ours</td><td>71.72 ± 0.49</td><td>66.40±0.37</td></tr><tr><td>71.96 ± 0.35 +0.24</td><td>66.93 ± 0.49 +0.53</td></tr><tr><td rowspan="2">DLP_ResNet101 +ours</td><td>75.59± 0.30</td><td>74.83 ± 0.56</td></tr><tr><td>75.89 ± 0.10 +0.30</td><td>74.99 ± 0.23 +0.16</td></tr></table></body></html>

2022) and Multi-Atlas Labeling Beyond the Cranial Vault (BTCV) (BA et al. 2015) datasets. AMOS is a diverse clinical dataset offering $3 0 0 \mathrm { C T }$ (Computed Tomography) scans and $6 0 ~ \mathrm { M R I }$ (Magnetic Resonance Imaging) scans with annotations. The public BTCV dataset consists of 30 abdominal CT scans sourced from patients with metastatic liver cancer or postoperative abdominal hernia.

Comparison Methods We evaluated the Pool Skip using nnUNet (Isensee et al. 2021) and V-Net (Milletari, Navab, and Ahmadi 2016). For nnUNet, our implementation closely follows the nnUNet framework (Isensee et al. 2021), covering data preprocessing, augmentation, model training, and inference. Scans and labels were resampled to the same spacing as recommended by nnUNet. We excluded nnUNet’s post-processing steps to focus on evaluating the model’s core segmentation performance. For a fair comparison, when reproducing nnUNet, we retained its default configuration. For V-Net (Milletari, Navab, and Ahmadi 2016), we adopted the preprocessing settings consistent with nnUNet.

For the BTCV dataset, 12 scans were assigned to the test set, and 18 to the training and validation set. From AMOS, 360 scans(containing CTs and MRIs) were divided into 240 for training and validation and 120 for testing, with a training-to-validation ratio of 4:1. We performed 5-fold cross-validation on all models, averaging their softmax outputs across folds to determine voxel probabilities. Our evaluation is based on the Dice Score (Milletari, Navab, and Ahmadi 2016), Normalized Surface Dice (NSD) (Nikolov et al. 2018), and mIoU metrics.

For the V-Net model, we only added the Pool Skip to the first two encoders due to the odd size of feature map outputs by the final encoder. As for the nnUNet model, we applied the Pool Skip in each convolutional layer on all encoders when training on the BTCV dataset. When training on the AMOS dataset, Pool Skips were employed on all encoders except for the final encoder.

Experimental Results As illustrated in Table 3, this Pool Skip architecture applies to networks for 3D medical imaging segmentation tasks. Enhancement on V-Net and nnUNet demonstrate the Pool Skip’s efficacy for complex image segmentation tasks.

# Further Analysis

Efficacy of Pool Skip in Deep CNNs To assess the effectiveness of the Pool Skip in deep CNNs, we utilized

Table 3: Performance results of medical image segmentation on two datasets.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="3">BTCV</td></tr><tr><td>DICE (%)↑</td><td>NSD (%) ↑</td><td>mIoU(%)↑</td></tr><tr><td>V-Net</td><td>78.32</td><td>70.77</td><td>68.08</td></tr><tr><td>+ours</td><td>79.70 +1.38</td><td>72.35+1.58</td><td>69.48 +1.40</td></tr><tr><td>nnUNet</td><td>81.52</td><td>76.10</td><td>72.14</td></tr><tr><td>+ours</td><td>82.47 +0.95</td><td>77.00 +0.90</td><td>73.08 +0.94</td></tr><tr><td></td><td></td><td>AMOS</td><td></td></tr><tr><td>V-Net</td><td>77.15</td><td>62.97</td><td>66.32</td></tr><tr><td>+ours</td><td>80.02 +2.87</td><td>67.32+4.35</td><td>69.81+3.49</td></tr><tr><td>nnUNet</td><td>89.75</td><td>85.58</td><td>83.13</td></tr><tr><td>+ours</td><td>89.78 +0.03</td><td>85.52-0.06</td><td>83.13 +0</td></tr></table></body></html>

![](images/d36981d68a0c40860428a6917679064fee2748fa7b4745bc6ed87f3c26c9ae63.jpg)  
Figure 3: Top-1 error rates $( \% )$ of deep ResNet on CIFAR10 and CIFAR100 datasets. The pool kernel size is 4 for CIFAR100 experiments and 2 for CIFAR10 experiments.

ResNet (He et al. 2016a) as our baseline architecture. Our experiments covered a range of network depths, including 50, 101, 152, 200, 350, and 420 layers. For networks with fewer than 152 layers, we adhered to the original ResNet architecture as proposed by He et al. (He et al. 2016a). For deeper architectures (i.e., 200, 350, and 420 layers), we followed the architectural specifications outlined in a prior study (Bello et al. 2021). Training settings were consistent with those outlined in Devries and Taylor’s work (DeVries and Taylor 2017).

The performance of the models across varying network depths is depicted in Figure 3. Initially, as the number of layers increases, the model’s performance improves before reaching a peak. The original ResNet achieves its best performance with 152 layers, achieving a top-1 error of 4.584 on CIFAR10 and 20.78 on CIFAR100. Subsequently, performance deteriorates rapidly. However, upon integrating the Pool Skip, performance improves, with a lower top-1 error of 4.562 on CIFAR10 with 350 layers and 20.752 on CIFAR100 with 152 layers. After stabilizing, performance begins to decline again. Nevertheless, there is a noticeable improvement in top-1 error (0.1-0.25 on CIFAR10 and 0.8-1 on CIFAR100) in deep networks. This underscores the efficacy of Pool Skip in enhancing the performance of deep CNNs. Thus, Pool Skip demonstrates promise in mitigating the degradation phenomenon that both BN and ResNet struggle to address, particularly with the increase in model depth.

![](images/962a2e9ebd42a9fab584bb5d5dcbb6989a3b3e9fd7b83db065dd76558f9f207e.jpg)  
Figure 4: $\frac { l _ { 2 } } { l _ { 1 } }$ value quantitative comparison in ResNet350 and ResNet420 on CIFAR10 and CIFAR100 Datasets. The $\frac { l _ { 2 } } { l _ { 1 } }$ values were computed based on the output sequence of the network, with and without the incorporation of the Pool Skip. The plot highlights a moderate alleviation of the network degradation issue in shallow layers upon the integration of Pool Skip. Note: The horizontal axis represents the layers of the network along the output direction, from left to right. The “Pool Skip S4” means the size of Pool operation kernel is 4, “Pool Skip S4” does 2.

Efficacy of Mitigating Elimination Singularities (ES) To assess weight sparsity, we calculated the $\frac { l _ { 2 } } { l _ { 1 } }$ ratio for each layer with and without the Pool Skip, as proposed by Hoyer et al. (Hoyer 2004). The higher the $\frac { l _ { 2 } } { l _ { 1 } }$ value is, the more zero the weights may contain (Wang et al. 2020; Hoyer 2004). Figure 4 illustrates the $\frac { l _ { 2 } } { l _ { 1 } }$ curve. Original ResNet350 and ResNet420 architectures suffered from severe degradation issues in shallow layers, despite the inclusion of batch normalization. However, integrating the Pool Skip noticeably alleviated this problem. Specifically, the $\frac { l _ { 2 } } { l _ { 1 } }$ ratios for ResNet350 decreased from a maximum of 0.7 to 0.1 on CIFAR100 and from 0.35 to 0.1 on CIFAR10. Similarly, for ResNet420, the ratios decreased from a maximum of 0.4 to 0.15 on CIFAR100 and from 1 to 0.15 on CIFAR10. This underscores the efficacy of the Pool Skip in mitigating elimination singularities, thereby enhancing model stability and maintaining feature integrity across layers.

Ablation Experiments We conducted ablation experiments with VGG16, as shown in Table 4, to assess the contribution of each block to the Pool Skip. Skip connections proved to be the most critical, with their removal causing a $13 \%$ increase in Top-1 error. Using only convolutions and skip connections increased the Top-1 error by $3 \%$ , while using only skip connection and pool design had minimal impact on performance.

Table 4: Top-1 error rates $( \% )$ of ablation experiments on VGG16: “Pool” denotes Max Pooling and Max Unpooling, “Conv” represents $3 \times 3$ convolutions, and “Skip” indicates skip connections.   

<html><body><table><tr><td>VGG16 +{Pool,Skip} +{Conv,Skip} +{Pool,Conv} +{Ours}</td><td></td><td></td><td></td></tr><tr><td>27.84</td><td>27.88</td><td>30.82</td><td>44.07 27.23</td></tr></table></body></html>

# Discussion

While Pool Skip holds promise for mitigating the elimination singularities issue in convolutional kernels and has demonstrated effectiveness across extensive datasets and network architectures, the proposed structure still has some limitations. Firstly, since each convolutional kernel is followed by a $3 \times 3$ convolution, the overall number of parameters in the network increases significantly, thereby adding a burden to both training and inference processes. Additionally, the sizes of Max Pooling kernels used in the experiments with deep ResNet are 2 and 4, while in compatibility experiments, it is 2. Additionally, the effectiveness of dimensional and affine compensations needs to be optimized through the adjustments of the pooling size in the Pool Skip.

Table 5: Top-1 error rates (Mean $\pm$ Std) of ViT-based models.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">CIFAR100 CIFAR10</td></tr><tr><td>Top-1 error(%)↓</td><td>Top-1 error(%)↓</td></tr><tr><td>ViT</td><td>25.37 ± 0.17</td><td>6.85 ± 0.23</td></tr><tr><td>+ours</td><td>25.07 ± 0.33 -0.30</td><td>7.07 ± 0.27 +0.22</td></tr><tr><td>CCT</td><td>19.16 ±0.20</td><td>3.87 ±0.19</td></tr><tr><td>+ours</td><td>18.61 ± 0.12 -0.55</td><td>3.89 ± 0.14 +0.02</td></tr><tr><td>CVT</td><td>22.92 ± 0.38</td><td>5.98 ±0.19</td></tr><tr><td>+ours</td><td>22.80 ± 0.40 -0.12</td><td>6.36 ± 0.36 +0.38</td></tr></table></body></html>

On the other hand, ViTs generally rely on one or more convolutional layers to generate image patches. To explore potential improvements in performance, we experimented with integrating the pool skip directly into the patch generation process. Our goal was to observe how this modification influences the overall effectiveness of the ViTs. We evaluated three ViT-based models including original ViT (Dosovitskiy et al. 2020), CCT (Hassani et al. 2021) and CVT (Wu et al. 2021) on the CIFAR10 and CIFAR100 datasets. According to Table 5, pool skip gains from 0.12 to 0.55 reduction in Top-1 error on CIFAR100, but from 0.02 to 0.38 deterioration. We believe that the observed performance changes are likely due to the pooling layer’s ability to extract key features, leading to performance improvements. However, the potential information loss caused by pooling may also contribute to performance deterioration. This also represents a limitation in the application of this structure.

# Conclusion

In this paper, we proposed Pool Skip, a simple architectural enhancement addressing elimination singularities in deep CNN training. Based on the Weight Inertia hypothesis, it offers affine and dimensional compensation, stabilizing training. Experiments on various datasets and models confirm its effectiveness in optimizing CNNs.
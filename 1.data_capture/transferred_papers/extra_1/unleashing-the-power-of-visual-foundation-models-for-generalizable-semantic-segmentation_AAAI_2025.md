# Unleashing the Power of Visual Foundation Models for Generalizable Semantic Segmentation

PeiYuan Tang1, Xiaodong Zhang2,3\*, Chunze Yang1, Haoran Yuan4, Jun $\mathbf { S u n } ^ { 5 }$ , Danfeng Shan1, Zijiang James Yang4,6\*

1School of Computer Science and Technology, Xi’an Jiaotong University 2School of Computer Science and Technology, Xidian University 3Shaanxi Key Laboratory of Network and System Security, Xidian University 4Synkrotron, Inc. 5Singapore Management University 6University of Science and Technology of China tangpeiyuan $@$ stu.xjtu.edu.cn, Zhangxiaodong $@$ xidian.edu.cn

# Abstract

Deep learning models often suffer from performance degradation in unseen domains, posing a risk for safety-critical applications such as autonomous driving. To tackle this problem, recent studies have leveraged pre-trained Visual Foundation Models (VFMs) to enhance generalization. However, exsiting works mainly focus on designing intricate networks for VFMs, neglecting their inherent strong generalization potential. Moreover, these methods typically perform inference on low-resolution images. The loss of detail hinders accurate predictions in unseen domains, especially for small objects. In this paper, we argue that simply fine-tuning VFMs and leveraging high-resolution images unleash the power of VFMs for generalizable semantic segmentation. Therefore, we design a VFM-based segmentation network (VFMNet) that adapts VFMs to this task with minimal fine-tuning, preserving their generalizable knowledge. Then, to fully utilize high-resolution images, we train a Mask-guided Refinement Network (MGRNet) to refine VFMNet’s predictions combining detailed image features. Furthermore, we adopt a twostage coarse-to-fine inference approach. MGRNet is used to refine the low-confidence regions predicted by VFMNet to obtain fine-grained results. Extensive experiments demonstrate the effectiveness of our method, outperforming stateof-the-art methods by $3 . 3 \%$ on the average mIoU in syntheticto-real domain generalization.

Code — https://github.com/tpy001/VFMSeg

# Introduction

Deep learning has significantly advanced computer vision tasks like semantic segmentation (Chen et al. 2017; Xie et al. 2021; Cheng et al. 2022). These successes ususally rely on the basic assumption that the training and testing data should come from the same distribution. When models are deployed in the real world, they might encounter unseen scenarios outside of their training data. This may lead to significant performance drops, posing a threat to safety-critical applications, such as autonomous driving. Collecting and labeling data for all possible scenarios is the most effective way to solve this problems, but it often requires a significant amount of time and money. Therefore, Domain Generalization (Yang, Gu, and Sun 2023; Fan et al. 2023; Wang et al. 2023b) has been widely investigated to enhance model generalization ability. It aims to train models on labeled data from source domains (e.g., synthetic data) such that it performs well on unseen domains (e.g., real-world data).

![](images/9c507ce0be01b8b15de4f1a554643325aba5387642bba336f78c46886ef0f766.jpg)  
Figure 1: Comparison of predictions using VFMs at low and high resolutions. When inferring on low-resolution images, VFMs often struggle with exact boundaries and small objects in the distance. High-resolution inference using a sliding window approach produces fine-grained results.

In the field of Domain Generalization Semantic Segmentation (DGSS), existing works mainly focus on domaininvariant feature learning (Xu et al. 2022; Peng et al. 2022) and data augmentation techniques (Peng et al. 2021; Zhong et al. 2022) to prevent overfitting to the source domains. In recent years, Visual Foundation Models (VFMs) have demonstrated remarkable performance across various computer vision tasks (Radford et al. 2021; Kirillov et al. 2023; Oquab et al. 2023). Through pre-training on large-scale image data, VFMs acquire universal visual representations that can be transfered to other domains. Their strong image understanding and generalization ability make them ideal for developing generalizable models.

Previous works (Wei et al. 2024; H¨ummer et al. 2023) have leveraged VFMs for DGSS by employing their encoders as the feature extractor and then training a complex decoder like Mask2Former (Cheng et al. 2022). We observe that while complex decoders enhance model capacity, they potentially lead to overfitting to the source domain, undermining the generalizability of the VFMs. Since VFMs are trained on large-scale data, we hypothesize that their prior knowledge may help recognize long-tail classes in unseen domains. Therefore, We argue that fine-tuning the VFM with a simple decoder can adapt the model to a specific domain while preserving its generalizable prior knowledge.

High-resolution images with rich detials might improve the segmentation result. Howerev, most VFMs struggle with high-resolution images partly due to the length extrapolation problem (Song et al. 2024), i.e. the inconsistency in token length between the training and prediction impairs performance. Therefore, previous methods (Wei et al. 2024) had to perform inference on downsampled images, leading to poor segmentation results, as shown in Fig. 1. Sliding window inference mitigates this problem by dividing the image into fixed-size patches and then making predictions for each patch. But it still faces challenges in generalizing to higher resolutions. As resolution increases, each patch contains less content, leading to a lack of context and degraded performance. For instance, if a bicycle underneath a person does not appear in the patch, it can be challenging to distinguish whether this person is a pedestrian or a cyclist.

To address this issue, we propose leveraging lowresolution semantic predictions to guide inference on highresolution patches. The low-resolution predictions provide initial labels for each pixel, although they may contain errors such as ignoring small objects. Based on these class priors, the model is trained to retain labels that match the image features and adjust those that do not. Therefore, by extracting useful information from low-resolution semantic predictions, the model can effectively overcome the lack of contextual information.

In this work, we focus on the task of domain generalization semantic segmentation (DGSS) and introduce a novel framework to fine-tune VFMs and enable high-resolution inference. To achieve this, we design a simple yet effective VFM-based segmentation network (VFMNet) to learn the global layout and content of input images, alongside a Mask-Guided Refinement Network (MGRNet) that focuses on details. The coarsed mask predicted by VFMNet is used as a guidance for MGRNet to refine high-resolution image features. During inference, we employ a two-stage coarseto-fine approach to effectively combine high-resolution and low-resolution predictions. The main contributions of this work are summarized as follows:

• We design a simple yet effective network to utilize VFMs for sematntic segmentation while preserving their rich generalizable knowledge. • We introduce a multi-scale training framework along with a coarse-to-fine sliding window inference method to enable high-resolution inference. • Extensive experiments are conducted on various benchmarks and backbones to validate the effectiveness of our method and demonstrate the superior performance of our approach over the state-of-the-art methods.

# Related Works

# Domain Generalized Semantic Segmentation

Domain Generalization aims to train a generalizable model on labeled source domains such that it performs well across multiple domains. In Domain Generalization Semantic Segmentation (DGSS), existing methods can be divided into two categories: (1) domain-invariant feature learning forces the model to learn domain-agnostic features. Some approaches (Pan et al. 2018; Peng et al. 2022; Choi et al. 2021) leverage Instance Normalization (IN) or Instance Whitening (IW) to standardize global features. Other approaches (Huang et al. 2023; Yang, Gu, and Sun 2023) project images into a feature space to reduce style variations. This method effectively removes domain-specific statistics, but it is only implemented on simple backbones like ResNet, leaving its effectiveness on other transformer-based backbones unclear. (2) Data augmentation has proven to be a simple and effective technique in DGSS. (Peng et al. 2021; Zhong et al. 2022) randomizes the style or texture of images, increasing the diversity of training data. Other approaches (Jia et al. 2024; Benigmim et al. 2024) leverage generative models like Stable Diffusion (Rombach et al. 2022) to synthesize new images. However, this method significantly increases the training time, and the model’s performance is unstable as it depends on the quantity and quality of the generated data.

# Visual Foundation Models

Visual Foundation Models (VFMs) are base models trained on large-scale image data in a self-supervised or semisupervised manner (Bommasani et al. 2021). By being pretrained on millions of images, they acquire general knowledge, allowing them to easily adapt to various downstream visual tasks (Zhang et al. 2023; Wang et al. $2 0 2 3 \mathrm { a }$ ; Hu¨mmer et al. 2023). CLIP (Radford et al. 2021) is a visionlanguage model that learns high-quality visual representations through contrastive learning with large-scale imagetext pairs. SAM (Kirillov et al. 2023) is an interactive image segmentation model trained on a large-scale segmentation dataset, demonstrating zero-shot segmentation ability even for unseen objects. DINOv2 (Oquab et al. 2023) is pretrained on carefully curated datasets with self-supervised learning methods, achieving general visual representations without task-specific annotations.

Due to the superior performance of VFMs, many recent works have utilized them for generalized segmentation. Rein (Wei et al. 2024) proposed an effective fine-tuning method that maintains the generalization ability of VFMs. CLOUDS (Benigmim et al. 2024) designs a framework that combines multiple VFMs to leverage their strengths. (Pak et al. 2024) design a textual query-driven transformer that leverages domain-invariant semantic knowledge from text to enhance generalization. However, most previous methods ignore the challenges faced by VFM when processing high-resolution images. In this paper, we address this issue by developing a novel framework.

![](images/702f28316f058092a0bebfc5e56dd49222b5d3756fa43667316b544988193af9.jpg)  
Figure 2: The overall framework of our method consisting of VFM-based Segmentation Network (VFMNet) and Maskedguided Refinement Network (MGRNet). (a) VFMNet and MGRNet are trained on resized and cropped images, respectively. Both networks share the same encoder from the VFM but use different decoders. To introduce contextual information, the coarse mask from VFMNet is used as a class prior for MGRNet. (b) During inference, a two-stage coarse-to-fine inference method combines high-resolution features with low-resolution predictions for fine-grained results.

# Method

# Problem Definition

In this work, we focus on single-source domain generalization for semantic segmentation. We define the source domain $S$ as a set of image-label pairs $S ~ = ~ \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , where $x _ { i } ~ \in ~ \mathbb { R } ^ { H \times W \times 3 }$ denotes an RGB image, and $y _ { i } \in$ $\{ 0 , 1 \} ^ { H \times W \times N }$ is the one-hot encoded class label for e ∈h pixel, where $N$ is the number of classes.

Our goal is to train a generalizable semantic segmentation model on a single source domain $S$ such that it performs well on various unseen domians $T = \{ T _ { 1 } , T _ { 2 } , . . . \bar { T } _ { m } \}$ . We first present the overall framework and our main ideas, and discuss details such as model architectures in later sections.

# Framework Overview

We aim to adapt VFMs to this tasks while enabling highresolution inference. To achieve this, we train two networks: one to capture global context and the other to focus on local details. Their predictions are then combined during inference, as shown in Fig. 2. Specifically, we design the VFMbased segmentation network (VFMNet) and the Maskedguided Refinement Network (MGRNet), denoted as $f _ { \theta }$ and $g _ { \phi }$ , respectively.

VFMNet is trained on low-resolution images, focusing on learning the global layout and content of the image. Conversely, MGRNet is trained on high-resolution image crops to capture fine-grained details. To supplement the missing context, we input the coarse segmentation mask of VFMNet to MGRNet as the class prior. This allows MGRNet to effectively refine the image features guided by the mask.

We share the weights of the VFM’s encoder between the two networks. This not only reduces the number of trainable parameters but also facilitates learning of both global and local features.

Training. We consider semantic segmentation as a pixelwise classification task and employ the cross-entropy loss to train two networks, as shown in the following equations:

$$
\mathcal { L } _ { s e g } ( y , \hat { y } ) = - \sum _ { i = 1 } ^ { H } \sum _ { j = 1 } ^ { W } \sum _ { c = 1 } ^ { N } y _ { i , j , c } \log \hat { y } _ { i , j , c }
$$

We denote the loss of VFMNet as content loss $\mathcal { L } _ { c }$ and the loss of MGRNet as detail loss $\mathcal { L } _ { d }$ . The total loss is a weighted sum of individual network losses:

$$
\mathcal { L } = \mathcal { L } _ { c } + \lambda \mathcal { L } _ { d }
$$

where $\lambda$ is a hyperparameter that balances the contribution of the detail loss relative to the content loss.

Inference. Since VFMNet excels at capturing global context, while MGRNet focuses on local details, we propose a two-stage inference strategy to leverage both strengths, as illustrated in Fig. 2.

Given an input image $x _ { i } \in \mathbb { R } ^ { H \times W \times 3 }$ , the first stage involves generating the coarse prediction $\hat { y }$ using VFMNet $f _ { \theta }$ on the resized low-resolution image, defined as:

$$
\hat { y } = f _ { \theta } ( \mathrm { r e s i z e } ( x _ { i } ) )
$$

In the second stage, we first divide both the input image $x _ { i }$ and the coarse prediction $\hat { y }$ into overlapped patches $\mathcal { P } = \{ \mathcal { P } _ { 1 } , \mathcal { P } _ { 2 } , . . . , \mathcal { P } _ { k } \}$ , $\hat { y } = \{ \hat { y } _ { 1 } , \hat { y } _ { 2 } , . . . , \hat { y } _ { k } \}$ , using a sliding window approach. Then MGRNet refines the coarse prediction $\hat { y _ { k } }$ for each patch $\mathcal { P } _ { k }$ . However, not all patches need to be refined. We only process the patches with low confidence, as these may contain fine-grained details not captured by VFMNet. The confidence of each patch is calculated based on the softmax probabilities of the segmentation logits.

$$
p _ { k } ^ { ( i , j , c ) } = \frac { \exp ( \hat { y } _ { k } ^ { ( i , j , c ) } ) } { \sum _ { n = 1 } ^ { N } \exp ( \hat { y } _ { k } ^ { ( i , j , n ) } ) }
$$

$$
C _ { k } = \frac { 1 } { h w } \sum _ { i = 1 } ^ { h } \sum _ { j = 1 } ^ { w } \left[ \operatorname* { m a x } _ { c } p _ { k } ^ { ( i , j , c ) } > \theta \right]
$$

![](images/948ecd4a967bc70bc2feb5bb6c87fa1f638b85c12e818240fa445c570727b9ec.jpg)  
Figure 3: The architecture of the proposed VFMNet and MGRNet. (a) VFMNet fine-tunes the ViT encoder of VFM using LoRA and employs a deconvolution-based decoder. (b) To prevent MGRNet from over-relying on high-resolution features, some feature tokens are randomly masked with a learnable token. (c) The decoder of MGRNet uses the feature-to-semantics attention to combine high-resolution features with low-resolution semantic predictions. (d) Details of the feature-to-semantics attention, where image features serve as queries to retrieve semantic information from the coarse mask. Note that MGRNet shares the encoder with VFMNet and fuses multi-scale features using a linear layer, which we have omitted here for brevity.

where p(ki,j,c) defines the softmax probabilities, [·] denotes the Iverson bracket, $\theta$ is a predefined threshold. $C _ { k }$ indicates the certainty, reflecting the proportion of high-confidence pixels within patch $\mathcal { P } _ { k }$ .

Then, the MGRNet $g _ { \phi }$ refines the predictions $\hat { y } _ { k }$ from the VFMNet based on a predefined confidence threshold $C _ { \tau }$ :

$$
\begin{array} { r } { \tilde { y } _ { k } = \left\{ \begin{array} { l l } { \hat { y } _ { k } } & { \mathrm { i f } \ C _ { k } > C _ { \tau } } \\ { g _ { \phi } ( \mathcal { P } _ { k } , \hat { y } _ { k } ) } & { \mathrm { o t h e r w i s e } } \end{array} \right. } \end{array}
$$

# VFM-Based Segmentation Network (VFMNet)

To leverage the VFM for DGSS, we need to adapt the model for the segmentation task with minimal fine-tuning in order to preserve its pre-trained generalizable knowledge. Therefore, we design a simple and effective segmentation network based on encoder-decoder structure.

Encoder. We utilize the encoder of VFM and fine-tune it using the Parameter-Efficient Fine-Tuning (PEFT) method. This approach efficiently adapts the VFMs while preserving their generalizable knowledge. Here we employ Low-Rank Adaptation (LoRA) (Hu et al. 2022) for its effectiveness and negligible impact on inference speed.

Specifically, LoRA freezes the weight matrix W ∈ Rd×k in the original model and introduces extra trainable low-rank matrices $\mathbf { \bar { A } } \in \mathbb { R } ^ { r \times k }$ and $B \in \mathbb { R } ^ { d \times r }$ to update the weight. The updated weight $W ^ { \prime }$ is given by:

$$
\boldsymbol { W } ^ { \prime } = \boldsymbol { W } + \frac { \alpha } { r } \boldsymbol { B } \boldsymbol { A }
$$

where $\alpha$ is a scaling factor that controls the contribution of the original weight and the LoRA weight.

Decoder. Most VFMs produce single-scale and lowresolution features, making them less effective for semantic segmentation. To address this problem, we propose a simple and effective decoder, as shown in Fig. 3 (a). We begin by extracting multi-scale features $F _ { i }$ from different depths of the VFM backbone (at depths of $1 / 4 , 1 / 2 , 3 / 4$ , and full depth). Each feature map, of size $h \times w \times c$ , is then concatenated along the channel dimension and passed through a linear layer to produce the fused features. Next, two transposed convolutions, each with a kernel size of 2 and a stride of 2, are applied to upsample the feature maps. Finally, a 1x1 convolution layer generates the segmentation mask $\hat { y }$ .

# Mask-Guided Refinement Network (MGRNet)

We train the MGRNet on high-resolution image crops to capture fine-grained details. However, the lack of context hinders its ability to recognize large objects. To address this, we feed the low-resolution coarse mask into MGRNet as class priors. The MGRNet then generates refined output based on high-resolution image features and the lowresolution coarse mask.

We first obtain the coarse mask $\hat { y }$ by cropping the lowresolution prediction from the VFMNet. To match the size of the image features, we process the mask through two $2 \mathrm { x } 2$ convolutional layers with a stride of 2 to obtain the mask embedding $Y$ . High-resolution features $F$ are extracted using a LoRA-based VFM encoder and fused with a linear layer. We then apply feature-to-semantics attention to integrate both features, as shown in Fig. 3 (d). This process can be described as follows:

Table 1: Performance comparison with existing domain generalized methods. The results are shown in mIoU, with the best ones highlighted. ’-’ indicates no results reported in the paper or no official code available to reproduce the results. $\dagger$ indicates that we reproduced the results using the official pre-trained checkpoints on the same testing size.   

<html><body><table><tr><td rowspan="2">Backbone</td><td rowspan="2">Method</td><td colspan="4">Trained on GTAV</td><td colspan="3">Trained on Citys</td></tr><tr><td>Citys</td><td>BDD</td><td>Map</td><td>Avg.</td><td>BDD</td><td>Map</td><td>Avg.</td></tr><tr><td rowspan="5">ResNet101</td><td>SAN-SAW (Peng et al. 2022)</td><td>45.33</td><td>41.18</td><td>40.77</td><td>42.43</td><td>54.73</td><td>61.27</td><td>58.00</td></tr><tr><td>WildNet (Lee et al. 2022)</td><td>45.79</td><td>41.73</td><td>47.08</td><td>44.87</td><td>50.94</td><td>58.79</td><td>54.87</td></tr><tr><td>SHADE (Zhao et al. 2022)</td><td>46.66</td><td>43.66</td><td>45.50</td><td>45.27</td><td>50.95</td><td>60.67</td><td>55.81</td></tr><tr><td>TLDR (Kim,Kim,and Kim 2023)</td><td>47.58</td><td>44.88</td><td>48.80</td><td>47.09</td><td>一</td><td>1</td><td>1</td></tr><tr><td>FAMix (Fahes et al.2024)</td><td>49.47</td><td>46.40</td><td>51.97</td><td>49.28</td><td>1</td><td>1</td><td>一</td></tr><tr><td rowspan="2">Swin-L</td><td>HGFormer (Ding et al. 2023)</td><td>一</td><td>1</td><td></td><td>1</td><td>61.50</td><td>72.10</td><td>66.80</td></tr><tr><td>CMFormer (Bi, You,and Gevers 2024)</td><td>55.31</td><td>49.91</td><td>60.09</td><td>55.10</td><td>62.60</td><td>73.60</td><td>68.10</td></tr><tr><td rowspan="3">CLIP-L</td><td>VLTSeg (Himmer et al.2023)</td><td>55.60</td><td>52.70</td><td>59.60</td><td>55.97</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Rein (Wei et al. 2024)</td><td>57.10</td><td>54.70</td><td>60.50</td><td>57.43</td><td></td><td>一</td><td></td></tr><tr><td>Ours</td><td>62.31</td><td>56.09</td><td>66.47</td><td>61.62</td><td>60.62</td><td>73.27</td><td>66.95</td></tr><tr><td rowspan="2">SAM-H</td><td>Rein (Wei et al. 2024)</td><td>59.60</td><td>52.00</td><td>62.10</td><td>57.90</td><td>1</td><td>1</td><td>一</td></tr><tr><td>Ours</td><td>64.05</td><td>55.59</td><td>67.71</td><td>62.45</td><td>59.79</td><td>70.83</td><td>65.31</td></tr><tr><td rowspan="3">EVA02-L</td><td>VLTSeg (Huimmer et al. 2023)</td><td>65.60</td><td>58.40</td><td>66.50</td><td>63.50</td><td>64.40</td><td>76.40</td><td>70.40</td></tr><tr><td>Rein (Wei et al. 2024)</td><td>65.30</td><td>60.50</td><td>64.90</td><td>63.57</td><td>64.10</td><td>69.50</td><td>66.80</td></tr><tr><td>Ours</td><td>69.53</td><td>61.14</td><td>69.97</td><td>66.88</td><td>64.70</td><td>76.43</td><td>70.56</td></tr><tr><td rowspan="2">DINOv2-L</td><td>Reint (Wei et al. 2024)</td><td>69.20</td><td>60.65</td><td>70.16</td><td>66.67</td><td>65.00</td><td>76.09</td><td>70.54</td></tr><tr><td>Ours</td><td>73.87</td><td>62.91</td><td>73.52</td><td>70.10</td><td>66.16</td><td>77.08</td><td>71.62</td></tr></table></body></html>

$$
Q = \operatorname { L i n e a r } ( F ) , K = \operatorname { L i n e a r } ( Y ) , V = \operatorname { L i n e a r } ( Y )
$$

$$
W = \mathrm { s o f t m a x } \left( { \frac { Q \cdot K ^ { T } } { \sqrt { d } } } \right)
$$

Here, three different linear layers are used to project $F$ and $Y$ into the same dimension. The matrix $W$ reflects the similarity between the image features and the coarse mask. We then fuse the high-resolution features $F$ and the lowresolution mask $Y$ as follows:

$$
F ^ { \prime } = F + W \cdot V
$$

The feature-to-semantics attention allows the model to extract valuable information from the low-resolution coarse mask, thereby refining the high-resolution features.

# Feature Masking

During the training process, we observed that the MGRNet might exhibit an overreliance on fine-grained highresolution features, potentially neglecting the coarse mask. To mitigate this bias and enhance performance, we implemented a feature masking strategy. Specifically, we randomly mask a portion of feature tokens and replacing them with a learnable token.

Let $F \in \mathbb { R } ^ { H \times W \times D }$ denote the high-resolution features and $T \in \mathbb { R } ^ { 1 \times D }$ denote the learnable masking token. We generate a random binary mask M ∈ {0, 1}H×W as follows:

$$
M = [ v > p ] \quad { \mathrm { w h e r e ~ } } v \sim U ( 0 , 1 )
$$

Here, [·] denotes the Iverson bracket, and $U ( 0 , 1 )$ is the uniform distribution between 0 and 1, $p$ is the masking ratio. The masked feature map $\hat { F }$ is then computed by:

$$
\hat { F } = F \odot M + T \odot ( 1 - M )
$$

# Experiments

# Experimental Setups

Datasets. Following previous studies (Wei et al. 2024), we evaluate our method on both synthetic and real-world datasets. The synthetic dataset is GTAV (Richter et al. 2016), which contains 24,966 street-view images rendered by a computer game engine with the resolution of $1 9 1 4 \mathrm { x } 1 0 5 2$ . For real-world datasets, we use Cityscapes (Cordts et al. 2016), a large-scale semantic segmentation dataset for autonomous driving, with 2,975 training images and 500 validation images, all with a resolution of $2 0 4 8 \times 1 0 2 4$ . BDD100K (Yu et al. 2020) is another realworld dataset that contains diverse urban driving scene images with the resolution of $1 2 8 0 \times 7 2 0$ . The last real-world dataset we use is Mapillary (Neuhold et al. 2017), which consists of highresolution images with a minimum resolution of $1 9 2 0 \times 1 0 8 0$ collected from around the world. BDD100K and Mapillary provide 1000 and 2000 validation images, respectively. For brevity, we refer to Cityscapes, BDD100K, and Mapillary as Citys, BDD, and Map, respectively

Visual Foundation Models. We conduct experiments using CLIP (Radford et al. 2021), EVA02 (Fang et al. 2024), SAM (Kirillov et al. 2023), and DINOv2 (Oquab et al. 2023) as backbones to evaluate the effectiveness of our method. Following previous approaches (Wei et al. 2024), we employ the ViT-L architecture for all models except SAM, which uses the ViT-H architecture.

Table 2: Ablation study of different components. Components are sequentially incorporated to show their impact. The baseline model uses a frozen DINOv2 backbone with a linear decoder and is trained on GTAV.   

<html><body><table><tr><td>Config</td><td>Citys</td><td>BDD</td><td>Map</td><td>Avg.</td></tr><tr><td>Baseline</td><td>59.80</td><td>54.83</td><td>61.57</td><td>58.73</td></tr><tr><td>+ VFMNet</td><td>72.16</td><td>62.98</td><td>71.88</td><td>69.00</td></tr><tr><td>+ MGRNet</td><td>72.94</td><td>62.56</td><td>72.56</td><td>69.35</td></tr><tr><td>+ Feat.Mask</td><td>73.87</td><td>62.91</td><td>73.52</td><td>70.10</td></tr></table></body></html>

Implementation Details. Our implementation is based on the MMSegmentation framework. We use the AdamW optimizer with learning rates of 1e-5 for the backbone and 1e-4 for all decode heads. Training is conducted for 40,000 iterations with a batch size of 2 and crop size of 512x512. We employ basic data augmentation techniques including random cropping, random horizontal flipping, photo-metric transformation and rare class sampling (Hoyer, Dai, and Van Gool 2022). During training, we set $\lambda = 1 . 0$ , $r = \alpha = 3 2$ , and $p \ = \ 0 . 2$ . During inference, we use a sliding window approach with a window size of $5 1 2 \mathrm { x } 5 1 2$ and a stride of 320. The $\theta$ and $C _ { \tau }$ are set to 0.968 and 0.8 respectively.

# Comparison with State-of-the-Art Methods

Compared Methods. We compare our method with several DGSS methods: SAN-SAW (Peng et al. 2022), WildNet (Lee et al. 2022), SHADE (Zhao et al. 2022), TLDR (Kim, Kim, and Kim 2023), FAMix (Fahes et al. 2024), HGFormer (Ding et al. 2023), CMFormer (Bi, You, and Gevers 2024), VLTSeg (H¨ummer et al. 2023), and Rein (Wei et al. 2024).

Main Results. We compare our method with existing methods in two generalization settings: $\mathrm { G T A } \to \mathrm { C i t y s } \ +$ $\mathbf { B D D + M a p }$ , and Citys $ \mathrm { B D D + M a p }$ , as shown in Table 1. Our method with DINOv2 backbone outperforms existing approaches in both synthetic-to-real (trained on GTAV) and real-to-real (trained on Citys) generalization, achieving average mIoU improvements of $3 . 4 \%$ and $1 . 1 \%$ over the state-ofthe-art, respectively. This demonstrates the effectiveness of our approach. We find that our method shows a more significant improvement in synthetic-to-real generalization compared to real-to-real generalization. We hypothesize that this is due to the larger domain gap between synthetic and real data, which requires more prior knowledge for effective generalization. It also shows that our approach can fully leverage the capabilities of VFM to bridge this gap effectively.

Comparison with Various VFM Backbones. As shown in Table 1, our method can effectively integrate with different VFMs and consistently outperforms Rein, demonstrating its effectiveness. Notably, using DINOv2 and Eva 02 as the backbone yields the better results, highlighting its strong generalization capabilities. Howerev, the results of CLIP and SAM are relatively poor. This may be due to the fact that

![](images/b54c491e919d8a2c59a3cd07cb2efc5ff9baa2c08c57aab1728c212db10c47f8.jpg)  
Figure 4: The average mIoU comparison under $\mathrm { \bf G T A V  }$ Citys $+ \mathrm { B D D + M a p }$ generalization at different test resolutions . The short edge of the input image is scaled to 512, 1024, 1536, 2048, respectively.

Table 3: Ablation study of different decoders for VFMNet. Models are trained on GTAV with the DINOv2 backbone.   

<html><body><table><tr><td>Decoder</td><td>Citys</td><td>BDD</td><td>Map</td><td>Avg.</td></tr><tr><td>Linear</td><td>71.17</td><td>61.75</td><td>70.89</td><td>67.94</td></tr><tr><td>SegFormer</td><td>70.72</td><td>62.24</td><td>71.16</td><td>68.04</td></tr><tr><td>Mask2Former</td><td>71.22</td><td>60.12</td><td>71.59</td><td>67.64</td></tr><tr><td>Ours</td><td>72.16</td><td>62.98</td><td>71.88</td><td>69.00</td></tr></table></body></html>

CLIP tends to extract features with richer semantics, which leads to the neglect of details. On the other hand, SAM itself is a backbone designed specifically for segmentation, which may focus too much on details, potentially hindering the generalization ability of the model.

Inference Under Different Resolutions. We conducted inference at various resolutions by scaling the short edge of images to 512, 1024, 1536, and 2048, comparing the performance of different methods. As shown in Fig 4, our method consistently achieved the best performance across all resolutions. For most methods, increasing resolution initially improves performance, but later leads to degradation. This partly due to a mismatch between training and inference resolutions, preventing effective generalization to higher resolutions. In contrast, the performance of our method improves with resolution increases, with the highest resolution achieving a $4 . 7 \%$ improvement compared to the lowest resolution. This demonstrates the effectiveness of our model in adapting to increasing image resolutions.

# Ablation Studies

Analysis of the Key Components. The proposed method integrates three key components: VFMNet, MGRNet, and a feature masking strategy. In our ablation study, our baseline is a frozen DINOv2 encoder with a linear head. We then incorporated different components sequentially to show their impact. The results are shown in Table 2. First, the frozen DINOv2 only achieves $5 8 . 7 \%$ mIoU, highlighting the limitations of using pre-trained models without adaptation. Then, it is observed that our VFMNet allows effective adaptation for semantic segmentation tasks, improving performance by $1 0 . 3 \%$ compared to the baseline. Furthermore, MGRNet boosts the performance of VFMs by $0 . 3 5 \%$ , demonstrating the effectiveness of combining highresolution features with low-resolution predictions. Additionally, the introduction of the feature masking strategy led to a further performance increase of $0 . 7 5 \%$ , highlighting its critical role.

![](images/454adf3609cbcdda29628fad7e424dfb31c5c69b60c1579ae30131ff44f4e491.jpg)  
Figure 5: Qualitative Comparison under GT $\mathrm { 4 V \to C i t y s + \bar { \Omega } }$ BDD $^ +$ Map generalization setting.

Comparison of Different Decoders in VFMNet. We evaluated several decoder designs for VFMNet, with the results shown in Table. 3. Notably, even a simple linear head achieved $67 . 9 \%$ mIoU after fine-tuning the backbone with LoRA. Interestingly, more complex decoder heads do not necessarily improve performance, e.g., Mask2Former yielded similar results to the linear head. Our decoder uses deconvolution to upsample image features, creating highresolution features while preserving the pre-trained knowledge of VFMs. These experimental results validate the effectiveness of our decoder design.

# Sensitivity to Hyper-parameters

Impact of Mask Ratio and Detail Loss. As shown in Table. 4, we investigated the impact of mask ratio and detail loss weights on performance. The optimal result was achieved with a detail loss of 1.0, indicating that detail loss and content loss are equally important. For the mask ratio, a value of 0.2 yields the best performance. Higher mask ratios may reduce information from high-resolution images, causing the model to rely too much on low-resolution predictions, while lower mask ratios can excessively focus on high-resolution features, neglecting class priors from lowresolution predictions. This balance highlights the need to integrate information from both high and low resolutions for

Table 4: Ablation study on mask ratio and detail loss weight.   

<html><body><table><tr><td colspan="6">(a) Choice of mask ratio p</td></tr><tr><td>p</td><td>0</td><td>0.1</td><td>0.2</td><td>0.3</td><td>0.4</td></tr><tr><td>Avg. mIoU</td><td>69.35</td><td>69.56</td><td>70.10</td><td>69.87</td><td>69.33</td></tr><tr><td colspan="6">(b) Choice of detail loss weight λ</td></tr><tr><td>入</td><td>0.2</td><td>0.5</td><td>1.0</td><td>1.5</td><td>2.0</td></tr><tr><td>Avg. mIoU</td><td>69.27</td><td>69.61</td><td>70.10</td><td>69.63</td><td>69.48</td></tr></table></body></html>

optimal performance.

# Quantitative Results

As shown in Fig 5, we present a visual comparison of the segmentation results on the $\mathrm { G T A V } \to \{ \mathrm { C i t y s + B D D + M a p } \}$ generalization setting. It can be observed that our approach exhibits a strong ability to identify small objects in the distance. This clearly demonstrates that our method can fully leverage high resolution, as well as the importance of highresolution images for accurate semantic segmentation.

# Conclusion

In this work, we explored the benefits of leveraging VFMs for generalizable semantic segmentation. We first proposed a simple yet effective network, VFMNet, to adapt VFMs to this task while preserving their generalizable knowledge. Then, we designed MGRNet to capture details guided by the prediction of VFMNet. Additionally, we proposed a two-stage inference method to enhance inference on highresolution images. Extensive experiments demonstrate the superiority of our method over the state-of-the-art on various benchmarks. In the future, we will apply this method to other non-foundation models and further explore more efficient ways to perform inference on high-resolution images.

# Acknowledgments

This work was supported by the National Natural Science Foundation of China (NSFC) under grants No. 62232008 and No. 62032010, National Natural Science Foundation of China (Youth Program) under Grant No. 62402367, Fundamental Research Funds for the Central Universities under Grant No.20101247556.
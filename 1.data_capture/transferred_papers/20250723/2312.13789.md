# TinySAM: Pushing the Envelope for Efficient Segment Anything Model

Han Shu1,2, Wenshuo $\mathbf { L i } ^ { 2 }$ , Yehui Tang2, Yiman Zhang2, Yihao Chen2, Houqiang $\mathbf { L i } ^ { 1 }$ , Yunhe $\mathbf { W a n g ^ { 2 * } }$ , Xinghao Chen2\*

1University of Science and Technology of China 2Huawei Noah’s Ark Lab {han.shu, xinghao.chen, yunhe.wang}@huawei.com

# Abstract

Recently segment anything model (SAM) has shown powerful segmentation capability and has drawn great attention in computer vision fields. Massive following works have developed various applications based on the pre-trained SAM and achieved impressive performance on downstream vision tasks. However, SAM consists of heavy architectures and requires massive computational capacity, which hinders the further application of SAM on computation constrained edge devices. To this end, in this paper we propose a framework to obtain a tiny segment anything model (TinySAM) while maintaining the strong zero-shot performance. We first propose a full-stage knowledge distillation method with hard prompt sampling and hard mask weighting strategy to distill a lightweight student model. We also adapt the post-training quantization to the prompt-based segmentation task and further reduce the computational cost. Moreover, a hierarchical segmenting everything strategy is proposed to accelerate the everything inference by $2 \times$ with almost no performance degradation. With all these proposed methods, our TinySAM leads to orders of magnitude computational reduction and pushes the envelope for efficient segment anything task. Extensive experiments on various zero-shot transfer tasks demonstrate the significantly advantageous performance of our TinySAM against counterpart methods.

# Code — https://github.com/xinghaochen/TinySAM

# Introduction

Object segmentation is an important and foundational task in computer vision fields. Extensive visual applications such as object localization and verification rely on accurate and fast object segmentation. Tremendous prior works have focused on segmentation tasks which include semantic segmentation (Long, Shelhamer, and Darrell 2015; Strudel et al. 2021), instance segmentation (Bolya et al. 2019; Liu et al. 2018) and panoptic segmentation (Cheng et al. 2022; Kirillov et al. 2019). Recently, Kirillov et al. (Kirillov et al. 2023) introduce a powerful segment anything model (SAM), together with a massive segmentation dataset SA-1B that contains over 1 billion masks on 11 million images. With the strong capability to segment objects with arbitrary shapes and categories, SAM has become a foundation framework for many downstream tasks such as object tracking (Cheng et al. 2023), image inpainting (Yu et al. 2023) and 3D vision (Cen et al. 2023) etc. Moreover, the powerful zero-shot segmentation ability of SAM has benefited research area with less data like medical imaging (Ma and Wang 2023).

Although SAM has achieved impressive performance on downstream vision tasks, complicated architecture and huge computational cost make SAM difficult to be deployed on resource constrained devices. The inference time of SAM model for a $1 0 2 4 \times 1 0 2 4$ image could take up to 2 seconds on a modern GPU (Zhao et al. 2023). Some recent attempts have tried to obtain a more computation efficient segment anything model. For example, MobileSAM (Zhang et al. 2023) tries to replace the heavy component of image encoder with a lightweight architecture of TinyViT (Wu et al. 2022). However, it only accesses the image encoder network with a decoupled knowledge distillation strategy by training the compact image encoder network with the supervision of image embeddings from the teacher network. This partial training strategy inevitably causes performance decay without the supervision of final mask prediction. FastSAM (Zhao et al. 2023) transfers the segment anything task to an instance segmentation task with only one foreground category with YOLOv8 (Jocher, Chaurasia, and Qiu 2023). To fulfill the function of prompt-based segmentation, FastSAM applies a post-process strategy together with the instance segmentation network. However, this reformulated framework could not achieve comparable performance as SAM on downstream zero-shot tasks.

To further push the envelope for efficient segment anything model, in this paper we propose a full framework to obtain TinySAM that greatly reduces the computational cost while maintaining the zero-shot segmentation ability to maximum extent. Specifically, we propose a hard mining full-stage knowledge distillation method to improve the capability of the compact student network. The student network is distilled in an end-to-end manner with the supervision of teacher network from different network stages. A mask-weighted distillation loss is proposed to efficiently transfer the information from teacher to student through massive various SA-1B masks. Besides, an online hard prompt sampling strategy is proposed to make the distillation process attend more to hard examples and thus improves the final performance. We also adapt the post-training quantization to the prompt-based segmentation task and further reduce the computational cost. Moreover, we find that it takes tremendous computational cost for segmenting everything in an image since massive masks have to be generated from grid prompt points. To this end, a hierarchical segmenting everything strategy is proposed to accelerate the everything inference by $2 \times$ with almost no performance degradation. With all these proposed methods, our TinySAM leads to orders of magnitude computational reduction and pushes the envelope for efficient segment anything task. For example, TinySAM can achieve $1 0 0 \times$ acceleration for segment anything task compared with the original SAM. Extensive experiments on various zero-shot transfer tasks demonstrate the significantly advantageous performance of our TinySAM against counterparts.

![](images/8eab402c971117fe0f11e48d8526f762b1855b8d23b20e3ef3a0836577bce8e1.jpg)  
Figure 1: (a) The overall framework of our proposed method. Consisting the modules of the hard mining full-stage knowledge distillation, the post training quantization and the hierarchical everything inference, the computation cost is down-scaled by magnitudes. (b) The proposed TinySAM can save considerable computation cost while maintaining the performance. The latency is tested with TensorRT on NVIDIA T4 GPU.

# Related Work

# Segment Anything Model

Recently proposed segment anything model (SAM) (Kirillov et al. 2023) proves its generalization on object segmentation and downstream vision tasks. SAM consists of three subnetworks, i.e., image encoder, prompt encoder and mask decoder. The image encoder is a heavy vision transformerbased network (Dosovitskiy et al. 2020), which extracts the input image into image embedding. The prompt encoder is designed to encode input points, boxes, arbitrary-shaped masks and free-form text with positional information. The geometric prompt and text prompt are processed with different networks. The mask decoder, which contains a twoway transformer, takes the output of image encoder and prompt encoder to generate the final mask prediction. Together with the proposed SA-1B dataset, which contains 11 million high-resolution images and more than 1 billion high-quality segmentation masks, SAM shows impressive high quality segmentation ability for objects of any category and shape. Moreover, SAM demonstrates powerful generalization on zero-shot downstream vision tasks including edge detection, object proposal, instance segmentation and text-to-mask prediction. Due to the flexible prompt mode and high quality segmentation capability, SAM has been regarded as a foundation model for vision applications. However, SAM, especially the image encoder network, consists of large parameters and requires high computation capacity for deployment. Therefore, it is not easy to apply SAM on edge devices with constrained resources. The compression and acceleration of SAM becomes an important research topic (Zhao et al. 2023; Zhang et al. 2023; Chen et al. 2024).

# Knowledge Distillation

Hinton et al. (Hinton et al. 2015) propose the knowledge distillation method to supervise the training of lightweight student network via the output of teacher network. Since then knowledge distillation has been an important approach to improve the performance of compact networks during training process. Knowledge distillation methods can be roughly divided into two categories, i.e. distillation for network outputs (Hinton et al. 2015) and for intermediate features (Romero et al. 2014). Majority of research of knowledge distillation methods have focused on image classification task (Park et al. 2019; Peng et al. 2019; Dong et al. 2023; Li et al. 2022b). Subsequent works (Chen et al. 2017; Liu et al. 2019; Guo et al. 2021; Chen et al. 2020; Deng, Kong, and Murakami 2019) propose knowledge distillation methods for high-level computer vision tasks such as object detection and semantic segmentation. Zhang et al. (Zhang et al. 2023) propose to use the distillation method to obtain an efficient segment anything model (MobileSAM). However, MobileSAM only accesses the image encoder network with the supervision of corresponding image embeddings from original SAM. This partial distillation strategy could cause considerable performance decay since there is no guidance of mask-level information for lightweight student network from either teacher network or labeled data.

# Quantization

Model quantization is also one of the commonly used model compression methods, which quantizes weights or activations from higher bit-width to lower bit-width to reduce both storage requirements and computational complexity with limited accuracy degradation. There are two types of model quantization methods, quantization-aware training (QAT) (Choi et al. 2018; Esser et al. 2019) and post-training quantization (PTQ) (Choukroun et al. 2019). QAT methods require a labeled training dataset and extensive training cost, while PTQ methods only need a small unlabeled calibration dataset and thus are more efficient. Many prior PTQ methods (Liu et al. 2023; Nagel et al. 2020) have proposed to search for appropriate quantization parameters for convolutional neural networks. As vision transformers (Dosovitskiy et al. 2020; Liu et al. 2021a) achieve remarkable performance on various visual tasks, recent works (Liu et al. 2021b; Yuan et al. 2022; Tai, Lin, and Wu 2023; Li et al. 2022c) investigate how to apply post-training quantization for vision transformers and have achieved good performance with 8-bit quantization configuration. However, there is rare exploration for quantization of prompt-based segmentation task, especially for segment anything models.

# Methodology Overview of TinySAM

This paper proposes a framework to get a highly efficient SAM, as described in Figure 1. Firstly, we introduce a hard mining full-stage knowledge distillation specifically designed for SAM. To further activate the distillation process, the proposed hard mask weighting and hard prompt sampling strategy are utilized to mine the essential knowledge from the teacher network to the student network. Secondly, a post-training quantization method is adapted to promptbased segmentation task and applied to the lightweight student network. Thirdly, a hierarchical everything inference mode is designed for segmenting everything task, which can avoid massive redundant computation only with negligible accuracy loss and speedup the inference time by $2 \times$ .

# Hard Mining Full-Stage Knowledge Distillation

SAM consists of three subnetworks, i.e. image encoder, prompt encoder and mask decoder. The image encoder network is based on vision transformer (Dosovitskiy et al. 2020) and consumes great computation cost. Inspired by MobileSAM (Zhang et al. 2023), we use the lightweight TinyViT (Wu et al. 2022) to replace the original heavy image encoder network. Considerable performance decay exists for this simple substitution. Therefore, we propose a hard mining full-stage knowledge distillation strategy to guide the lightweight image encoder during learning procedure from multiple knowledge levels.

Besides the conventional loss between the predicted results and ground-truth labels, we introduce multiple distillation losses on different stages as described in Figure 2. Specifically, we select several nodes of teacher network to guide the learning of student network from multiple level of knowledge. Firstly, we choose the output feature of image

𝐸𝑖𝑇𝑚𝑔 𝒯 𝑇 DT 𝑚𝑎𝑠𝑘 𝑀𝑇 ℒ𝑒𝑚𝑏𝑒𝑑𝑑𝑖𝑛𝑔 ℒ𝑡𝑜𝑘𝑒𝑛 ℒ𝑜𝑢𝑡𝑝𝑢𝑡 ℋ𝑖 𝑀𝑖𝑆 𝑖S𝑚 𝒯 𝑆 𝐷𝑆𝑚𝑎𝑠𝑘 GT Iter. 0 Iter. 1 Iter. t 扁电曲 ROSEDEE

encoder, i.e. image embedding, as a distillation information. Image embedding concentrates the information from input image, which is the fundamental knowledge during the prediction. For an input image of $I$ , the distillation loss function for image embedding can be expressed as,

$$
\mathcal { L } _ { e m b e d d i n g } = \mathcal { L } ( E _ { i m g } ^ { T } ( I ) , E _ { i m g } ^ { S } ( I ) ) ,
$$

where $E _ { i m g } ^ { S }$ and $E _ { i m g } ^ { T }$ denote the image encoder for student and teacher network, respectively. Since image level information does not directly relate to the mask prediction, features more close to the final output are essential for this segmentation task. Naturally, the final output of the teacher network is chosen to be a distillation point. The output distillation loss $\mathcal { L } _ { o u t p u t }$ can be described as,

$$
\mathcal { L } _ { o u t p u t } = \mathcal { L } ( D _ { m a s k } ^ { T } ( E _ { i m g } ^ { T } ( I ) , q ) , D _ { m a s k } ^ { S } ( E _ { i m g } ^ { S } ( I ) , q ) ) ,
$$

where DS $D _ { m a s k } ^ { S }$ mask and D Tmask are mask decoders for student and teacher, respectively. $q$ denotes the query of the mask decoder, which is the concatenation of prompt embedding and output tokens. Since the structure of SAM is rather complicated, the previously mentioned two distillation losses could be inconsistent and thus hard for lightweight student to learn. We further propose to distill the output tokens from the two-way transformer of the mask decoder, which interacts information from prompt embedding and image embedding. It captures the target mask information in a more abstract way. The corresponding distillation losses $\mathcal { L } _ { t o k e n }$ can be described as,

$$
\mathcal { L } _ { t o k e n } = \mathcal { L } ( \mathcal { T } ^ { T } ( E _ { i m g } ^ { T } ( I ) , q ) , \mathcal { T } ^ { S } ( E _ { i m g } ^ { S } ( I ) , q ) ) ,
$$

where $\mathcal { T } ^ { S }$ and $\mathcal { T } ^ { T }$ are the two-way transformer module of mask decoder and $\mathcal { L }$ denotes the loss function. We empirically find that the numerical values of feature difference could make the conventionally used MSE loss $\lfloor \ell _ { 2 }$ distance) too small to be well optimized. Thus we use $\ell _ { 1 }$ distance function instead. The overall distillation loss function

$\mathcal { L } _ { d i s t i l l }$ can be expressed as,

$\mathcal { L } _ { d i s t i l l } = \alpha * \mathcal { L } _ { e m b e d d i n g } + \beta * \mathcal { L } _ { t o k e n } + \gamma * \mathcal { L } _ { o u t p u t } ,$ (4) where $\alpha , \beta , \gamma$ represent the hyper-parameters for each distillation loss. The total training loss is a linear combination of distillation loss, ground truth loss for mask prediction $\mathcal { L } _ { m a s k }$ and IoU prediction $\mathcal { L } _ { i o u s }$ , where $\mathcal { L } _ { m a s k }$ is a combination of focal loss (Lin et al. 2017) and dice loss (Milletari, Navab, and Ahmadi 2016), $\mathcal { L } _ { i o u s }$ is $\ell _ { 1 }$ loss function between predicted IoUs and calculated IoUs.

$$
\mathcal { L } _ { t o t a l } = \mathcal { L } _ { d i s t i l l } + \mathcal { L } _ { m a s k } + \mathcal { L } _ { i o u s } .
$$

Hard Mask Weighting. To make the knowledge distillation more effective, we design a hard mask weighting strategy when calculating the losses. There is an observation that masks could be extremely various in a single image of SA1B dataset since the fine-grained granularity and no semantic constraints. As shown in Figure 2, segmenting the flag with complex boundary could be difficult while segmenting the rectangular window with high contrast color could be easy. The hard mask should reasonably be assigned with larger weight for student to learn. Specifically, we calculate the gap of student and teacher network output to indicate the mask hardness $\mathcal { H } _ { i }$ .

$$
\mathcal { H } _ { i } = \mathrm { s i g m o i d } ( \frac { \mathrm { I o U } ( M _ { i } ^ { T } , M _ { i } ^ { G T } ) } { \mathrm { I o U } ( M _ { i } ^ { S } , M _ { i } ^ { G T } ) + \epsilon } - 1 ) ,
$$

where $M _ { i } ^ { T } , M _ { i } ^ { S } , M _ { i } ^ { G T }$ represent the mask prediction of student network, the mask prediction of teacher network and the ground truth for $i$ th mask, respectively. Thus the distillation loss could be updated with

$$
\mathcal { L } _ { d i s t i l l } ^ { * } = \alpha * \mathcal { L } _ { e m b e d d i n g } + \beta * \mathcal { L } _ { t o k e n } + \gamma * \sum _ { i = 1 } ^ { N } \mathcal { H } _ { i } * \mathcal { L } _ { o u t p u t } ^ { i } .
$$

Hard Prompt Sampling. Generally, random sampling from labeled training data could be adopted to generate the prompts to drive the end-to-end training of prompt-based mask prediction network as SAM. To further ease the learning process of the distillation between teacher and lightweight student network, we propose a hard prompt sampling strategy, which makes the training samples concentrate in the difficult area for prediction. Taking points prompt as an example, points $P _ { 0 }$ are initially sampled inside the labeled mask region $M _ { g t }$ . These initial points are fed into the network with input image to get the predicted mask region $M _ { 0 }$ . Then we sample the prompt points from the difference set of $M _ { g t }$ and $M _ { 0 }$ , and we conduct the procedure iteratively. The $( i + 1 )$ -th round sampling points $P _ { i }$ are sampled from the difference set of $M _ { g t }$ and $M _ { i }$ , i.e.

$$
P _ { i + 1 } \in M _ { g t } - M _ { i } , i = 0 , 1 , 2 , \ldots
$$

where

$$
M _ { i } = D _ { m a s k } \big ( E _ { p r o m p t } ( P _ { i } ) , E _ { i m g } ( I ) \big ) .
$$

When applied on the training process, the $i$ -th iteration is random sampled from 0 to 9, which makes the difficulty of sampled prompts in a constrained range. The bottom of Figure 2 shows the location change of the sampling prompts with iterations, the green stars denote the sampled point prompts with online hard prompt sampling strategy. With more iterations, the sampling points are more close to the edge region of the ground truth mask.

# Quantization

Quantization aims to project floating point tensor $x$ to $b$ -bit integer tensor $x _ { q }$ with a scaling factor $s$ . The uniform symmetric quantization could be formulated as follows,

$$
x _ { q } = Q ( b , s ) = \mathrm { c l i p } ( \mathrm { r o u n d } ( \frac { x } { s } ) , - 2 ^ { b - 1 } , 2 ^ { b - 1 } - 1 ) .
$$

For a matrix multiplication $O \ = \ A B$ , it can be quantized with two scaling factors $s _ { A }$ and $s _ { B }$ , and the quantized matrix is denoted as $\hat { O } = \hat { A } \hat { B }$ . The metric for measuring the distance between $\hat { O }$ and $O$ is vitally important for optimizing $\hat { A }$ and $\hat { B }$ . Following the successful practice of quantization methods in image classification models (Tai, Lin, and Wu 2023; Yuan et al. 2022; Frantar et al. 2022; Wu et al. 2020), we perform hessian guided metric as the distance to solve the scaling factors, which is more consistent with task loss. Different from classification tasks, the promptbased segmentation task of SAM outputs segmentation predictions which contains fine-grained masks. Thus we use the Kullback-Leible (KL) divergence of masks and IoUs as the task loss and use some calibration data to calculate the hessian matrix, the task loss is formulated as,

$$
L = \mathrm { K L } ( \hat { y } _ { p r e d } , y _ { p r e d } ) + \mathrm { K L } ( \hat { y } _ { i o u } , y _ { i o u } ) ,
$$

where $y _ { p r e d }$ and $y _ { i o u }$ are the outputs of the floating point model, $\hat { y } _ { p r e d }$ and $\hat { y } _ { i o u }$ are the outputs after quantization.

After specifying the distance metric, we could solve $s _ { A }$ and $s _ { B }$ as an alternate iterative grid search problem. With calibration data we get the maximum value of $A$ and $B$ , which is $A _ { m a x }$ and $B _ { m a x }$ respectively, and use two parameters $\theta _ { l }$ and $\theta _ { u }$ to specify the search range for $s _ { A }$ and $s _ { B }$ , $\begin{array} { r } { [ \theta _ { l } \frac { A _ { m a x } } { 2 ^ { b - 1 } } , \theta _ { u } \frac { A _ { m a x } } { 2 ^ { b - 1 } } ] } \end{array}$ and $\begin{array} { r } { \big [ \dot { \theta _ { l } } \frac { B _ { m a x } } { 2 ^ { b - 1 } } , \theta _ { u } \frac { B _ { m a x } } { 2 ^ { b - 1 } } \big ] } \end{array}$ . These two search ranges are linearly divided into $n$ candidate options separately. $\hat { A }$ and $\hat { B }$ are optimized in an alternate manner.

The input of matrix multiplication after softmax is unevenly distributed at both ends of the interval [0,1], while the feature after GELU varies greatly between the positive and negative ranges. These two circumstances go far from the assumption of uniform quantization, i.e., the activation in neural networks obeys Gaussian distribution. The violation will result in high quantization error. Thus we split feature into two groups and use two scaling factors to reduce the quantization error.

# Hierarchical Segmenting Everything

SAM proposes an automatic mask generator which samples points as a grid to segment everything. However, we find that dense point grid leads to over fine-grained segmentation results and also occupies massive computing resources. On the one hand, for a complete object, too many sampling points may cause slightly different parts of the object to be incorrectly segmented as separate masks. On the other hand, since the image encoder has been largely shrunk by the proposed method, the time cost of everything mode inference is mainly in the mask decoder part. For the default setting of SAM automatic mask generator, it samples $3 2 \times 3 2 = 1 0 2 4$ points as the prompts, which means the mask decoder is inferred by 1024 times. It costs 16ms for image encoder and $8 9 4 \mathrm { { m s } }$ for mask decoder on a single V100 GPU.

![](images/cbd11e6923098446e3a05bb616ac8f4e84568298ceccc424bcce4ce49b247db3.jpg)  
Figure 3: Comparison between our hierarchical strategy and the original strategy. (a) Points sampling (take points per side $\scriptstyle \cdot = I 6$ as an example) of original everything mode. (b) Segmentation results of original strategy. (c) First step of our hierarchical strategy, only $1 / 1 6$ points are sampled. (d) Get high confidence area from (c) and ignore points in this area. The high confidence area is shown as white mask. (e) Segmentation results of our hierarchical strategy.

To reduce the time cost of everything mode, we propose a hierarchical mask generating method. The comparison between our hierarchical strategy and the original one is shown in Figure 3. Different from original everything mode, in the first step we only use $1 / 4$ points in each side so the total points is $1 / 1 6$ of the original settings, as shown in Figure 3(c). Then we infer the prompt encoder and mask decoder with these prompts and get the results.

Then we filter out some masks with confidence exceeding a threshold $\tau$ , and mark the corresponding regions as areas that could be considered as final predictions. For these areas, since they are considered as the segmentation results of instances with high confidences, there is no need to regenerate point prompts. Thus we sample points as the same density with original setting but ignore points in the above area. As shown in Figure 3(d), most points on the grass and body of the front cow are ignored. Meanwhile, the points on the back cow and the sky are kept to be further segmented. Specifically, the back cow is incorrectly segmented as the same object with the front cow in the initial round. This strategy can avoid redundant cost of inference time and over finegrained segmentation of the object. Then we utilize the point prompts sampled in the second round to get the mask predictions. Finally, the results of these two round are merged and post-processed to get the final masks. More than $5 0 \%$ points are ignored by our method thus brings in significant latency reduction.

# Experiments Implementation Details

We utilize the TinyViT-5M (Wu et al. 2022) as the lightweight student image encoder and SAM-H as the teacher model, following prior work (Zhang et al. 2023). $1 \%$ of SA-1B dataset is used as the training data for fullstage distillation. We adopt Adam optimizer and train the student network for 8 epochs. For each iteration, we sample 64 prompts according to hard prompt sampling strategy. To accelerate the distillation process, the image embeddings from the teacher network have been computed and stored in advance. Therefore, the heavy image encoder of teacher network is not necessary to compute repeatedly during training time. For post training quantization, we set $\theta _ { l } = \mathrm { \bar { 0 . 0 1 } } , \theta _ { u } = 1 . 2 , n = 1 0 0 , r o u \mathrm { \bar { } { } } i \mathrm { \bar { } s } = 3$ for iterative search. We calibrate quantized model on SA-1B dataset using 8 images. We conduct zero-shot evaluation on downstream tasks like instance segmentation and point prompt segmentation. Following the suggestions by SAM (Kirillov et al. 2023), the multi-output mode is adopted and the final mask prediction is the one with highest IoU prediction.

# Zero-Shot Instance Segmentation

For zero-shot instance segmentation task, we strictly follow the experimental settings of SAM and use the object detection results of ViTDet-H (Li et al. 2022a) as the box prompt for instance segmentation. We evaluate the zero-shot instance segmentation task for models on the benchmark of COCO (Lin et al. 2014) dataset and LVIS v1 (Gupta, Dollar, and Girshick 2019). We compare our TinySAM with different variants of SAM (Kirillov et al. 2023), and also with prior efficient models like FastSAM (Zhao et al. 2023), MobileSAM (Zhang et al. 2023), EfficientSAM (Xiong et al. 2024) and SlimSAM (Chen et al. 2024). As shown in Table 1, the proposed TinySAM obtained superior performance when compared with prior methods. Specifically, our TinySAM outperforms FastSAM (Zhao et al. 2023) in terms of MACs and instance segmentation accuracy, i.e., about $4 \%$ AP improvement with only $9 . 5 \%$ MACs and $2 5 \%$ latency. With the same computational cost, our TinySAM also achieves $1 . 3 \% +$ AP on COCO dataset than MobileSAM (Zhang et al. 2023) and $1 . 9 \% +$ AP on LVIS v1 dataset, respectively. With similar performance on COCO dataset, TinySAM is $2 \times$ faster than EfficientSAM (Xiong et al. 2024). Our W8A8 quantized variant of TinySAM (QTinySAM) also obtains competitive performance across different methods. Specifically, Q-TinySAM achieves $0 . 1 \% +$ AP on COCO and $0 . 2 \% +$ on LVIS v1 dataset than SlimSAM (Chen et al. 2024), with only $3 9 \%$ MACs and $2 1 . 8 \%$ latency. Visual results on COCO validation set and LVIS dataset shows that our proposed TinySAM captures more clear and smooth boundaries compared with other efficient variants of SAM.

# Zero-shot Points Valid Mask Evaluation

In this section, we evaluate the performance of our TinySAM for segmenting an object from several points as the prompts. We use the same points selection metric as previous work (Kirillov et al. 2023; Gupta, Dollar, and Girshick 2019), which calculates the distance transform of false positive and false negative masks, and then sample points at a maximal value. We calculate the mIoU of each dataset to evaluate the performance of different models.

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2">MACs</td><td rowspan="2">Lat.(ms)</td><td colspan="4">COCO</td><td colspan="4">LVIS v1</td></tr><tr><td>AP</td><td>APS</td><td>ApM</td><td>ApL</td><td>AP</td><td>APS</td><td>APM</td><td>ApL</td></tr><tr><td>ViTDet-H (Li et al. 2022a)</td><td>1</td><td>1</td><td>51.0</td><td>32.0</td><td>54.3</td><td>68.9</td><td>46.6</td><td>35.0</td><td>58.0</td><td>66.3</td></tr><tr><td colspan="9">zero-shot transfer methods (segmentation module only):</td><td></td><td></td></tr><tr><td>SAM-H (Kirillov et al. 2023)</td><td>2976G</td><td>2392</td><td>46.6</td><td>30.8</td><td>51.0</td><td>61.7</td><td>44.7</td><td>32.5</td><td>57.6</td><td>65.5</td></tr><tr><td>SAM-L (Kirillov et al.2023)</td><td>1491G</td><td>1146</td><td>46.2</td><td>30.2</td><td>50.1</td><td>60.5</td><td>43.5</td><td>31.1</td><td>56.3</td><td>65.1</td></tr><tr><td>SAM-B (Kirillov et al. 2023)</td><td>487G</td><td>368.8</td><td>43.4</td><td>28.5</td><td>45.5</td><td>53.4</td><td>40.8</td><td>29.1</td><td>52.8</td><td>60.7</td></tr><tr><td>FastSAM (Zhao et al. 2023)</td><td>443G</td><td>153.6</td><td>37.9</td><td>23.9</td><td>43.4</td><td>50.0</td><td>34.5</td><td>24.6</td><td>46.2</td><td>50.8</td></tr><tr><td>EfficientSAM-Ti (Xiong et al. 2024)</td><td>106G</td><td>81.0</td><td>42.3</td><td>26.7</td><td>46.2</td><td>57.4</td><td>39.9</td><td>28.9</td><td>51.0</td><td>59.9</td></tr><tr><td>SlimSAM-77 (Chen et al. 2024)</td><td>51.7G</td><td>110</td><td>41.3</td><td>25.7</td><td>44.9</td><td>57.4</td><td>38.3</td><td>26.7</td><td>49.7</td><td>59.0</td></tr><tr><td>MobileSAM (Zhang et al. 2023)</td><td>42.0G</td><td>38.4</td><td>41.0</td><td>24.4</td><td>44.5</td><td>58.6</td><td>37.0</td><td>24.7</td><td>47.8</td><td>59.1</td></tr><tr><td>TinySAM (Ours)</td><td>42.0G</td><td>38.4</td><td>42.3</td><td>26.3</td><td>45.8</td><td>58.8</td><td>38.9</td><td>27.0</td><td>50.3</td><td>60.2</td></tr><tr><td>Q-TinySAM (Ours)</td><td>20.3G</td><td>24.0</td><td>41.4</td><td>25.6</td><td>45.1</td><td>57.9</td><td>38.5</td><td>26.6</td><td>49.8</td><td>59.8</td></tr></table></body></html>

Table 1: Zero-shot instance segmentation results on COCO and LVIS v1 dataset. Zero-shot transfer methods are prompted with the detection boxes from fully-supervised ViTDet model. TinySAM and quantized Q-TinySAM demonstrate advantageous performance on average precision. The latency is tested on NVIDIA T4 GPU.

LVIS DOORS BBBC038v1 TimberSeg 1.0 Mean   
0.75 0.86 0.84 0.8 0.80 0.82   
0.70 0.84 0.80 0.7 0.75   
0.65 SMAoMbi-lBeSAM 0.82 SMAoMbi-lBeSAM 0.768 SMAoMbi-lBeSAM 0.6 SMAoMbi-lBeSAM 0.70 SMAoMbi-lBeSAM 0.80   
0.60 TSiAnMy-SLHAM 0.78 TSiAnMy-SLHAM 0.724 TSiAnMy-SLHAM 0.5 TSiAnMy-SLHAM 0.65 SAM-LH 2 4 6 8 2 4 6 8 2 4 6 8 2 4 6 8 2 4 6 8

Figure 4: Results of zero-shot points valid mask evaluation. X-axis represents the number of prompts points and Y-axis represents the mIoU across all masks. The proposed TinySAM outperforms MobileSAM and achieves results close to SAM-B.   
Table 2: Comparison of original point grid strategy and our hierarchical strategy. Evaluation on the first 100 images of COCO val2017 set.   

<html><body><table><tr><td>Strategy</td><td>Model</td><td>mIoU</td><td>Time (s)</td></tr><tr><td>Original</td><td>MobileSAM</td><td>0.5963</td><td>1.6719</td></tr><tr><td>Hierarchical (Ours) Original</td><td>MobileSAM SAM-H</td><td>0.5958 0.7047</td><td>0.8462</td></tr><tr><td>Hierarchical (Ours)</td><td>SAM-H</td><td></td><td>2.4549</td></tr><tr><td></td><td></td><td>0.7055</td><td>1.3537</td></tr><tr><td>Original</td><td>TinySAM</td><td>0.6137</td><td>1.7790</td></tr><tr><td>Hierarchical (Ours)</td><td>TinySAM</td><td>0.6061</td><td>0.9303</td></tr></table></body></html>

We choose a subset of total 23 datasets used in (Kirillov et al. 2023) for efficient evaluation, which contains BBBC038v1 (Caicedo et al. 2019), DOORS (Pugliatti and Topputo 2022), TimberSeg (Fortin et al. 2022) and LVIS (Gupta, Dollar, and Girshick 2019). To make fair comparison, we follow the settings of SAM (Kirillov et al. 2023) to sample the images and masks, and the first $N$ masks in the corresponding split are used in the evaluation.

The evaluation results are shown in Figure 4. Our TinySAM outperforms MobileSAM (Zhang et al. 2023) significantly on LVIS and TimberSeg dataset and obtains similar performance on DOORS dataset. Moreover, TinySAM achieves better results on BBBC038v1 when fewer points are utilized as prompts. We also report the mean IoU of all four datasets, as shown in the right of Figure 4. The proposed TinySAM achieves higher mIoU than MobileSAM and obtains close performance to that of SAM-B.

![](images/a357acb4af692b424517279e04494ebae7eaac3f2ab6f89a49e1bf6d632f97fe.jpg)  
Figure 5: Visualization for the process hierarchical everything strategy. (a) shows the intermediate result of highconfidence regions after 1st sparse prompt points with white mask and remained 2nd dense prompt points with green stars. (b) shows the final segmentation result and the small objects can be accurately segmented.

# Everything Mode Acceleration

We evaluate our proposed hierarchical everything inference strategy on COCO validation set. Latency benchmarks are conducted on a single NVIDIA V100 GPU for everything mode. We sample 100 images with the least img id from val2017 and conduct everything mode inference on these images. The threshold values used in the everything mode are all kept the same as default. The results are shown in

Table 3: Effect of distillation loss, online hard prompt sampling and quantization respectively, evaluated on zero-shot instance segmentation on COCO validation dataset.   

<html><body><table><tr><td>Ind.</td><td>Setings</td><td>AP (%)</td></tr><tr><td>0</td><td>Baseline</td><td>40.7</td></tr><tr><td>1</td><td>+ Knowledge Distillation Loss</td><td>41.4</td></tr><tr><td>2</td><td>+ Hard Prompt Sampling</td><td>41.9</td></tr><tr><td>3</td><td>+ HardMask Weighting</td><td>42.3</td></tr><tr><td>4</td><td>+ Quantization</td><td>41.4</td></tr></table></body></html>

<html><body><table><tr><td>Embedding Loss</td><td>Token Loss</td><td>Output Loss</td><td>AP (%)</td></tr><tr><td></td><td>-</td><td>√</td><td>41.6</td></tr><tr><td>√</td><td></td><td>√</td><td>41.7</td></tr><tr><td>√</td><td>√</td><td>√</td><td>41.9</td></tr><tr><td>√</td><td>√</td><td>√(HMW)</td><td>42.3</td></tr></table></body></html>

Table 4: Ablation study on combinations of knowledge distillation losses for zero-shot instance segmentation on COCO val set.

Table 2. We apply the same threshold and stability score on the same model evaluated with different strategies to make a fair comparison, but they can be different between these models. Our hierarchical strategy achieves comparable results compared with original $3 2 \times 3 2$ points grid strategy while the cost of inference time is reduced by about $5 0 \%$ . Figure 5 shows the intermediate visual results of the hierarchical strategy. We can see that the 1st round of sparse inference has segmented and removed the large objects, the remained points focus more on the small objects. This selfadaptive hierarchical strategy efficiently reduces the computation redundancy and maintains the high accuracy.

# Ablation Studies

In this section, we conduct ablation studies of the proposed method on zero-shot instance segmentation task on COCO validation dataset. The experimental setting is the same as described in zero-shot instance segmentation.

Impacts of different modules. We first evaluate the effects of different modules, i.e., full-stage knowledge distillation loss, hard prompt sampling, hard mask weighting and post quantization, respectively. As shown in Table 3, utilizing our proposed full-stage distillation strategy improve the performance from $4 0 . 7 \%$ to $4 1 . 4 \%$ . Incorporated with the online hard prompt sampling strategy, our method could obtain $0 . 5 \%$ AP gain. With the hard mask weighting loss, the performance can further increase to $4 2 . 3 \%$ . Using post-training quantization results in $0 . 9 \%$ AP degradation but greatly reduces the computational cost.

Impacts of different distillation losses. For detailed fullstage knowledge distillation process, we investigate the necessity of the proposed three-level distillation from the teacher network. Table 4 shows the ablation results with different combinations of distillation losses. The output distillation loss takes important part since it is close to the supervision information and the similarity with teacher network directly reflects in the evaluation metric. Token loss and embedding loss both prove to be beneficial since they are related to key nodes of teacher network, which reflects the image-level information and the interaction of prompts with the image, respectively. Hard mask weighting for output loss can further boost the performance.

Table 5: Ablation on point density and threshold for hierarchical strategy.   

<html><body><table><tr><td>Points per side 1st/2nd</td><td>Thresh.T</td><td>mIoU</td><td>Time (s)</td></tr><tr><td>4/16</td><td>8.5</td><td>0.5521</td><td>0.3571</td></tr><tr><td>8/32</td><td>8.5</td><td>0.6061</td><td>0.9303</td></tr><tr><td>10/32</td><td>8.5</td><td>0.6078</td><td>1.2774</td></tr><tr><td>8/32</td><td>7.0</td><td>0.6018</td><td>0.8154</td></tr><tr><td>8/32</td><td>10.0</td><td>0.6067</td><td>1.1819</td></tr><tr><td>32/-</td><td></td><td>0.6137</td><td>1.7790</td></tr></table></body></html>

<html><body><table><tr><td>Model</td><td>AP (%)</td><td>MACs (G)</td></tr><tr><td>MobileSAM</td><td>41.0</td><td>42.0</td></tr><tr><td>+ W8A8</td><td>39.8</td><td>20.28</td></tr><tr><td>+W6A6</td><td>36.3</td><td>18.45</td></tr><tr><td>TinySAM(Ours)</td><td>42.3</td><td>42.0</td></tr><tr><td>+ W8A8</td><td>41.4</td><td>20.28</td></tr><tr><td>+ W6A6</td><td>39.0</td><td>18.45</td></tr></table></body></html>

Table 6: Ablation study for different bit width of quantization for zero-shot instance segmentation on COCO val set.

Point density and threshold for hierarchical strategy. In Table 5, we conduct ablation study with different settings of point density and high-confidence mask threshold $\tau$ . More points and higher threshold $\tau$ lead to more precise results but longer inference time. The point density of 2nd round is more sensitive compared to the 1st one. Considering both accuracy and efficiency, the setting in bold is a good balance and used for other experiments of everything inference.

Different bits for quantization. We here explore the influence of different bit width. Table 6 reports the average precision on COCO dataset. From the results, we can conclude that quantization to 8-bit results in only slight performance drop. We also demonstrate the performance by further reducing the quantization bit width to 6.

# Conclusion

In this paper, we propose a framework to push the envelope for segment anything task and obtain a highly efficient model named TinySAM. We first propose a full-stage knowledge distillation method with hard mask weighting and hard prompt sampling strategy to distill a lightweight student model. We also adapt the post-training quantization to the prompt-based segmentation task and further reducing the computational cost. Moreover, a hierarchical segmenting everything strategy is proposed to accelerate the everything inference by $2 \times$ with almost no performance degradation. With all these proposed methods, our TinySAM leads to orders of magnitude computational reduction and push the envelope for efficient segment anything task. Extensive experiments on various zero-shot transfer tasks demonstrate the significantly advantageous performance of our TinySAM against counterpart methods. We hope the proposed TinySAM brings beneficial perspective for designing a highly efficient segment anything model.
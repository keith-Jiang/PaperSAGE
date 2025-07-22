# A Layer Selection Approach to Test Time Adaptation

Sabyasachi Sahoo1,2, Mostafa ElAraby2,3, Jonas Ngnawe1,2, Yann Batiste Pequignot1, Fre´de´ric Precioso4, Christian Gagne´1,2,5

1IID, Universite´ Laval 2Mila 3Universite´ de Montre´al 4Universite´ Cote d’Azur, CNRS, INRIA, I3S, Maasai 5Canada CIFAR AI Chair sabyasachi.sahoo.1 $@$ ulaval.ca

# Abstract

Test Time Adaptation (TTA) addresses the problem of distribution shift by adapting a pretrained model to a new domain during inference. When faced with challenging shifts, most methods collapse and perform worse than the original pretrained model. In this paper, we find that not all layers are equally receptive to the adaptation, and the layers with the most misaligned gradients often cause performance degradation. To address this, we propose GALA, a novel layer selection criterion to identify the most beneficial updates to perform during test time adaptation. This criterion can also filter out unreliable samples with noisy gradients. Its simplicity allows seamless integration with existing TTA loss functions, thereby preventing degradation and focusing adaptation on the most trainable layers. This approach also helps to regularize adaptation to preserve the pretrained features, which are crucial for handling unseen domains. Through extensive experiments, we demonstrate that the proposed layer selection framework improves the performance of existing TTA approaches across multiple datasets, domain shifts, model architectures, and TTA losses.

# 1 Introduction

Distribution shifts (Gulrajani and Lopez-Paz 2021) present significant challenges when deploying deep learning models in real-world scenarios. Test Time Adaptation (TTA) (Liang, He, and Tan 2023) has emerged as a promising approach for adapting pretrained models to novel domains during inference. However, these methods often falter when confronted with severe or diverse distributional changes. To mitigate potential performance degradation, various regularization strategies have been proposed (Niu et al. 2022; Shin et al. 2024). Nevertheless, these strategies might not effectively address all types of shifts or TTA losses (Burns and Steinhardt 2021; Zhao et al. 2023a). Moreover, the selection of layers in the existing TTA approaches typically remains unchanged across different shifts (Wang et al. 2024), which may not be optimal. In contrast, layer selection has demonstrated substantial improvements in related fields such as domain generalization (Chattopadhyay, Balaji, and Hoffman 2020), fine-tuning (Lee et al. 2023), multi-task learning (Wallingford et al. 2022), and continual learning (Zhao et al.

![](images/06b66c6a5d1823cf47ca60f457d08f0f50c12a3b51ea063090709bb05e55f0ec.jpg)  
Figure 1: Intuition for proposed approaches: (a) As the model reaches closer to minima, the individual sample gradients start to be misaligned with gradients of previous samples (Mahsereci et al. 2017; Forouzesh and Thiran 2021; Agarwal, D’souza, and Hooker 2022). We leverage this misalignment to identify trainable layers. (b) While effective in moving in the direction of most aligned gradients, the introduced criterion based on angular deviation could prevent adaptation when a direction change is needed, even if the following updates (or gradients) are aligned. A reset of the past horizon (i.e., gradients of previous samples) considered in the alignment condition can help resolve such situations.

2023b), underscoring the importance and broad potential of layer selection. Still, the question of optimal layer selection remains largely unexplored in the context of TTA.

In this paper, we study layer selection for TTA and show that not all layers of a given model are equally receptive to adaptation. Our findings suggest that adapting the right layer can lead to meaningful improvement, while adapting the wrong layer can cause significant performance degradation in TTA approaches. Specifically, we find that while adapting a certain layer may benefit one shift, it may be detrimental to another. Additionally, we find that on a given shift, the effect of adapting a certain layer also depends on the loss used. Therefore, while we observe an important potential in selecting the right layer to adapt in each situation, identifying these layers at test time can be challenging.

To address the challenges of layer selection, we propose Gradient-Aligned Layer Adaptation, GALA, a novel criterion to identify good layers for adaptation at test time.

![](images/c58f16ceb96e8882699c310a6cdd1f9900f66d22ae7be4072124a5b34694cf84.jpg)  
Figure 2: Gradient-Aligned Layer Adaptation or GALA framework adapts the most gradient-aligned layer per sample. It adapts all the layers for the first sample in a reset window (e.g., $x _ { 1 } , x _ { n } , \ldots )$ . For all the other samples, it adapts the most gradientaligned layer per sample. It can also skip the adaptation on a given sample if all the layers are misaligned. We use a reset window to periodically reset the anchor parameters to allow for a change in direction.

GALA ranks all the layers of a model based on the gradient alignment of the current adaptation step. As the model approaches the optimization minima, the variance in gradient updates increases (Mahsereci et al. 2017; Forouzesh and Thiran 2021; Agarwal, D’souza, and Hooker 2022), leading to potential overfitting and performance degradation. Building on this insight, for each layer, we propose to measure the angle deviation of the proposed gradient update from the average of all gradient updates performed so far (including the proposed one). This measure can also be expressed as the cosine between the proposed update and the (anticipated) total displacement of the parameters from their pretrained values. This allows us to compare the updates for each layer on a common scale and only perform the update of the layer with the smallest angle.

Our extensive experiments on Domainbed (Gulrajani and Lopez-Paz 2021) and Continual TTA benchmark (Wang et al. 2022) demonstrate that GALA consistently surpasses all layers and ERM (no adaptation) baselines and other existing layer selection baselines across various datasets, various neural network backbones, and various losses. Further analysis reveals that GALA can identify the good layers, which exhibit significant displacement in a single direction and higher gradient alignment. This layer selection strategy enhances the model’s ability to adapt to novel domains by mitigating performance degradation and potentially serves as a regularization mechanism, reducing catastrophic forgetting of source domain knowledge. Ablation studies reveal that GALA’s performance is robust to hyperparameter choices.

The contributions of our paper are summarized as follows:

1. We study the problem of layer selection for TTA and find that while adapting specific layers can enhance performance, the optimal set of layers for adaptation is not universal but rather contingent upon the particular distribution shift encountered and the TTA loss function employed during inference.

2. We introduce GALA, a novel layer selection criterion to identify good layers to adapt per sample that can be applied across various distribution shifts and TTA loss functions at test time.   
3. Through extensive experiments across different backbones, datasets, and TTA losses, we show that GALA outperforms standard ERM (no adaptation), all layers baselines, and other layer selection baselines (i.e., AutoRGN and AutoSNR (Lee et al. 2023)) for TTA.

# 2 Proposed Approach

In the following, we describe the Gradient-Aligned Layer Adaptation (GALA) framework for Test Time Adaptation (TTA). We first introduce our layer selection framework for TTA (Sec. 2.1), before describing the cosine distance criterion proposed to identify the most trainable layers (Sec. 2.2), and then present the reset window strategy used to improve performances with the proposed cosine criterion (Sec. 2.3).

# 2.1 Layer selection framework for TTA

Let $f _ { \theta _ { \mathrm { s r c } } }$ denote the model parameterized by parameters $\theta _ { \mathrm { s r c } }$ trained beforehand on the source domain $\mathcal { D } _ { \mathrm { s r c } }$ . Let us also assume that target domain samples $\{ x _ { i } \} _ { i = 1 } ^ { n }$ are coming in an online fashion at test time. For some sample $x _ { i }$ at test time, TTA adapts the model to obtain $\theta _ { i }$ before performing inference (Sun et al. 2020; Liang, He, and Tan 2023). We set $\theta _ { 0 } = \theta _ { \mathrm { s r c } }$ and, at each step, $\theta _ { i }$ is obtained by updating $\theta _ { i - 1 }$ using the following equation:

$$
\theta _ { i } = \theta _ { i - 1 } + \mathbf { u } _ { i } ,
$$

where $\mathbf { u } _ { i }$ is a parameter update specific to the TTA algorithm. Typically, if SGD optimizer is used with learning rate $\eta$ , this update takes the form $\mathbf { u } _ { i } = - \eta \nabla \mathcal { L } ( x _ { i } ; \theta _ { i - 1 } )$ , where $\mathcal { L }$ is the unsupervised loss specific to the TTA method.

In this section, we consider single-step TTA performed online on a single input sample using an SGD optimizer for notation simplicity. Throughout, we assume the deep learning model is written as a certain composition of functions, which we simply refer to as layers, though any granularity would do. This allows us to write the model at step $\mathbf { \chi } _ { i }$ as $f _ { \theta _ { i } } = f _ { \theta _ { i , L } } \circ \cdot \cdot \cdot \circ f _ { \theta _ { i , 1 } }$ , where $\theta _ { i , l }$ denote the parameters of layer $l$ at step $i$ . The update equation at step $i$ can be written for each layer as:

![](images/b70c465b537ee260f1a3e30742d978d17c9b242118cc9eee5f4f3e5a12b02c6b.jpg)  
Figure 3: Illustration of proposed criterion based on angular deviation. Different layers can be ranked based on their alignments with previous gradient updates. In the figure, updates drawn in red are discarded, while green updates are applied, adding up to $\mathbf { T D } _ { i - 1 }$ . The update under scrutiny $\mathbf { u _ { i } }$ is drawn in cyan, and its sum with $\mathbf { T D } _ { i - 1 }$ is drawn in blue. Application of update $\bf { u _ { i } }$ or not is based on the angle $\alpha _ { i }$ .

$$
\theta _ { i , l } = \theta _ { i - 1 , l } + \mathbf { u } _ { i , l } .
$$

To perform layer selection, we modify this update equation by introducing a mask:

$$
\theta _ { i , l } = \theta _ { i - 1 , l } + m _ { i , l } { \bf u } _ { i , l } ,
$$

where $m _ { i , l } \in \{ 0 , 1 \}$ is the value of the binary mask applied to the update ui,l.

# 2.2 Cosine distance criterion

Existing works have shown that gradient descent happens in a tiny subspace (Gur-Ari, Roberts, and Dyer 2018). Moreover, as the model reaches closer to the minima, the gradients across the samples get noisy (Mahsereci et al. 2017; Forouzesh and Thiran 2021; Agarwal, D’souza, and Hooker 2022). We aim to identify the layers with the most beneficial gradient updates to the model for adapting to the new domain. Let us assume that the total displacement of parameters of layer $l$ at the start of the $i ^ { \mathrm { { t h } } }$ step is given by:

$$
\mathbf { T D } _ { i - 1 , l } = \sum _ { j = 1 } ^ { i - 1 } m _ { j , l } \mathbf { u } _ { j , l } = \theta _ { i - 1 , l } - \theta _ { 0 , l } .
$$

Our proposed criterion relies on the angular deviation of the update $\mathbf { u } _ { i , l }$ from the direction of the total displacement that would result from making this update:

$$
\cos ( \alpha _ { i , l } ) = \frac { \mathbf { u } _ { i , l } \cdot ( \mathbf { u } _ { i , l } + \mathbf { T D } _ { i - 1 , l } ) } { \Vert \mathbf { u } _ { i , l } \Vert _ { 2 } \Vert \mathbf { u } _ { i , l } + \mathbf { T D } _ { i - 1 , l } \Vert _ { 2 } } .
$$

This angle can be interpreted as the deviation of the update under consideration from the anticipated average update, which has the same direction as the anticipated total displacement $\mathbf { u } _ { i , l } + \mathbf { T } \mathbf { D } _ { i - 1 , l } - \mathrm { s e e }$ this illustrated in Fig. 3.

Comparing our criterion across layers allows us to define which update is performed by defining the mask:

$$
m _ { i , l } = { \left\{ \begin{array} { l l } { 1 } & { { \mathrm { i f } } \cos ( \alpha _ { i , l } ) > \lambda } \\ { 0 } & { { \mathrm { o t h e r w i s e } } } \end{array} \right. } ,
$$

where $\lambda$ is the selection threshold. The fact that the cosine metric lies in the $[ - 1 , 1 ]$ domain allows us to compare the alignment of updates for layers with different sizes of parameters. We set a single $\lambda > 0$ for thresholding over all layers, which prevents the adaptation of updates that are misaligned with the updates applied in the past. A $\lambda$ close to 1 will only allow adaptation of updates aligned with past updates, while a lower $\lambda$ would be less restrictive.

# 2.3 Cosine distance with reset

While the cosine distance can stop adaptation for noisy gradients, our criterion may fail, especially when the gradient update trajectory needs to change direction after a certain point. If the gradient updates meet an inflection point in the loss landscape, cosine distance will prevent further adaptation, and the model will remain stuck at this point even if the gradient update is informative. To solve such cases, we propose to use resets for the computation of the total displacement of a layer. We use a fixed window scheme for resetting the initial parameter point, which we will call the anchor point. This corresponds to:

$$
\mathbf { T D } _ { i , l } = \theta _ { i , l } - \theta _ { r , l } ,
$$

where $\theta _ { r , l }$ is the parameter at last reset step $\begin{array} { r } { r = \lfloor \frac { i - 1 } { s } \rfloor } \end{array}$ , and $s$ is the size of the reset window. The anchor point changes only when the reset window changes, as illustrated in Fig. 2.

# 3 Experiments

This section compares our proposed approaches with existing baselines on Domainbed (Gulrajani and Lopez-Paz 2021), a popular benchmark with large single distribution shifts, and Continual TTA, a popular benchmark with multiple distribution shifts.

TTA losses Two popular TTA losses are considered: Pseudo-Labeling (PL) (Lee et al. 2013) and SHOT (Liang, Hu, and Feng 2020). We perform hyperparameter selection based on Zhao et al. (2023a), where we report the performance for the best hyperparameter set found by sweeping over a range of values.

Baselines We compare the TTA performance obtained by adapting All layers vs. the layers proposed by our approach. We also report the ERM (no adaptation) performance of the pretrained model. In Domainbed, we also compare against AutoRGN and AutoSNR (Lee et al. 2023), two popular baselines proposed to identify optimal layers in fine-tuning setup.

Table 1: Accuracy $( \% )$ of various layer selection methods on Domainbed benchmark (setup described in Sec. 3.1). The best method for a given TTA loss and backbone is in bold.   

<html><body><table><tr><td></td><td>TTA</td><td>Method</td><td>PACS↑</td><td>VLCS↑</td><td>Terra ↑</td><td></td><td>Office ↑</td><td>Mean 个</td></tr><tr><td></td><td>ERM</td><td></td><td>80.99 (±0.9)</td><td>75.14 (±1.2)</td><td>40.80 (±0.2)</td><td></td><td>62.18 (±0.4)</td><td>64.78</td></tr><tr><td></td><td>PL</td><td>All layers AutoRGN AutoSNR GALA</td><td>81.79 (±0.7) 82.82 (±0.6) 80.58 (±1.2) 83.56 (±0.6)</td><td>65.69 (±1.5) 72.63 (±1.3) 65.72 （±1.8)</td><td>35.40 (±9.7) 38.18 （±6.1) 35.01 (±10.4)</td><td>62.38 59.82</td><td>60.20 (±1.4) (±0.2) （±0.9)</td><td>60.77 64.00 60.28</td></tr><tr><td></td><td>SHOT</td><td>All layers AutoRGN AutoSNR</td><td>83.48 (±0.3) 84.10 (±0.5) 83.43 （±0.3)</td><td>75.48 （±1.2） 66.23 (±2.8) 69.78 (±1.3) 66.26 （±2.7)</td><td>44.19 (±1.1) 33.81 (±1.3) 37.37 (±0.7)</td><td>63.09</td><td>62.67 (±0.2) 63.03 (±0.4) (±0.2)</td><td>66.47 61.64 63.59</td></tr><tr><td></td><td>ERM</td><td>GALA</td><td>83.92 （±0.8) 82.84 (±0.5)</td><td>76.23 (±1.1) 75.83 (±0.9)</td><td>33.75 (±1.2) 42.13 (±1.4) 46.14 (±2.3)</td><td>63.02</td><td>(±0.4) 63.32 (±0.3) 66.93 (±0.3)</td><td>61.62 66.40 67.93</td></tr><tr><td></td><td></td><td>All layers AutoRGN</td><td>82.36 (±2.8) 85.03 (±1.4)</td><td>69.22 (±1.4) 75.34 (±4.4）)</td><td>42.28 (±3.2) 48.44 ±2.4）</td><td></td><td>61.54 (±3.3) 66.9 (±0.3)</td><td>63.85 68.94</td></tr><tr><td>SHOT</td><td></td><td>GALA All layers AutoRGN AutoSNR GALA</td><td>84.87 （±0.8） 85.15 (±1.1) 86.34 (±1.1) 85.51 （±0.5） 86.13 (±0.8)</td><td>76.88 (±1.6) 64.25 (±1.1) 70.2 (±0.9) 64.26 (±1.3) 76.48 (±1.0)</td><td>50.10 （±2.5） 35.33 (±3.1) 40.59 （±1.3) 34.97 ±3.2) 45.94 (±1.6)</td><td>68.10 68.13 (±0.3)</td><td>67.34 (±0.3) 67.37 (±0.3) (±0.4) 67.33 (±0.2)</td><td>69.80 63.03 66.31 63.02</td></tr></table></body></html>

Implementational details We report results for GALA with window size of 20 and selection threshold of 0.75 with single-layer granularity. It appears that GALA is not overly sensitive to hyperparameters, and those values work well overall – see Sec. 5 for more discussion on hyperparameter values and the design choices. We also scale the updates for a few initial samples in the reset window to reduce their impact on incorrect layer selection.

# 3.1 Domainbed results

For the experiments on Domainbed, we follow the evaluation protocol as described in Iwasawa and Matsuo (2021), including dataset splits for the following four datasets: PACS (Li et al. 2017), VLCS (Fang, Xu, and Rockmore 2013), Terra Incognita (Beery, Van Horn, and Perona 2018), and Office-Home (Venkateswara et al. 2017). Results are reported on two backbones (i.e., ResNet-18 and ResNet-50) with batch normalization layers, while the pretrained models are made using default hyperparameters described in Gulrajani and Lopez-Paz (2021). Mean and standard deviation are reported over three repetitions with different random seeds. See Appendix A.1 for further details.

Key takeaways from results are reported in Tab. 1:

• GALA outperforms ERM (no adaptation) by $2 \%$ overall and All layers TTA baselines by more than $5 \%$ overall across all losses, backbones, and datasets. • Existing layer selection baselines like AutoRGN or AutoSNR can improve performance compared to all layers TTA in most setups, especially AutoRGN, but fail to improve against no adaptation baselines for some datasets like VLCS or TerraIncognita or some TTA losses like SHOT. GALA consistently demonstrates equivalent or superior performance across all datasets and TTA losses, achieving an overall improvement of about $2 \%$ .

Table 2: Accuracy $( \% )$ of layer selection methods on Continual TTA benchmark (with the setup described in Sec. 3.2). The best method for a given TTA loss is in bold.   

<html><body><table><tr><td>TTA</td><td>Method</td><td>CIFAR10C↓</td><td>CIFAR100C↓</td></tr><tr><td>ERM</td><td></td><td>43.50 (±18.7)</td><td>46.40 (±15.7)</td></tr><tr><td rowspan="2">PL</td><td>All layers</td><td>88.72 (±1.2)</td><td>98.63 (±1.5)</td></tr><tr><td>GALA</td><td>28.68 (±6.6)</td><td>33.69 (±5.7)</td></tr><tr><td rowspan="2">SHOT</td><td>All layers</td><td>89.33 (±2.3)</td><td>97.32 (±4.8)</td></tr><tr><td>GALA</td><td>20.46 (±7.7)</td><td>32.87 (±5.6)</td></tr></table></body></html>

• GALA improves over Domainbed large shift datasets (i.e., PACS, OfficeHome) similar to AutoRGN and AutoSNR while comfortably outperforming the ERM baseline. On small shift datasets (i.e., VLCS, TerraIncognita), existing baselines struggle to outperform the no adaptation baseline while GALA appears to prevent degradation caused by over-adaptation, thereby enhancing performance over the ERM baseline and safeguard against further degradation.

# 3.2 Continual TTA results

We follow the evaluation protocol as described in Wang et al. (2022), evaluating performance on two datasetsbackbones: 1) CIFAR10C (Hendrycks and Dietterich 2019) with WideResNet-28 (Zagoruyko and Komodakis 2016) and CIFAR100C (Hendrycks and Dietterich 2019) with ResNeXt-29 (Xie et al. 2017). The pretrained models are trained as described in Robustbench (Croce et al. 2021). Mean and standard deviation are reported across the 15 corruption types. Further details are given in Appendix A.2.

The key takeaways based on the results from Tab. 2 are:

• Performance degradation by training all layers is worse in the Continual TTA benchmark containing multidomain shifts than degradation in the Domainbed benchmark containing single-domain shifts. Moreover, more severe degradations are observed in CIFAR100C, which has 100 classes, compared to CIFAR10, which includes 10 classes, despite similar ERM performance on both datasets.   
• GALA consistently outperforms ERM by about $1 5 \%$ and all layers TTA baseline by about $65 \%$ , despite severe degradation.

# 4 Layer Selection Study

In this section, we evaluate the importance of layer selection for test time adaptation on the Domainbed benchmark and provide some analysis and motivation for GALA. We use the Domainbed benchmark with the ResNet-18 backbone, which contains four blocks of layers. We study the effect of choosing one block over another by performing adaptation on a single block while freezing all the other blocks of the model. We refer to blocks and layers interchangeably in this section. We report the difference between TTA and ERM accuracy over all blocks for each loss and dataset shift setting. Otherwise, we rely on the same setup and evaluation protocol described in Sec. 3.1.

![](images/80bf7af73effb91ec77378f7a77ea232eaf0e30ea283296949a1629880a9bbdf.jpg)  
Figure 4: Heatmap of Performance improvement $( \% )$ per-block on Domainbed benchmark. Performance improvement is the difference between the TTA accuracy of a given block/layer and ERM accuracy for the same shift. Positive performance improvements are shown in green, and negative performance improvements (or degradation) are in red. Using the bounding box, we highlight the best block per loss and dataset shift. Further details in Sec. 4.

# 4.1 Layer selection matters

In Fig. 4, we observe that not all layers are equally receptive to adaptation. We refer to a layer as good or bad based on the accuracy improvement of selecting a given layer w.r.t. performance of a pretrained model on the same shift. We compare against Empirical Risk Minimization (ERM) or the frozen pretrained model’s performances, as we are interested in measuring the performance improvement or degradation brought by individual layers during adaptation. Also, the ERM model performs better on average than all layer TTA, as seen in Sec. 3, and it becomes a natural baseline that can help contrast different layers.

The selection of layers in existing TTA approaches typically remains unchanged in all adaptation settings. We find it can be a suboptimal strategy and one of the major causes of degradation in existing TTA approaches – no single layer adaptation is suitable for all settings. Therefore, layer selection is essential for TTA, and we propose GALA to improve the performance of existing TTA approaches in various settings.

# 4.2 What affects the adaptability of a layer?

Using the same setup and evaluation protocol (cf., Sec. 3.1), we are making the following observations on Fig. 4 about the factors affecting the adaptability of good layers:

• Location of good layers in a model can change across shifts of a given dataset, despite using pretrained models trained on the same class labels. Similar observations have also been made in fine-tuning setups(Lee et al.

2023). There is a need for a good layer selection criterion that depends on target samples observed by the model at test time.

• We also find that good layers in a model can change with different TTA loss functions, even for the same shift and dataset. Hence, a good layer selection criterion must also depend on the TTA loss function used to adapt the model at inference.

Since gradients depend on the shift and TTA loss function used, GALA uses layerwise gradients to identify the adaptability of each layer in the model.

# 4.3 How do good layers differ from bad layers?

To perform a detailed per-layer analysis, we created the Tiny-Domainbed benchmark, which was made as a smaller version of Domainbed. It consists of all the critical shifts with the brightest red/green layers (displayed in Fig. 4), whose good layers can also change with the TTA method. We follow the benchmark and evaluation protocol described in Sec. 3.1, with further details given in Appendix A.3. Based on Tab. 3, the following are the differences between good and bad layers:

• Adaptation with Worst Block results in poor TTA accuracy, poorer generalization to the target domain, and higher forgetting. Since training all layers involves training the worst layer, this could explain why training all layers results in poorer TTA accuracy. On the other hand, Best Block results in better generalization to the target domain. This implies that TTA with good layers can poten

<html><body><table><tr><td>Method</td><td>TTA Acc. ↑</td><td>Gen.↑</td><td>Forget.↓</td><td>Rank corr. 个</td></tr><tr><td>All Blocks</td><td>53.6</td><td>46.5</td><td>31.3</td><td>N/A</td></tr><tr><td>Worst Block (oracle)</td><td>43.5</td><td>38.7</td><td>39.9</td><td>-1</td></tr><tr><td>Best Block (oracle)</td><td>64.1</td><td>63.9</td><td>28.7</td><td>1</td></tr><tr><td>Random Block</td><td>53.1</td><td>49.1</td><td>13.1</td><td>0</td></tr><tr><td>GALA</td><td>59.4</td><td>58.0</td><td>9.3</td><td>0.76</td></tr></table></body></html>

Table 3: Effect of various layer selection methods on TTA Accuracy $( \% )$ , Generalization $( \% )$ , Forgetting $( \% )$ and Spearman correlation with Best Block $( \in [ - 1 , 1 ] )$ averaged over different shifts on Tiny-Domainbed benchmark (with the setup described in Sec. 4.3). TTA Acc is the accuracy of testing samples from the target domain seen during adaptation. Generalization is the accuracy of the held-out split of the target domain after adaptation. Forgetting is the drop in accuracy on the held-out split of source domains after adaptation. Rank correlation is the Spearman correlation of layer selection rank between the oracle and the method. Bold and underlined denote best and second-best, respectively.

tially learn target domain features better than TTA with bad layers.

• We observe that Best Block results in reduced source forgetting compared to Worst Block. This implies that TTA with good layers strikes an improved balance between learning new features on the target domain while retaining useful pretrained features from the source domain.

Therefore, we propose GALA to identify good layers for adaptation, which can help balance adaptation to the new domain while reducing source forgetting.

# 4.4 How does GALA compare to oracle strategies?

To analyze GALA’s layer selection behavior, we compare it to the oracle strategies given by Best block and Worst block on the Tiny-Domainbed benchmark (Tab. 3).

GALA well approximates the oracle layer selection GALA substantially improves over All Blocks, Worst Block, and Random Block method. In some sense, Best Block method acts as an empirical upper-bound performance if we have access to a target domain with labels while incurring the high computational cost of brute forcing over individual layers of the model. GALA comes close to this upper bound performance without requiring any target labels using a cheap layer selection criterion. As a result, GALA also effectively balances computational cost with performance.

GALA is more conservative than the oracle GALA selects the layers for adaptation with the most aligned gradients. It can stop adaptation if the gradients are noisy or no longer aligned to prevent further degradation. In Tab. 3, we see that it may have aggressively stopped a few useful updates compared to the Best Block, our empirical upper bound. As a result, it gets much better at avoiding forgetting but is a bit lower on TTA accuracy and generalization.

GALA tends to select more often the blocks with better accuracy Oracle TTA performance, as measured in Fig. 4, ranks the four blocks for each configuration. Similarly,

Table 4: Accuracy $( \% )$ under different experimental conditions. The values are averaged on Domainbed for the first four settings and Continual TTA for the last.   

<html><body><table><tr><td>Setting</td><td>Condition</td><td>Accuracy ↑</td></tr><tr><td rowspan="3">Partitioning</td><td>Single block</td><td>67.64</td></tr><tr><td>Single layer</td><td>68.57</td></tr><tr><td>Multiple layers</td><td>66.48</td></tr><tr><td rowspan="3">Window Size</td><td>5</td><td>68.46</td></tr><tr><td>20</td><td>68.57</td></tr><tr><td>8</td><td>68.37</td></tr><tr><td rowspan="3">Threshold</td><td>0.5</td><td>68.57</td></tr><tr><td>0.75</td><td>68.57</td></tr><tr><td>0.99</td><td>68.6</td></tr><tr><td rowspan="2">Batch Size = 1</td><td>All Layers</td><td>33.47</td></tr><tr><td>GALA</td><td>67.28</td></tr><tr><td rowspan="2">Continual TTA</td><td>No Reset</td><td></td></tr><tr><td>With Reset</td><td>69.9 71.1</td></tr></table></body></html>

GALA chooses to update each layer with a particular frequency during TTA, leading to a ranking of the four blocks. We assess the relationship between these two different ways of ranking blocks using Spearman rank correlation and find $\rho = 0 . 7 6$ (cf. Tab. 3), which seems to indicate that the selection strategy used by GALA is a good proxy for the oracle TTA performance achieved when adapting always the same layer.

# 5 Analysis of GALA

In this section, we evaluate the impact of different design choices and hyperparameters of GALA in Tab. 4, supporting choices presented in Tab. 1. For the partitioning setting, Single block means a single block of many layers is updated at each iteration, Single layer corresponds to the best layer selected for the update, and Multiple layers corresponds to individually best layers selected for the update based on the cosine distance and the threshold. Also, a window size of $\infty$ implies no reset. Some important observations stemming from Tab. 4:

• Layer granularity performs better than block granularity. At layer granularity, GALA has better fine-grained control over choosing the layers to adapt, improving performances in all cases tested.   
• Adaptation with the best single layer is much better than with the best multiple layers. Cosine distance can correctly identify the single best layer to train, although it may still struggle to determine the best set of multiple layers to update.   
• Optimal reset-window size can improve performance. We see that a reset window size of 20 works reasonably well across the backbones and the TTA losses tested on Domainbed.   
• The choice of selection threshold is not very sensitive. A threshold of 0.75 seems to work across the board without being too restrictive.

In the following section, we briefly analyze some aspects of the proposed approach.

![](images/d4ca20baefb53d748a43e3dc177dc11271807c072f57c26a9bae5b791938b314.jpg)  
Figure 5: Effect of magnitude of $u$ on cosine distance criterion. Left: Consider two vectors such that $u _ { 1 }$ is smaller than $u _ { 2 }$ but is better aligned with its displacement. For large displacements $( T )$ , alignment becomes crucial and GALA selects $u _ { 2 }$ . For small displacements $( T ^ { ' } )$ , the update’s magnitude can dominate the criterion, and GALA selects $u _ { 1 }$ . Middle and Right: Plot of cosine metric values with level curves. Alignment prevails for small updates compared to the total displacement (Middle). But, for updates with large magnitude compared to total displacement (Right), large cosine values can be obtained even for misaligned updates.

Proposed cosine distance criterion effectively balances gradient magnitude and direction. Let us first rewrite the GALA criterion in Eq. 5 for a given layer $l$ in terms of ${ \cal T } = \| { \bf T } { \bf D } _ { i - 1 , l } \|$ , $u \ = \ \Vert \mathbf { u } _ { i , l } \Vert$ and the angle $\beta$ between $\mathbf { T } \mathbf { D } _ { i - 1 , l }$ and $\mathbf { u } _ { i , l }$ . Using the Pythagorean theorem, we obtain:

$$
\cos ( \alpha ) = \frac { T \cos ( \beta ) + u } { \sqrt { ( T + u \cos ( \beta ) ) ^ { 2 } + ( u \sin ( \beta ) ) ^ { 2 } } } .
$$

We observe that our criterion depends on the norm $T$ of the total displacement, the norm $u$ of the update, and their alignment, given by the angle $\beta$ between these vectors. Fig. 5 shows the cosine metric plots. We see that while alignment is crucial for large displacements, the update’s magnitude can also dominate for small displacements. For example, consider two layers with the same norm $T$ but different updates $\mathbf { u } _ { 1 }$ and ${ \bf u } _ { 2 }$ . If $\left\| \mathbf { u } _ { 1 } \right\|$ is smaller than $\left. \mathbf { u } _ { 2 } \right.$ but $\mathbf { u } _ { 1 }$ is more aligned with its displacement, two scenarios arise:

1. For larger $T$ , GALA selects layer 1, favoring the alignment and exploiting the learned direction. This scenario would seem more common during TTA.   
2. For small $T$ , GALA selects layer 2, favoring the magnitude, and can explore over different directions. This can occur for initial samples.

Consequently, GALA effectively balances the gradient magnitude and the direction of gradients for selecting the best layer. More discussion is in Appendix A.4.

Proposed layer selection framework offers a more flexible adaptation strategy for TTA. The selection of layers in existing TTA approaches typically remains unchanged across different shifts. On the other hand, sample selectionbased TTA (Niu et al. 2022) approaches aim to improve performance by skipping the adaptation of all layers on a few unreliable samples. Based on Eq. 3 , we can see that GALA is more flexible and general than the existing layer selection and sample selection strategies in TTA for performing layerwise adaptation.

Reset mechanism seems beneficial in multi-domain shift settings. Comparing GALA with and without reset on Tab. 4, we see that while reset yields only marginal improvement on Domainbed, a single-domain shift benchmark, its benefits are more evident on a multi-shift benchmark like Continual TTA. This indicates that the reset mechanism’s ability to facilitate slight adjustments in the overall gradient update direction may be advantageous in a continuously changing testing domain.

GALA is quite robust on single sample adaptation. In Tab. 4, we show that in the adverse setting of batch size of 1, while existing TTA approaches witness severe performance degradation, GALA improves on all layers baseline on Domainbed.

# 6 Conclusion

In this paper, we introduce Gradient Aligned Layer Adaptation (GALA), a novel layer selection framework explicitly designed for Test Time Adaptation (TTA). Our comprehensive study reveals that layers in neural networks exhibit varying receptiveness to adaptation, and the optimal set of layers for adaptation depends on both the specific distribution shift and the loss function employed during inference. Building on these insights, we propose GALA, a dynamic layer selection criterion that ranks layers based on gradient alignment, effectively mitigating overfitting and performance degradation. Extensive experiments across diverse datasets, model architectures, and TTA losses demonstrate GALA’s superior performance compared to existing methods, including standard ERM, all-layers adaptation, and other layer selection baselines.

# Acknowledgments

This work is supported by the DEEL Project CRDPJ 537462-18 funded by the Natural Sciences and Engineering Research Council of Canada (NSERC) and the Consortium for Research and Innovation in Aerospace in Que´bec (CRIAQ), together with its industrial partners Thales Canada inc, Bell Textron Canada Limited, CAE inc and Bombardier inc. Computations were made on the cedar, and beluga supercomputers, managed by Calcul Que´bec and the Digital Research Alliance of Canada (Alliance). We extend our gratitude to the members of the #lunch-at-mila and #deel ood for their valuable input, with special thanks to Vineetha Kondameedi for her essential feedback in enhancing the quality of this paper.
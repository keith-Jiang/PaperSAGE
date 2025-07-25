# Region-Based Optimization in Continual Learning for Audio Deepfake Detection

Yujie Chen 1, Jiangyan Yi 3,\*, Cunhang Fan 1, Jianhua Tao 3, 4, Yong Ren 2, Siding Zeng 2, Chu Yuan Zhang 3, Xinrui Yan 2, Hao Gu 2, Jun Xue 1, Chenglong Wang 2, Zhao Lv 1, Xiaohui Zhang 2,\*

1 School of Computer Science and Technology, Anhui University, China 2 Institute of Automation, Chinese Academy of Sciences 3 Department of Automation, Tsinghua University 4 Beijing National Research Center for lnformation Science and Technology, Tsinghua University e22201148@stu.ahu.edu.cn, yijy $@$ tsinghua.edu.cn

# Abstract

Rapid advancements in speech synthesis and voice conversion bring convenience but also new security risks, creating an urgent need for effective audio deepfake detection. Although current models perform well, their effectiveness diminishes when confronted with the diverse and evolving nature of real-world deepfakes. To address this issue, we propose a continual learning method named Region-Based Optimization (RegO) for audio deepfake detection. Specifically, we use the Fisher information matrix to measure important neuron regions for real and fake audio detection, dividing them into four regions. First, we directly fine-tune the less important regions to quickly adapt to new tasks. Next, we apply gradient optimization in parallel for regions important only to real audio detection, and in orthogonal directions for regions important only to fake audio detection. For regions that are important to both, we use sample proportion-based adaptive gradient optimization. This region-adaptive optimization ensures an appropriate trade-off between memory stability and learning plasticity. Additionally, to address the increase of redundant neurons from old tasks, we further introduce the Ebbinghaus forgetting mechanism to release them, thereby promoting the model’s ability to learn more generalized discriminative features. Experimental results show our method achieves a 21.3 percent improvement in EER over the stateof-the-art continual learning approach RWM for audio deepfake detection. Moreover, the effectiveness of $\mathtt { R e g O }$ extends beyond the audio deepfake detection domain, showing potential significance in other tasks, such as image recognition.

# Introduction

Recently, with the rapid development of speech synthesis and voice conversion technologies, the distinction between real and fake audio has become increasingly blurred, posing significant security risks to society. Consequently, researchers are increasingly focusing on audio deepfake detection mechanisms (Yi et al. 2023b). Community-led initiatives, such as the ASVspoof Challenges (Kinnunen et al. 2017; Todisco et al. 2019; Yamagishi et al. 2021) and the Audio Deepfake Detection (ADD) Challenges (Yi et al. 2022, 2023a), significantly advance the field of fake audio detection. In addition, the introduction of pre-trained audio models significantly improves the effectiveness of audio deepfake detection, achieving impressive performance on publicly available datasets. (Tak et al. 2022; Wang and Yamagishi 2022; Hsu et al. 2021)

Despite the significant advancements in audio deepfake detection models, their performance is still limited when confronted with diverse and unseen forged audio in realworld scenarios. To address this challenge, two primary approaches have been developed. The first approach utilizes data augmentation and multi-feature fusion techniques to extract robust audio features, improving the generalization of model across various datasets (Wang et al. 2023; Fan et al. 2024). The second approach is based on continual learning (Ma et al. 2021), where the model incrementally learns from both new and old datasets, allowing it to integrate previously learned discriminative information. This enhances its detection capability for diverse and unseen deepfakes, achieving a balance between memory stability (the model’s ability to retain performance on old tasks) and learning plasticity (the model’s ability to perform on new tasks). Currently, the most advanced continual learning methods for audio deepfake detection, Regularized Adaptive Weight Modification (RAWM) (Zhang et al. 2023) and Radian Weight Modification (RWM) (Zhang et al. 2024), overcome catastrophic forgetting by introducing trainable gradient correction directions to optimize weights. While RAWM and RWM exhibit notable effectiveness in overcoming catastrophic forgetting, the use of Recursive Least Squares (RLS) (Shah, Palmieri, and Datum 1992) to approximate the gradient plane of previous tasks can introduce errors (Liavas and Regalia 1999). Moreover, applying gradient modification to all neurons restricts the model’s learning plasticity for new tasks.

To address the aforementioned issues, we propose a continual learning method for audio deepfake detection, named Region-Based Optimization (RegO). Under the same acoustic environments, real audio exhibits a more compact feature distribution compared to fake audio (Ma et al. 2021; Zhang et al. 2023, 2024), so they can be seen as a whole from the same dataset. Based on this observation, we propose that for new tasks, gradient updates for real audio should be parallel to the gradient directions of previous tasks, while the updates for fake audio should be orthogonal to the previous task gradients. Specifically, we utilize the Fisher information matrix (FIM) to measure the importance of neurons in the model, and then calculate the FIM for both real and fake audio detection separately. Following the fundamental principle of not constraining unimportant neurons to allow the model to quickly adapt to new tasks, and optimizing important neurons to overcome catastrophic forgetting, we divide the neurons into four regions for fine-grained region-adaptive gradient optimization: neurons that are unimportant for both real and fake audio detection are finetune directly to quickly adapt to new tasks; neurons that are important only for real audio detection are updated in parallel to the previous task gradients; neurons important only for fake audio detection are updated orthogonally to the previous task gradients; and neurons important for both real and fake audio detection are optimized through adaptive gradient updates based on the ratio of real to fake samples to maintain memory stability.

However, when the number of neurons in the model remains fixed as tasks increase, two problems arise: First, the number of neurons in less important regions decreases, making it progressively harder for the model to adapt to new tasks. Second, redundant neurons appear—those that are beneficial for only a few specific tasks but ineffective for others. To address these challenges, we draw inspiration from the non-linear nature of human memory forgetting (Loftus 1985; Ebbinghaus 2013). Over time, the memories that persist are generally those involving deeply understood knowledge, while other redundant Knowledge are forgotten. Based on this principle, we further propose a neuron forgetting mechanism inspired by the Ebbinghaus forgetting curve (Wo´zniak, Gorzelan´czyk, and Murakowski 1995) to release redundant neurons from previous tasks. This mechanism enables the model to learn knowledge from new tasks more efficiently, ensuring quicker adaptation to new tasks and the acquisition of more generalized discriminative information.

Our experiments on the Evolving Deepfake Audio (EVDA) benchmark (Zhang, Yi, and Tao 2024) demonstrate that our method outperforms several mainstream continual learning methods and state-of-the-art continual audio deepfake detection methods, including RAWM and RWM, in terms of balancing stability and plasticity. Furthermore, our method can be easily extended to other domains. General study conducted on image recognition tasks also showed competitive results, highlighting the potential significance of our approach across different machine learning fields.

In summary, we make the following contributions:

• We propose a continual learning method for audio deepfake detection, named RegO, which partitions the neural network parameter space into four regions using the FIM. This method facilitates fine-grained, region-adaptive gradient optimization, ensuring an optimal trade-off between memory stability and learning plasticity. • We further propose a neuron forgetting mechanism based on Ebbinghaus forgetting curve, which releases redundant neurons from previous tasks to ensure faster adaptation to new tasks and to learn more generalizable discriminative information. • We conduct extensive experiments on the EVDA benchmark to validate the effectiveness of our method. Furthermore, we perform general study, and the results indicate that our approach holds potential significance in other domains, such as image recognition, without being limited to a specific field.

# Related Work

Continual learning is a machine learning paradigm that aims to enable models to retain and use previously learned knowledge while continuously learning new tasks, thereby overcoming catastrophic forgetting. Current mainstream methods can be categorized into the following types: Regularization-based methods, which balance new and old tasks by selectively adding regularization terms to the changes in network parameters, such as Elastic Weight Consolidation (EWC) (Kirkpatrick et al. 2017), Synaptic Intelligence (SI) (Zenke, Poole, and Ganguli 2017), Gradient Episodic Memory (GEM) (Lopez-Paz and Ranzato 2017), Orthogonal Weight Modification (OWM) (Zeng et al. 2019) etc. (Qiao et al. 2024; Elsayed and Mahmood 2023) Replay-based methods, which carefully select training samples from previous tasks, store them in a buffer, and mix them with new task training samples to ensure accuracy for old tasks, such as Greedy Sampler and Dumb Learner (GDumb) (Prabhu, Torr, and Dokania 2020) and CWRStar (Lomonaco et al. 2020); Architecture-based methods, which use parameter isolation and dynamic expansion of the parameter space to protect previously acquired knowledge. (Wang et al. 2022; Razdaibiedina et al. 2023)

Mainstream continual learning methods have achieved significant success in various fields, including image classification (Razdaibiedina et al. 2023; Yoo et al. 2024), object detection (Menezes et al. 2023), and semantic segmentation (Camuffo and Milani 2023; Zhu et al. 2023). Continual learning methods are also effective in audio deepfake detection, empowering models to recognize diverse and unseen fake audio by leveraging continual learning principles. Currently, most continual learning methods for audio deepfake detection are regularization-based, such as Detecting Fake Without Forgetting (DFWF) (Ma et al. 2021), RAWM (Zhang et al. 2023), and RWM (Zhang et al. 2024), and have demonstrated impressive results. However, the approximations made by regularization methods can lead to cumulative errors during continual learning, affecting the balance between model stability and plasticity. In contrast, our method selectively adjusts the gradients of parameters that are crucial for previous tasks while fine-tuning the remaining parameters to learn new tasks. This allows our method to adapt more quickly to new tasks while preventing catastrophic forgetting of old ones.

# Methodology

In this section, the Region-Based Optimization (RegO) architecture, as illustrated in Figure 1, encompasses the principles and implementation of three core modules: Importance Region Localization (IRL), Region-Adaptive Optimization (RAO), and the Ebbinghaus Forgetting Mechanism (EFM).

![](images/a098857841175faa1700e6de7ca9e59e9a2e8bb062d059332f33ba5a4db291e2.jpg)  
Figure 1: Illustration of RegO Architecture. (i).After the training of each task, we calculate the region matrix $\mathbf { R }$ through the IRL module. (ii).From the second task onward, gradient optimization is performed during backpropagation (shown by dark red arrows and boxes). The $\mathbf { E } _ { k }$ is obtained through the EFM module and then combined with the historical $\mathbf { R }$ to generate $\overline { { \mathbf { R } } }$ . (iii).Based on $\overline { { \mathbf { R } } }$ , the RAO module updates the model weights as follows: Region A: fine-tuning (i.e. $g )$ ; Region B: gradient update in the projection direction (i.e. $g _ { p } \mathrm { , }$ ); Region C: gradient update in the orthogonal direction (i.e. $g _ { o }$ ); Region D: adaptive gradient update based on the number of samples (i.e. $\widetilde { g } ,$ ).

# Definitions and Notation

In Continual Learning, we define a sequence of tasks $\{ T _ { 1 } , T _ { 2 } , T _ { 3 } , \dots , T _ { N } \}$ of $N$ tasks. For the $k$ -th task $T _ { k }$ , there is a corresponding trainin et $\mathcal { D } _ { k } \ = \ \{ ( x _ { k } ^ { i } , y _ { k } ^ { i } ) \} _ { i = 1 } ^ { N _ { k } }$ and $\theta _ { k }$ on input $x$ is denoted by $f ( x ; \theta _ { k } )$ . During the training of task $k$ , we define the loss function as follows:

$$
\mathcal { L } ( \theta _ { k } , \mathcal { D } _ { k } ) = \frac { 1 } { | \mathcal { D } _ { k } | } \sum _ { ( x , y ) \in \mathcal { D } _ { k } } C E ( f ( x ; \theta _ { k } ) , y )
$$

where $C E$ is the standard cross-entropy loss. The gradient is shown as follows:

$$
g _ { k } = \nabla _ { \boldsymbol { \theta } _ { k } } \mathcal { L } ( \boldsymbol { \theta } _ { k } , D _ { k } )
$$

# Importance Region Localization

First, we need to identify the neuron regions that are important for the previous tasks. Inspired by (Kirkpatrick et al. 2017; Husza´r 2018), we choose the Fisher Information Matrix (FIM) as a measure of neuron importance. After completing the training of the $k$ -th task, we pass real and fake audio through the model separately to calculate the corresponding FIM, which encapsulate the importance measures of the weights for both real and fake audio detection. The FIM is defined as follows:

$$
\mathbf { F } _ { k } ^ { ( c ) } = \mathbb { E } \left[ \nabla _ { \theta } \log p ( D _ { k } ^ { ( c ) } | \theta ) \nabla _ { \theta } \log p ( D _ { k } ^ { ( c ) } | \theta ) ^ { \top } \Big | _ { \theta = \theta _ { k } ^ { * } } \right]
$$

Here, $c$ is 0 (fake) or 1 (real), $D _ { k }$ represents the training dataset corresponding to $T _ { k }$ , and $\theta _ { k } ^ { * }$ denotes the optimal parameters for $T _ { k }$ . Note that the log probability of the data $D _ { k } ^ { ( c ) }$ given the parameters lo $\mathrm { g } p ( D _ { k } ^ { \overline { { ( c ) } } } | \boldsymbol { \theta } )$ is simply the negative of the loss function $- \mathcal { L } ( \theta _ { k } , \widetilde { \mathcal { D } _ { k } } )$ for task $T _ { k }$ . Based on this, by setting the threshold $\alpha$ , we locate important neurons, resulting in a localization matrix, as shown in Equation 4.

$$
\mathbf { L } _ { k } ^ { ( c ) } [ i ] [ j ] = \left\{ \begin{array} { l l } { 2 } & { \mathrm { i f } c = 0 \mathrm { a n d } \mathbf { F } _ { k } ^ { ( c ) } [ i ] [ j ] \geq \mathrm { P } _ { \alpha } ( \mathbf { F } _ { k } ^ { ( c ) } ) } \\ { 1 } & { \mathrm { i f } c = 1 \mathrm { a n d } \mathbf { F } _ { k } ^ { ( c ) } [ i ] [ j ] \geq \mathrm { P } _ { \alpha } ( \mathbf { F } _ { k } ^ { ( c ) } ) } \\ { 0 } & { \mathrm { o t h e r w i s e } } \end{array} \right.
$$

where $i , j$ represents the neuron index, $P _ { \alpha }$ represents the $\alpha$ - percentile. Based on this, by summing the localization matri

ces for both real and fake audio, we can identify four distinct regions, as shown in Equation 5.

$$
\mathbf { R } _ { k } [ i ] [ j ] = \mathbf { L } _ { k } ^ { 0 } [ i ] [ j ] + \mathbf { L } _ { k } ^ { 1 } [ i ] [ j ]
$$

In the $\mathbf { R } _ { k }$ , regions with values $0 , 1 , 2$ , and 3 are denoted by the letters A, B, $\mathbf { C }$ , and D, respectively. Region A represents neurons that are not important for $T _ { k }$ . Region B represents neurons that are important for real audio detection in $T _ { k }$ . Region C represents neurons that are important for fake audio detection in $T _ { k }$ . Region D represents neurons that are important for both real and fake audio detection in $T _ { k }$ .

# Region-Adaptive Optimization

During training, starting from the second task, we merge all region matrices $\mathbf { R }$ from the previous tasks to obtain $\overline { { \mathbf { R } } }$ . For the four regions $A , B , C$ , and $D$ within $\overline { { \mathbf { R } } }$ , we adaptively optimize each region according to the following principles.

Firstly, because the neurons in region A have minimal impact on previous tasks, but are likely to become important for new tasks, we allow them to quickly adapt and learn the knowledge of the new tasks. Therefore, we do not apply additional gradient optimization to the neurons in region A and instead update them using fine-tuning, the gradient of region A is defined in Equation 6.

$$
g _ { A } = g \odot \mathbb { I } _ { \{ \overline { { \mathbf { R } } } [ i ] [ j ] = 0 \} }
$$

where $\odot$ denote the Hadamard product, $\mathbb { I } _ { \{ \overline { { \mathbf { R } } } [ i ] [ j ] = 0 \} }$ is an indicator function that takes a value of 1 when $\overline { { \mathbf { R } } } [ i ] [ j ]$ equals 0 and 0 otherwise.

Second, as mentioned above, real audio has a more compact feature distribution compared to fake audio and can be considered as coming from the same dataset. Therefore, to largely retain the knowledge of real audio detection from previous tasks, the neurons in region $\mathbf { B }$ should project the current gradient $g$ onto the direction of the old task gradient $\hat { g }$ for gradient optimization, as shown in Equation 7.

$$
\begin{array} { r l } & { { { g } _ { p } } = \frac { { { g } \cdot { \hat { g } } } } { \| \hat { g } \| ^ { 2 } } \hat { g } } \\ & { { { g } _ { B } } = { { g } _ { p } } \odot \mathbb { I } _ { \{ \overline { { \mathbf { R } } } [ i ] [ j ] = 1 \} } } \end{array}
$$

Thirdly, due to the diversity of speech synthesis and voice conversion methods, there is a wide variance in feature distributions among fake audio samples. Therefore, to reduce the forgetting of discriminative information about fake audio from previous tasks while learning discriminative information for fake audio in new tasks, we update the gradient direction of neurons in region C to be orthogonal to the gradient direction of the old tasks, as shown in Equation 8.

$$
\begin{array} { c } { g _ { o } = g - g _ { p } } \\ { g _ { C } = g _ { o } \odot \mathbb { I } _ { \{ \overline { { \mathbf { R } } } [ i ] [ j ] = 2 \} } } \end{array}
$$

Fourth, region $\mathrm { ~ D ~ }$ is crucial for both real and fake audio discrimination from previous tasks. Therefore, to balance the retention of knowledge for both real and fake audio detection in neurons of region D, we need to optimize the gradient update direction to achieve an optimal trade-off.

# Require: Training data from different datasets, $\eta$ (learning rate), $\mathcal { R }$ (Region matrix set)

Algorithm 1: Region-Based Optimization   

<html><body><table><tr><td colspan="2">1:for every dataset k do</td></tr><tr><td>2:</td><td>for every batch b do</td></tr><tr><td>3:</td><td>if k=1 then</td></tr><tr><td>4:</td><td>Update0k:0k ←0k-nw</td></tr><tr><td>5:</td><td>else</td></tr><tr><td>6:</td><td>Compute memory matrix Mk by Equ.(13)</td></tr><tr><td>7:</td><td>Compute Ebbinghaus matrix Ek by Equ.(14)</td></tr><tr><td>8:</td><td>Compute Rk by combining region matrix set R</td></tr><tr><td>9:</td><td>gA ← g①I{R[][j]=0}</td></tr><tr><td>10:</td><td>9p←g</td></tr><tr><td>11:</td><td>gB←gpI{R[[i]=1}</td></tr><tr><td>12:</td><td>go←g-9p</td></tr><tr><td>13:</td><td>gc ←goIR[i[]=2}</td></tr><tr><td>14:</td><td>β↑ ∑=N ∑N</td></tr><tr><td>15:</td><td>g←β*gp+(1-β)*go</td></tr><tr><td>16:</td><td>gD ←gI{Rk[[j]=3}</td></tr><tr><td>17:</td><td>Initialization: w ←0</td></tr><tr><td>18:</td><td>w ←gA+gB+gc+gD</td></tr><tr><td>19:</td><td>Update 0k: 0k ←0k- nw</td></tr><tr><td>20:</td><td>end if</td></tr><tr><td>21:</td><td>end for</td></tr><tr><td>22:</td><td>Compute the k-th Region Matrix Rk by Equ.(1)(2)(3)</td></tr><tr><td>23: 24: end for</td><td>R←Rk</td></tr></table></body></html>

Specifically, we adaptively determine whether the gradient update direction should lean more towards the projection direction or the orthogonal direction based on the proportion of real and fake audio samples, as shown in Equations 9 and 10.

$$
\beta = \frac { \sum _ { l = 1 } ^ { u } N ^ { l } } { \sum _ { l = 1 } ^ { u + v } N ^ { l } }
$$

$$
\widetilde { g } = \beta * g _ { p } + ( 1 - \beta ) * g _ { o }
$$

$$
g _ { D } = \widetilde { g } \odot \mathbb { I } _ { \{ \overline { { \mathbf { R } } } [ i ] [ j ] = 3 \} }
$$

In Equation 9, $u$ and $v$ repr esent the number of classes with similar feature distributions and the remaining classes, respectively. In deepfake audio detection, $u$ and $v$ are both set to 1, indicating the two classes of real and fake audio. In image recognition, $u$ and $\boldsymbol { v }$ represent the number of classes with similar feature distributions and the number of classes with dissimilar feature distributions, respectively. $N ^ { l }$ denotes the number of samples in the batch for the $l$ -th class.

Finally, the total gradient update for a batch is defined as:

$$
w = g _ { A } + g _ { B } + g _ { C } + g _ { D }
$$

# Ebbinghaus Forgetting Mechanism

During the continual learning process, when the number of neurons remains constant, neurons in region A gradually diminish, and redundant neurons that only benefit individual tasks begin to emerge, which undermines the model’s adaptability and generalization ability. To address this issue, inspired by Ebbinghaus forgetting theory (Loftus 1985; Ebbinghaus 2013), we propose a neuron forgetting mechanism based on the Ebbinghaus memory curve (Woz´niak, Gorzelan´czyk, and Murakowski 1995). The approximation function is defined as Equation 12.

<html><body><table><tr><td rowspan="2">Continual Learning Methods</td><td colspan="8">EER(↓) on each experience</td></tr><tr><td>Exp1</td><td>Exp2</td><td>Exp3</td><td>Exp4</td><td>Exp5</td><td>Exp6</td><td>Exp7</td><td>Exp8 Avg</td></tr><tr><td>Replay-All</td><td>2.80</td><td>5.68</td><td>1.52</td><td>0.76</td><td>1.84</td><td>7.96 5.76</td><td>2.56</td><td>3.61</td></tr><tr><td>Finetune-Exp1</td><td>2.20</td><td>24.80</td><td>23.16</td><td>16.84</td><td>23.80 34.12</td><td>26.44</td><td>15.52</td><td>20.86</td></tr><tr><td>Finetune</td><td>5.16</td><td>15.56</td><td>8.20</td><td>2.32</td><td>4.08 21.72</td><td>9.64</td><td>3.04</td><td>8.72</td></tr><tr><td>EWC</td><td>3.72</td><td>13.92</td><td>7.32</td><td>2.12</td><td>3.56</td><td>17.40 10.24</td><td>3.16</td><td>7.68</td></tr><tr><td>GDumb</td><td>4.72</td><td>14.12</td><td>7.32</td><td>4.60</td><td>6.56</td><td>24.28 15.28</td><td>11.40</td><td>11.03</td></tr><tr><td>GEM</td><td>5.60</td><td>16.56</td><td>6.28</td><td>2.60</td><td>9.60</td><td>24.44 11.88</td><td>4.28</td><td>10.15</td></tr><tr><td>CWRStar</td><td>5.12</td><td>27.92</td><td>22.88</td><td>29.36</td><td>45.52</td><td>43.20 49.92</td><td>18.32</td><td>30.28</td></tr><tr><td>SI</td><td>6.96</td><td>10.88</td><td>5.92</td><td>1.60</td><td>4.04</td><td>18.96</td><td>10.04 3.32</td><td>7.71</td></tr><tr><td>OWM</td><td>27.28</td><td>33.72</td><td>29.32</td><td>33.12</td><td>47.28</td><td>49.52</td><td>48.80 26.32</td><td>36.92</td></tr><tr><td>RAWM</td><td>9.28</td><td>16.04</td><td>6.76</td><td>2.60</td><td>3.60</td><td>19.52 9.64</td><td>3.40</td><td>8.85</td></tr><tr><td>RWM</td><td>4.44</td><td>14.92</td><td>6.28</td><td>1.92</td><td>4.44</td><td>18.92</td><td>10.04 3.52</td><td>8.06</td></tr><tr><td>RegO (Ours)</td><td>4.36</td><td>10.64</td><td>3.76</td><td>1.20</td><td>3.16</td><td>15.72</td><td>9.16</td><td>2.72 6.34</td></tr></table></body></html>

Table 1: The EER $( \% )$ of our method compared with various methods.

$$
\phi ( t ) = e ^ { - \frac { t } { k } }
$$

where $t$ represents the time steps and $k$ denote the $k$ -th task. Specifically, we define the Ebbinghaus forgetting curve function based on the number of tasks processed so far, and calculate the memory weights using this forgetting curve function. Then, we allocate memory weights to the region matrix $\mathbf { R }$ of the old tasks. By Equation 13 and 14, we compute the memory matrix $\mathbf { M }$ , which contains the accumulated memory weights for each neuron. Finally, we set a threshold $\gamma$ . When the memory weight of a neuron is less than $\gamma$ , that neuron is released, resulting in the Ebbinghaus matrix $\mathbf { E }$ .

$$
\mathbf { M } _ { k } [ i ] [ j ] = \sum _ { t = 1 } ^ { k - 1 } \phi ( t ) * \mathbb { I } _ { \{ \mathbf { R } _ { t } [ i ] [ j ] \in \{ 0 , 1 , 2 , 3 \} \} }
$$

$$
\mathbf { E } _ { k } [ i ] [ j ] = \left\{ { \begin{array} { l l } { 1 } & { { \mathrm { i f } } \mathbf { M } _ { k } [ i ] [ j ] > \gamma } \\ { 0 } & { { \mathrm { i f } } \mathbf { M } _ { k } [ i ] [ j ] \leq \gamma } \end{array} } \right.
$$

# Experiments

We conduct a series of experiments to evaluate the effectiveness of our approach. The experiments are performed on a continual learning benchmark EVDA (Zhang, Yi, and Tao 2024) for speech deepfake detection, which includes eight publicly available and popular datasets specifically designed for incremental synthesis algorithm audio deepfake detection. Additionally, we carry out a general study in the field of image recognition using the well-established continual learning benchmark, CLEAR (Lin et al. 2021).

# Experimental Setup

Datasets and Metrics In this paper, we refer to each dataset as “Exp” (e.g., $\mathrm { E x p } _ { 1 }$ , $\mathrm { E x p _ { 2 } , . . . , E x p _ { 1 0 } ) }$ , representing the different datasets used in our experiments. The EVDA benchmark from $\mathrm { E x p } _ { 1 }$ to $\mathrm { E x p } _ { 8 }$ are FMFCC (Zhang et al. 2021), In the Wild (Mu¨ ller et al. 2022), ADD 2022 (Yi et al. 2022), ASVspoof2015 (Wu et al. 2017), ASVspoof2019 (Todisco et al. 2019), ASVspoof2021 (Yamagishi et al. 2021), FoR (Reimao and Tzerpos 2019), and HAD (Yi et al. 2021). For the EVDA baseline, 2000 samples are randomly sampled from each dataset as the training set, and 5000 samples are sampled as the test set. The EVDA baseline dataset configuration includes cross-lingual (Chinese and English) and cross-task (whole-segment and partial-segment fake detection) scenarios to simulate the unseen and diverse realworld forgery conditions. The final model in this study refers to the model trained sequentially on these eight datasets and evaluated on each dataset. We use the standard metric Equal Error Rate (EER) (Wu et al. 2017) in the field of audio deepfake detection to evaluate the performance of our model.

Model We employ the pre-trained speech model Wav2vec 2.0 (Baevski et al. 2020) as the feature extractor, the parameters of Wav2vec 2.0 are loaded from the pretrained model XLSR-53 (Conneau et al. 2021). Given the robustness of the speech features obtained from the pre-trained model, we opt for a 5-layer SimpleMlp as the backend, which consists of fully connected layers with the following dimensions: 1024 to 512, 512 to 512 (x3), and 512 to 2. The code is available at https://github.com/cyjie429/RegO.

Training Details We use the Adam optimizer to finetune the SimpleMlp, with a learning rate $\eta$ of 0.0001 and a batch size of 32, on an NVIDIA A100 80GB GPU. To evaluate the performance of our method, we compare it with six widely used continual learning methods, finetuning, and two advanced continual learning methods specifically designed for audio deepfake detection: RAWM (Zhang et al. 2023), RWM (Zhang et al. 2024). Additionally, we present the training results on all datasets (Replay-All), which are considered the lower bound for all mentioned continual learning methods (Parisi et al. 2019).

<html><body><table><tr><td rowspan="2">Ablation Study</td><td colspan="9">EER（↓) on each experience</td></tr><tr><td>Exp1</td><td>Exp2</td><td>Exp3</td><td>Exp4</td><td>Exp5</td><td>Exp6</td><td>Exp7</td><td>Exp8</td><td>Avg</td></tr><tr><td>RegO (Ours)</td><td>4.36</td><td>10.64</td><td>3.76</td><td>1.20</td><td>3.16</td><td>15.72</td><td>9.16</td><td>2.72</td><td>6.34</td></tr><tr><td>W/o EFM</td><td>4.48</td><td>10.48</td><td>5.56</td><td>1.32</td><td>3.12</td><td>16.48</td><td>9.68</td><td>3.32</td><td>6.80</td></tr><tr><td>W/o IRL</td><td>5.32</td><td>12.28</td><td>4.72</td><td>1.52</td><td>3.48</td><td>19.08</td><td>9.76</td><td>2.92</td><td>7.38</td></tr><tr><td>w/o RAO</td><td>4.68</td><td>14.68</td><td>5.52</td><td>2.48</td><td>4.44</td><td>16.16</td><td>9.80</td><td>3.32</td><td>7.63</td></tr></table></body></html>

Table 2: The EER $( \% )$ results of the ablation study for our method.

# Comparison with Other Methods

In this experiment, we compare RegO with other methods. Here, Finetune- $\cdot \mathrm { E x p } _ { 1 }$ shows the results of training on $\mathrm { E x p } _ { 1 }$ and evaluating on the other Exps, highlighting the significant differences between the various Exps. As shown in Table 1, after training on 8 Exps, our method achieves the best performance on 7 of the Exps and the second-best performance on Exp1. The overall evaluation metrics demonstrate that our method is only slightly inferior compared to Replay-ALL, which is considered the upper bound for continual learning performance. Additionally, among the eight Exps, $\mathrm { E x p } _ { 1 }$ , $\mathrm { E x p } _ { 3 }$ , and $\mathrm { E x p } _ { 8 }$ are Chinese datasets, while the remaining ones are English datasets. Notably, $\mathrm { E x p } _ { 3 }$ consists of low-quality speech data, and $\mathrm { E x p } _ { 8 }$ is a partial-fake spoofing dataset. The experimental results demonstrate that our method shows promising potential in both cross-lingual and cross-task scenarios, indicating its capability to handle relatively diverse real-world audio deepfake environments.

# Ablation Study

We conduct an ablation study to evaluate the effectiveness of the proposed modules. The results, shown in Table 3, are as follows: “w/o EFM” denotes the removal of the Ebbinghaus forgetting mechanism, “w/o IRL” indicates no division between real and fake regions, and “w/o RAO” refers to applying orthogonal gradient optimization to all weights. The results for “w/o RAO” suggest that, compared to orthogonally optimizing all weights, applying region-adaptive gradient optimization to critical regions alone achieves a better balance between model memory stability and learning plasticity. Additionally, we observe that “w/o EFM” performs better than RegO on some of the earlier datasets, but its overall capability is inferior to RegO. We attribute this to the significant differences among the eight audio deepfake detection datasets, which result in a substantial number of redundant neurons. The EFM module effectively filters out these redundant neurons, enabling the model to adapt more quickly to other tasks.

# Hyperparameter Study

We conduct a hyperparameter study to evaluate the impact of $\alpha$ and $\gamma$ on our method RegO. Notably, the $\alpha$ study is performed with $\gamma$ fixed at 0.1, while the $\gamma$ study is conducted with $\alpha$ fixed at 0.75. The experimental results show that our method performs best when $\alpha$ is set to 0.75. As $\alpha$ increases, the region of important neurons (i.e., regions B, C, and D) shrinks, reducing model stability. Conversely, as $\alpha$ decreases, the region of less important neurons (i.e., region A) diminishes, leading to reduced model plasticity. Both scenarios result in a decline in model performance. In the $\gamma$ experiments, increasing $\gamma$ causes more neurons to be classified as redundant, including important neurons effective across multiple early tasks that are mistakenly classified as redundant. This misclassification reduces model stability, leading to a decline in performance.

![](images/45b0d006db18998bad6d7599f29f08416e8966d8d2f81fc3ad8777a5930812e8.jpg)  
Figure 2: The average EER $( \% )$ results of the hyperparameter study for our method RegO.

# General Study

Experimental Setup We use the CLEAR benchmark (Lin et al. 2021) to evaluate the scalability of our method. CLEAR is a classic continual image classification benchmark, with datasets based on the natural temporal evolution of visual concepts in the real world. It adopts task-based sequential learning by dividing the temporal stream into 10 buckets, each composed of labeled subsets for training and evaluation (with 300 images in the training set and 150 in the test set), resulting in a series of 11-way classification tasks. A small labeled subset $( \mathrm { E x p } _ { 1 }$ , $\mathrm { E x p } _ { 2 }$ , ..., $\mathrm { E x p } _ { 1 0 } ,$ ) consists of 11 temporally dynamic categories, including examples like computers, cosplay, etc. We use classification accuracy to evaluate model performance. For the image recognition model, we use a pre-trained ResNet-50 (He et al. 2016) as the feature extractor, which is frozen during continual learning, generating 2048-dimensional features. The downstream classifier has three linear layers: 2048 to 1024, 1024 to 512, and 512 to 2. We set the initial learning rate to 0.1, a batch size of 512, and used SGD optimizer with 0.9 momentum.

Table 3: The accuracy $( \% )$ of the models trained on all CLEAR experiments. All results are reproduced by us.   

<html><body><table><tr><td rowspan="2">Continual Learning Methods</td><td colspan="10">Accuracy(↑) on each experience</td></tr><tr><td>Exp1</td><td>Exp2</td><td>Exp3</td><td>Exp4</td><td>Exp5</td><td>Exp6</td><td>Exp7</td><td>Exps</td><td>Exp9</td><td>Exp10</td></tr><tr><td>Replay-All</td><td>94.34</td><td>94.44</td><td>94.44</td><td>94.85</td><td>95.66</td><td>94.14</td><td>93.94</td><td>95.86</td><td>94.24</td><td>95.56</td></tr><tr><td>Finetune-Exp1</td><td>57.27</td><td>56.67</td><td>59.60</td><td>58.89</td><td>59.39</td><td>55.05</td><td>56.16</td><td>54.75</td><td>54.65</td><td>55.15</td></tr><tr><td>Finetune</td><td>90.40</td><td>89.80</td><td>90.10</td><td>92.73</td><td>90.71</td><td>90.40</td><td>90.10</td><td>89.90</td><td>90.40</td><td>92.42</td></tr><tr><td>EWC</td><td>91.72</td><td>91.62</td><td>91.31</td><td>92.12</td><td>91.31</td><td>90.40</td><td>91.11</td><td>90.61</td><td>90.71</td><td>93.13</td></tr><tr><td>GEM</td><td>91.62</td><td>90.51</td><td>90.30</td><td>92.93</td><td>91.62</td><td>89.39</td><td>90.30</td><td>90.10</td><td>89.49</td><td>93.33</td></tr><tr><td>GDumb</td><td>90.20</td><td>87.78</td><td>89.60</td><td>89.09</td><td>89.19</td><td>86.57</td><td>87.47</td><td>88.18</td><td>87.88</td><td>88.38</td></tr><tr><td>CWRStar</td><td>87.68</td><td>87.98</td><td>87.58</td><td>88.79</td><td>88.79</td><td>86.77</td><td>87.58</td><td>86.77</td><td>86.77</td><td>90.00</td></tr><tr><td>SI</td><td>89.39</td><td>89.29</td><td>90.00</td><td>91.11</td><td>89.79</td><td>88.69</td><td>89.39</td><td>89.90</td><td>88.79</td><td>90.40</td></tr><tr><td>OWM</td><td>73.03</td><td>71.41</td><td>70.30</td><td>73.13</td><td>71.01</td><td>68.99</td><td>69.70</td><td>67.27</td><td>70.30</td><td>69.70</td></tr><tr><td>RAWM</td><td>85.25 87.17</td><td>84.95</td><td>82.83</td><td>83.84</td><td>84.14</td><td>81.62</td><td>81.52</td><td>83.64</td><td>83.84</td><td>82.42</td></tr><tr><td>RWM</td><td></td><td>86.26</td><td>87.68</td><td>89.29</td><td>87.17</td><td>85.66</td><td>88.18</td><td>85.15</td><td>86.87</td><td>85.86</td></tr><tr><td>RegO (Ours)</td><td>91.92</td><td>93.03</td><td>92.63</td><td>93.64</td><td>93.94</td><td>92.32</td><td>92.53</td><td>92.53</td><td>92.42</td><td>94.75</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2">Ablation Study</td><td colspan="10">Accuracy (↑) on each experience</td></tr><tr><td>Exp1</td><td>Exp2</td><td>Exp3</td><td>Exp4</td><td>Exp5</td><td>Exp6</td><td>Exp7</td><td>Exp8</td><td>Expg</td><td>Exp10</td></tr><tr><td>RegO (Ours)</td><td>91.92</td><td>93.03</td><td>92.63</td><td>93.64</td><td>93.94</td><td>92.32</td><td>92.53</td><td>92.53</td><td>92.42</td><td>94.75</td></tr><tr><td>W/o EFM</td><td>92.93</td><td>93.93</td><td>93.33</td><td>94.65</td><td>94.34</td><td>91.82</td><td>93.23</td><td>93.64</td><td>93.13</td><td>94.85</td></tr><tr><td>W/o IRL</td><td>90.61</td><td>89.70</td><td>90.61</td><td>91.21</td><td>91.41</td><td>88.89</td><td>90.71</td><td>89.49</td><td>89.49</td><td>92.83</td></tr><tr><td>w/o RAO</td><td>88.59</td><td>87.37</td><td>88.48</td><td>89.49</td><td>88.99</td><td>87.98</td><td>87.67</td><td>88.48</td><td>87.37</td><td>90.91</td></tr></table></body></html>

Table 4: The accuracy $( \% )$ results of the ablation study for our method on the CLEAR experiences.

Comparison with Other Methods We compare RegO with several classic continual learning methods. As shown in Table 3, after training on $1 0 \ \mathrm { E x p s }$ , the performance of RegO is second only to Replay-All, which is considered the upper bound for continual learning performance. Additionally, Table 3 shows that RWM and RAWM perform better in the earlier subset $( \mathrm { E x p _ { 1 } - E x p _ { 5 } } ) \$ ) compared to the later ones, indicating that these algorithms are more focused on mitigating forgetting, but are less adaptable to new tasks. Our method overcomes catastrophic forgetting by optimizing the gradients of important neurons, while fine-tuning less critical neurons directly to ensure rapid adaptation to new tasks, ensuring an appropriate stability-plasticity trade-off.

Ablation Study We conduct an ablation study to evaluate the effectiveness of the proposed modules, with the results shown in Table 4. Compared to Table 3, we observe an interesting phenomenon: in the image recognition task, removing the EFM module leads to better model performance, which contrasts with the ablation results for audio deepfake detection. We speculate that this is because the CLEAR benchmark represents a decade-long natural temporal evolution of real-world visual concepts, where the appearance of major categories such as computers, cameras, etc. has remained relatively unchanged over the years. The results of Finetune- $\mathrm { E x p } _ { 1 }$ support this hypothesis: after training on $\mathrm { E x p } _ { 1 }$ , the accuracy remains similar from $\mathrm { E x p } _ { 1 }$ to $\mathrm { E x p } _ { 5 }$ but shows a slight decline from $\mathrm { E x p } _ { 5 }$ to $\mathrm { E x p } _ { 1 0 }$ . This indicates that retaining old knowledge might interfere with performance across multiple tasks. On the other hand, in the audio deepfake detection task, where differences in synthesis or conversion algorithms are more distinct (as shown by the Finetune- $\mathrm { E x p } _ { 1 }$ results in Table 1), the role of EFM becomes more critical. Nevertheless, regardless of whether the EFM is integrated, both versions of our method consistently outperform other methods.

# Conclusion

In this paper, we propose an effective continual learning algorithm, Region-Based Optimization (RegO), aimed at improving the generalization of audio deepfake detection models against diverse and unseen forgeries in real-world scenarios. The core idea of $\mathtt { R e g O }$ is to avoid constraints on less important neurons, allowing the model to quickly adapt to new tasks, while applying region-based adaptive gradient optimization to important neurons to overcome catastrophic forgetting, achieving a suitable balance between memory stability and learning plasticity. Experimental results demonstrate that our method outperforms SOTA method RWM for audio deepfake detection, proving its robustness against diverse forgery techniques. Additionally, we conduct the general study and achieve competitive results, indicating our method has potential significance in other domains, such as image recognition. Moreover, we plan to explore how to extend our method to address other challenges in machine learning, such as multi-task learning (Langa 2021).
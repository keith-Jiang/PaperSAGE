# ALRMR-GEC: Adjusting Learning Rate Based on Memory Rate to Optimize the Edit Scorer for Grammatical Error Correction

Zhixiao Wu, Yao Lu\*, Jie Wen\*, Guangming Lu

Harbin Institute of Technology, Shenzhen, China wzxnh24428@gmail.com, luyao2021@hit.edu.cn, jiewen pr $@$ 126.com, luguangm@hit.edu.cn

# Abstract

Edit-based approaches for Grammatical Error Correction (GEC) have attracted volume attention due to their outstanding explanations of the correction process and rapid inference. Through exploring the characteristics of the generalized and specific knowledge learning for GEC, we discover that efficiently training GEC systems with satisfactory generalization capacity prefers more generalized knowledge rather than specific knowledge. Current gradient-based methods for training GEC systems, however, usually prioritize minimizing training loss over generalization loss. This paper proposes the strategy of Adjusting Learning Rate Based on Mermory Rate to optimize the edit-based GEC scorer (ALRMRGEC). Specifically, we introduce the memory rate, a novel metric, to provide an explicit indicator for the model’s state of learning generalized and specific knowledge, which can effectively guide the GEC system to adjust the learning rate timely. Extensive experiments, conducted by optimizing the published edit scorer on the BEA2019 dataset, have shown our ALRMR-GEC significantly enhances the model generalization ability with stable and satisfactory performance nearly irrespective of the initial learning rate selection. Also, our method can accelerate the training over tenfold faster in certain cases. Finally, the experiments indicate the memory rate introduced in our ALRMR-GEC guides the GEC editscorer to learn more generalized knowledge.

# Introduction

Grammatical Error Correction (GEC) involves automatically identifying the errors and converting the source text to its clean version. Edit-based approaches have attracted volume attention due to their outstanding explanations of the correction process and rapid inference (Omelianchuk et al. 2020; Tarnavskyi, Chernodub, and Omelianchuk 2022). The editscorers (Sorokin 2022) use pre-trained Transformerbased models as encoders to identify the correctness of generated edits by GECToR (Omelianchuk et al. 2020) and achieve state-of-the-art quality. However, how to further unleash the potential of GEC models to get satisfactory performance remains an important issue.

GEC models aim to learn generalized knowledge like grammar rather than specific knowledge like word combinations and spelling. However, common gradients-based methods (Duchi, Hazan, and Singer 2011; Tieleman 2012; Kingma and Ba 2014) fail to guide models to enhance generalization ability considering gradients alone do not guarantee the models to enhance generalization ability since the gradients only provide the information about how to minimize the training loss. For example, the training period using Adamw optimizer with 1e-5 as the learning rate gets worse performance on the test dataset (Figure 1b) although it minimizes the training loss faster than the period using 1e-6 (Figure 1a). Therefore, researchers have to adjust the learning rate artificially to ensure satisfactory generalization performance, which is costly and labor-intensive.

Finding a solution to separate generalized knowledge and specific knowledge learning is an urgent issue. In our analysis, generalized knowledge and specific knowledge represent the commonality and individuality of a model when learning from different data. For specific data, generalized knowledge can be acquired from other data while specific knowledge cannot. Therefore, we introduce the accuracy of the model on a specific subset not involved in later training (memorability) to indicate whether the current learning rate favors the generalized knowledge accumulation. Exploratory experiments are conducted in Figure 2 and the analysis can be seen as the following.

Firstly, we explore the influence of learning rates on memorability in Figure 2a and observe that memorability is notably diminished when the learning rate is far away from the favorable learning rate. The above observation implies that memorability is a trade-off of two competing processes, designated as A and B. When the learning rate is excessively low (high), A (B) dominates and leads to low memorability.

Secondly, we explore the characteristics of the above processes in Figure 2b. Applied by small learning rates including 1e-7, process A dominates and the memorability tends to increase during the later training period. Correspondingly, applied by large learning rates including 5e-8, process B dominates and the memorability tends to decrease.

Finally, the correlation between generalization ability and learning rates can be seen in Figure 2c. Models trained with learning rates that keep memorability at a high level attain favorable performance on the test dataset. Specifically, 1.5e-6 is the favorable learning rate with nearly $8 4 . 8 \%$ accuracy on the test set and keeps memorability at a higher level

![](images/e3fc26e4aba03e057bd2d23272af629103157428519fa5848304b8ce48eb025a.jpg)  
Figure 1: Gradients and memorability on GEC task.   
Figure 2: The influence of learning rates on generalization ability and memorability. 1.5e-6 is named the favorable learning rate due to its favorable impact on the acquisition of generalized knowledge. The favorable learning rate is used as the benchmark $\ b { x } = 1$ ) and the value of the $\mathbf { \boldsymbol { x } }$ -axis represents the multiples of the benchmark.

1 + epoch1 2e-7 1e-7 1 0.85 2e-7 ↑ 1e-7 + epoch5 epoch11 0.8 4-7-6 4.57-6 epoch21 0.7 4e-6 e-6 1e-5 1e-5 0.80 1/10 1/5 10 0 20 40 60 0 20 40 60 Learning Rate Epoch Epoch   
(a) Learning rates and memorability (b) Trends of memorability (c) Accuracy on test dataset

in Figure 2b compared to other learning rates. Upon comparative analysis of Figures 1 and 2, memorability distinctively differentiates the performance of models across learning rates, surpassing the gradients as a more informative metric to guide the model to learn generalized knowledge.

Based on the above observations, we realize the importance of memorability and its tight correlation to the favorable learning rate. We further explore the selection of benchmark data used to calculate the memorability (Figure 5) and solidify the memorability as Memory Rate, the accuracy of the particular subset selected from the train set that was classified incorrectly in the previous epoch but correctly in the current epoch and will not participate in the subsequent training periods. Furthermore, ALRMR-GEC is proposed to optimize the editscorers automatically and the code can be seen at https://github.com/rearchwzx/ALRMR-GEC. The contributions of our work can be summarized by answering the following questions:

• Is there a metric for overcoming the drawback of gradients in enhancing the generalization ability? Yes, the proposed memory rate provides an explicit indicator of the model’s state of generalized knowledge learning, which can guide the GEC systems to adjust the learning rates to the favorable learning rate timely and thus enhance the generalization ability of the editscorer. • Can the learning rate be automatically adjusted efficiently and hence itself no longer be a hyperparameter that needs to be carefully adjusted artificially? Yes, the proposed ALRMR-GEC has two stages: the fast start stage and the slow adjustment stage. Based on

the aforementioned stages, the ultimate performance is nearly regardless of the initial learning rate selection. • Can the method further unleash the potential of pretrained Transformer-based models for GEC? Yes, we use ALRMR-GEC to further train more powerful and stable GEC editscorers effectively based on the Roberta-base and the Roberta-large models. The optimized models can be stably maintained at the favorable generalization state. In specific cases, the time to reach a certain validation accuracy can be shortened to 1/10 compared to the origin training process.

# Related Work

Pre-trained Transformer-based models, trained with massive language data, have demonstrated remarkable efficacy in NLP tasks (Hu et al. 2023; Zhong et al. 2022; Li et al. 2022, 2023a; Rossiello et al. 2023). Specifically, for GEC (Gong et al. 2022; Zhang et al. 2022; Fang et al. 2023), the architecture of multiple-layer multi-head attention mechanisms helps the systems (Gong et al. 2022; Sorokin 2022; Li et al. 2023b) exhibit enhanced feature extraction capabilities compared to their counterparts.

Most strategies used to unleash the potential of pretrained Transformer-based models upon GEC are based on gradients (Smith and Topin 2019; Li and Arora 2020; Loshchilov and Hutter 2022; Iyer et al. 2023). Currently, Adam (Kingma and Ba 2014) and its variant AdamW (Loshchilov and Hutter 2017) are among the most popular optimizers (Schmidt, Schneider, and Hennig 2021). It retains the first and second moment information of parameters to facilitate adaptive learning step size. Shuaipeng Li (Li et al. 2024) establishes a scaling law between favorable learning rates and batch sizes for Adam-style optimizers. However, strictly following the derivative-guided route (Sutskever et al. 2013; Duchi, Hazan, and Singer 2011; Zeiler 2012; Tieleman 2012; Kingma and Ba 2014; Shazeer and Stern 2018; Smith and Topin 2019; Li and Arora 2020; Chen et al. 2024) does not guarantee the models to enhance generalization ability. We observe that some scientists (Fung, Yoon, and Beschastnikh 2018; Ozdayi, Kantarcioglu, and Gel 2021) adjust the learning rate based on global features to defend against backdoor attacks, which inspires us to find better metrics to adjust the learning rate.

# Our Approach

In this section, a theoretical framework is constructed to elucidate the empirical phenomena and we introduce ALRMRGEC, a novel strategy that dynamically adjusts the learning rate based on the characteristics of knowledge learning.

# Specific Knowledge and Generalized Knowledge

The parameters of model can be denoted as $\begin{array} { r l } { \theta _ { t } } & { { } = } \end{array}$ $[ \theta _ { t } ^ { 1 } , \theta _ { t } ^ { \dot { 2 } } , \cdot \cdot \cdot , \theta _ { t } ^ { i } , \cdot \cdot \cdot , \theta _ { t } ^ { n } ]$ and the knowledge learned at the $t ^ { t h }$ backpropagation can be expressed as $\Delta \theta _ { t }$ . Assuming an optimal target model is maximizing the generalization ability within the current architecture, parameters of the optimal model can be denoted as $\begin{array} { r l } { \theta _ { b e s t } } & { { } = } \end{array}$ $[ \theta _ { b e s t } ^ { 1 } , \theta _ { b e s t } ^ { 2 } , \hat { \textbf { \ i } } ^ { . } \cdot \textbf { \ i } , \theta _ { b e s t } ^ { i } , \hat { \textbf { \ i } } \cdot \hat { \textbf { \ i } } , \theta _ { b e s t } ^ { n } ]$ .

Generalized knowledge refers to the knowledge that helps the current model to approach the optimal model. Specific knowledge refers to the knowledge that belongs to specific data that hinders the current model from approaching the optimal model. By introducing $[ \alpha _ { t } ^ { 1 } , \alpha _ { t } ^ { 2 } , \cdot \cdot \cdot , \bar { \alpha } _ { t } ^ { i } , \cdot \cdot \cdot , \bar { \alpha _ { t } ^ { n } } ] ( \alpha _ { t } ^ { i } \bar { > }$ 1) into definitions to better fit the real situation, the learned generalized and specific knowledge at the $t ^ { t h }$ backpropagation can be defined as follows:

$$
\begin{array} { r l } & { \Delta \theta _ { t } ^ { g e n ( s p e c ) } = \overset { \left[ \Delta \theta _ { t , 1 } ^ { g e n ( s p e c ) } , \Delta \theta _ { t , 2 } ^ { g e n ( s p e c ) } , \cdots , \right. } { \left. \Delta \theta _ { t , i } ^ { g e n ( s p e c ) } , \cdots , \right. } } \\ & { \left. \Delta \theta _ { t , i } ^ { g e n ( s p e c ) } , \cdots , \Delta \theta _ { t , n } ^ { g e n ( s p e c ) } \right] ^ { \mathrm { ? } } } \\ & { \Delta \theta _ { t , i } ^ { g e n ( s p e c ) } = \left\{ \alpha _ { t } ^ { i } * \Delta \theta _ { t } ^ { i } \right. \qquad \left. s g n ( \Delta \theta _ { t } ^ { i } ) = ( \neq ) s g n ( \Delta \theta _ { t , i } ^ { b e s t } ) \right. } \\ & { \left. \qquad \quad ( 1 - \alpha _ { t } ^ { i } ) * \Delta \theta _ { t } ^ { i } \quad s g n ( \Delta \theta _ { t } ^ { i } ) \neq ( = ) s g n ( \Delta \theta _ { t , i } ^ { b e s t } ) . \right. } \end{array}
$$

$\Delta \theta _ { t } ^ { g e n }$ and $\Delta \theta _ { t } ^ { s p e c }$ are complementary and together constitute the learned knowledge $\Delta \theta _ { t } . s g n ( \cdot )$ outputs $- 1$ and 1 based on the direction of parameter variation. Distinct outputs $( s g n ( \cdot ) \ne s g n ( \cdot ) )$ indicate the opposite directions.

# Knowledge Learning

Symbols $S P E C$ and $G E C$ represent the specific knowledge and generalized knowledge respectively. $T$ represents the number of backpropagations during the subsequent training periods. The close-knit correlation between the edits instigates two adversarial periods is shown as follows:

Knowledge degradation : $\Delta X . D E _ { x , y } ^ { Y }$ in Eqns. 3 and 4 refers to the detrimental influence of knowledge $Y$ learned at $y ^ { t h }$ backpropagation on knowledge $X$ learned at $x ^ { t h }$ backpropagation $( x ^ { - } \leq y , X , Y \in \{ \hat { S P E C } , G E N \} )$ .

$$
\begin{array} { r } { \Delta X _ { \textrm { - } } D E _ { x , y } ^ { Y } = \underset { \Delta X _ { \textrm { - } } D E _ { x , y , i } ^ { Y } , \textrm { - } \cdot \textrm { - } , \Delta X _ { \textrm { - } } D E _ { x , y , n } ^ { Y } } { \Delta X _ { \textrm { - } } D E _ { x , y , 1 } ^ { Y } , \textrm { - } \cdot \textrm { - } , \Delta X _ { \textrm { - } } D E _ { x , y , n } ^ { Y } ] } } \end{array}
$$

$$
\Delta X \_ D E _ { x , y , i } ^ { Y } = \left\{ \begin{array} { l l } { \alpha _ { x } ^ { i } * \Delta \theta _ { y , i } ^ { Y } } & { s g n ( \Delta \theta _ { x , i } ^ { X } ) \neq s g n ( \Delta \theta _ { y , i } ^ { Y } ) , } \\ { s g n ( \Delta \theta _ { x , i } ^ { X } ) = s g n ( \Delta \theta _ { x } ^ { i } ) } & { s g n ( \Delta \theta _ { x , i } ^ { Y } ) } \\ { ( 1 - \alpha _ { x } ^ { i } ) * \Delta \theta _ { y , i } ^ { Y } } & { s g n ( \Delta \theta _ { x , i } ^ { X } ) \neq s g n ( \Delta \theta _ { y , i } ^ { Y } ) , } \\ { 0 } & { s g n ( \Delta \theta _ { x , i } ^ { X } ) \neq s g n ( \Delta \theta _ { x } ^ { i } ) } \\ { 0 } & { o t h e r s } \end{array} \right.
$$

Knowledge complement : $\Delta X \lrcorner C O _ { x , y } ^ { Y }$ in Eqns. 5 and 6 refers to the positive influence of knowledge Y learned $y ^ { t h }$ backpropagation on knowledge $X$ learned at $x ^ { t h }$ backpropagation $\mathsf { \bar { ( } } x \mathsf { \bar { \leq } } y , X , Y \in \{ S P \mathsf { \bar { E } } C , G E N \} \mathsf { ) }$ .

$$
\begin{array} { r } { \Delta X _ { - } C O _ { x , y } ^ { Y } = \overset { [ \Delta X _ { - } C O _ { x , y , 1 } ^ { Y } , \Delta X _ { - } C O _ { x , y , 2 } ^ { Y } , \cdots ~ , } { \Delta X _ { - } C O _ { x , y , n } ^ { Y } } , } \end{array}
$$

$$
\Delta X . C O _ { x , y , i } ^ { Y } = \left\{ \begin{array} { l l } { \alpha _ { x } ^ { i } * \Delta \theta _ { y , i } ^ { Y } } & { s g n ( \Delta \theta _ { x , i } ^ { X } ) = s g n ( \Delta \theta _ { y , i } ^ { Y } ) , } \\ { s g n ( \Delta \theta _ { x , i } ^ { X } ) = s g n ( \Delta \theta _ { x } ^ { i } ) } & { s g n ( \Delta \theta _ { x } ^ { Y } ) } \\ { ( 1 - \alpha _ { x } ^ { i } ) * \Delta \theta _ { y , i } ^ { Y } } & { s g n ( \Delta \theta _ { x , i } ^ { X } ) = s g n ( \Delta \theta _ { y , i } ^ { Y } ) , } \\ { 0 } & { s g n ( \Delta \theta _ { x , i } ^ { X } ) \neq s g n ( \Delta \theta _ { x } ^ { i } ) } \\ { 0 } & { o t h e r s } \end{array} \right.
$$

As $\Delta X \_ C O _ { x , y } ^ { Y }$ and $\Delta X \lrcorner D E _ { x , y } ^ { Y }$ have covered all variations of the parameters when $X = { \overset { \triangledown } { \boldsymbol { Y } } } ( X , Y \in \{ S P E C , G E N \} )$ , it is unnecessary to consider the impact of $\Delta X \lrcorner C O _ { x , y } ^ { Y }$ and $\Delta X \lrcorner D E _ { x , y } ^ { Y }$ swthen $X \neq Y$ c. eTarreatnindgcokmnoprwelheedngse bilned.epen

Characteristic of specific knowledge learning : Specific knowledge is characterized by an average of zero on the gradients and randomness in directions. From the perspective of knowledge degradation, the detrimental influence of specific knowledge learned from subsequent data $\Delta S P E \bar { C } . D E _ { i , t + 1 } ^ { S P E C } ( t \bar { \geq } i )$ continuously overwrites $\Delta \theta _ { i } ^ { s p e c }$ until $\Delta \theta _ { i } ^ { s p e c }$ is completely forgotten when $T \to \infty$ , which can be depicted as follows:

$$
\operatorname* { l i m } _ { T  \infty } \sum _ { t = i } ^ { i + T } \Delta S P E C . D E _ { i , t + 1 } ^ { S P E C } = - \Delta \theta _ { i } ^ { s p e c } .
$$

Specific knowledge represents the non-commonality in parameter variation. From the perspective of knowledge complement, the positive influence of specific knowledge learned from subsequent data $\Delta S P E C { \overset { \cdot } { C } } C O _ { i , t + 1 } ^ { S P E C } ( t \geq \overset { \cdot } { i } )$ is nearly non-existent when $T \to \infty$ , which can not comaplbeomvenatn $\Delta \theta _ { i } ^ { s p e c }$ ,anrebseultdienpgicitned cs→ufmol∞ulloawtisv:e effect of 0. The

$$
\operatorname* { l i m } _ { T  \infty } \sum _ { t = i } ^ { i + T } \Delta S P E C . C O _ { i , t + 1 } ^ { S P E C } = 0 .
$$

Characteristic of generalized knowledge learning : Generalized knowledge is characterized by a consistent directionality on the gradient, representing the commonalities. From the perspective of knowledge degradation, the detrimental influence of generalized knowledge learned from subsequent data $\Delta G \bar { E } N \_ D E _ { i , t + 1 } ^ { G E N } ( t \geq i )$ is nearly nonexistent when $T \to \infty$ as the consistency in directions ensures that generalization knowledge does not cancel each other out, resulting in a cumulative effect of 0, which can be depicted as follows:

$$
\operatorname* { l i m } _ { T  \infty } \sum _ { t = i } ^ { i + T } \Delta G E N . D E _ { i , t + 1 } ^ { G E N } = 0 .
$$

Generalization knowledge represents the parameter variation that converge toward the optimal model. From the perspective of knowledge complement, the positive influence of generalized knowledge learned from subsequent data $\Delta G \bar { E } N \_ C O _ { i , t + 1 } ^ { G E N } ( t \geq i )$ continuously augments the unmastered generalization knowledge until $\theta _ { i }$ is the same as the optimal model’s parameters set $\breve { \theta } _ { b e s t }$ when $T \to \infty$ . The above analysis can be depicted as follows:

$$
\operatorname* { l i m } _ { T  \infty } \sum _ { t = i } ^ { i + T } \Delta G E N _ { - } C O _ { i , t + 1 } ^ { G E N } = \theta _ { b e s t } - \theta _ { i } .
$$

# Memory Rate

$\Delta S P E C _ { i } ^ { e p o c h . i }$ represents the accumulation of specific knowledge learned at $i ^ { t h }$ backpropagation during the current epoch. Specifically, epoch $\mathbf { \Omega } _ { - } i$ refers to the number of aftermost backpropagation during the epoch which includes $i ^ { t h }$ backpropagation. The detrimental and positqiuvenitndflauteancaenobf srpepcriefsic nktendowalse $\bar { \Delta S P E C . D E _ { i , t + 1 } ^ { S P E C } }$ basned$\Delta S P E C . C O _ { i , t + 1 } ^ { S P E C } ( t \geq i )$ , respectively. The mathematical description of the above process can be presented as follows:

$$
\begin{array} { r l } { \displaystyle \Delta S P E C _ { i } ^ { e p o c h . i } = \Delta \theta _ { i } ^ { s p e c } + \sum _ { t = i } ^ { e p o c h . i } ( \Delta S P E C . D E _ { i , t + 1 } ^ { S P E C } } & { } \\ { \displaystyle + \Delta S P E C . C O _ { i , t + 1 } ^ { S P E C } ) . } & { } \end{array}
$$

A large amount of data is one of the characteristics of the training set. Therefore, for the subset trained at the early stage (represented as $A$ ), the number of backpropagations during the subsequent training periods $T$ can be seen as $T  \infty$ . According to Eqns. 7 and 8, its specific knowledge is almost overwritten and cannot be accumulated by the latter learned specific knowledge, resulting in a cumulative effect close to $\mathrm { { \bar { 0 } } }$ , which can be depicted as follows:

$$
\Delta S P E C _ { i } ^ { e p o c h . i } \approx 0 ( i \in \cal { A } ) .
$$

However, the size of the dataset is limited. Therefore, for the fresh subset trained at $\{ i | i \notin A \}$ , the number of backpropagations during the subsequent training periods $T$ can not be seen as $T \to \infty$ . Therefore, its specific knowledge can not be completely overwritten by the latter learned specific knowledge, which contributes to the principal part of the learned specific knowledge at the current epoch, resulting in a non-zero cumulative effect. The mathematical description can be presented as follows:

$$
\Delta S P E C _ { i } ^ { e p o c h . i } > 0 ( i \notin A ) .
$$

$\Delta G E N _ { i } ^ { e p o c h . i }$ represents the accumulation of generalized knowledge learned at $i ^ { t h }$ backpropagation during the current epoch. The detrimental and positive inqfluent  otaf cgaen brae rzepd skennotweld dags $\Delta G E N \_ D E _ { i , t + 1 } ^ { G E N }$ asned$\Delta G E N \_ C O _ { i , t + 1 } ^ { G E N } ( t \geq i )$ , respectively. The mathematical description of the above process can be presented as follows:

$$
\begin{array} { r } { \begin{array} { l } { { \Delta G E N _ { i } ^ { e p o c h . i } = \Delta \theta _ { i } ^ { g e n } + \displaystyle \sum _ { t = i } ^ { e p o c h . i } ( \Delta G E N . D E _ { i , t + 1 } ^ { G E N } } } \\ { { + \Delta G E N . C O _ { i , t + 1 } ^ { G E N } ) . } } \end{array} } \end{array}
$$

The performance of models upon the data trained at $i ^ { t h }$ backpropagation after the current epoch is a trade-off adversarial process based on ∆GEN iepoch i a nd ∆SP ECiepoch i. ∆GEN epoch i contributes to the optimization of the model with theihindrance of $\Delta S P E C _ { i } ^ { e p o c h . i } \ : ( i \notin A )$ .

$$
\Delta \theta _ { i } ^ { e p o c h . i } = \Delta G E N _ { i } ^ { e p o c h . i } + \Delta S P E C _ { i } ^ { e p o c h . i } .
$$

Based on improper small learning rates, the unsatisfactory generalized knowledge $\Delta \theta _ { i } ^ { g e n }$ learned at $i ^ { t h }$ backpropagation leads to a low memory rate. The memory rate will increase as the insufficient generalized knowledge will be supplemented by other data in subsequent training periods according to Eqn. 14. Based on improper large learning rates, the preference of models on fresh data is strengthed. The accumulation of specific knowledge, depicted in Eqn. 13, prevails compared to the accumulation of generalized knowledge. The memory rate is expected to decrease in the latter training periods according to Eqn.12.

# Learning Rate Evolution Strategy

At the fast-start stage, ALRMR-GEC introduces a backtracking mechanism to initiate improper training and find appropriate learning rates with a complexity of ${ \bf \bar { \cal O } } ( l o g _ { 2 } n )$ . At the fine-turning stage, ALRMR-GEC keeps the learning rate at a favorable status to elaborately optimize the model and ensure its stability, thereby preventing overfitting.

# Algorithm 1 ALRMR-GEC

<html><body><table><tr><td>Require :Learning rate lr>O, the editscorer modelθ Require :model stage differentiates the fast-start phase O and the fine-tune phase 1 Require : correct flag differentiates the recalculation time O and the adjustment time 1 Initialize : correct flag, modelstage = 0 while training do if correctflag is O and modelstageis O then Backtrack modelθ to the initial state. end if if correct flag is1 then Train modelθ with lr Calculate the memory rate nowmero if modelstage is O then if The variation or the continuous change of nowmeroexceedsthethresholdthen Use binary search to find the favorable learning rate correct flag = 0 else correctflag,modelstage=1 end if else if The variation or the continuous change of nowmero exceeds the threshold then lr=lr*α(β)，αandβ(<1) correct flag=0 else correct flag=1 end if end if else Update the benchmark dataset Calculate the memory rate lastmero</td></tr></table></body></html>

Table 1: Performance of ALRMR-GEC based on Roberta-base and Roberta-large using Faster Simultaneous Decoding.   

<html><body><table><tr><td rowspan="2" colspan="2">Model</td><td colspan="2">Roberta-base</td><td colspan="2">Roberta-large</td></tr><tr><td>Pie_bea-gector</td><td>ALRMR-GEC</td><td>Clang_large_ft2</td><td>ALRMR-GEC</td></tr><tr><td rowspan="4">Threshold = 0.5</td><td>P</td><td>57.43</td><td>65.45</td><td>63.18</td><td>65.95</td></tr><tr><td>R</td><td>31.68</td><td>32.31</td><td>37.40</td><td>35.27</td></tr><tr><td>F</td><td>49.40</td><td>54.31</td><td>55.53</td><td>56.18</td></tr><tr><td>Acc(%)</td><td>78.51</td><td>85.11</td><td>84.80</td><td>85.90</td></tr><tr><td rowspan="4">Threshold = 0.7</td><td>P</td><td>63.24</td><td>65.92</td><td>66.60</td><td>66.67</td></tr><tr><td>R</td><td>28.16</td><td>31.76</td><td>35.67</td><td>34.80</td></tr><tr><td>F</td><td>50.63</td><td>54.25</td><td>56.76</td><td>56.35</td></tr><tr><td>Acc(%)</td><td>82.62</td><td>85.16</td><td>86.01</td><td>86.13</td></tr><tr><td rowspan="4">Threshold = 0.9</td><td>P</td><td>76.20</td><td>67.12</td><td>72.17</td><td>67.67</td></tr><tr><td>R</td><td>13.97</td><td>31.16</td><td>30.52</td><td>33.84</td></tr><tr><td>F</td><td>40.29</td><td>54.53</td><td>56.69</td><td>56.40</td></tr><tr><td>Acc(%)</td><td>83.24</td><td>85.33</td><td>86.59</td><td>86.33</td></tr></table></body></html>

Table 2: Performance of ALRMR-GEC based on Roberta-base using Better Stagewise Decoding.   

<html><body><table><tr><td colspan="2" rowspan="2">Model</td><td colspan="3">Threshold=0.7</td><td colspan="2">Threshold=0.8</td><td colspan="2">Threshold=0.9</td></tr><tr><td>origin</td><td>ALRMR-GEC</td><td>origin</td><td>ALRMR-GEC</td><td>origin</td><td>ALRMR-GEC</td></tr><tr><td rowspan="3">Stage = 1</td><td>P</td><td>72.25</td><td>71.28</td><td>75.14</td><td>73.33</td><td>76.90</td><td>75.68</td></tr><tr><td>R</td><td>17.70</td><td>19.90</td><td>16.19</td><td>19.52</td><td>13.95</td><td>18.63</td></tr><tr><td>F</td><td>44.70</td><td>47.01</td><td>43.48</td><td>47.27</td><td>40.43</td><td>46.94</td></tr><tr><td rowspan="3">Stage = 3</td><td>P</td><td>68.41</td><td>67.49</td><td>72.34</td><td>70.12</td><td>74.99</td><td>73.56</td></tr><tr><td>R</td><td>27.80</td><td>31.80</td><td>24.78</td><td>30.84</td><td>20.31</td><td>28.76</td></tr><tr><td>F</td><td>52.94</td><td>55.12</td><td>52.27</td><td>55.89</td><td>48.74</td><td>56.09</td></tr><tr><td rowspan="3">Stage = 5</td><td>P</td><td>67.72</td><td>66.57</td><td>71.87</td><td>69.42</td><td>74.93</td><td>73.08</td></tr><tr><td>R</td><td>30.24</td><td>34.41</td><td>26.68</td><td>33.23</td><td>21.62</td><td>30.84</td></tr><tr><td>F</td><td>54.27</td><td>56.08</td><td>53.68</td><td>57.00</td><td>50.18</td><td>57.37</td></tr><tr><td rowspan="3">Stage = 7</td><td>P</td><td>67.46</td><td>66.59</td><td>71.69</td><td>69.37</td><td>74.89</td><td>73.11</td></tr><tr><td>R</td><td>30.99</td><td>35.26</td><td>27.24</td><td>33.98</td><td>22.00</td><td>31.53</td></tr><tr><td>F</td><td>54.60</td><td>56.54</td><td>54.05</td><td>57.41</td><td>50.57</td><td>57.85</td></tr></table></body></html>

# Experimental Results and Analysis Preliminary

The implementation follows the proposed editscorer and employs the same training data and experimental setup (Sorokin 2022). The gector variants are generated using edit generators on the BEA 2019 Shared Task data. The Base strategy in Figure 5 is used as the default approach to select the benchmark. Furthermore, there are two decoding strategies used in our experiments. Faster Simultaneous Decoding is an offline approach in which the editscorer calculates a collection that satisfies the criteria whose probability exceeds the predefined threshold and then chooses the edits that perform higher scores and do not contradict other edits. At Better Stagewise Decoding, the editscorer selects the most probable edit at first. Then the editscorer applies it to the current input sentence and removes all the edits with intersecting spans repeatedly until the most probable edit is “do nothing” or its probability is below the threshold.

# Superiority of Our Approach

Superiority of Generalization Ability: The models optimized by ALRMR-GEC can surpass original editscorers within Roberta-base models both in Faster Simultaneous

Decoding (Table 1) and Better Stagewise Decoding (Table 2). Our approach improves Acc by $6 . 9 \%$ when using 0.5 as the threshold during the inference period in Faster $S i$ - multaneous Decoding. In Better Simultaneous Decoding, the proposed ALRMR-GEC enhances the Roberta-base models in recall and f-measure with a minor decrease in precision. The above observations suggest that the memory rate can enhance the generalization ability of GEC models.

The models optimized by ALRMR-GEC can not get the same amazing improvement within Roberta-large models compared to Roberta-base models in Faster Simultaneous Decoding. We speculate that this is because smaller models are more reliant on the choice of learning rate and the memory rate is based on the accuracy so that the models are more inclined to improve the Acc rather than the other metrics. Adjusting the definition of the memory rate may alter the direction of model improvement.

Superiority of Efficiency: The red regions exhibit two characteristics compared to the blue regions: a steady increase and a narrower range (Figure 3), which respectively align with our method’s advantages in stability and efficiency. Therefore, the editscorers can be optimized nearly regardless of the initial learning rate selection and do not need to be adjusted manually. Specifically, ALRMR-GEC

![](images/4a2d0fb98f048d9f9e96fc8be0e704f928edb6776b5494de15b19bc18ece709b.jpg)  
Figure 3: Performance with different selections of initial learning rates. Accuracy represents the accuracy of the trained model upon the test dataset. \* represents the training period using the ALRMR-GEC method. Blue represents the area of the origin training period and Red depicts the area of the training period using ALRMR-GEC.

0.85 0.85 0.85 ·4e-6 2e-6 1e-7 0.84 1e-8 A0.83 M !v +4e-6\* 0.84 +2e-6\* +1e-7\* c .82 ·1e-8\* 0.82 0.83 0.82 0.80 0.81+ 0 100200300400 0 100200300400 0 100200300400 0 100200300400 Epoch Epoch Epoch Epoch (a) Improvement upon overfitting (b) Improvement upon underfitting

can accelerate the training period especially when the learning rate is improperly small. Figure 4b shows that even after 400 epochs (nearly 166 hours) of training, the model trained by 1e-8 is still far from achieving a satisfactory performance. However, it takes only 37 epochs (nearly 15 hours) to train the Roberta-base models to touch $84 \%$ on validation accuracy when optimized by ALRMR-GEC. The results indicate that the proposed ALRMR-GEC can accelerate the models to learn the generalized knowledge.

Superiority of Stability: According to Figure 4, the related models suffer from overfitting in the origin training period even if both 2e-6 and 4e-6 are favorable learning rates at the beginning. Optimized by ALRMR-GEC, Roberta-base models can be stably maintained at the peak performance state upon the validation accuracy $( 8 4 . 8 \% )$ . Therefore, overfitting is not an inevitable trend in model training. Properly adjusting the learning rate enables the model to increase accuracy on the training set without compromising its accuracy on the test set. Overfitting arises owing to the acquisition of detrimental specific knowledge. The results reply that the memory rate constrains the model from overly focusing on specific knowledge within GEC sentences.

# Ablation Study

# 1. The approaches of choosing the benchmark dataset

According to Figure 5a, for the Base/Reverse strategy, the memory rate varies from 0.6 to 0.8 whose fluctuations become much more pronounced compared to the Random

![](images/13fa7cb908363821dab0edd75fb1473f66fb7f633b0405fb095fa6225fded4ce.jpg)  
Figure 4: Performance based on Roberta-base model with different selections of initial learning rates during 400 epochs. Acc represents the accuracy of the trained model upon the test dataset. \* represents the training period using the ALRMR-GEC method. In the actual training environment, each training epoch approximately consumes 25 minutes.   
Figure 5: The approaches of choosing the benchmark dataset to calculate the memory rate. Base (Reverse) refers to selecting the subset of the training set that was classifed incorrectly (correctly) in the previous epoch but correctly (incorrectly) in the current epoch. Random refers to randomly selecting a subset of the training set.

# Train Set

# A Data Relation

# Fresh data

Specific Knowledge "T","know”,"amazing”,"scientist”, "So”,"if',"you”,“visit”,etc Generalized Knowledge Subject-Verb-Object structure, Prepositions and Conjunctions,etc

I know it isa amazing to be scientist . I know it is a amazing to be a scientist So if you want to visit any country Iam going to recommend Italy . So if you want to visit anycountry, Iam going to recommend Italy .

![](images/e1da02188f21bd3e062f6663f69f6637a6bb875e82ade7193c6aa260493b9704.jpg)

# Benchmark

# Test Set

But we still sold it not bought it . Butwe still sold it,notbought it

Wait a minute am i so angry to make this? Wait a minute, am i so angry to make this?

# B. Knowledge Learning

Specific Knowledge

$$
\begin{array} { l } { { \displaystyle \operatorname* { l i m } _ { \mathrm { T  \infty } } \sum _ { { \mathrm { t } } = i } ^ { i + T } \Delta S P E C _ { { \mathrm { - } } } D E _ { i , t + 1 } ^ { S P E C } = - \Delta \theta _ { i } ^ { s p e c } } \ { \displaystyle  \bar { \boldsymbol { \Xi } } \boldsymbol { \vartheta } _ { j }  } } \\ { { \displaystyle \operatorname* { l i m } _ { \mathrm { T  \infty } } \sum _ { { \mathrm { t } } = i } ^ { i + T } \Delta S P E C _ { { \mathrm { - } } } } \underline { { C O _ { i , t + 1 } ^ { S P E C } } } = 0 } \end{array}
$$

![](images/3975813882086fe2c3d9eadd2ccd31c6407337d152050be943fd7532ed5a413c.jpg)

# Generalized Knowledge

![](images/6368fdceb8774c33a9183d09189061330ef9de4a36dc9274982b62f425a3edaf.jpg)

$$
\begin{array} { r l } & { \underset { \mathrm { T }  \infty } { \operatorname* { l i m } } \underset { { \mathrm { t } = i } } { \overset { i + T } { \sum } } \Delta G E N _ { - } D E _ { i , t + 1 } ^ { G E N } = 0 } \\ & { \underset { \mathrm { T }  \infty } { \operatorname* { l i m } } \underset { t = i } { \overset { i + T } { \sum } } \Delta G E N _ { - } C O _ { i , t + 1 } ^ { G E N } = \theta _ { b e s t } - \theta _ { i } } \end{array}
$$

![](images/37077674d30a96240d4724c062896bf45763a5ac08cad045809a5272695072fc.jpg)  
Figure 6: Visualization of knowledge learning. $\Delta \theta ^ { i } = \Delta \theta _ { t } ^ { i } - \Delta \theta _ { 0 } ^ { i } > 0$ is named as the positive direction of $\theta ^ { i }$ . Therefore, the parameter variation in the negative (positive) direction is reflected by black (red). Data Relation investigates the source of learned knowledge. Secondly, Knowledge Learning experimentally demonstrates the characteristics of specific and generalized knowledge accumulation. Finally, the entire training process and the analysis of memory rate are reviewed in Training.

strategy in which the memory rate varies from 0.85 to 0.9. According to Figure 5b, the memory rate based on Random keeps 0.9 during the training and fails to reflect the state of generalized knowledge learning. Conversely, the benchmark datasets on the Base/Reverse strategy contain more knowledge that has not been grasped by the model and the variation in accuracy can still intuitively reflect the suitability of the hyperparameters adopted in the current training period.

# 2. Visualization of knowledge learning

Data Relation: For Benchmark in Figure 6A, inserting “,” into the “sold it” and “not bought it.” needs the model to learn the generalized knowledge about the Subject-VerbObject structure. The specific knowledge of Fresh data (“I”, “know”, etc) contributes to the major part of the learned specific knowledge during the current training period (Eqn. 13). The specific knowledge learned at the early training (“So”, “if”, etc) will be forgotten according to Eqn. 8. But the generalized knowledge can be accumulated with proper learning rates according to the discussion about Eqn. 15.

Knowledge Learning: Within large learning rates, the model’s preference for fresh data dominates, leading to significant parameter variation that mainly reflects the characteristics of specific knowledge in different fresh data. According to Figure 6B, the directions of parameter variation in different specific knowledge are so random that their cumulative effects cancel each other out. Conversely, with small learning rates, the model’s emphasis on accumulating generalized knowledge prevails. The similar parameters variation implies the universality of generalized knowledge.

Training: On one hand, higher learning rates accelerate the accumulation of generalized knowledge. On the other hand, the bias of the specific knowledge (“I”, “know”, “amazing”, etc) is also intensified as the gradients of fresh data focus on details about word spelling without considering the learned grammar (Subject-Verb-Object structure) used in the sentences “But we still sold it, not bought it”. The favorable learning rate is a trade-off point of the above adversarial periods, which can be depicted in Figure 6C. Memory rate, analogous to the performance of learners upon error-correction notebooks without reviewing in human grammar learning, serves as a global feature to guide the ALRMR-GEC to indicate whether the current learning rate favors the generalized knowledge accumulation.

# Conclusion

Grammatical Error Correction aims to learn generalized knowledge like grammar rather than specific knowledge like word combinations and spelling. Generalized knowledge and specific knowledge represent the commonality and individuality of a model when learning from different data. Generalized knowledge can be acquired from other data while specific knowledge cannot. Based on this analysis, we introduce memory rate to guide the ALRMR-GEC to find the optimal learning rate timely.

# Acknowledgments

This work was supported in part by the NSFC fund (NO. 62206073, 62176077), in part by the Shenzhen Key Technical Project (NO. JSGG20220831092805009, JSGG20220831105603006, JSGG20201103153802006, KJZD20230923115117033, KJZD20240903100712017), in part by the Guangdong International Science and Technology Cooperation Project (NO. 2023A0505050108), in part by the Shenzhen Fundamental Research Fund (NO. JCYJ20210324132210025), and in part by the Guangdong Provincial Key Laboratory of Novel Security Intelligence Technologies (NO. 2022B1212010005), and in part by the Natural Science Foundation of Shenzhen General Project under Grant JCYJ20240813110007010, in part by the Natural Science Foundation of Guangdong Province under Grant 2023A1515010893, in part by the Shenzhen Doctoral Initiation Technology Plan under Grant RCBS20221008093222010, in part by the Shenzhen Pengcheng Peacock Startup Fund.
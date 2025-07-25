# MABR: Multilayer Adversarial Bias Removal Without Prior Bias Knowledge

Maxwell J. Yin, Boyu Wang, Charles Ling

University of Western Ontario jyin97@uwo.ca, bwang@csd.uwo.ca, charles.ling@uwo.ca

# Abstract

Models trained on real-world data often mirror and exacerbate existing social biases. Traditional methods for mitigating these biases typically require prior knowledge of the specific biases to be addressed, and the social groups associated with each instance. In this paper, we introduce a novel adversarial training strategy that operates withour relying on prior bias-type knowledge (e.g., gender or racial bias) and protected attribute labels. Our approach dynamically identifies biases during model training by utilizing auxiliary bias detector. These detected biases are simultaneously mitigated through adversarial training. Crucially, we implement these bias detectors at various levels of the feature maps of the main model, enabling the detection of a broader and more nuanced range of bias features. Through experiments on racial and gender biases in sentiment and occupation classification tasks, our method effectively reduces social biases without the need for demographic annotations. Moreover, our approach not only matches but often surpasses the efficacy of methods that require detailed demographic insights, marking a significant advancement in bias mitigation techniques.

# Code — https://github.com/maxwellyin/MABR

# Introduction

Neural natural language processing (NLP) models are known to exhibit social biases, with protected attributes like gender and race serving as confounding variables (Blodgett et al. 2020; Bansal 2022). These attributes can create spurious correlations with task response variables, leading to biased predictions. This issue manifests across various NLP tasks, such as machine translation (Cho et al. 2019; Stanovsky, Smith, and Zettlemoyer 2019), dialogue generation (Liu et al. 2020), and sentiment analysis (Kiritchenko and Mohammad 2018).

Adversarial approaches are widely used to reduce bias related to protected attributes. In these methods, the text encoder strives to obscure protected attributes so that the discriminator cannot identify them (Li, Baldwin, and Cohn 2018; Zhang, Lemoine, and Mitchell 2018; Han, Baldwin, and Cohn 2021b). However, these methods require training examples labeled with protected attributes, which presents several challenges. First, we may not be aware of the specific biases, like gender or age bias, that require mitigation (Orgad and Belinkov 2023). Second, obtaining protected labels can be difficult due to privacy regulations and ethical concerns, leading to few users publicly disclosing their protected attributes (Han, Baldwin, and Cohn 2021a). Moreover, prior research has typically focused on mitigating a single type of bias (Schuster et al. 2019; Clark, Yatskar, and Zettlemoyer 2019; Utama, Moosavi, and Gurevych 2020). However, in practice, corpora often contain multiple types of biases, each with varying levels of detection difficulty.

In this paper, we address the challenge of bias removal without prior knowledge of bias labels by proposing a Multilayer Adversarial Bias Removal (MABR) framework. We introduce a series of auxiliary classifiers as bias detectors. The rationale behind using multiple classifiers is to capture different aspects and levels of bias present in the data. Each classifier operates on different layers of the main model’s encoder, based on the insight that different layers of the encoder may capture different aspects of bias. Lower-level feature maps may capture word-level biases, such as associating words like “nurse” or “secretary” predominantly with female pronouns or contexts, and words like “engineer” or “pilot” with male pronouns. Higher-level feature maps may capture more subtle gender biases, such as associating leadership qualities with male-associated terms or nurturing qualities with female-associated terms, or inferring competence and ambition based on gendered names or contexts. These biases manifest in more nuanced ways, such as assuming managerial roles are more suited to one gender over another, reflecting societal stereotypes in professional settings.

Once biased samples are detected, we apply adversarial training to mitigate these biases. We introduce domain discriminators at each layer of the main model’s encoder. The goal of the adversarial training is to make the representations learned by the main model invariant to the biases identified by the auxiliary classifiers. To achieve this, we employ a Reverse Gradient Layer during backpropagation, which ensures that the main model generates feature representations that are indistinguishable with respect to the domain discriminators. This process encourages the alignment of feature distributions between biased and unbiased samples, thereby reducing the influence of biased features on the model’s predictions.

However, this approach alone is insufficient. The bias detector tends to detect relatively easy samples where the biased features are obvious or the sentence structure is simple. Building on the findings of Liu et al. (2021), we recognize that standard training of language models often results in models with low average test errors but high errors on specific groups of examples. These performance disparities are particularly pronounced when spurious correlations are present. Therefore, we also consider training examples misclassified by the main model as hard biased samples, supplementing the samples detected by the bias detector.

We conduct experiments on two English NLP tasks and two types of social demographics: sentiment analysis with gender and occupation classification with race. Our MABR method successfully reduces bias, sometimes even outperforming methods that use demographic information. This indicates that MABR may offer a more robust solution for bias mitigation compared to other existing methods.

Our contributions are as follows:

1. We introduce MABR, an adversarial bias removal method that does not require prior knowledge of specific biases.   
2. We enhance bias detection in MABR by enabling it on all layers of the main model’s encoder, capturing various types of biases.   
3. We demonstrate that MABR can successfully reduce bias without protected-label data and is robust across different tasks and datasets.

# Related Work

Research suggests various methods for mitigating social biases in NLP models applied to downstream tasks. Some approaches focus on preprocessing the training data, such as converting biased words to neutral alternatives (De-Arteaga et al. 2019) or to those that counteract bias (Zhao et al. 2018), or balancing each demographic group in training (Zhao et al. 2018; Wang et al. 2019; Lahoti et al. 2020; Han, Baldwin, and Cohn 2022). Others focus on removing demographic information from learned representations, for instance, by applying post-hoc methods to the neural representations of a trained model (Ravfogel et al. 2020, 2022; Iskander, Radinsky, and Belinkov 2023). Adversarial training is also a common strategy (Li, Baldwin, and Cohn 2018; Zhang, Lemoine, and Mitchell 2018; Elazar and Goldberg 2018; Wang et al. 2019; Han, Baldwin, and Cohn 2021b). However, all these methods require prior knowledge of the specific bias to be addressed, such as gender bias. Furthermore, many of these approaches depend on demographic annotations for each data instance. For example, to address gender bias, each data sample must be annotated to indicate whether it pertains to a male or female subject. In contrast, our method does not require any prior knowledge about the bias. Additionally, while the authors of these studies select hyperparameters based on the fairness metrics they aim to optimize, we choose our hyperparameters without explicitly measuring fairness metrics.

Recent studies have also explored fairness in machine learning through alternative approaches, such as discovering intersectional unfairness (Xu et al. 2024), learning fairness across multiple subgroups (Shui et al. 2022b), and aligning representations implicitly for fair learning (Shui et al. 2022a). Other work in related areas has proposed leveraging prompt-based learning (Yin, Wang, and Ling 2024) or masking mechanisms (Yin et al. 2024) to mitigate domain gaps. These methods contribute to advancing fairness research but still differ from our approach, which avoids both demographic annotations and prior bias knowledge.

# Methodology

# Problem Formulation

We consider the problem of general multi-class classification. The dataset $\mathcal D \ = \ \{ ( x _ { i } , \overline { { y _ { i } } } , z _ { i } ) \} _ { i = 1 } ^ { N }$ comprises triples consisting of an input $x _ { i } \in { \mathcal { X } }$ , a label $y _ { i } \in \mathcal { V }$ , and a protected attribute $z _ { i } \in \mathcal { Z }$ , which corresponds to a demographic group, such as gender. The attribute $z _ { i }$ is unknown, meaning it is not accessible during training stages. Our objective is to learn a mapping $f _ { M } : \mathcal { X } \to \mathbb { R } ^ { | \mathcal { V } | }$ , where $f _ { M }$ , referred to as the main model, is resilient to demographic variations introduced by $z _ { i }$ , with $| \mathcal { V } |$ denoting the number of classes.

The model’s fairness is evaluated using various metrics. A fairness metric maps a model’s predictions and the associated protected attributes to a numerical measure of bias: $M : ( \bar { \mathbb { R } } ^ { | y | } , \mathcal { Z } ) \to \mathbb { R }$ . The closer the absolute value of this measure is to 0, the fairer the model is considered to be.

# Bias Detection

Since the protected attribute $\mathcal { Z }$ is unknown, we detect possible biased samples automatically and dynamically. To achieve this, we introduce a bias detector for each layer of the encoder, as depicted in Fig. 1. Given the embedding output of a specific layer, the bias detector on that layer is trained to predict whether the main model will successfully predict the correct label for the main task for each training sample. Let $L$ denote the total number of layers in the encoder. It is formulated as $f _ { B _ { l } } : g _ { l } ( \pmb { \chi } )  \mathbb { R } ^ { | s _ { l } | }$ for each layer $l$ , where $g _ { l } ( x )$ represents the output embedding of the $l _ { t h }$ layer and $l$ ranges from 1 to $L$ . Here, $s _ { l }$ is an indicator function defined as: $s _ { l } = \mathbb { I } ( f _ { M } ( x ) = y )$ , which is dynamic and changes across different epochs of the training process. Notably, the bias detector has no knowledge of the original task, and the prediction is made without access to the main task label. The intuition behind this approach is that if the bias detector can successfully predict the main model’s behavior based solely on a single embedding layer output, without access to the task label, it indicates that the main model likely relies on a specific bias feature as a shortcut, leading to shallow decision-making.

Initially, we train both the main model and the bias detectors using the standard training process, where both models are optimized using cross-entropy loss.

The cross-entropy loss for the main model, represented as ${ \mathcal { L } } _ { \mathrm { m a i n } }$ , is defined in the equation below:

![](images/8ab15498c4e07986682543fb0000f51f86a14e42595f2ae4ffb276b38f9b75e8.jpg)  
Figure 1: Schematic Overview of the MABR Framework. The left panel illustrates the overall architecture of the model for main task and bias detection. The right panel details the domain adversarial training process upon each encoder layer.

$$
\mathcal { L } _ { \operatorname* { m i n } } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \sum _ { c = 1 } ^ { | \mathcal { V } | } y _ { i , c } \log \left( f _ { M , c } ( x _ { i } ) \right)
$$

where $N$ is the number of training samples, $y _ { i , c }$ is a binary indicator (0 or 1) indicating whether class label $c$ is the correct classification for sample $i$ , and $f _ { M , c } ( \boldsymbol { x } _ { i } )$ is the predicted probability of the main model for class $c$ .

The cross-entropy loss for the bias detector at the $l _ { t h }$ layer, $\mathcal { L } _ { \mathrm { b i a s } }$ , is defined as follows:

$$
\begin{array} { r l } { \left. { \mathcal { L } _ { \mathrm { b i a s } } ^ { l } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left( s _ { i } ^ { l } \log \left( f _ { B _ { l } } ( g _ { l } ( x _ { i } ) \right) \right) } } \\ & { + \ ( 1 - s _ { i } ^ { l } ) \log \left( 1 - f _ { B _ { l } } ( g _ { l } ( x _ { i } ) ) \right) \right) } \end{array}
$$

where $s _ { i } ^ { l } = \mathbb { I } ( f _ { M } ( x _ { i } ) = y _ { i } )$ is an indicator function that denotes whether the main model’s prediction is correct for sample $i$ at layer $l$ , and $f _ { B _ { l } } ( g _ { l } ( x _ { i } ) )$ is the predicted probability of the bias detector at layer $l$ .

The total loss for the bias detectors across all layers, ${ \mathcal { L } } _ { \mathrm { b i a s } }$ , is obtained by summing the losses from each layer, as formulated below:

$$
\mathcal { L } _ { \mathrm { b i a s } } = \sum _ { l = 1 } ^ { L } \mathcal { L } _ { \mathrm { b i a s } } ^ { l }
$$

After the initial training phase, we utilize the bias detectors $\boldsymbol { B } = \{ f _ { B _ { l } } \} _ { l = 1 } ^ { L }$ to identify biased samples. If the bias detector can predict whether the main model is correct or incorrect on a sample (i.e., $\sigma ( f _ { B } ( x ) )$ is high), without knowing the task at hand, then the sample likely contains some simple but biased features. This intuition aligns with the claim that in the context of complex language understanding tasks, all simple feature correlations are spurious (Gardner et al. 2021; Orgad and Belinkov 2023). Therefore, the samples for which the bias detector predicts a score higher than the threshold $\tau$ are considered biased samples, where $\tau$ is a hyperparameter.

Nevertheless, this approach tends to detect samples with more apparent biases or simpler sentence structures. To address this limitation, we incorporate insights from Liu et al. (2021), which highlight that language models trained with standard methods can achieve low average test errors while exhibiting high errors on certain groups due to spurious correlations. Consequently, we also consider misclassified training examples as hard biased samples. Formally, for a sample $x _ { i }$ , if $\dot { \hat { y } } _ { i } = f _ { M } ( x _ { i } ) \neq y _ { i }$ , it is deemed a hard biased sample. This supplementary set of hard biased samples enhances the identification of biased instances beyond those detected by the bias detector alone.

# Adversarial Training

As illustrated in the right part of Fig. 1, we employ an adversarial training process to mitigate the biases identified by the bias detectors and hard biased samples. This process involves two primary components: the main model $f _ { M }$ and a set of domain discriminators $\mathcal { G } = \{ G _ { l } \} _ { l = 1 } ^ { L }$ . The goal of adversarial training is to make the representations learned by the main model invariant to the identified biases.

The main model $f _ { M }$ can be decomposed into an encoder $g$ and a classifier $h _ { M }$ , such that $f _ { M } = h _ { M } \circ g$ . Each domain discriminator $G _ { l }$ attempts to predict whether a sample is biased or not based on the representations generated by $g _ { l }$ .

For adversarial training, we employ the Reverse Gradient Layer (Ganin and Lempitsky 2014) to ensure that the main model learns to generate representations that are invariant to the identified biases. The Reverse Gradient Layer functions by reversing the gradient during backpropagation, thereby encouraging the main model to produce feature representations that are indistinguishable with respect to the domain discriminators.

The adversarial training is conducted at each layer of the encoder separately. The adversarial loss for a sample $x _ { i }$ at layer $l$ is computed as follows:

$$
\begin{array} { c l } { \displaystyle \mathcal { L } _ { \mathrm { a d v } } ^ { l } = - \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \big ( z _ { i } ^ { l } \log \big ( G _ { l } \big ( g _ { l } \big ( x _ { i } \big ) \big ) \big ) } \\ { \displaystyle + \big ( 1 - z _ { i } ^ { l } \big ) \log \big ( 1 - G _ { l } \big ( g _ { l } \big ( x _ { i } \big ) \big ) \big ) \big ) } \end{array}
$$

where $z _ { i } ^ { l }$ is an indicator variable that denotes whether the sample $x _ { i }$ is considered biased (i.e., identified by the bias detector or misclassified by the main model).

The total loss for the adversarial training is a combination of the main model’s cross-entropy loss, the bias detector’s cross-entropy loss, and the adversarial loss at each layer. The combined loss function is given by:

$$
\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { m a i n } } + \sum _ { l = 1 } ^ { L } \mathcal { L } _ { \mathrm { b i a s } } ^ { l } + \sum _ { l = 1 } ^ { L } \mathcal { L } _ { \mathrm { a d v } } ^ { l }
$$

During backpropagation, the weights of the encoder $g$ are updated to minimize the total loss ${ \mathcal { L } } _ { \mathrm { t o t a l } }$ . Let $\theta _ { g }$ represent the weights of the encoder $g$ . The update rule for $\theta _ { g }$ is:

$$
\theta _ { g } \gets \theta _ { g } - \eta \left( \frac { \partial \mathcal { L } _ { \mathrm { m a i n } } } { \partial \theta _ { g } } - \sum _ { l = 1 } ^ { L } \frac { \partial \mathcal { L } _ { \mathrm { a d v } } ^ { l } } { \partial \theta _ { g } } \right)
$$

It is important to note that the gradient contribution from the adversarial loss ${ \mathcal { L } } _ { \mathrm { a d v } } ^ { l }$ is reversed by the Reverse Gradient Layer, and the gradient from the bias detectors is not used for updating $\theta _ { g }$ . The whole procedure is detailed in Algorithm 1.

# Experiments

# Tasks and Models

In our experiments, we investigate two classification tasks, each associated with a distinct type of bias:

Sentiment Analysis and Race Following the methodology of previous research (Elazar and Goldberg 2018; Orgad and Belinkov 2023), we employ a dataset from Blodgett, Green, and O’Connor (2016) that consists of 100,000 tweets to explore dialect differences in social media language. This dataset allows us to analyze racial identity by categorizing each tweet as either African American English (AAE) or Mainstream US English (MUSE), commonly referred to as Standard American English (SAE). The classification leverages the geographical information of the tweet authors. Additionally, Elazar and Goldberg (2018) used emojis embedded in tweets as sentiment indicators to facilitate the sentiment classification task.

Algorithm 1: Adversarial Training with Bias Detection and

<html><body><table><tr><td>Mitigation</td></tr><tr><td>Require: Dataset D= {(xi, yi, zi)}=</td></tr><tr><td>Require:Encoder g,Classifier hM，Bias Detectors B {fB1}L-1,Domain Discriminators D = {D)}-1</td></tr><tr><td>Require: Threshold T,Learning rate n</td></tr><tr><td>1: Initialize the main model fM = hM  g</td></tr><tr><td>2:Initialize bias detectors B and domain discriminators D</td></tr><tr><td>3:Phase 1: Initial Training (1 epoch)</td></tr><tr><td>4:for each mini-batchMin D do</td></tr><tr><td>5: Compute main model outputs: fM(x)</td></tr><tr><td>6: Compute cross-entropy loss: Lmain</td></tr><tr><td>7: Update main model parameters to minimize Lmain 8: Compute bias detector outputs: fB(gt(x))</td></tr><tr><td>9: Compute cross-entropy loss for bias detectors: Lbias</td></tr><tr><td>10: Update bias detector parameters to minimize Lbias</td></tr><tr><td>11:end for</td></tr><tr><td>12:Phase 2: Adversarial Training (T epochs)</td></tr><tr><td>13:for epoch= 1 to Tdo 14: foreach mini-batch Min D do</td></tr><tr><td>15: Compute main model outputs: fM(x)</td></tr><tr><td>16: Compute bias detector outputs: fBt(gt(x))</td></tr><tr><td>17: Identify biased samples using threshold T and mis-</td></tr><tr><td>classified main model samples</td></tr><tr><td>18: Compute adversarial loss for domain discrimina tors:Cld</td></tr><tr><td>19: Computetota loss oal=i+∑as+</td></tr><tr><td>∑-1Cady 20:</td></tr><tr><td></td></tr><tr><td>21: end for</td></tr><tr><td>22:end for</td></tr><tr><td></td></tr><tr><td>23:Output: Trained main model fM,bias detectors B,and domain discriminators D</td></tr></table></body></html>

Occupation Classification and Gender Bias Following previous research (Orgad and Belinkov 2023), we utilize the dataset provided by De-Arteaga et al. (2019), which comprises 400,000 online biographies, to examine gender bias in occupational classification. The task involves predicting an individual’s occupation using a portion of their biography, specifically excluding the first sentence that explicitly mentions the occupation. The protected attribute in this context is gender, and each biography is labeled with binary gender categories based on the pronouns used within the text, reflecting the individual’s self-identified gender.

# Metrics

Research by Orgad and Belinkov (2022) demonstrates that different fairness metrics can respond variably to debiasing methods. Specifically, methods designed to improve fairness according to one metric may actually worsen outcomes when measured by another. Therefore, to achieve a comprehensive analysis of the performance of our method and previous baselines, we measure multiple metrics.

True Positive Rate gap The True Positive Rate (TPR) gap indicates the difference in performance between two demographic groups, such as females versus males. For gender, we measure the TPR gap for label $y$ as $G A { \breve { P _ { T P R , y } } } ~ = ~ | T P R _ { y } ^ { F } ~ - ~ T P R _ { y } ^ { M } |$ . To provide a more comprehensive assessment, we calculate the root-meansquare form of the TPR gap (denoted $T P R _ { R M S } )$ , which is $\begin{array} { r } { \sqrt { \frac { 1 } { | Y | } \sum _ { y \in Y } ( G A P _ { T P R , y } ) ^ { 2 } } } \end{array}$ , following previous research (Ravfogel et al. 2020, 2022; Orgad and Belinkov 2023).

Independence This metric evaluates the statistical independence between the model’s predictions and the protected attributes. According to the independence rule (demographic parity), the probability of a positive prediction should be the same regardless of the protected attribute. To measure this, we calculate the Kullback-Leibler (KL) divergence between two distributions: $K L ( P ( \hat { Y } ) , P ( \hat { Y } | Z = z ) ;$ ), $\forall z \in { \mathcal { Z } }$ . Summing these values over $z$ gives a single measure reflecting the model’s independence. This metric does not consider the true labels (gold labels); instead, it intuitively measures how much the model’s behavior varies across different demographic groups.

# Sufficiency

This metric measures the statistical dependence between the target label given the model’s prediction and the protected attributes. It uses the Kullback-Leibler divergence between two distributions: $K L ( P ( y | r ) , P ( y | r , z = z ) { \bar { ) } }$ , for all $\boldsymbol { r } \in \mathcal { V }$ and $z \in { \mathcal { Z } }$ . The values are summed over $r$ and $z$ to produce a single measure. Related to calibration and precision gap, this metric assesses if a model disproportionately favors or penalizes a specific demographic group (Liu, Simchowitz, and Hardt 2019).

# Implementation Details

We experiment with BERT (Devlin et al. 2018) and DeBERTa-v1 (He et al. 2020) as backbone models, utilizing the transformer model as a text encoder with its output fed into a linear classifier. The text encoder and linear layer are fine-tuned for the downstream task. We implement the MABR framework using the Huggingface Transformers library (Wolf et al. 2020). The batch size is set to 64, enabling dynamic adversarial training per batch. We set the learning rate to 1e-3 for the bias detector and domain classifier, and 2e-5 for the model. The threshold $\tau$ is selected to ensure approximately $30 \%$ of samples fall outside it after initial training. For training epochs, we balance task accuracy and fairness using the “distance to optimum” (DTO) criterion introduced by Han, Baldwin, and Cohn (2022). Model selection is performed without a validation set with demographic annotations, choosing the largest epoch while limiting accuracy reduction. We use 0.98 of the maximum achieved accuracy on the task as the threshold to stop training. Other hyperparameters follow the default settings provided by the Transformers library.

# Baselines

We compare MABR with the following baselines:

Finetuned The MABR model architecture, trained for downstream tasks without any debiasing mechanisms.

INLP (Ravfogel et al. 2020) A post-hoc method that trains linear classifiers to predict attributes, then projects representations onto their null-space to remove attribute information and mitigate bias.

R-LACE (Ravfogel et al. 2022) Eliminates specific concepts from neural representations using a constrained minimax optimization framework. It employs a projection matrix to remove the linear subspace corresponding to the targeted concept, preventing linear predictors from recovering it.

BLIND (Orgad and Belinkov 2023) Identifies biased samples through an auxiliary model and reduces their weight during training. Effective for single biases but lacks broader anti-bias capabilities.

JTT (Liu et al. 2021) A two-stage framework: first identifies high-loss examples with empirical risk minimization (ERM), then upweights these in the final training to boost worst-group performance without group annotations.

# Results

# Overall Results

Tables 1 and 2 present the performance metrics for various models on the sentiment analysis and occupation classification tasks, respectively. The vanilla fine-tuning baseline yields the highest accuracy but also the worst bias (highest fairness metrics) for both BERT and RoBERTa, and across both tasks. This outcome is expected due to the inherent trade-off between fairness and performance.

Sentiment Analysis For the sentiment analysis task (Table 1), MABR effectively reduces bias. On BERT, compared to the finetuned baseline, MABR lowers $T P R _ { R M S }$ by $1 0 . 6 \%$ and Independence by $3 . 6 \%$ , with only a $0 . 5 \%$ drop in accuracy. While R-LACE achieves comparable fairness, its accuracy decreases by $3 . 2 \%$ . JTT and BLIND show similar accuracy to MABR but fall short in mitigating bias consistently across metrics, with varying performance depending on the metric.

For RoBERTa, MABR also demonstrates strong bias reduction, decreasing $T P R _ { R M S }$ by $1 3 . 5 \%$ and Independence by $2 . 9 \%$ , with a $1 . 6 \%$ drop in accuracy. Although INLP achieves the largest $T P R _ { R M S }$ reduction $( 1 7 . 3 \% )$ , it suffers a substantial $1 3 . 2 \%$ accuracy drop. JTT marginally outperforms MABR in sufficiency reduction, but overall, MABR offers the best balance of fairness and accuracy.

Occupation Classification For the occupation classification task (Table 2), the finetuned baseline shows less pronounced bias, likely due to the dataset’s lower inherent bias. Nevertheless, MABR significantly improves fairness, reducing $T P R _ { R M S }$ by $2 . 4 \%$ and Independence by $5 . 0 \%$ , with only a $1 . 2 \%$ accuracy drop. R-LACE achieves slightly higher accuracy but offers limited bias reduction, with $T P R _ { R M S }$ decreasing by only $0 . 2 \%$ .

For RoBERTa, MABR matches R-LACE in accuracy but provides superior fairness, reducing $T P R _ { R M S }$ by $2 . 3 \%$ and

Table 1: Performance metrics on the sentiment analysis task, averaged over 5 independent experimental runs.   

<html><body><table><tr><td rowspan="2"></td><td colspan="4">BERT</td><td colspan="4">RoBERTa</td></tr><tr><td>Acc ↑</td><td>TPRRMs ↓</td><td>Indep↓</td><td>Suff ←</td><td>Acc ↑</td><td>TPRRMs↓</td><td>Indep √</td><td>Suff↓</td></tr><tr><td>Finetuned</td><td>0.771</td><td>0.243</td><td>0.039</td><td>0.028</td><td>0.779</td><td>0.261</td><td>0.035</td><td>0.031</td></tr><tr><td>INLP</td><td>0.753</td><td>0.198</td><td>0.021</td><td>0.025</td><td>0.647</td><td>0.088</td><td>0.010</td><td>0.030</td></tr><tr><td>RLACE</td><td>0.739</td><td>0.140</td><td>0.009</td><td>0.021</td><td>0.751</td><td>0.157</td><td>0.014</td><td>0.032</td></tr><tr><td>JTT</td><td>0.762</td><td>0.191</td><td>0.014</td><td>0.028</td><td>0.753</td><td>0.185</td><td>0.013</td><td>0.026</td></tr><tr><td>BLIND</td><td>0.759</td><td>0.202</td><td>0.029</td><td>0.024</td><td>0.741</td><td>0.213</td><td>0.024</td><td>0.033</td></tr><tr><td>MABR</td><td>0.766</td><td>0.137</td><td>0.003</td><td>0.021</td><td>0.763</td><td>0.126</td><td>0.006</td><td>0.028</td></tr><tr><td>-multi</td><td>0.768</td><td>0.145</td><td>0.010</td><td>0.025</td><td>0.762</td><td>0.162</td><td>0.014</td><td>0.033</td></tr><tr><td>-hard</td><td>0.768</td><td>0.139</td><td>0.006</td><td>0.022</td><td>0.763</td><td>0.142</td><td>0.009</td><td>0.031</td></tr><tr><td>JTT-Disc + AdvTrain</td><td>0.768</td><td>0.173</td><td>0.013</td><td>0.027</td><td>0.763</td><td>0.169</td><td>0.017</td><td>0.034</td></tr><tr><td>Bias-Disc + Upweight</td><td>0.766</td><td>0.168</td><td>0.011</td><td>0.025</td><td>0.762</td><td>0.160</td><td>0.015</td><td>0.033</td></tr></table></body></html>

<html><body><table><tr><td rowspan="2"></td><td colspan="4">BERT</td><td colspan="4">RoBERTa</td></tr><tr><td>Acc ↑</td><td>TPRRMs↓</td><td>Indep↓</td><td>Suff ↓</td><td>Acc ↑</td><td>TPRRMS ↓</td><td>Indep ↓</td><td>Suff ↓</td></tr><tr><td>Finetuned</td><td>0.869</td><td>0.135</td><td>0.149</td><td>1.559</td><td>0.863</td><td>0.132</td><td>0.144</td><td>1.600</td></tr><tr><td>INLP</td><td>0.857</td><td>0.131</td><td>0.137</td><td>1.216</td><td>0.851</td><td>0.123</td><td>0.132</td><td>1.052</td></tr><tr><td>RLACE</td><td>0.868</td><td>0.133</td><td>0.144</td><td>1.413</td><td>0.852</td><td>0.124</td><td>0.127</td><td>1.362</td></tr><tr><td>JTT</td><td>0.849</td><td>0.132</td><td>0.132</td><td>1.417</td><td>0.844</td><td>0.139</td><td>0.139</td><td>1.397</td></tr><tr><td>BLIND</td><td>0.826</td><td>0.136</td><td>0.123</td><td>1.097</td><td>0.839</td><td>0.123</td><td>0.122</td><td>0.906</td></tr><tr><td>MABR</td><td>0.857</td><td>0.101</td><td>0.099</td><td>1.031</td><td>0.852</td><td>0.109</td><td>0.100</td><td>0.821</td></tr><tr><td>-multi</td><td>0.859</td><td>0.128</td><td>0.112</td><td>1.054</td><td>0.853</td><td>0.121</td><td>0.117</td><td>0.907</td></tr><tr><td>-hard</td><td>0.858</td><td>0.119</td><td>0.111</td><td>1.033</td><td>0.853</td><td>0.114</td><td>0.116</td><td>0.883</td></tr><tr><td>JTT-Disc+AdvTrain</td><td>0.855</td><td>0.129</td><td>0.118</td><td>1.061</td><td>0.850</td><td>0.133</td><td>0.125</td><td>0.876</td></tr><tr><td>Bias-Disc + Upweight</td><td>0.856</td><td>0.131</td><td>0.120</td><td>1.059</td><td>0.851</td><td>0.137</td><td>0.121</td><td>0.892</td></tr></table></body></html>

Table 2: Performance metrics on the occupation classification task, averaged over 5 independent experimental runs

Independence by $4 . 4 \%$ . MABR consistently outperforms baselines across models and metrics, underscoring its robustness in mitigating bias.

# Ablation Study

To better understand our proposed framework, we conducted ablation studies to evaluate the effectiveness of each component. The results are shown in Tables 1 and 2. The notation “-multi” denotes the removal of the multi-layer bias detection component, and the notation “-hard” signifies the omission of the adversarial training with hard examples. Additionally, we analyzed two variants: “JTT-Disc $^ +$ AdvTrain,” which combines JTT’s bias discovery with our adversarial training, and “Bias-Disc $^ +$ Upweight,” which integrates our bias detection with JTT’s upweighting strategy.

For the sentiment analysis task (Table 1), removing the multi-layer bias detection component (“-multi”) results in a slight increase in accuracy but worsens bias performance, with $T P R _ { R M S }$ rising by $0 . 8 \%$ and Independence by $0 . 7 \%$ . Similarly, omitting the hard example detection process (“- hard”) leads to an increase in bias metrics, with $T P R _ { R M S }$ increasing by $0 . 2 \%$ and Independence by $0 . 3 \%$ . The “JTTDisc $^ +$ AdvTrain” variant performs better than JTT alone, as adversarial training mitigates bias more effectively, but its simpler bias discovery mechanism limits its performance. Meanwhile, the “Bias-Disc $^ +$ Upweight” variant improves bias metrics compared to JTT but underperforms MABR, as upweighting at the final layer lacks the nuanced mitigation provided by multi-layer processing. These findings emphasize the importance of both multi-layer bias detection and hard example training, with the former having a more substantial impact.

For the occupation classification task (Table 2), removing the multi-layer bias detection component (“-multi”) leads to a decrease in accuracy and worsened bias performance, with $T P R _ { R M S }$ rising by $2 . 7 \%$ and Independence by $1 . 3 \%$ . Similarly, omitting the hard example detection process (“-hard”) increases $T P R _ { R M S }$ by $1 . 8 \%$ and Independence by $1 . 2 \%$ . The “JTT-Disc $^ +$ AdvTrain” and “Bias-Disc $^ +$ Upweight” variants follow similar trends as in the sentiment analysis task, improving upon JTT in bias mitigation but failing to reach MABR’s level of effectiveness due to limitations in either bias detection or mitigation strategies.

Considering the results across both tasks, MABR achieved a larger reduction in bias for sentiment analysis (e.g., a $1 0 . 6 \%$ decrease in $T P R _ { R M S }$ compared to the finetuning baseline) than for occupation classification (e.g., a $3 . 4 \%$ decrease in $T P R _ { R M S } ,$ ). This highlights the critical role of MABR’s multi-layer bias detection and adversarial training in addressing deeply ingrained biases, demonstrating their effectiveness in enhancing fairness across tasks with complex or subtle biases.

![](images/1d8e52bf4613a88668219d1596d23f83f20f92c940181e6afad18b4bb8975c92.jpg)  
Figure 2: Accuracies for each layer of domain adversarial training components when training with BERT on the sentiment classification task. The orange line represents pre-adversarial training and the blue line represents postadversarial training.

![](images/ab94d2cfa848b077b86fee7d1508d97693cfa4dbf9e405406ae0e9a543c2e2d3.jpg)  
Figure 3: Accuracies for each layer of domain adversarial training components when training with Roberta on the sentiment classification task.

# Layer Level Analysis

Figure 2 and 3 illustrate the accuracy for each layer of the bias detectors and domain classifiers before and after the adversarial training process for BERT and RoBERTa, respectively. Initially, the accuracy of the bias detectors is notably high. For BERT, all detectors achieve accuracies greater than 0.79 before adversarial training and remain above 0.76 afterward. Similarly, RoBERTa’s detectors maintain strong performance, with accuracies exceeding 0.74. This indicates that the bias detectors effectively determine whether the main model succeeds in its task without needing access to the main task labels. This observation supports our assumption that many samples identified by the bias detectors rely on biased features as shortcuts to make predictions, consistent with the findings of Orgad and Belinkov (2023).

Furthermore, we notice that the adversarial training process significantly reduces the accuracy of the bias detectors, demonstrating that the adversarial training has effectively mitigated the bias features in the embedding maps. This makes it harder for the bias detectors to identify easy biases. However, the bias detectors still maintain relatively high accuracy post-training because they are trained during the process simultaneously. As a result, the labels of the samples input to the domain classifier are dynamically refined, which is a significant difference over previous adversarial training methods (Elazar and Goldberg 2018; Wang et al. 2019; Han, Baldwin, and Cohn 2021b).

Table 3: Performance metrics with the bias detector and domain classifier removed at specific layer levels during adversarial training using the MABR method on the sentiment analysis task with BERT.   

<html><body><table><tr><td></td><td>Acc ↑</td><td>TPRRMs↓</td><td>Indep↓</td><td>Suff↓</td></tr><tr><td>MABR</td><td>0.766</td><td>0.137</td><td>0.003</td><td>0.021</td></tr><tr><td>- layer[1:5]</td><td>0.766</td><td>0.140</td><td>0.005</td><td>0.022</td></tr><tr><td>- layer[6:9]</td><td>0.767</td><td>0.142</td><td>0.005</td><td>0.022</td></tr><tr><td>- layer[10:12]</td><td>0.767</td><td>0.142</td><td>0.007</td><td>0.023</td></tr></table></body></html>

Table 4: Performance metrics with the bias detector and domain classifier removed at specific layer levels during adversarial training using the MABR method on the sentiment analysis task with Roberta.   

<html><body><table><tr><td></td><td>Acc ↑</td><td>TPRRMs ↓</td><td>Indep↓</td><td>Suff ↓</td></tr><tr><td>MABR</td><td>0.763</td><td>0.126</td><td>0.006</td><td>0.028</td></tr><tr><td>- layer[1:5]</td><td>0.766</td><td>0.152</td><td>0.013</td><td>0.031</td></tr><tr><td>- layer[6:9]</td><td>0.765</td><td>0.148</td><td>0.012</td><td>0.030</td></tr><tr><td>- layer[10:12]</td><td>0.767</td><td>0.158</td><td>0.012</td><td>0.028</td></tr></table></body></html>

We also observe that different layers respond differently to the adversarial training process. As depicted in Figure 2 and 3, the early layers behave similarly. The reduction in the accuracy of the bias detector is relatively low, and the accuracy of the domain classifiers remains quite high. This suggests that the lower layers capture fundamental features that are less susceptible to bias, thereby leaving limited room for mitigating bias features without compromising the final accuracy. However, this does not imply that mitigation at the lower levels is unimportant. As evidenced by the data in Tables 3 and 4, if we remove the adversarial training process from the lower layers (layer 1 to 5), the fairness metrics still degrade significantly.

# Conclusion

In this paper, we introduced MABR, a novel adversarial training strategy that mitigates biases across various encoder layers of LLMs. By employing multiple auxiliary classifiers to capture different aspects and levels of bias, our approach effectively identifies and reduces social biases without prior knowledge of bias types or demographic annotations. This method significantly improves fairness in tasks such as sentiment analysis and occupation classification, matching or exceeding the performance of models requiring detailed demographic insights. Our findings underscore the importance of leveraging the distinct capabilities of different model layers in capturing nuanced bias features, marking a significant advancement in bias mitigation techniques.
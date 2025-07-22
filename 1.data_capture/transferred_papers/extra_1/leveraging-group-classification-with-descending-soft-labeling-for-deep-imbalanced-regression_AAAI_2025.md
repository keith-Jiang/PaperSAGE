# Leveraging Group Classification with Descending Soft Labeling for Deep Imbalanced Regression

Ruizhi $\mathbf { P u } ^ { 1 * }$ , Gezheng ${ \bf X } { \bf u } ^ { 1 }$ , Ruiyi Fang1, Bing-Kun Bao2, Charles Ling1†, Boyu Wang1†

1 Department of Computer Sceince, Western University 2 School of Computer Science, Nanjing University of Posts and Telecommunications

# Abstract

Deep imbalanced regression (DIR), where the target values have a highly skewed distribution and are also continuous, is an intriguing yet under-explored problem in machine learning. While recent works have already shown that incorporating various classification-based regularizers can produce enhanced outcomes, the role of classification remains elusive in DIR. Moreover, such regularizers (e.g., contrastive penalties) merely focus on learning discriminative features of data, which inevitably results in ignorance of either continuity or similarity across the data. To address these issues, we first bridge the connection between the objectives of DIR and classification from a Bayesian perspective. Consequently, this motivates us to decompose the objective of DIR into a combination of classification and regression tasks, which naturally guides us toward a divide-and-conquer manner to solve the DIR problem. Specifically, by aggregating the data at nearby labels into the same groups, we introduce an ordinal groupaware contrastive learning loss along with a multi-experts regressor to tackle the different groups of data thereby maintaining the data continuity. Meanwhile, considering the similarity between the groups, we also propose a symmetric descending soft labeling strategy to exploit the intrinsic similarity across the data, which allows classification to facilitate regression more effectively. Extensive experiments on realworld datasets also validate the effectiveness of our method.

Appendix — https://github.com/RuizhiPu-CS/Group-DIR

# Introduction

Data imbalance exists ubiquitously in real-world scenarios, posing significant challenges to machine learning tasks as certain labels may be less observed than others or even missed during training. Although the imbalanced problem has been extensively studied in the field of classification (He and Garcia 2009), how to tackle deep imbalanced regression (DIR) is still under-explored.

Due to the continuity of the label space and the dependence of data across nearby targets (Yang et al. 2021), previous solutions in DIR primarily focused on estimating

MSE ↓ MODELPredicted Labels 价 Classification Regulizations Symmetric Descending Soft GroupLabels   
groupa ↓ Regressora Classifier Regressorb MSE   
groupb ↑ Regressor c Group Classification Regulizations groupc

accurate imbalanced label density, such as label distribution smoothing (LDS), feature distribution smoothing (FDS) (Yang et al. 2021) and re-weighting (Cui et al. 2019; Branco, Torgo, and Ribeiro 2017). Meanwhile, (Ren et al. 2022) proposed a balanced Mean Square Error (B-MSE) loss to accommodate the imbalanced distribution in the label space. Recent works incorporated classification regularizers with the Mean Square Error (MSE) in DIR, such as contrastive regularization (Zha et al. 2023a; Keramati, Meng, and Evans 2023), entropy regularization (Zhang et al. 2023), and feature ranking regularization (Gong, Mori, and Tung 2022), which have achieved significant performance improvements. Moreover, (Pintea et al. 2023) formalized the relation between the balanced and imbalanced regression and empirically investigated how classification can help regression.

Although incorporating classification regularizers in DIR has already achieved enhanced output, the relationship between the objectives of classification and DIR remains elusive. In the meantime, these classification regularizers would also force the model to focus more on the discriminative feature which is inappropriate for regression tasks. For example, for a facial-image-based age regression task, the images corresponding to nearby labels exhibit both continuity and similarity. A photo of a 40-year-old person should resemble those of both 35-year-olds and 45-year-olds, and also reflect an intermediate stage in age-related features. However, such property (data similarity) has been always ignored in existing classification-based methods.

In this paper, to investigate the connections between the classification and DIR, we revisit the objective of DIR from a Bayesian perspective. We show that the objective of DIR can be decomposed into the combination of both group classification and sample regression within each group. Inspired by this finding, we can explicitly leverage the classification to help DIR in a divide-and-conquer manner.

Specifically, considering that data with nearby labels would naturally be similar (Yang et al. 2021; Pintea et al. 2023) in DIR, we aggregate the data of close labels as the same groups. Hereby, we divide the whole dataset into continuous but disjoint groups and convert the DIR into a classification problem. In the meantime, these divided groups can not only preserve the ordinal information as their original labels but also provide us with a feasible way to explore the connection between the group classification and DIR.

Subsequently, since the decomposition of the DIR objective would also split the imbalance into both objectives of group classification and regression, inspired by (Liu et al. 2021) that feature representations learned by self-supervised learning can exhibit imbalance-robust, we introduce an ordinal group contrastive learning to learn an ordinal highquality feature representation to build a solid foundation for both classification and regression tasks. Afterward, we make the group prediction for each learned representation on a classifier (Divide). With this group estimation, we employ a multi-experts regressor to regress the representation on its corresponding predicted group (Conquer). The difference between our proposed method and the previous works can be found in Fig.1, where the previous works handle all data simultaneously while our work first divides the data into different groups and then conquers them with each expert regressor given their corresponding groups.

However, empirical observation shows that it is difficult to make an accurate group estimation under standard classification loss such as cross-entropy (CE) loss. For example, in Fig.2, the data samples from group 1 to 5 (minorities) are rarely correctly predicted. Instead, most of the data samples are over-estimated into groups 6,8,12, and 14 (majorities). The primary cause of this inaccurate prediction is the data dependence of nearby groups (images of close groups). (Yang et al. 2021) as each group also exhibits different levels of similarity between each other. As shown in Fig.2, groups with minority data samples would be easily misclassified into their neighboring groups with majority data samples.

As a result, these imprecise group predictions would misguide the data samples into the incorrect expert regressors and result in performance degradation. To tackle this problem, we propose a symmetric descending soft labeling strategy that leverages the intrinsic label similarity of the data for the group prediction. Since the labels can not only present the discrepancy information but also reflect the relative similarity between the data in DIR, we encode the group label into the soft labels which descend symmetrically from their group label until the end of the groups to capture the similarities between the groups.

![](images/6f08bb0d750dbb827322cce753403693fbdc45cbe46c3d74b58f8f8ed924edf2.jpg)  
Figure 2: Comparison between the (Logarithm scale) Ground Truth (GT) label and estimated label based on CE. X: group id Y: number of samples.

In a nutshell, by incorporating the classification with the symmetric descending soft labeling into DIR, we provide a novel framework to address the DIR in a divide-and-conquer manner. More importantly, we also conduct comprehensive experiments with various real-world datasets, demonstrating the effectiveness of our method.

In summary, we conclude our contributions as follows:

• We revisit the objective of DIR from a Bayesian perspective, which motivates us to address the DIR in a divideand-conquer manner. • We incorporate an ordinal group-aware contrastive learning to learn a high-quality feature representation to provide a solid foundation for both classification and regression tasks in our decomposed objective. • We introduce a multi-experts regressor to handle different groups of data with different expert regressors and we propose a symmetric descending soft labeling strategy to capture the similarity across the data in DIR.

# Motivation

# Preliminary

We study the DIR problem in this paper. In DIR, we assume that we have a training set $\{ x _ { i } , y _ { i } \} _ { i = 1 } ^ { N }$ with size $N$ where $x _ { i } \in \mathbb { R } ^ { d }$ is the input with dimension $d$ and $y _ { i } \in \mathbb { R }$ is the label. Meanwhile, the distribution of this training set $p _ { t r }$ is always highly skewed. The objective of DIR is to learn a model from this highly skewed training set to generalize well on an unseen test set with the balanced distribution $p _ { b a l }$ . In this paper, we aim to learn a feature extractor $f$ with parameter ${ \mathbf { w } } _ { f }$ , a classifier $h$ with parameter ${ \bf w } _ { h }$ and a set of regressors $\varphi = [ \varphi _ { 0 } , \ldots , \varphi _ { | G | - 1 } ]$ with parameter $\mathbf { w } _ { \varphi } = [ \mathbf { w } _ { \varphi _ { 0 } } , \ldots , \mathbf { w } _ { \varphi _ { | G | - 1 } } ]$ simultaneously, the parameters of the model consists of $\boldsymbol { \theta } = \{ \mathbf { w } _ { f } , \mathbf { w } _ { h } , \mathbf { w } _ { \varphi } \}$ .

# Motivation

We first revisit our goal from a Bayesian perspective. In DIR, our goal is to learn a model with parameter $\theta$ via the MSE loss to model the imbalanced training distribution $p _ { t r } ( y | x )$ and generalize well on the unseen balanced test distribution $p _ { b a l } ( y | x )$ . Since directly adopting MSE loss in DIR is in fact to model the $p ( y | x )$ for an underlying Gaussian distribution (Ren et al. 2022), a model learned from an imbalanced set would consequently underestimate rare labels, limiting its ability to generalize to an unseen balanced set (Groups are mapped from labels, e.g., for a mapping $\textstyle g = \left\lfloor { \frac { y } { | G | } } \right\rfloor )$ .

Therefore, we can review the conditional distribution of training data $p _ { t r } ( y | x )$ as follows:

Lemma 1 (Group-aware Bayesian Distribution Modeling for DIR). The conditional distribution of $p _ { t r } ( y | x )$ in the training of DIR can be decomposed into a combination of both classification and regression tasks summing over distinct groups:

$$
\begin{array} { l } { \displaystyle p _ { t r } ( y | x ) = \frac { p _ { t r } ( x , y ) } { p _ { t r } ( x ) } = \frac { \sum _ { g \in G } p _ { t r } ( x , y , g ) } { p _ { t r } ( x ) } } \\ { \displaystyle \quad = \frac { \sum _ { g \in G } p _ { t r } ( g | x ) p _ { t r } ( x ) p _ { t r } ( y | x , g ) } { p _ { t r } ( x ) } } \\ { \displaystyle \quad = \sum _ { g \in G } p _ { t r } ( g | x ) p _ { t r } ( y | x , g ) } \end{array}
$$

where $G$ is the set of groups, and $| G |$ is the number of groups, and we abbreviate $g$ as the group label.

We take a step forward by taking negative logarithm at both sides 1, we can obtain the learning objective of DIR in the form of loss as:

$$
\begin{array} { r l r } {  { - \log p _ { t r } ( y | x ) \leq \sum _ { g \in G } - \log ( p _ { t r } ( g | x ) p _ { t r } ( y | x , g ) ) } } \\ & { } & { = \sum _ { g \in G } \underbrace { - \log p _ { t r } ( g | x ) } _ { g r o u p s \ c l a s s i f i c a t i o n } \underbrace { - \log p _ { t r } ( y | x , g ) } _ { l a b e l s \ r e g r e s s i o n } } \end{array}
$$

Remark: The learning objective of DIR can be decomposed into two perspectives, 1) the objective of imbalance group classification to predict the group label, 2) the objective of imbalance regression to regress the data labels, showcasing that we can solve the DIR in a divide-and-conquer manner. Empirical results from Fig.3 also validate the effectiveness of objective decomposition in Lemma 1 for addressing DIR.

As we can observe from Fig.32, if we train a vanilla model with MSE loss only (regression-only), the training MSE loss curve and the validation MSE curve converges with different scales and the convergence speed diverges a lot, demonstrating that the training from the imbalanced set would result in unsatisfying results on the balanced validation set due to the imbalance in the training set.

![](images/4fdb7a76f4e7906e8b36fc1185ab2b4e5fe3803bf003d7a717654fe4adbd8b8d.jpg)  
Figure 3: MSE results between model trained with MSE and model trained with the decomposition loss from Lem. 1 (20 groups) on imbalanced Train & balanced Validation set (AgeDB-DIR). Row: Epoch, Column: MSE (Note: column is in logarithmic scale for an easier observation).

Instead, when we substitute the MSE loss with classification loss $p ( g | x )$ (e.g. CE) and the classification-guided MSE loss $p ( \boldsymbol { y } | \boldsymbol { x } , \boldsymbol { g } )$ as Lemma 1, the classification-guided MSE loss exhibits a sharp converge compared with the vanilla model. Moreover, the validation MSE of the classificationguided regression also converges more sharply compared with that of the vanilla model and not even at the same scale, demonstrating that the representation learned from classification can help to address the DIR and showcasing the effectiveness of alleviating the negative impact of imbalance in DIR with our classification guided regression. This motivates us to perform the divide-and-conquer by first estimating the groups $p ( g | x )$ and then guiding the groupcorresponding-regressors to regress the labels of the data samples $p ( \boldsymbol { y } | \boldsymbol { x } , \boldsymbol { g } )$ .

Hereby, we connect the classification with the objective of the DIR (to model $p _ { t r } ( y | x )$ from $p _ { t r } ( y | x ; \theta ) )$ . Furthermore, the above lemma also demonstrates that the objective of DIR can be upper-bounded by both classification and regression. By minimizing the empirical risk of the classification of the groups (to model $p _ { t r } ( { \bar { g } } | x )$ with $p _ { t r } ( g | x ; \mathbf { w } _ { f } , \mathbf { w } _ { h } ) )$ and guiding the predictions of groups to minimize the empirical risk of the regression (to model $p _ { t r } ( y | x , g )$ with $p _ { t r } \big ( y | x , g ; \mathbf { w } _ { f } , \mathbf { w } _ { \varphi } \big ) \big )$ simultaneously, we can properly address the DIR from a Bayesian perspective. More importantly, this motivates us to solve the DIR problem in a divideand-conquer manner as we can leverage the group classification $p { \bar { ( g | x ) } }$ to guide the learning of regression $p \bar { ( \boldsymbol { y } | \boldsymbol { x } , \boldsymbol { g } ) }$ .

# Methodology

In this section, we introduce ordinal group-aware constrastive learning to learn a high-quality feature representation which is beneficial for both classification and regression. Then, we leverage a multi-experts regressor to conduct regression under the guidance of the group predictions to fully exploit the benefits from classification to help regression in a divide-and-conquer manner. Furthermore, we propose a symmetric descending soft labeling strategy to capture the data similarity across groups.

# Ordinal Group-aware Contrastive Learning

As in DIR, label space is not only continuous but also ordinal. Consequently, we introduce an ordinal group-aware contrastive learning to learn a high-quality feature representation. Meanwhile, this high-quality representation can also act as a solid foundation for both classification $p ( g | x )$ and the regression $p ( \boldsymbol { y } | \boldsymbol { x } , \boldsymbol { g } )$ tasks as described in our objective decomposition from Lemma 1.

Inspired by (Zha et al. 2023b; Xiao et al. 2023), we introduce ordinal contrastive learning in a group-aware manner. Since data samples at nearby labels would have similar features (e.g., facial samples from 30 to 40 would be similar to each other), we cluster the data samples with their corresponding groups. Different from (Zha et al. 2023b), we concentrate on investigating relationships between these groups to help to learn a high-quality feature representation.

As these distinct groups would preserve the ordinal as their original labels (e.g., the label of arbitrary sample in group 0 would always be smaller than the label of arbitrary sample in group 1), we focus on constructing an ordinal group-aware contrastive learning framework where the learned feature representations can also preserve this ordinal characteristics between the groups. To achieve this goal, for an anchor group label $\mathbf { \chi } _ { i }$ and another arbitrary group label $j$ , we push away other samples whose group label distances are more distant than $i$ and $j$ . If two samples are in the same group, we pull them together at the feature space. In this way, data samples with different distances in group labels would be pushed away in various degrees, as the closer groups would be pushed less than the distant groups.

Hereby, we formulate the ordinal group-aware contrastive loss as the following:

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { g r c } ( \mathbf { w } _ { f } ) = } \ ~ } \\ { { \displaystyle ~ - \frac { 1 } { B ( B - 1 ) } \sum _ { i = 1 } ^ { B } \sum _ { j = 1 , \atop j \neq i } ^ { B } \log \frac { s ( z _ { i } , z _ { j } ) } { \sum _ { k = 1 } ^ { B } \mathbf { 1 } _ { [ \phi ( i , j , k ) ] } s ( z _ { i } , z _ { k } ) } } } \end{array}
$$

where for the index $i , j , k$ of three arbitrary data samples in a batch, $z$ is the feature representation, $s ( i , j )$ is the abbreviate of $e x p ( s i m ( z _ { i } , z _ { j } ) / t )$ and $s i m ( \cdot )$ denotes the similarity function (e.g., cosine similarity), $e x p ( \cdot )$ is the exponential function, $\phi ( i , j , k ) \triangleq \{ k \neq i , d ( g _ { i } , g _ { k } ) \geq d ( g _ { i } , g _ { j } ) \}$ is the condition of the zero-one indicator 1 (return 1 where $\phi$ satisfies and 0 vice verse), $g$ denotes the group label of the data sample, $t$ is the temperature hyper-parameter, $B$ is the batch size, and $d ( \cdot )$ denotes the distance function (e.g., L1 distance). By comparing the relative distance of group labels between two arbitrary samples, we can achieve the group ordinal as that of the labels in the feature space.

# Classification Guided Multi-experts Regression : Modeling $p ( y | x , g )$

With the acquired contrastive representations, we introduce a multi-experts regressor to tackle each group of data in a divide-and-conquer manner. In the training phase, given the ground truth group label of each data sample, we perform regression on its corresponding expert regressor. In the testing phase, each data sample is first classified into a group. Since each predicted group corresponds to an expert regressor, we conduct regression on the predicted expert regressor during the testing phase.

Therefore, we formulate the multi-expert regression MSE loss as follows:

$$
\mathcal { L } _ { m s e } ( \mathbf { w } _ { f } , \mathbf { w } _ { \varphi } ) = \sum _ { g = 0 , y \in [ g ] } ^ { | G | - 1 } ( y _ { \varphi _ { g } } - \hat { y } _ { \varphi _ { g } } ) ^ { 2 }
$$

where $y \in [ g ]$ denotes the label $y$ belongs to group $g$ . We have $\hat { y } _ { \varphi _ { g } } = \mathbf { w } _ { \varphi _ { g } } ( z _ { \varphi _ { g } } )$ for the group of data samples whose ground truth group labels are $g$ and their learned representations $z$ are then forwarding to their corresponding regressors $\varphi _ { g }$ to obtain the prediction $\hat { y } _ { \varphi _ { g } }$ . We abbreviate the ground truth target labels as $y _ { \varphi _ { g } }$ . Moreover, LDS can also be utilized to further tackle the intra-group imbalance. Since the final MSE is calculated on each data sample and each data sample corresponds to each group, we accumulate the MSE loss over all groups.

# Symmetric Descending Soft Labeling for Group Classification : Modeling p(g x)

However, since the nearby label data would exhibit data dependence (Yang et al. 2021) in DIR, in our framework, the nearby group data would also exhibit data dependence. Consequently, the data dependence of nearby groups and inherent group imbalance would hinder us from making accurate group estimations for regression.

When we directly utilize the standard CE loss to estimate the group label of the data, as can be observed from Fig.2, the predictions mostly fall into the groups with the majority of samples. Meanwhile, when we adopt logits adjustment (LA) (Menon et al. 2021), which is one of the most effective imbalance classification solutions, to predict the group labels, another empirical observation arises in Fig.4 that this method over-estimate the groups with minority samples.

Therefore, empirical results in Fig.2 and Fig.4 have shown that classification loss such as CE and LA perform poorly in group prediction of DIR. More importantly, as can be observed from Fig.5, in both CE and LA, the data dependence of the groups would lead the predictions to mainly fall into nearby groups (high absolute difference of misclassification in Fig.5). The reason for this is the classification solutions would focus on the discriminative information (as we stated above) while ignoring the data similarities across the groups.

In standard classification loss such as CE, the ground truth for one group label $g$ is encoded as a vectorized label $l _ { g t } = [ \ldots , 0 , 0 , 1 , 0 , 0 , \ldots ]$ , where 1 is at the position of $g$ -th index, and 0 is at the rest indexes and the CE loss $\mathcal { L } _ { c e } = - \log p ^ { g }$ is only calculated on the prediction at the index $g$ for the group prediction $p = [ p ^ { 0 } , \dotsc , p ^ { g } , \dotsc , p ^ { | { \cal G } | - 1 } ]$ .

![](images/b5292d488bcd1395e2859afd44e586b21b4da85227cb4bfdb45519b9afacf08b.jpg)  
Figure 4: Comparison between the (Logarithm scale) Ground Truth (GT) label and estimated label based on LA. X: group id Y : number of samples.

As a result, only the information on the ground truth index is provided (set to 0) while others are all overlooked (set to 0). However, since the group label is not merely continuous, it’s also essential to recognize that labels within DIR encode intrinsic relative similarities between them. Thus, directly adopting classification loss in group prediction is not applicable in DIR.

Therefore, we introduce a symmetric descending soft labeling strategy into the group classification to fully exploit the similarity nature of the groups. To convert a scalar group label into a vectorized label for training, for a group with ground truth label $g$ , we assign the $g$ -th index in the label vector with the highest value of $| G |$ and decrease it symmetrically from the position of the current index until the end. Thus, the soft label of the scalar group label $g$ would be encoded as $l _ { s o f t } = [ \ldots , | G | - 2 \beta , | G | - \beta , | G | , | G | - \beta , | G | -$ $2 \beta , \ldots ]$ , where, $| G |$ is at the index of $g$ in the label vector, $\beta$ is a hyper-parameter e.g., $\beta = 1$ and it denotes the relative distance between two neighboring labels. We formulate the soft label $q _ { s o f t }$ of a data sample from the ground truth group label as: $\begin{array} { r } { q _ { s o \dot { f } t } ~ = ~ \sigma ( l _ { s o f t } ) } \end{array}$ where $\sigma$ denotes the SoftMax function, as $\begin{array} { r } { \sigma ( q _ { i } ) = \frac { e ^ { q _ { i } } } { \sum _ { j = 1 } ^ { | G | } e ^ { q _ { j } } } } \end{array}$ . Moreover, we briefly show two extreme cases for our soft label, in the case when $g = 0$ , the $l _ { g t _ { - } s o f t } = [ | G | , | G | - 1 , | G | - 2 , \dots , 1 ]$ , and in the case when $g = | G |$ , the $l _ { g t - s o f t } = [ 1 , \dots , | G | - 2 , | G | - 1 , | G | ]$ .

The soft label cross-entropy loss for a data sample with group label $g$ (corresponding with the regressor $g$ ) in a batch $B$ as the following:

$$
\mathcal { L } _ { s o f t } \big ( \mathbf { w } _ { f } , \mathbf { w } _ { \varphi _ { g } } \big ) = \sum _ { j = 1 } ^ { B } \sum _ { g = 0 } ^ { | G | - 1 } q _ { j } ^ { g } \log p _ { j } ^ { g }
$$

where $\mathit { p _ { j } ^ { g } }$ denotes group prediction and $q _ { j } ^ { g }$ is the soft label in $q _ { s o f t }$ of sample $j$ at index $g$ .

By encoding the ground truth labels into soft labels, we can preserve the relative group information of all groups in one single label, providing comprehensive data information for the group classification and also contributing to the regression. Comparison between different classification criteria (Soft Labeling/CE/LA) also shows the effectiveness of our proposed method.

# Final Loss

By aggregating the above losses together, the final objective of our proposed method is :

$$
\mathcal { L } _ { f i n a l } = \mathcal { L } _ { g r c } + \lambda _ { 1 } \mathcal { L } _ { m s e } + \lambda _ { 2 } \mathcal { L } _ { s o f t }
$$

where $\lambda _ { 1 }$ and $\lambda _ { 2 }$ are hyper-parameters to balance the losses.

# Experiments

# Datasets

We validate our proposed method with the following realworld dataset which includes both visual tasks and natural language processing tasks:

IMDB-WIKI-DIR is a large-scale real-world human facial dataset constructed by (Rothe, Timofte, and Van Gool 2018) and re-organized for imbalance tasks by (Yang et al. 2021), it contains 235K face images. There are 191.5K imbalance training images, 11K balanced validation images, and 11K balanced test images. The dataset was manually divided given the bin length of 1 year (each bin can be regarded as the target label as in (Yang et al. 2021)).

AgeDB-DIR is another real-world human facial dataset constructed by (Moschoglou et al. 2017) and also reorganized by (Yang et al. 2021). It contains 12.2K image training data, 2.1K image validation data, and 2.1K image test data. The bin length is also 1 year but the minimum age is 0 and the maximum age is 101.

STS-B-DIR is a text similarity score dataset constructed by (Wang et al. 2018) and re-constructed by (Yang et al. 2021). It is collected from news headlines, videos, image captions, and natural language inference data. The dataset is a set of sentence pairs annotated with an average similarity score, and the range of scores varies from 0 to 5. There are $5 . 2 \mathrm { K }$ pairs for the training, 1K balanced pairs for validation, and 1K balanced pairs for test. Each bin length is 0.1.

# Implementation Details

Baselines and experiment set up We conducted our experiments with the backbone based on ResNet-50 for AgeDBDIR & IMDB-WIKI-DIR dataset. For STS-B-DIR, we follow the same standard experiment setting as in (Yang et al. 2021; Ren et al. 2022), we adopted the $\mathrm { \ B i L S T M + G l o V e }$ word embeddings and preprocessed them in the experiment. Moreover, we follow the training procedures and hyperparameters (e.g., temperature $t$ ) as (Zha et al. 2023a), but apart from (Zha et al. 2023a) which only used a sub-sample of both datasets (e.g., 32K for IMDB-WIKI-DIR), we stick to the setting of (Yang et al. 2021) and use the full training set with the batch size of 128 for training. Same as (Yang et al. 2021; Branco, Torgo, and Ribeiro 2017), the train data distribution is always highly skewed while the test distribution is balanced. More details can be found in the Appendix.

# Result Analysis

AgeDB-DIR $\because$ In the dataset AgeDB-DIR, it is obvious that our method outperforms most of other methods in Tab.1.

![](images/b257b04dcfa87c203f10d0187e8f90bbce512f058efdbe4db46fc22347eac422.jpg)  
Figure 5: Comparison of the absolute difference (Diff) between group predictions and ground truth in CE, LA, and Ours on AgeDB-DIR. Lower denotes more accurate group predictions. X : group numbers. ${ \mathrm { Y } } : { \mathrm { } }$ Absolute value.

In particular, we show that our method can better deal with the majority and median, without greatly sacrificing the performance of the minorities as in the previous works. Compared with existing works, our work achieved a state-of-art (SOTA) performance with an overall MAE of 6.87. Meanwhile, our work has lower GM, which shows that our results of MAE are averagely smaller than other works, showing the effectiveness of our proposed method.

We show that in Fig.5, we use the absolute difference between the ground truth labels and the estimated labels to identify if our proposed method can help the classification (how accurate the group estimation can be given the ground truth). Our proposed symmetric descending soft labeling significantly outperforms others in group estimation (compared with Fig.2 & 4), that is because the soft labels can help the representations to fully exploit the similarity characteristics of the data from other labels. Consequently, it contributes to a more accurate group estimation than other existing works, resulting in minimizing the $\log p ( g | x )$ and the gap $\Delta$ at the same time.

Another interesting observation from Fig.5 arises that, directly using the CE and LA would also make the predictions in the tail groups almost fail, that is because the tail groups are always the minorities. Also, our Soft labeling can capture the information from other groups to help minorities in group imbalance. As in Fig.7, the classification performance of the soft labeling is consistently better than that of the CE and LA, such as in 20 and 25 groups, the soft labeling has a $5 \%$ more improvement compared to others, which validates that the label similarity is one crucial characteristic in DIR and leverage the group similarities as that of the label similarities can help to take advantages of classification in helping DIR as (Pintea et al. 2023).

IMDB-WIKI-DIR: In the dataset IMDB-WIKI-DIR, which is also the largest DIR real-world dataset, our overall performance in Tab.2 achieved a satisfying result and is better than most of the current solutions. Specifically, we show that our method can have a better performance on the median and the few shot, it shows that our proposed method can exploit more information on the median and the few shots with the soft-label, resulting in an overall performance improvement. Moreover, we show that our result in GM is also better than others both in the Med. and Few., showing a consistent superiority of our work over others and validating that our proposed method can better address the Med. and Few shots in the extreme case of imbalance. As we can observe from Fig.6, the comparison of b-MAE results between the stateof-art DIR solutions and our proposed method also validates the effectiveness of our method, showcasing our method can have a better performance on the balanced sets, especially on the minority samples.

![](images/80ff9aeeb5f94b926d8077c4f77d84b9e5093587a9b52b5dae334ae43c4c5b16.jpg)  
Figure 6: Comparison between various DIR solutions of bMAE Results in Majority, Median, and Minority on IMDBWIKI-DIR. Y : b-MAE.

![](images/dde9b352dee406f514be2b936095294a193925875b6016af4449f32f7d19d1ee.jpg)  
Figure 7: Group prediction accuracy comparison between our Soft labeling/CE/LA on AgeDB-DIR, X: group number Y: group prediction accuracy.

STS-B-DIR: In the dataset STS-B-DIR, it is easy to observe that the under-represented median & minority data samples significantly outperform others in Tab.3, contributing to the overall performance enhancement. Moreover, our work simultaneously improved the performance of median & minority shots in Pearson Correlation compared to others, that is because our soft labeling can help us to preserve the data similarity as that of labels in representation learning, which can enhance the Pearson correlations and consistent lutions in Tab.1 and Fig.8, which also shows the prominence of our proposed method.

Table 1: Evaluation on AgeDB-DIR.   

<html><body><table><tr><td></td><td rowspan="2">Shot</td><td colspan="4">MAE↓</td><td colspan="4">GM↓</td></tr><tr><td>Method</td><td>All</td><td>Many.</td><td>Med.</td><td>Few.</td><td>All</td><td>Many.</td><td>Med.</td><td>Few.</td></tr><tr><td>VANILLA</td><td></td><td>7.77</td><td>6.62</td><td>9.55</td><td>13.67</td><td>5.05</td><td>4.23</td><td>7.01</td><td>10.75</td></tr><tr><td>SMOTER</td><td></td><td>8.16</td><td>7.39</td><td>8.65</td><td>12.28</td><td>5.21</td><td>4.65</td><td>5.69</td><td>8.49</td></tr><tr><td>SMOGN</td><td>8.26</td><td></td><td>7.64</td><td>9.01</td><td>12.09</td><td>5.36</td><td>4.90</td><td>6.19</td><td>8.44</td></tr><tr><td>RRT</td><td>7.74</td><td>6.98</td><td></td><td>8.79</td><td>11.99</td><td>5.00</td><td>4.50</td><td>5.88</td><td>8.63</td></tr><tr><td>RRT+LDS</td><td>7.72</td><td>7.00</td><td>8.75</td><td></td><td>11.62</td><td>4.98</td><td>4.54</td><td>5.71</td><td>8.27</td></tr><tr><td>FOCAL-R</td><td>7.64</td><td>6.68</td><td>9.22</td><td>13.00</td><td></td><td>4.90</td><td>4.26</td><td>6.39</td><td>9.52</td></tr><tr><td>SQINV</td><td>7.81</td><td>7.16</td><td>8.80</td><td>11.20</td><td></td><td>4.99</td><td>4.57</td><td>5.73</td><td>7.77</td></tr><tr><td>SQINV +LDS</td><td>7.67</td><td>6.98</td><td>8.86</td><td>10.89</td><td></td><td>4.85</td><td>4.39</td><td>5.80</td><td>7.45</td></tr><tr><td>LDS+FDS</td><td>7.55</td><td>7.01</td><td>8.24</td><td>10.79</td><td></td><td>4.72</td><td>4.36</td><td>5.45</td><td>6.79</td></tr><tr><td>VAE</td><td>7.63</td><td>6.58</td><td>9.21</td><td>13.45</td><td></td><td>4.86</td><td>4.11</td><td>6.61</td><td>10.24</td></tr><tr><td>DER</td><td>8.09</td><td>7.31</td><td>8.99</td><td>12.66</td><td></td><td>5.19</td><td>4.59</td><td>6.43</td><td>10.49</td></tr><tr><td>Con-R</td><td>7.20</td><td>6.50</td><td>8.04</td><td>9.73</td><td></td><td>4.59</td><td>3.94</td><td>4.83</td><td>6.39</td></tr><tr><td>RankSim</td><td>7.02</td><td>6.49</td><td>7.84</td><td>9.68</td><td></td><td>4.53</td><td>4.13</td><td>5.37</td><td>6.89</td></tr><tr><td>VIR</td><td>6.99</td><td>6.39</td><td>7.47</td><td>9.51</td><td></td><td>4.41</td><td>4.07</td><td>5.05</td><td>6.23</td></tr><tr><td>LDS+FDS+DER</td><td>8.18</td><td>7.44</td><td>9.52</td><td></td><td>11.45</td><td>5.30</td><td>4.75</td><td>6.74</td><td>7.68</td></tr><tr><td>Ours</td><td>6.87</td><td>6.54</td><td>6.96</td><td>9.83</td><td></td><td>4.30</td><td>4.10</td><td>4.39</td><td>6.45</td></tr></table></body></html>

Table 2: Evaluation on IMDB-WIKI-DIR.   

<html><body><table><tr><td rowspan="2">Method</td><td rowspan="2">Shot</td><td colspan="4">MAE↓</td><td colspan="4">GM↓</td></tr><tr><td>All</td><td>Many.</td><td>Med.</td><td>Few.</td><td>All</td><td>Many.</td><td>Med.</td><td>Few.</td></tr><tr><td>VANILLA</td><td></td><td>8.06</td><td>7.23</td><td>15.12</td><td>26.33</td><td>4.57</td><td>4.17</td><td>10.59</td><td>20.46</td></tr><tr><td>SMOTER</td><td></td><td>8.14</td><td>7.42</td><td>14.15</td><td>25.28</td><td>4.64</td><td>4.30</td><td>9.05</td><td>19.46</td></tr><tr><td>SMOGN</td><td>8.03</td><td>7.30</td><td></td><td>14.02</td><td>25.93</td><td>4.63</td><td>4.30</td><td>8.74</td><td>20.12</td></tr><tr><td>SMOGN+LDS</td><td>8.02</td><td>7.39</td><td>13.71</td><td></td><td>23.22</td><td>4.63</td><td>4.39</td><td>8.71</td><td>15.80</td></tr><tr><td>RRT+LDS</td><td>7.79</td><td>7.08</td><td>13.76</td><td></td><td>24.64</td><td>4.34</td><td>4.02</td><td>8.72</td><td>16.92</td></tr><tr><td>SQINV+LDS</td><td>7.83</td><td>7.31</td><td>12.43</td><td></td><td>22.51</td><td>4.42</td><td>4.19</td><td>7.00</td><td>13.94</td></tr><tr><td>FOCAL-R+LDS</td><td>7.90</td><td>7.10</td><td>14.72</td><td></td><td>25.84</td><td>4.47</td><td>4.09</td><td>10.11</td><td>19.14</td></tr><tr><td>BMC</td><td>8.08</td><td>7.52</td><td>12.47</td><td>23.29</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GAI</td><td>8.12</td><td>7.58</td><td>12.27</td><td></td><td>23.05</td><td></td><td></td><td></td><td></td></tr><tr><td>VAE</td><td>8.04</td><td>7.20</td><td>15.05</td><td>26.30</td><td></td><td>4.57</td><td>4.22</td><td>10.56</td><td>20.72</td></tr><tr><td>DER</td><td>7.85</td><td>7.18</td><td>13.35</td><td>24.12</td><td></td><td>4.47</td><td>4.18</td><td>8.18</td><td>15.18</td></tr><tr><td>Con-R</td><td>7.33</td><td>6.75</td><td>11.99</td><td>22.22</td><td></td><td>4.02</td><td>3.79</td><td>6.98</td><td>12.95</td></tr><tr><td>RankSim</td><td>7.50</td><td>6.93</td><td>12.09</td><td>21.68</td><td></td><td>4.19</td><td>3.97</td><td>6.65</td><td>13.28</td></tr><tr><td>VIR</td><td>7.19</td><td>6.56</td><td>11.81</td><td>20.96</td><td></td><td>3.85</td><td>3.63</td><td>6.51</td><td>12.23</td></tr><tr><td>LDS +FDS+DER</td><td>7.24</td><td>6.64</td><td>11.87</td><td>23.44</td><td></td><td>3.93</td><td>3.69</td><td>6.64</td><td>16.00</td></tr><tr><td>Ours</td><td>7.22</td><td>6.71</td><td>11.42</td><td>20.25</td><td></td><td>3.88</td><td>3.68</td><td>5.74</td><td>11.13</td></tr></table></body></html>

Table 3: Evaluation on STS-B-DIR.   

<html><body><table><tr><td rowspan="2">Shot Method</td><td rowspan="2"></td><td colspan="4">MSE↓</td><td colspan="4">Pearson Correlation↑</td></tr><tr><td>All</td><td>Many.</td><td>Med.</td><td>Few.</td><td>All</td><td>Many.</td><td>Med.</td><td>Few.</td></tr><tr><td>VANILLA</td><td>0.974</td><td>0.851</td><td>1.520</td><td>0.984</td><td>74.2</td><td>72.0</td><td></td><td>62.7</td><td>75.2</td></tr><tr><td>SMOTER</td><td>1.046</td><td>0.924</td><td>1.542</td><td>1.154</td><td>72.6</td><td></td><td>69.3</td><td>65.3</td><td>70.6</td></tr><tr><td>SMOGN</td><td>0.990</td><td>0.896</td><td>1.327</td><td>1.175</td><td>73.2</td><td>70.4</td><td></td><td>65.5</td><td>69.2</td></tr><tr><td>SMOGN + LDS</td><td>0.962</td><td>0.880</td><td>1.242</td><td>1.155</td><td>74.0</td><td>71.5</td><td></td><td>65.2</td><td>69.8</td></tr><tr><td>RRT</td><td>0.964</td><td>0.842</td><td>1.503</td><td>0.978</td><td>74.5</td><td></td><td>72.4</td><td>62.3</td><td>75.4</td></tr><tr><td>RRT+LDS</td><td>0.916</td><td>0.817</td><td>1.344</td><td>0.945</td><td>75.7</td><td></td><td>73.5</td><td>64.1</td><td>76.6</td></tr><tr><td>FOCAL-R</td><td>0.951</td><td>0.843</td><td>1.425</td><td>0.957</td><td>74.6</td><td></td><td>72.3</td><td>61.8</td><td>76.4</td></tr><tr><td>INV</td><td>1.005</td><td>0.894</td><td>1.482</td><td>1.046</td><td>72.8</td><td></td><td>70.3</td><td>62.5</td><td>73.2</td></tr><tr><td>INV + LDS</td><td>0.914</td><td>0.819</td><td>1.31</td><td>0.95</td><td>75.6</td><td></td><td>73.4</td><td>63.8</td><td>76.2</td></tr><tr><td>VAE</td><td>0.968</td><td>0.833</td><td>1.511</td><td>1.102</td><td>75.1</td><td></td><td>72.4</td><td>62.1</td><td>74.0</td></tr><tr><td>LDS +FDS</td><td>0.907</td><td>0.802</td><td>1.363</td><td>0.942</td><td>76.0</td><td></td><td>74.0</td><td>65.2</td><td>76.6</td></tr><tr><td>DER</td><td>1.001</td><td>0.912</td><td>1.368</td><td>1.055</td><td>73.2</td><td></td><td>71.1</td><td>64.6</td><td>74.0</td></tr><tr><td>RankSim</td><td>0.903</td><td>0.908</td><td>0.911</td><td>0.804</td><td>75.8</td><td></td><td>70.6</td><td>69.0</td><td>82.7</td></tr><tr><td>VIR</td><td>0.892</td><td>0.795</td><td>0.899</td><td>0.781</td><td>77.6</td><td></td><td>75.2</td><td>69.6</td><td>84.5</td></tr><tr><td>LDS+FDS +DER</td><td>1.007</td><td>0.880</td><td>1.535</td><td>1.086</td><td>72.9</td><td></td><td>71.4</td><td>63.5</td><td>73.1</td></tr><tr><td>Ours</td><td>0.887</td><td>0.897</td><td>0.891</td><td>0.779</td><td>77.4</td><td></td><td>74.9</td><td>70.7</td><td>85.8</td></tr></table></body></html>

with the smoothing-based methods (e.g., as VIR (Wang and Wang 2023; Yang et al. 2021)).

![](images/6b4deec0d2b521cb6d0c7daef3d61bdef0c8b8a86a5b0640a64b978915a62b93.jpg)  
Figure 8: Comparison on Group Numbers vs MAE in Majority, Median and Minority. Y: MAE, X: group numbers.

# Conclusion

# Ablation Study on Group Numbers

We also provide a detailed ablation study on the group numbers with the group prediction accuracy and the MAE on AgeDB in Fig.7 and Fig.8. With the increasing of the group numbers, the prediction accuracy gradually drops, the reason why this phenomenon occurs comes from the data dependence over the groups. Therefore, we proposed soft labeling which can leverage the data dependence across the groups and yield a satisfying outcome.

In Fig.8, we can observe that each portion of data (Majority, Median, and Minority) varies slightly with the increasing of group numbers. Specifically, in 15, 20, 25, and 40 group settings, the performance of the majority shots is always close to each other while the median is varied slightly. Meanwhile, most of them always outperform other DIR so

In this work, we present a symmetric descending Soft labeling guided group-aware ordinal contrastive learning framework to learn a high-quality representation that both exhibits discriminative and similar characteristics simultaneously to address the DIR with a multi-expert regressor in a divideand-conquer manner motivated by our theoretical analysis. Extensive experiments on various real-world datasets verify the superiority of our method. Our analysis of the results further validates the effectiveness of our proposed method.

# Acknowledgments

We appreciate constructive feedback from anonymous reviewers and meta-reviewers. Thanks to Dr.Qi Chen for her valuable suggestions. This work is supported by the Natural Sciences and Engineering Research Council of Canada
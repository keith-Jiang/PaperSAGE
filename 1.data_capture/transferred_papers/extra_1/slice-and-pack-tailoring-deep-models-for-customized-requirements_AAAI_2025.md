# Slice-and-Pack: Tailoring Deep Models for Customized Requirements

Ruice Rao\*1,2, Dingwei $\mathbf { L i } ^ { * 1 , 2 }$ , Ming Li†1,2

1National Key Laboratory for Novel Software Technology, Nanjing University, China 2School of Artificial Intelligence, Nanjing University, China {raorc, lidw, lim}@lamda.nju.edu.cn

# Abstract

The learnware paradigm aims to establish a learnware market such that users can build their own models by reusing appropriate existing models in the market without starting from scratch. It is often the case that a single model is insufficient to fully satisfy the user’s requirement. Meanwhile, offering multiple models can lead to higher costs for users alongside an increase in hardware resource demands. To address this challenge, this paper proposes the “Slice-and-Pack” (S&P) framework to empower the market to provide users with only the required model fragments without having to offer entire abilities of all involved models. Our framework first slices a set of models into small fragments and subsequently packs selected fragments according to user’s specific requirement. In the slicing stage, we extract units layer by layer and connect these units to create numerous fragments. In the packing stage, an encoder-decoder mechanism is employed to assemble these fragments. These processes are conducted within data-limited constraints due to privacy concerns. Extensive experiments validate the effectiveness of our framework.

# Introduction

Machine learning has achieved great success in various domains (Bengio, Lecun, and Hinton 2021; Silver et al. 2016; OpenAI 2023; Luo et al. 2022; Radford et al. 2021). However, building a satisfactory learning model from scratch is often time-consuming and requires significant expertise, and personalized customization is often necessary for specific situations. Moreover, the access to raw data is often limited by privacy concerns, making it even challenging to train well-behaved models from scratch. To address these issues, Zhou (2016) proposed the learnware paradigm, aiming to establish a learnware market such that users can build their own models by reusing appropriate existing models in the market without starting from scratch. Unlike LLM, the learnware market enables model construction and reuse in broader application scenarios such as physics, chemistry, geology, medicine, manufacturing, and more. A learnware is a pre-trained model associated with a specification describing the model’s specialty and utility. Developers can sub

Developers User Upload Query A model Skin for normal disease Persist Return models Combine(big) Infection Previous framework vs Split-and-Pack Upload Query □ Stomach   
SHpoescpiiatlailsst Factorize Persist FrRaegqumireendts A(ss emalbl)le Clinic

mit trained models to the market with a few data points describing their tasks, and the market will assign a specification upon accepting this submitted model. When users try to tackle a learning task, the market can recommend helpful learnwares whose specifications match the requirement.

However, in many cases it is hard to find a single model that fully meets the user’s requirements, especially in the early stages of the learnware market. In real-world scenarios, the abilities desired by users might spread across several models. Offering all involved models would make users feel compelled to purchase for potentially redundant abilities, and also requires more hardware resource for deployment. For instance, consider the community clinics that try to use patient records for diagnosis in Figure 1. Due to the constraints in equipment and treatment capabilities, these clinics are primarily tasked with addressing normal diseases. In the learnware market, there are several existing wellbehaved models available from specialist hospitals trained on private data. However, these models are designed to diagnose as many diseases as possible in a specific field. Since the requirements of community clinics are abilities to diagnose normal diseases, these rare abilities of the model are redundant for them. It is to be mentioned that models equipped with more capabilities typically demand more computational power for inference, leading to the need for more robust devices and higher acquisition costs. As highlighted in (Liang et al. 2022), the cost of commercial models significantly escalates as the size and capability of the models increase, ranging from $\$ 169$ to upwards of $\$ 10,000$ . Furthermore, the utilization of these stronger models often necessitates much more computational resources, which could impose further financial burden on community clinics.

Table 1: Characteristics of problem settings   

<html><body><table><tr><td>Setting</td><td>Data limited</td><td>Increasing diversity</td><td>Cheap</td><td>Few parameters</td><td>Plug&Play</td><td>Heterogeneous model</td></tr><tr><td>Train from scratch</td><td>X</td><td></td><td>X</td><td></td><td>X</td><td></td></tr><tr><td>Transfer learning</td><td>√</td><td>×</td><td>√</td><td></td><td>X</td><td>X</td></tr><tr><td>Model reuse</td><td>√</td><td>X</td><td>X</td><td>√</td><td>X</td><td>√</td></tr><tr><td>Model decomposition</td><td>√</td><td>X</td><td>√</td><td>√</td><td>X</td><td>×</td></tr><tr><td>Ensemble</td><td>√</td><td>√</td><td></td><td></td><td>√</td><td>√</td></tr><tr><td>S&P</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table></body></html>

To address this issue, we propose the “Slice-and-Pack” (S&P) framework, which empowers the market to provide users with only the required model fragments, eliminating the need to offer entire abilities of all involved models. These fragments can be combined in various ways to satisfy different users’ specific requirements. As illustrated in Figure 1, to meet the clinic’s requirements, we can slice models into fragments of different diseases and combine fragments of normal diseases. This framework offers superior flexibility and convenience compared to using the entire model, minimizing the cost of redundant abilities while providing easy plug-and-play functionality. Whenever the change of requirements happens, users can easily discard outdated fragments and purchase new ones as needed. Furthermore, the framework can also greatly benefit the market development, as there is no need for the market to consider all possible combinations of abilities, thereby expanding the market’s capacity to match that of a market several times its size.

To achieve above desired properties, our framework works in two stages: slicing and packing. In the slicing stage, we extract fragments from the original model, each corresponding to a sub-abilities. We first identify units that are significant to the sub-abilities layer by layer and then progressively connect each layer’s selected units. This process creates individual fragments that perform specific abilities. In the packing stage, we use an encoder-decoder mechanism to pack the previously stored fragments. We take the fragments identified in the slicing part and assemble them to form a new model tailored to the user’s specific requirements. All these processes simply use the limited data provided by developers for constructing the learnware specification. We summarize our contributions in the following.

• We are the first to tackle the problem of increasing model diversity in the market, which holds significant commercial viability compared to previous problems. • We introduce the “Slice-and-Pack” (S&P) framework, a novel method with plug-and-play feature that expands the market in data-limited environments. • Our empirical evaluations show that the S&P can generate highly accurate packed models and expand the market’s capacity by many times. Moreover, the minimal time required for packing makes it readily available to users.

# Problem Setting

The learnware paradigm (Zhou 2016; Zhou and Tan 2022) presents a promising framework where a vast number of models are submitted by developers from various tasks without the availability of their original training data. It poses significant challenges for users to identify and reuse helpful models in the market $\mathrm { \Delta W u }$ et al. 2021; Tan et al. 2022). Assume that there are N models {M(i)}iN=1 developed by different developers in the learnware market. Each model $\mathcal { M } ^ { ( i ) }$ has an ability set representing the abilities that the model has. In the following, we focus on classification task and each ability is the function to identify a single class. Let $\mathscr { C } ^ { ( i ) }$ be the class set of $i$ -th model $\mathcal { M } ^ { ( i ) }$ , $\textstyle { \mathcal { U } } = { \bar { \bigcup } } _ { i = 1 } ^ { N } { \mathcal { C } } ^ { ( i ) }$ be the set of all classes in $\{ \mathcal { M } ^ { ( i ) } \} _ { i = 1 } ^ { N }$ , $\mathcal { D } ^ { ( i ) }$ be theSlimited samples of model $\mathcal { M } ^ { ( i ) }$ uploaded by developers, and $\mathcal { D } _ { c } ^ { ( i ) } \subset \mathcal { D } ^ { ( i ) }$ be the samples of class $c \in \mathcal { C } ^ { ( i ) }$ .

The problem in this paper is based on the learnware paradigm: Given a set of models and a few samples $\mathsf { \bar { \{ } } ( \mathcal { M } ^ { ( i ) } , \mathcal { D } ^ { ( i ) } ) \} _ { i = 1 } ^ { N }$ , how to get a set of fragments, allowing for the rapid and easy construction of a smallest possible model with any subset of classes $c \subset { \mathcal { U } }$ .

Let $\mathcal { F } ^ { c }$ be the fragment of class $c$ . Our goal can be formalized as optimizing the following objective:

$$
\underset { \mathcal { F } ^ { c } } { \operatorname* { m i n } } \ \mathbb { E } _ { \mathcal { C } \subset \mathcal { U } , ( x , y ) \sim \mathcal { P } _ { \mathcal { C } } } \mathcal { L } \left( \mathcal { F } _ { \mathcal { C } } ( x ) , y \right)
$$

where $\mathcal { P } _ { \mathcal { C } }$ is the distribution of class $c$ ’s samples, ${ { \mathcal F } _ { { \mathcal C } } } \mathrm { ~ = ~ }$ $\operatorname { U n i o n } ( \{ \mathcal { F } ^ { c } \} _ { c \in \mathcal { C } } )$ is the final model by packing required fragments, $\mathcal L ( \cdot , \cdot )$ is some loss function.

# Related Work

The difference between other settings is shown in Table 1. Transfer learning (Pan and Yang 2010) and domain adaptation (Ben-David et al. 2006; Sun et al. 2023) are techniques aiming to transfer knowledge from the source domain to the target domain. They assume that raw data from one (Huang et al. 2006; Pan et al. 2010; Fernando et al. 2013) or multiple domains (Shu et al. 2021; Nguyen et al. 2021; Yang et al. 2022) are accessible when training the target model. However, in the learnware paradigm, the raw source data is unavailable at the time of training the target model, making these techniques inapplicable.

Hypothesis transfer learning (Kuzborskij and Orabona 2013, 2017) and model reuse (Ding and Zhou 2020; Zhao, Cai, and Zhou 2020) attempt to exploit pre-trained models to handle current jobs of learners. These techniques assume that the pre-trained models are helpful for the current job (Li et al. $2 0 2 0 \mathrm { a }$ ; Shen et al. 2021). However, this assumption is not suitable for our problem as the abilities required by users may be distributed across several models, making it difficult to exactly find a single model that is helpful to users. Additionally, we aim to slice pre-trained models into smaller models with different sub-abilities, rather than replicating the original functionality.

![](images/4a59dfd5500b0d603bf1b4df57e79b50d615d10a17179c3b84306d9f814019c4.jpg)  
Figure 2: The overview of Slice-and-Pack.

Multi-party learning (Pathak, Rane, and Raj 2010; Wu, Liu, and Zhou 2019) aims to unite local data to solve the same or similar task in privacy-preserving ways, rather than using the existing pre-trained models. Some recent attempts show some possibilities of leveraging extracted components of existing models for specific tasks (Yang, Ye, and Wang 2022). These approaches necessitate developers to share substantial amounts of data, which is unacceptable for learnware markets due to privacy concerns of the model provider.

Model decomposition is a technique used to increase parameter-efficiency in deep learning models by breaking down parameter-heavy layers into multiple lightweight ones. This approach involves using various techniques such as low-rank decomposition (Li et al. 2020b) and weight decomposition (Kim et al. 2015; Zhang et al. 2015; Hameed et al. 2022). They aim to generate smaller models of similar abilities with comparable performance for distributed runtime environments, but not models with different functionalities.

# The Slice-and-Pack Framework

In this section, we present the details of our Slice-and-Pack (S&P) framework, which is composed of two parts: Slicing and Packing, as illustrated in Figure 2. The algorithm is shown in Appendix A.

# Slicing the Network

In this subsection, we denote $\mathcal { M } , \mathcal { D }$ as the shorthand for $\mathcal { M } ^ { ( i ) } , \mathcal { D } ^ { ( i ) }$ . The slicing method aims to obtain the fragment $\mathcal { F } ^ { c }$ with class $c$ from the network model $\mathcal { M }$ . In this paper, fragment refers to the model obtained from the feature extraction component of the original network. We regard $\mathcal { M }$ as a series of layers $m _ { i }$ composed of a linear layer and some nonlinear layers. Almost all network layers can be seen as a collection of units. The unit is a filter for the convolution layer, and the unit is a neuron for the linear layer. Suppose $\mathcal { M }$ has $L$ layers, we can express $\mathcal { M }$ as:

$$
\mathcal { M } = m _ { 1 } \circ m _ { 2 } \circ \cdot \cdot \cdot \circ m _ { L - 1 } \circ m _ { L }
$$

We adopt a dual-stage strategy to extract the fragment $\mathcal { F } ^ { c }$ of class $c$ from the network model $\mathcal { M }$ , which is designed to ease the risk of overfitting. In the first stage, we select units which are important to the class $c$ layer by layer. For each layer $m _ { i }$ , we use techniques Ability Pooling and Ability Adaption to get selected units layer $f _ { i }$ , as shown in Figure 3. In the second stage, each $f _ { i }$ is progressively combined and finally we get the fragment $\mathcal { F } ^ { c }$ of class $c$ .

Ability Pooling. In Slice-and-Pack setting, it is necessary to identify the units that are important to a specific class $c$ rather than the whole original model $\mathcal { M }$ . While some methods use parameter weights as regularization or criticism to make the model sparser or clip it, high-value parameters are essential to the overall model rather than a specific class. To address this, we use the activation of class $c$ ’s samples at the $i$ -th layer $m _ { i }$ as the means of identifying important units. We first use this activation as a regularization, and then we select units based on it.

Let $\mathcal { Z } _ { i } ^ { c }$ represent the output of layer $m _ { i }$ with $\mathcal { D } _ { c }$ as input, $\mathcal { M } ^ { \prime }$ be the model having parameters completely the same as $\mathcal { M }$ , and $u _ { i }$ be the number of units in $m _ { i }$ . To enhance the sparsity of weights among each units rather than the whole layer, we use $\ell _ { 2 , 1 }$ -norm (Yang et al. 2011) of $\mathcal { Z } _ { i } ^ { c }$ as follows:

$$
\left\| \mathcal { Z } _ { i } ^ { c } \right\| _ { 2 , 1 } = \frac { 1 } { | \mathcal { Z } _ { i } ^ { c } | } \sum _ { \mathbf { z } \in \mathcal { Z } _ { i } ^ { c } } \sum _ { j = 1 } ^ { u _ { i } } \left\| \mathbf { z } ^ { j } \right\| _ { 2 } ,
$$

where $\mathbf { z } ^ { j }$ is the output of the $j$ -th unit. We fine-tune $m _ { i }$ with the pooling loss containing $\ell _ { 2 , 1 }$ -norm of $\mathcal { Z } _ { i } ^ { c }$ and the $\ell _ { 2 }$ -loss

![](images/80a5ef6cfecbaa48e8b376682ab7042a5d708e136662cb7a0391459f58921ad3.jpg)  
Figure 3: The process of factorizing layer $m _ { i }$ . In ability pooling, we will centralize the functionality of class $c$ to less units and then select units to get $f _ { i }$ . In ability adaption, We use the adaption loss to adapt $f _ { i }$ for the change of structure.

function $\mathcal { L } _ { i } ^ { P }$ between $\mathcal { M }$ and $\mathcal { M } ^ { \prime }$ , expressed as

$$
\mathcal { L } _ { i } ^ { P } = \mathbb { E } _ { x \in \mathcal { D } } \left[ \left. \boldsymbol { \mathcal { M } } ^ { \prime } ( \boldsymbol { x } ) - \boldsymbol { \mathcal { M } } ( \boldsymbol { x } ) \right. _ { 2 } ^ { 2 } \right] + \alpha \left. \mathcal { Z } _ { i } ^ { c } \right. _ { 2 , 1 } ,
$$

where $\alpha$ is a hyperparameter. We can obtain a sparser layer $m _ { i } ^ { \prime }$ from $\mathcal { M } ^ { \prime }$ by using the loss function $\mathcal { L } _ { i } ^ { P }$ .

Then we select units by either threshold or rate. Let $U _ { i }$ be the set of selected unit indices as follows:

$$
\begin{array} { r l r } & { U _ { i } = \left\{ j \ | \ | ( \mathcal { Z } _ { i } ^ { c } ) _ { j } | > 1 - \beta \right\} } & { \mathrm { ( t h r e s h o l d ) } } \\ & { U _ { i } = \left\{ j \ | \ j \in \mathrm { T o p } { - } \beta \left( \{ | ( \mathcal { Z } _ { i } ^ { c } ) _ { j } | \} _ { j = 1 } ^ { u _ { i } } \right) \right\} , } & { \mathrm { ( r a t e ) } } \end{array}
$$

where $\beta$ is a hyperparameter in the range from 0 to 1. Finally we get the new layer $f _ { i }$ from $m _ { i } ^ { \prime }$ for class $c$ .

Ability Adaption. After selecting the units, we need to adapt the layer $f _ { i }$ to account for changes in the structure of the neural network. However, there are two challenges that we need to address: size mismatch and ability collapse.

Size mismatch refers to the discrepancy between the output size of the $i$ -th layer $f _ { i }$ and the input size of the $( i + 1 )$ - th layer $f _ { i + 1 }$ . Simply removing the parameters of $( i + 1 )$ -th layer without considering the context of the layer’s situation is wasteful, especially when dealing with limited data. To solve this problem, we assume that selected units contain sufficient information. This allows us to recover the output of dropped units by using output of the remaining units. We achieve this by employing a linear layer $l _ { i }$ . It takes the output of $f _ { i }$ as input and produces features with the same size as the dropped units. The output of $l _ { i }$ can then be used to fill in the missing results from the removed units in $m _ { i } ^ { \prime }$ , which we refer to as the repair operation, as shown in the right side of Figure 3. Moreover, $l _ { i }$ can be fused into the next layer, thereby avoiding the introduction of additional parameters (Li et al. 2020a). The proof is in Appendix B. Let ${ \widetilde { f } } _ { i } = l _ { i } \circ f _ { i }$ , and we use $r _ { i } ^ { \tilde { f } _ { i } }$ to refer to the repair of $f _ { i }$ with $\widetilde { f _ { i } }$ .e

Ability collapse refers to the situation that the ability of the extracted layer $f _ { i }$ deteriorates, especially when available data is limited. Previous approaches mitigate this issue by ensuring that the predictions of new layers or models remain consistent with original models (Hinton, Vinyals, and Dean 2015; Zhang, Zhu, and Ye 2019). However, in our situation, directly aligning $f _ { i }$ and $m _ { i }$ as their semantic spaces is not appropriate, due to the fact that the extracted layer $f _ { i }$ focuses on a different task compared to $m _ { i }$ . To solve this problem, we assume that different classes share some common abilities. This is particularly evident in the shallow layers of the network, where more low-level features are being abstracted. For example, in a DNN classifier for cars, horses, and dogs, information about edges and corners is captured first, followed by information about colors and contours, and finally information about object parts. Therefore, it is highly probable that these three classes share common sub-abilities to extract features, such as the ability to capture contours. We can put $f _ { i }$ back into the model and fine-tune it using samples from other classes to address this issue. By doing so, $f _ { i }$ will be updated when there is a sub-ability in $f _ { i }$ that is also a sub-ability of another class. Let ${ \overline { { f } } } _ { i }$ be the layer composed of the dropped units from $m _ { i }$ during the Ability pooling process. We simultaneously use ${ \overline { { f } } } _ { i }$ to repair $f _ { i }$ , denoted as $r _ { i } ^ { \overline { { f } } _ { i } }$ .

Let $\mathcal { M } ( \cdot ; p _ { i : j } )$ represent the replacement of layers from $i$ -th layer to $j$ -th layer with repair $r _ { i } ^ { p _ { i } } , \cdot \cdot \cdot , r _ { j } ^ { p _ { j } }$ , which can be expressed as

$$
{ \mathcal { M } } ( \cdot ; p _ { i : j } ) = m _ { 1 } \circ \cdot \cdot \cdot \circ m _ { i - 1 } \circ r _ { i } ^ { p _ { i } } \circ \cdot \cdot \cdot \circ r _ { j } ^ { p _ { j } } \circ m _ { j + 1 } \circ \cdot \cdot \cdot \circ m _ { L } ,
$$

so we can utilize $\mathcal { M } ( \cdot ; \widetilde { f } _ { i : j } )$ to represent a model that employs $\widetilde { f _ { i } }$ defined in size meismatch to repair layers, and use $\mathcal { M } ( \cdot ; \overline { { f } } _ { i : j } )$ to denote a model that utilizes ${ \overline { { f } } } _ { i }$ to repair lay

ers. Now we can define the adaption loss $\mathcal { L } _ { i } ^ { A }$ as:

$$
\mathcal { L } _ { i } ^ { A } = \mathbb { E } _ { ( x , y ) \in \mathcal { D } } \left[ \mathcal { L } _ { \mathrm { c l s } } ( x , y ; i , i ) + \gamma \mathcal { L } _ { \mathrm { d i s } } ( x ; i , i ) \right]
$$

$$
\mathcal { L } _ { \mathrm { c l s } } ( x , y ; i , j ) = \mathcal { L } _ { c } \left( \mathcal { M } ( x ; \widetilde { f } _ { i : j } ) \circ \mathcal { H } , y \right)
$$

$$
\mathcal { L } _ { \mathrm { d i s } } ( \boldsymbol { x } ; i , j ) = \left. \mathcal { M } ( \boldsymbol { x } ; \overline { { f } } _ { i : j } ) - \mathcal { M } ( \boldsymbol { x } ) \right. _ { 2 } ^ { 2 } ,
$$

where $\mathcal { L } _ { c }$ is the cross-entropy loss, $\gamma$ is the hyperparameter, and $\mathcal { H }$ is a classifier to output classification result. $\mathcal { L } _ { \mathrm { c l s } }$ works as the classification error to train $\ell _ { i }$ and $f _ { i }$ . ${ \mathcal { L } } _ { \mathrm { d i s } }$ works as the reconstruction loss to update $f _ { i }$ by utilizing common subabilities from dropped units, solving ability collapse problem. All other classes are considered as a single negative class.

Combining $f _ { i }$ layer by layer. A series of extracted layer $f _ { i }$ has been obtained, we start to construct ${ \mathcal { F } } ^ { c }$ . However, since each $f _ { i }$ is trained separately, we need to combine them one by one from front to end. The combining loss function $\mathcal { L } _ { i } ^ { C }$ can be expressed as

$$
\mathcal { L } _ { i } ^ { C } = \mathbb { E } _ { ( x , y ) \in \mathcal { D } } \left[ \mathcal { L } _ { \mathrm { c l s } } ( x , y ; 1 , i ) + \gamma \mathcal { L } _ { \mathrm { d i s } } ( x ; 1 , i ) \right] .
$$

By optimizing the loss function from $\mathcal { L } _ { 1 } ^ { C }$ to $\mathcal { L } _ { L } ^ { C }$ , we get optimized model $\mathcal { M } ( \cdot ; p _ { 1 : L } )$ . Since $l _ { i }$ is the linear layer and repair operation is also the linear operation, we can fuse all $l _ { i }$ into the linear layer next to it and finally get $\mathcal { F } ^ { c }$ .

# Packing Fragments

The packing method aims to construct a new model by fragments $\{ \mathcal { F } ^ { c } \} _ { c \in \mathcal { C } }$ so that the new model has all abilities in $\mathcal { C }$ . The challenge here is how to effectively combine features generated by these fragments. One possible approach is to simply concatenate features, but it will result in a very long feature vector that is difficult to work with, especially when there are multiple fragments.

In order to obtain tight features, we minimize the reconstruction loss for each fragment. Let $\ E n c$ be the fusion layer used to combine the features, and let $\{ D e c _ { c } \} _ { c \in \mathcal { C } }$ be the decoders used to restore the output of each fragment $\mathcal { F } ^ { c }$ . The final model $\mathcal { F } _ { \mathcal { C } }$ , use $\mathcal { F }$ for simplicity, can be expressed as:

$$
\mathcal { F } ( \boldsymbol { x } ) = E n c \left( \mathrm { c o n c a t } \left( \{ \mathcal { F } ^ { c } ( \boldsymbol { x } ) \} _ { c \in \mathcal { C } } \right) \right) .
$$

To ensure that the output of $\mathcal { F }$ contains all the information from the $\mathcal { F } ^ { c }$ fragments, we minimize the distance between $\mathcal { F } ^ { c }$ and $\mathcal { F } \circ D e c _ { c }$ . We define the packing loss function as:

$$
\mathcal { L } ^ { K } = \mathbb { E } _ { c \in \mathcal { C } , ( x , y ) \in \mathcal { D } _ { c } } \left[ \mathcal { L } _ { c } ^ { K } ( x , y ) \right]
$$

$\mathcal { L } _ { c } ^ { K } ( x , y ) = \mathcal { L } _ { c } \left( \mathcal { F } ( x ) \circ \mathcal { H } ^ { K } , y \right) + \delta \left\| \mathcal { F } ( x ) \circ D e c _ { c } - \mathcal { F } ^ { c } \right\| _ { 2 } ^ { 2 }$ in which $\mathcal { H } ^ { K }$ is the classifier for the packed model, and $\delta$ is the hyperparameter.

In the process, we only update the parameters of $E n c ,$ , $\{ D e c _ { c } \} _ { c \in \mathcal { C } }$ and $\mathcal { H } ^ { K }$ , while leaving the fragments $\{ \mathcal { F } ^ { c } \} _ { c \in \mathcal { C } }$ unchanged. This allows for easy integration or removal of fragments to meet new requirements, promoting high reusability of these fragments. We also have the flexibility to combine fragments with different structures. Additionally, due to limited number of parameters in ${ E n c }$ , $\{ D e c _ { c } \} _ { c \in \mathcal { C } }$ and $\mathcal { H } ^ { K }$ , with each being a linear layer followed by an activation layer, the workload required to adjust parameters is significantly low, resulting in increased efficiency in practice.

# Experiments

We conducted extensive experiments on various datasets for image classification and sentiment analysis. Our code was implemented by PyTorch and executed on an NVIDIA A100 40GB PCIe GPU with AMD EPYC 7H12 64-Core Processor. The implement details are provided in Appendix C.

# Experimental Settings

Datasets, Model and Task. We conducted a series of experiments on four different datasets: CIFAR10, CIFAR100 (Krizhevsky, Hinton et al. 2009), TREC (Hovy et al. 2001), and SST-5 (Socher et al. 2013). CIFAR10 and CIFAR100 consist of 50,000 images for training and 10,000 for testing. The TREC Question Classification dataset contains 5,500 sentences in the training set and another 500 in the test set, with 6 classes. SST-5 dataset consists of 8,544 sentences in the training set and another 2,210 in the test set, with 5 classes. We use VGG16 (Simonyan and Zisserman 2014) and ResNet34 (He et al. 2016) as original models on CIFAR10 and CIFAR-100, and CNN $\mathrm { K i m } 2 0 1 4 )$ and RNN as original models on TREC and SST-5.

We have designed 7 tasks to construct target models from original models. Task $T _ { 1 }$ uses 3 original models trained on CIFAR10. Tasks $T _ { 2 }$ and $T _ { 3 }$ utilize 5 original models trained on CIFAR100, with the former having models from the same superclass and the latter having models from different superclasses. Task $T _ { 4 }$ involves 10 original models trained on CIFAR100. Tasks $T _ { 6 }$ and $T _ { 7 }$ use 2 original models trained on TREC and SST-5, respectively. In $T _ { 7 }$ , the two models share a common class. Our approach involves slicing the models into multiple fragments, with each fragment representing a class, and then packing the selected fragments. This is in contrast to other methods that directly obtain the target model without the slicing and packing process. For each original model, we randomly sample $k$ samples per class from corresponding dataset. More details and experiments with different datasets and architectures are in Appendix D.

# Experimental Results

Performance on Image data. We begin by presenting the results of our method on image data and compare it with the finetuning method, CA-MKD (Zhang, Chen, and Wang 2022), and NetGraft (Shen et al. 2021). Finetuning method fine-tunes one of the original models using data of the target task. CA-MKD is the multi-teacher knowledge distillation method which adaptively assigns sample-wise reliability for each teacher prediction with the help of ground-truth labels. NetGraft is an knowledge distillation approach utilizing limited data. In our experiments, we utilize NetGraft to partition models and train a classifier layer for the combining models. As shown in Table 2, our method achieves better accuracy with a small number of parameters, especially for S&P (T). While S&P (R) has relatively lower accuracy, its parameter count is smaller. It is worth noting that the standard deviation of many results is higher than usual because we calculate the value between different models in the task, rather than different seeds in typical experimental settings. CA-MKD requires a substantial amount of data to

<html><body><table><tr><td rowspan="3" colspan="2">Task Method</td><td colspan="4">VGG16</td><td colspan="4">ResNet34</td></tr><tr><td rowspan="2">Param (×106)</td><td colspan="3">Acc.(%)</td><td rowspan="2">Param (×106)</td><td colspan="3">Acc.(%)</td></tr><tr><td>k=5</td><td>k=10 k=20</td><td></td><td>k=5</td><td>k=10</td><td>k=20</td></tr><tr><td rowspan="5">T1</td><td>CA-MKD</td><td>14.7(33.3%)</td><td>(failed)</td><td>(failed)</td><td>(failed)</td><td>18.9(33.3%)</td><td>(failed)</td><td>(failed)</td><td>(failed)</td></tr><tr><td>NetGraft</td><td>11.5(26.1%)</td><td>47.21±11.1</td><td>49.46±6.73</td><td>52.18±11.6</td><td></td><td></td><td></td><td></td></tr><tr><td>Finetune</td><td>14.7(33.3%)</td><td>65.47±6.41</td><td>66.88±4.47</td><td>66.99±2.33</td><td>18.9(33.3%)</td><td>67.29±4.69</td><td>70.04±6.74</td><td>71.56±4.80</td></tr><tr><td>S&P (R)</td><td>11.5(26.1%)</td><td>71.13±2.55</td><td>74.90±6.56</td><td>74.47±7.52</td><td>19.0(33.5%)</td><td>75.43±4.91</td><td>78.49±8.15</td><td>80.43±6.01</td></tr><tr><td>S&P(T)</td><td>7.4(16.8%)</td><td>62.20±4.00</td><td>73.23±5.38</td><td>76.37±5.49</td><td>25.7(45.3%)</td><td>73.57±2.25</td><td>76.89±6.39</td><td>80.31±6.43</td></tr><tr><td rowspan="5">T2</td><td>CA-MKD</td><td>14.7(20.0%)</td><td>(failed)</td><td>(failed)</td><td>63.10±10.2</td><td>18.9(20.0%)</td><td>(failed)</td><td>(failed)</td><td>53.35±10.5</td></tr><tr><td></td><td>19.2(26.1%)</td><td></td><td></td><td>63.90±6.75</td><td></td><td></td><td></td><td></td></tr><tr><td>NetGraft</td><td>14.7(20.0%)</td><td>54.45.04</td><td>62.0001</td><td>68.35±8.21</td><td>18.9(20.0%)</td><td>65.20±9.69</td><td>71.40±8.64</td><td>73.30±9.60</td></tr><tr><td>S&P (R)</td><td>19.2(26.1%)</td><td>61.95±12.9</td><td>68.65±11.0</td><td>72.90±7.79</td><td>31.7(33.5%)</td><td>67.10±13.7</td><td>72.55±9.26</td><td>75.85±9.92</td></tr><tr><td>S&P(T)</td><td>27.4(37.2%)</td><td>68.40±12.0</td><td>78.10±6.08</td><td>81.40±4.15</td><td>58.6(62.0%)</td><td>77.00±10.6</td><td>78.90±10.2</td><td>84.50±7.92</td></tr><tr><td rowspan="5">T3</td><td>CA-MKD</td><td>14.7(20.0%)</td><td>(failed)</td><td>(failed)</td><td>62.42±7.37</td><td>18.9(20.0%)</td><td>(failed)</td><td>(failed)</td><td>50.41±10.5</td></tr><tr><td>NetGraft</td><td>19.2(26.1%)</td><td>49.50±5.16</td><td>56.85±10.3</td><td>61.85±5.59</td><td></td><td></td><td>1</td><td></td></tr><tr><td>Finetune</td><td>14.7(20.0%)</td><td>60.55±4.26</td><td>62.05±3.62</td><td>67.00±5.31</td><td>18.9(20.0%)</td><td>64.55±7.17</td><td>68.15±5.59</td><td>73.55±6.44</td></tr><tr><td>S&P(R)</td><td>19.2(26.1%)</td><td>64.80±8.32</td><td>70.05±8.33</td><td>72.15±6.70</td><td>31.7(33.5%)</td><td>72.80±5.66</td><td>72.20±5.38</td><td>79.40±5.26</td></tr><tr><td>S&P (T)</td><td>27.0(36.7%)</td><td>71.35±6.43</td><td>77.80±7.41</td><td>79.55±4.27</td><td>59.8(63.3%)</td><td>75.75±8.01</td><td>81.20±4.45</td><td>84.65±3.92</td></tr><tr><td rowspan="5">T4</td><td>CA-MKD</td><td>14.7(20.0%)</td><td>(failed)</td><td>(failed)</td><td>59.08±8.68</td><td>18.9(20.0%)</td><td>(failed)</td><td>(failed)</td><td>47.74±11.3</td></tr><tr><td></td><td>19.2(26.1%)</td><td></td><td></td><td>58.84±6.57</td><td></td><td></td><td></td><td></td></tr><tr><td>NetGraft</td><td>14.7(20.0%)</td><td>44.48.01</td><td>55.364547</td><td>58.72±5.39</td><td>18.9(20.0%)</td><td>53.90±5.63</td><td>59.04±4.42</td><td>63.32±5.47</td></tr><tr><td>S&P (R)</td><td>19.2(26.1%)</td><td>56.40±7.23</td><td>63.74±7.96</td><td>69.86±7.03</td><td>31.7(33.5%)</td><td>66.80±9.56</td><td>69.15±7.52</td><td>77.30±6.17</td></tr><tr><td>S&P(T)</td><td>14.8(20.1%)</td><td>60.62±5.98</td><td>65.96±6.47</td><td>73.72±5.38</td><td>53.1(56.2%)</td><td>70.10±6.15</td><td>74.90±5.73</td><td>79.46±4.12</td></tr><tr><td rowspan="5">T5</td><td>CA-MKD</td><td>14.7(10.0%)</td><td>(failed)</td><td>42.22±5.19</td><td>50.26±4.83</td><td>18.9(10.0%)</td><td>(failed)</td><td>34.01±5.88</td><td>40.97±6.11</td></tr><tr><td></td><td>38.4(26.1%)</td><td>43.42±6.65</td><td></td><td>59.16±6.69</td><td></td><td></td><td></td><td></td></tr><tr><td>NetGrat</td><td>14.7(10.0%)</td><td>34.54±4.20</td><td>53.18±441</td><td>42.11±3.44</td><td>18.9(10.0%)</td><td>40.04±4.95</td><td>45.38±5.07</td><td>50.81±4.44</td></tr><tr><td>S&P (R)</td><td>38.4(26.1%)</td><td>45.59±5.83</td><td>54.63±6.27</td><td>59.22±5.41</td><td>63.4(33.5%)</td><td>50.67±4.99</td><td>57.96±5.80</td><td>63.40±4.68</td></tr><tr><td>S&P(T)</td><td>27.7(18.8%)</td><td>48.31±5.58</td><td>56.35±5.32</td><td>65.03±4.75</td><td>101.8(53.9%)</td><td>59.67±6.03</td><td>66.10±4.89</td><td>71.73±4.25</td></tr></table></body></html>

Table 2: The performance on image datasets. S&P (R) denotes selecting units with rate, and S&P (T) denotes selecting units with threshold. The percentage value in parentheses of the parameter column indicates the number of parameters compared to the ensemble method. (failed) means that the method cannot run in the setting for the lack of sufficient samples. Due to the code for ResNet34 is not provided, there are no reported results using NetGraft on this architecture.

![](images/b2baf9f48d4d80111a0cedb0ab447a0033c28b77bca4d82c825d47d7e2220eba.jpg)  
Figure 4: Comparison of parameter amount and accuracy.

Table 3: Result on text datasets.   

<html><body><table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td rowspan="2">x10</td><td colspan="2">k=5=10</td></tr><tr><td></td><td></td></tr><tr><td rowspan="3">RN)</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td></td><td>10</td><td>89.3</td><td>83.4</td></tr><tr><td rowspan="3">T7 (CNN)</td><td>Finsemble</td><td></td><td></td><td></td></tr><tr><td></td><td>72.0</td><td>59.19</td><td>73.22</td></tr><tr><td>S&P</td><td>35.3</td><td>67.73</td><td>74.02</td></tr></table></body></html>

obtain the corresponding models. It cannot get result when the number of samples is small. NetGraft is designed to obtain smaller models while maintaining the same ability as original models. We also compare our results with the ensemble method when $k = 2 0$ , which integrates results of all original models, as shown in Figure 4. Although the ensemble method achieves nearly 5 points higher accuracy than S&P, the resulting model is significantly larger, with 2 to 10 times more parameters than other generated methods. Furthermore, the ensemble method means users must purchase all models, resulting in higher costs. Experiment of packing fragments with different architectures or from different datasets is shown in Appendix D.

Performance on Textual data. We evaluate the performance of our approach on text datasets TREC and SST-5 datasets for Task $T _ { 6 }$ and $T _ { 7 }$ . Table 3 presents the accuracy of our method in comparison with the ensemble and finetune methods. Ensemble method concatenates the features of the original models and trains a classifier based on the concatenated feature, while finetune method fine-tunes one of the original models. Our method achieves comparable results with the ensemble method and even outperforms the ensemble method on $T _ { 7 }$ when $k = 1 0$ . The results of finetune method show that the provided samples are insufficient to train a new model.

VGG16 Rate Resnet34 Rate VGG16 Threshold Resnet34 Threshold c1 2 c3 C4 C5 c1 2 3 C4 C5 c1 2 3 C4 C5 c1 2 3 C4 C5 ： 1 1 1 1   
c1 1.00 0.33 0.33 0.33 0.33 c1 1.00 0.18 0.18 0.18 0.19 c1 1.00 0.61 0.62 0.63 0.60 c1 1.00 0.58 0.59 0.59 0.55   
c2 0.33 1.00 0.33 0.33 0.33 c2 0.18 1.00 0.18 0.18 0.17 c2 0.61 1.00 0.62 0.61 0.62 c2 0.58 1.00 0.60 0.58 0.56   
c3 0.33 0.33 1.00 0.34 0.33 c3 0.18 0.18 1.00 0.18 0.17 c3 0.62 0.62 1.00 0.64 0.61 c3 0.59 0.60 1.00 0.62 0.53   
c4 0.33 0.33 0.34 1.00 0.34 c4 0.18 0.18 0.18 1.00 0.18 c4 0.63 0.61 0.64 1.00 0.58 c4 0.59 0.58 0.62 1.00 0.50   
c5 0.33 0.33 0.33 0.34 1.00 c5 0.19 0.17 0.17 0.18 1.00 c5 0.60 0.62 0.61 0.58 1.00 c5 0.55 0.56 0.53 0.50 1.00

![](images/411b9aa52fc804b0e2c69dcc661e641b3aad8680ab5e82bd2812351b57fa7e05.jpg)  
Figure 5: The Intersection over Union (IoU) of the selected units set $U _ { i }$ between fragments varies significantly based on the fragment’s class. The IoU of S&P with threshold is relatively high, as it maintains more parameters in the front layers.

![](images/b6b31d8b5798d3e3b2dc06cc485ef5802760b2afdbb412c9de1b08f0fd7c8d4b.jpg)  
Figure 6: Equivalent expansion rate (Y-axis) of the market on different market Acceptable Error (X-axis).   
Figure 7: The spent time per class (Y-axis, seconds) as the number of user required models $\mathrm { \Delta X }$ -axis) increases on $T _ { 1 }$ .

The Equivalent expansion rate of the Market. The equivalent expansion rate is a measure of how many times a method increases the number of models available in the market. Only models with an accuracy higher than $1 - e$ is considered usable, in which $e$ is the market Acceptable Error predefined by market managers. To measure the equivalent expansion rate of our framework, we randomly generate 100 different combinations of 5 classes from 5 different original models, where each class is drawn from a different model. These original models are the same as those used in $T _ { 3 }$ . We plot the relation between the equivalent expansion rate and the market Acceptable Error when $k = 5$ , 10, 20 in Figure 6. Our results demonstrate that our framework can effectively expand the market. When we set the market acceptable error to 0.2, the market is amplified to about 200,000 times its original size. Similarly, when we set the market acceptable error to 0.3, the market is expanded to about 600,000 times its original size, representing a significant improvement.

Difference between Fragments. Next, we investigate the differences between each pair of fragments from the same original model in our method. Figure 5 shows the Intersection over Union (IoU) of each layer’s selected units set $U _ { i }$ on different fragments in task $T _ { 3 }$ . We calculate the average IoU among all layers for each pair of fragments in the target of the task. Here, $c _ { 1 } , c _ { 2 } , \cdots , c _ { 5 }$ represent classes in the original model. Our results show that fragments of different classes are significantly different in terms of the selection of units. Compared with selecting units by rate, selecting units by threshold has higher IoU. This is because it prefers to maintain more parameters in the front layers and drop more parameters in the last layers. This observation aligns with the intuition that there are more common sub-abilities in shallow layers of networks. More details are shown in Appendix E.

Running time of Slicing-and-Packing. We show the running time of our method on $T _ { 1 }$ using VGG16 and ResNet34 as original models. Figure 7 displays the time spent per class as the number of models required by users increases. As the slope of lines in figure is small, our method only needs little time to pack required fragments, which is insignificant compared with the slicing part. Besides, since the time needed to pack fragments is little, our method has plug-and-play property and user can add or remove fragments at any time.

# Conclusion and Future Work

In this paper, we developed a novel framework called Sliceand-Pack. This framework involves slicing existing models into fragments and packing required fragments with necessary abilities, allowing users to purchase models at a lower cost. Additionally, it provides a commercially viable solution for markets and enables them to offer a wider variety of models, demonstrating the capabilities of a larger market. Our experiments have shown that our framework can obtain highly accurate packed models with minimal time spent, particularly on packing, which enables plug-and-play functionality. In our future work, we plan to delve into a more flexible approach to slicing models and leveraging shared capabilities or classes across original models. Moreover, we intend to explore a wider range of network architectures in order to augment our research even further.

# Acknowledgments

This research was supported by NSFC (62076121, 61921006) and Major Program (JD) of Hubei Province (2023BAA024). The authors would like to thank Prof. Peng Zhao for his helpful feedback on drafts of the paper.
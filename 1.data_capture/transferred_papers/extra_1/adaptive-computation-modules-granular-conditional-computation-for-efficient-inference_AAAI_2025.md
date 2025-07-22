# Adaptive Computation Modules: Granular Conditional Computation for Efficient Inference

Bartosz W´ojcik1, 2, Alessio Devoto3, Karol Pustelnik4, Pasquale Minervini5, 6, Simone Scardapane3

1IDEAS NCBR   
2Jagiellonian University   
3Sapienza University of Rome   
4University of Warsaw   
5University of Edinburgh   
6Miniml.AI

# Abstract

While transformer models have been highly successful, they are computationally inefficient. We observe that for each layer, the full width of the layer may be needed only for a small subset of tokens inside a batch and that the “effective” width needed to process a token can vary from layer to layer. Motivated by this observation, we introduce the Adaptive Computation Module (ACM), a generic module that dynamically adapts its computational load to match the estimated difficulty of the input on a per-token basis. An ACM consists of a sequence of learners that progressively refine the output of their preceding counterparts. An additional gating mechanism determines the optimal number of learners to execute for each token. We also propose a distillation technique to replace any pre-trained model with an “ACMized” variant. Our evaluation of transformer models in computer vision and speech recognition demonstrates that substituting layers with ACMs significantly reduces inference costs without degrading the downstream accuracy for a wide interval of user-defined budgets.

# Introduction

Driven by their constantly improving capabilities, state-ofthe-art neural networks have been experiencing a continued growth in size in the last decade (Pugliese, Regondi, and Marini 2021). This progress was made possible by algorithmic and model design improvements (in particular with the introduction of transformer models) and the increasing computational power of modern GPUs. Scaling up the model architecture frequently results in improved performance (Devlin et al. 2018; Zagoruyko and Komodakis 2016), causing the size and computational costs of state-of-the-art models to keep increasing steadily. These escalating costs limit applications in latency-sensitive and energy-constrained scenarios and contribute to higher carbon emissions, exacerbating environmental concerns (Schwartz et al. 2020; Patterson et al. 2022).

Several approaches from the literature aim to mitigate this problem. For example, quantization reduces inference time by quantizing the weights and activations into low-precision floating-point (Courbariaux, Bengio, and David 2014) or integer values (Wu et al. 2020; Dettmers et al. 2022). Knowl(a) High-level overview of the Adaptive Computation Module (b) Example images with computation spatial load maps edge distillation methods (Hinton, Vinyals, and Dean 2015; Aguilar et al. 2020) can transfer knowledge from an ensemble or single larger model into a smaller network. Finally, sparsification methods (LeCun, Denker, and Solla 1989; Han et al. 2015; Hoefler et al. 2021) yield either sparse weight or sparse activation tensors.

![](images/e2360cb21819aaaf786456a00b56cfd7283671e9499022afa5904d0da93ca7b8.jpg)

![](images/ca7e756af93494d3ffa9b2153e07f8841ba97588056776992a1c1e7059587208.jpg)  
Figure 1: ACMs adapt their computational load for each input on a per-token basis by selecting the number of learners to execute via a trainable gate. In the example on the top, a background token (green) is allocated fewer learners than a content-rich token (orange). This results in a spatially varying computational load, as shown on the bottom.

While they incur a slight decrease in downstream model performance, such methods show that some neural modules can often be replaced with computationally cheaper alternatives. Based on this observation, we argue that in transformer models (Vaswani et al. 2017), the representational capacity of a whole layer is not needed for every input token. In other words, we hypothesize that the same transformation could be achieved in a more computationally efficient manner. To explore this hypothesis, we introduce Adaptive Computation Modules (ACMs), a neural network module that adapts its computational burden to match the difficulty of the current input token. An ACM consists of a sequence of learners and a single gating network. The task of each learner is to improve upon the combined output of previous learners, while the gating network determines the optimal number of learners to execute for each input token. Since all token-level decisions are independent, the resulting model is a highly granular application of the conditional computation paradigm (Bengio 2013; Bengio et al. 2015), and thus allows for spatial adaptability in transformer models (Figurnov et al. 2017; Han et al. 2021). Figure 1 provides an outline of the proposed model.

To enable the use of a vast range of publicly available models, we propose a straightforward conversion procedure in which we: (1) substitute the selected blocks of the model with ACMs, (2) initialize the learners by distilling knowledge from the substituted blocks, (3) pre-train the gating networks using artificially generated labels, and (4) train the entire model in an end-to-end manner.

Our approach can significantly decrease the model’s computational footprint without sacrificing performance. We evaluate our method on the ImageNet-1k (Russakovsky et al. 2015) dataset with Vision Transformer (ViT) models and on speech recognition with pre-trained Wav2Vec networks (Baevski et al. 2020), and show that in both cases we achieve a better performance-efficiency trade-off than other conditional computation methods. We make the following contributions:

• We show that individual layers in transformer models can be computationally inefficient and that their entire expressiveness is often needed only for a small subset of input tokens.   
• We introduce ACM, an easy-to-train, modular, and general-purpose module that offers a granular approach for reducing the computational cost of transformers.   
• We propose a smart initialization strategy for the parameters of ACMs that relies on module-wise knowledge distillation from pre-trained models.   
• We provide an efficient GPU implementation to demonstrate that ACMs effectively speed up inference.

# Related Work

We provide a brief overview of related works, focusing on our experimental comparison. A longer overview with a broader outlook can be found in the supplementary material.

# Conditional Computation

Conditional computation (CC) refers to the ability of a model to adapt its computational graph to its input. While CC can be used for problems such as continual learning (Lin, Fu, and Bengio 2019), most CC methods adjust their execution on a per-input basis to significantly reduce their average computational cost while maintaining performance (Scardapane et al. 2024).

In the following, we focus on CC algorithms that can be applied to any pre-trained transformer model with minimal modifications; architecture-specific CC algorithms (e.g., dynamic channel selection for CNNs (Chen et al. 2019; Li et al. 2021)) are discussed in the supplementary material. We broadly categorize the methods into three groups: early-exits (EEs) (Teerapittayanon, McDanel, and Kung 2016; Bolukbasi et al. 2017), mixture-of-experts (MoEs) (Yuksel, Wilson, and Gader 2012; Shazeer et al. 2017; Fedus, Dean, and Zoph 2022), and token dropping (TD) (Rao et al. 2021; Yin et al. 2022; Meng et al. 2022; Haurum et al. 2023). Finally, Cai et al. (2024) is a concurrent conditional computation work that is remarkably similar to the proposed ACMs. In particular, it also trains a router for each layer and converts a pre-trained dense model with a three-step method.

# Early-exits

In EE models, inputs are allowed to “exit” the architecture at intermediate layers via additional classifier heads that are trained together with the backbone network (Teerapittayanon, McDanel, and Kung 2016), or as a separate phase in a layerwise fashion (Han et al. 2021), and the predictions of multiple heads can be merged with a number of different techniques (Teerapittayanon, McDanel, and Kung 2016; Wołczyk et al. 2021; Scardapane et al. 2020). In EE networks, the execution is stopped for the entire input at once, as opposed to individual tokens as is done in the proposed ACMs. In addition, designing and placing exit heads is itself a non-trivial problem (Teerapittayanon, McDanel, and Kung 2016), and very little work has been done outside standard computer vision and natural language processing classification tasks. In contrast, ACMs are designed to replace individual blocks in a transformer model in a plug-and-play fashion.

# Token Dropping

TD strategies either prematurely stop execution (Rao et al. 2021; Yin et al. 2022; Haurum et al. 2023) or skip computation of selected layers (Meng et al. 2022) for individual tokens that are deemed less relevant or redundant. We can interpret them as dynamically allocating a variable depth to each token. Our proposed ACM can be seen as an orthogonal width variant of token dropping, in which each token is allocated a variable width for each layer in the network.

# Mixture-of-Experts

In MoEs, selected modules of a model are replaced by an independent set of blocks called experts, which are selectively activated for each token by an additional routing network (Puigcerver et al. 2023). MoEs have been successfully developed for replacing MLP layers in transformer models (Shazeer et al. 2017; Riquelme et al. 2021), attention layers (Zhang et al. 2022), entire blocks (Tan et al. 2023), and adapters (Zadouri et al. 2023). Although MoEs are typically trained from scratch or fine-tuned from existing models (Zadouri et al. 2023), a small number of works have investigated moefication procedures (Zhang et al. 2021; Qiu, Huang, and Fu 2023). Importantly, in MoEs, each token is allocated a fixed amount of compute depending on the routing strategy (e.g., top- $k$ routing (Riquelme et al. 2021)). ACM can be seen as a modification of MoEs in which the experts are ordered, and thus, the complexity of the routing problem is drastically reduced. The gating network decides how many learners instead of which experts to execute, therefore allowing for a variable amount of computation on a per-token basis. A concurrent work of Jain et al. (2024) defines nested experts, which results in a dynamic model similar to ACMs.

![](images/39ef5f044ab9cf1400aee0c4c5b7f9aa89af6d1669474294f204574277b7f49d.jpg)  
Figure 2: Architecture of an ACM block: the output is the sum of $k$ learners, where $k$ is determined on a per-token basis by a small gating network $g$ . The learners are executed in parallel. In the example, only the first two learners are executed, and the computation of the third (greyed out) is skipped.

# Method

An ACM is a conditional computation block that adapts its execution cost for each processed token via a trainable gating layer. In this paper, instead of training an ACMbased model from scratch, we focus on converting any pretrained transformer network into an equivalent “ACMized” version, i.e. one having similar accuracy while achieving a pre-defined computational budget on average. To this end, we propose an effective weight pre-initialization scheme. First, we substitute a subset of layers of the base model (e.g., MLPs or MHA projections) with an ACM of similar size. The ACMs are then trained independently but in parallel, first with a per-layer reconstruction loss to initialize the learners and then using a cross-entropy loss to initialize the gates. The actual training consists of fine-tuning the model in an end-to-end manner to allow it to adapt its weight to the dynamic inference setting. In the supplementary material, we demonstrate that this setup significantly speeds up the training of the ACMized networks in all cases.

# Adaptive Computation Module

Our ACM module aims to allow the model to work well with any required computational budget while still ensuring efficient execution on GPUs, a property most of the dynamic width methods lack (Li et al. 2021). Given an input token $z$ , consider a set of $N$ homogeneous modules, $s _ { n } \big ( z ; \phi _ { ( n ) } \big )$ , $n \in \{ 1 , . . . , N \}$ , each module having weights $\phi _ { ( n ) }$ . We refer to these modules as learners. In an ACM, each learner progressively refines the prediction of the previous ones such that the $k$ -th output of the ACM block, $k \in \{ 1 , . . . , N \}$ , is given by:

$$
h ( z , k ; \phi ) = \sum _ { n = 1 } ^ { k } s _ { n } ( z ; \phi _ { ( n ) } )
$$

All intermediate outputs $h ( z , 1 ; \phi ) , \ldots , h ( z , N ; \phi )$ are valid choices for the output of the ACM: a larger value for $k$ yields a more accurate result at the cost of a higher computational burden, while $k = 1$ means that only a single learner is executed. Note that once $k$ is known for a token, the learners can be executed in parallel.

For any token $z , k$ should be chosen as the smallest possible value that guarantees good network performance. To this end, at the beginning of the ACM block, we add a small trainable gating network $g$ with weights $\omega$ to select the number of learners $k$ to be executed. In practice, the gating network returns $N$ real-valued outputs, which are then discretized into a one-hot gating choice vector with the Gumbel-Softmax trick (Jang, Gu, and Poole 2016) to retain differentiability:

$$
\nu _ { n } = \frac { \exp ( ( \log ( g ( z ; \omega ) ) _ { n } + \gamma _ { n } ) / T ) } { \sum _ { n } \exp ( ( \log ( g ( z ; \omega ) ) _ { n } + \gamma _ { n } ) / T ) } .
$$

In Equation (2), $\gamma _ { 1 } , . . . , \gamma _ { N }$ are i.i.d samples from Gumbel $( 0 , 1 )$ , and $T$ is softmax temperature. In the forward pass, we discretize $\nu$ into a one-hot vector $\hat { \nu }$ , but the continuous values are used directly for gradient computation in the backward pass. The complete ACM architecture is outlined in Figure 2.

Design of the ACM blocks Any network architecture with the required input and output dimensionality can play the role of a learner. For simplicity, in this paper, we always use two dense layers with a GELU activation function in between. Since we replace selected modules of a pre-trained model with ACMs, we always pick $N$ and the size of a single learner such that the entire ACM module has approximately the same computational cost and the number of parameters as the substituted block. This is straightforward to achieve as the cost of a learner scales linearly with its hidden dimensionality. However, the output dimension has to be the same as in the replaced module. This results with output layer biases taking a larger share of learner parameters as we increase $N$ . To avoid this effect, we simply eliminate them from our architecture.

For the gating network, we also use a two-layer network and determine its hidden dimensionality such that the total computational cost of gating is around $1 \%$ of the overall model cost. When feasible – such as the module being placed under a residual connection – we set the minimum number of executable learners to 0 so that even the computation of the first learner can be skipped.

# ACM weight initialization

The costs of training large models are constantly increasing (Strubell, Ganesh, and McCallum 2019). On the other hand, there is a large number of publicly available pretrained models, so the capability of adapting them is a desired property for new methods. Since ACMs are designed to replace individual modules, we initially train them by distilling the knowledge from the corresponding trained modules that are being substituted. Specifically, we propose the following scheme. A trained static model $f ( \cdot )$ is cloned, and selected modules (e.g., every MLP module) are replaced with

ACMs in that cloned model $f _ { \mathrm { A C M i z e d } } ( \cdot )$ . Each learner from every ACM is initialized randomly. For each sample $x _ { i }$ from a mini-batch $B$ forwarded through the original model, we save every input token $z _ { i , j } ^ { l }$ and output token $\underset { . } { \dot { o } _ { i , j } ^ { l } }$ of each $l$ -th module that was replaced in the copied model. Then, every ACM is trained independently and in parallel by minimizing the mean squared error (MSE) applied for every possible choice of $k$ :

$$
\mathcal { L } ( \phi ) = \frac { 1 } { | B | S L N } \sum _ { i , j , l , n } \left\| h ( z _ { i , j } ^ { l } , n ; \phi ^ { l } ) - o _ { i } ^ { l } \right\| ^ { 2 }
$$

where $S$ is token sequence length, and $L$ is the number of substituted modules. Note that the gating network is not needed in this phase. The effectiveness of this training approach can be tested by setting a fixed $k$ for every ACM in the model and evaluating the model on the validation set.

With learners being able to reliably imitate the replaced modules, we subsequently freeze them and train the gating networks in a similar, layer-wise approach. We frame the problem as a classification task and generate artificial labels with the following heuristic. First, we consider the outputs of all learners and compute the distance to the original output:

$$
d _ { i , j } ^ { l } ( n ) = \| h ( z _ { i , j } ^ { l } , n ; \phi ) - o _ { i , j } ^ { l } \| _ { 2 }
$$

The target label is then set as:

$$
t _ { i , j } ^ { l } = \operatorname* { m i n }  n \in \lbrace 2 , . . . , N \rbrace | \frac { d _ { i , j } ^ { l } ( n ) } { d _ { i , j } ^ { l } ( n - 1 ) } \geq \tau  ,
$$

where $\tau$ is a threshold hyperparameter. In other words, we select the smallest number of learners such that the relative improvement from adding one more learner is lower than the threshold $\tau$ . With these labels, the gating networks are trained using standard cross-entropy loss:

$$
\mathcal { L } ( \omega ) = \frac { 1 } { | B | S L N } \sum _ { i , j , l , n } - t _ { i , j , n } ^ { l } \log ( \nu _ { i , j , n } ^ { l } ) .
$$

# End-to-end training

In the third phase, we finetune the entire model end-to-end to allow it to adapt its weights to the dynamic inference setting. To make the model more predictable in terms of its computational cost, we add an auxiliary loss term that penalizes for any deviation from the given target budget $\beta _ { \mathrm { t a r g e t } }$ on average:

$$
\mathcal { L } _ { \mathrm { b } } ( \theta ) = \left\| \frac { 1 } { | B | } \sum _ { i } \frac { \sum _ { j } \sum _ { l } k _ { i , j } ^ { l } p ^ { l } } { \sum _ { j } \sum _ { l } N p ^ { l } } - \beta _ { \mathrm { t a r g e t } } \right\| _ { 1 } ,
$$

where $p ^ { l }$ is the computational cost of a single learner from layer $l$ and $\beta _ { \mathrm { t a r g e t } } \ \bar { \in } \ ( 0 , 1 )$ is the targeted computational budget. This term still allows for the allocation of different amounts of computational resources to samples of varying complexity and only requires to be close to $\beta _ { \mathrm { t a r g e t } }$ on average. It also affects the computational cost of future training steps, potentially accelerating the training process.

While $\mathcal { L } _ { \mathrm { b } }$ does not prevent diversity of gating choices, it does not encourage it. In practice, we find that the gating networks collapse to a state where the same number of learners is chosen for every input token, effectively turning our dynamic model into a static one. To prevent this behavior and encourage diversity, we add two additional loss terms. The first auxiliary loss term maximizes the average normalized entropy of gating choices taken for tokens in a single image:

$$
\mathcal { L } _ { \mathrm { e } } ( \theta ) = \frac { 1 } { | B | L } \sum _ { i , l } \frac { \sum _ { n } a _ { n } \log ( a _ { n } ) } { \log ( N ) } ,
$$

where:

$$
a _ { i , n } ^ { l } = \frac { \sum _ { j } \hat { \nu } _ { i , j , n } ^ { l } } { \sum _ { n } \sum _ { j } \hat { \nu } _ { i , j , n } ^ { l } }
$$

represents a distribution between $N$ choices in batch $B$ for the $l$ -th ACM aggregated for an entire sequence of tokens from sample $i$ . The intuition behind this loss is that not every input region is equally important; hence, a non-uniform distribution of computation is required.

Finally, entire images may exhibit different difficulty levels, and enforcing diversity of gating choices for a single image at a time may not be enough. We address this with the second auxiliary loss, which is defined as:

$$
\mathcal { L } _ { \mathrm { d } } ( \theta ) = \frac { 1 } { \left| \boldsymbol { B } \right| ^ { 2 } } \sum _ { i } \sum _ { m } \left\| b _ { i } - b _ { m } \right\| _ { 1 } ,
$$

where:

$$
b _ { i } = \frac { \sum _ { j } \sum _ { l } k _ { i , j } ^ { l } } { \sum _ { j } \sum _ { l } N }
$$

denotes the fraction of learners executed on sample $i$ . It encourages the model to distribute its computational budget between easy and hard examples. The final loss that is minimized for classification tasks is given by:

$$
\begin{array} { c l c r } { \displaystyle \mathcal { L } ( \theta ) = \frac { 1 } { | B | } \sum _ { i } \sum _ { c } - y _ { i , c } \log ( \hat { y } _ { i , c } ) + \alpha _ { \mathrm { b } } \mathcal { L } _ { \mathrm { b } } ( \theta ) } \\ { \displaystyle + \alpha _ { \mathrm { e } } \mathcal { L } _ { \mathrm { e } } ( \theta ) + \alpha _ { \mathrm { d } } \mathcal { L } _ { \mathrm { d } } ( \theta ) , } \end{array}
$$

where $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { | B | }$ are samples from the current mini-batch $B$ , and $\alpha _ { \mathrm { b } }$ , $\alpha _ { \mathrm { e } }$ , $\alpha _ { \mathrm { d } }$ are hyperparameters for weighting the different terms. In practice, we always use $\alpha _ { \mathrm { b } } ~ = ~ 0 . 1$ , $\alpha _ { \mathrm { { e } } } =$ 0.05, and $\alpha _ { \mathrm { d } } ~ = ~ 0 . 0 5$ . In the analysis section, we present an ablation study that demonstrates the necessity of applying the proposed auxiliary losses and shows the positive impact of their interplay. Note that the auxiliary losses are taskagnostic, allowing ACMs to be used for any other task than classification.

# Experiments

Due to the widespread availability of pre-trained weights, we evaluate our method on popular image classification and speech recognition tasks. The aim of the evaluation is to compare the performance-efficiency trade-offs of different methods. To measure the computational efficiency, we track the number of FLOPs used by the model for each evaluated sample and report the averaged result. The source code for our experiments is available at https://github.com/ bartwojcik/adaptive computation modules.

![](images/a474741c6c23784573a484692c4fb931ae8529fb08b1865eb6e4cf5c577838d3.jpg)  
Figure 3: Performance-efficiency trade-offs of different conditional computation methods as measured on the ImageNet1k dataset. ACM-based ViT-B achieves the Pareto frontier for a wide range of computational budgets.

# Computer Vision

We select a ViT-B (Dosovitskiy et al. 2020) model pretrained on ImageNet-1k (Russakovsky et al. 2015) from the torchvison library1 as the base model for all methods. We compare ACMized models with three conditional computation techniques, each one coming from a different group: A-ViT (Yin et al. 2022) (Token Dropping), MoEfication (Zhang et al. 2021) (Mixture-of-Experts), and Zero Time Waste (Wołczyk et al. 2021) (Early Exiting). For the sake of a fair comparison, we assume the same training data budget of 100 epochs for every method.

Since MoEfication requires models using ReLU activation functions, we first replace GELU activations with ReLUs and finetune the model for 80 epochs, as described by Zhang et al. (2021). The remaining 20 epochs are used for training the routers. For Zero Time Waste, we train the earlyexit heads for 95 epochs and then train the ensembles for 5 epochs. For the ACMized ViT, we replace every MLP module and every linear projection in each MHA with an ACM module (the self-attention mechanism itself is not affected). Module-wise representation distillation is performed for 2 epochs, followed by 1 epoch for pre-training of the gating networks and 97 epochs of end-to-end finetuning of the entire model. We set the number of learners $N$ to 4 in every ACM. While MoEfication and Zero Time Waste can be evaluated with different numbers of selected experts and early exit confidence thresholds, A-ViT and ACMs need to be trained multiple times with different values of hyperparameters that approximately determine the final average computational budget. We emphasize that our A-ViT implementation includes fixes for two issues raised by GitHub users23, which may affect the final performance of A-ViT. The authors of A-ViT have not yet addressed them at the time of writing this article.

![](images/49ccc8382e75121d1d9961091dcdd5bfbc3eb167ac68a3a86317a694dc3bf228.jpg)  
Figure 4: Performance-efficiency trade-offs of different conditional computation methods as measured on the CommonVoice-es dataset. The model’s performance is reported in terms of Word Error Rate (WER). ACMs achieve lower WER for every evaluated computational budget.

We present the results in Figure 3. As projections in MHA are responsible for around $3 0 \%$ of the model cost, MoEfication seems to be hindered by the fact that it reduces only the cost of the MLP layers. A-ViT shows a gap in performance in relation to the pre-trained model, while Zero Time Waste is competitive only for higher computational budgets. ACMized ViT displays favorable performance for a wide range of computational budgets, and its advantage is especially significant below 12.5 GFLOPs.

# Speech-to-Text

For speech recognition tasks, we use the XLS-R-300M (Babu et al. 2021), a pre-trained Wav2Vec2 model (Baevski et al. 2020), fine-tuned on selected languages from the CommonVoice dataset (Ardila et al. 2020). Speech-to-text models introduce additional complexities for dynamic computation methods, as each token is decoded individually. Since A-ViT and Zero Time Waste were not designed for this setting, we restrict our baseline methods to MoEfication, the only task-agnostic method among those three.

We assume an equal training data budget of 10 epochs for all approaches. In the case of MoEfication, we substitute GELUs with ReLUs and train for eight epochs, followed by two epochs of router training. For ACMs, we replace every MLP block within the model with an ACM module with $N = 4$ . We subsequently perform five epochs of modulewise distillation, one of training the gating networks, and four epochs of end-to-end finetuning. We present results in Figure 4. We see that even if only the MLP blocks are ACMized, our method still obtains a better performanceefficiency trade-off than MoEfication.

# Analysis

Dynamic models allocate variable amounts of compute for different samples or input regions. In this section, we provide motivation for our method, examine the distribution of computational load, and show that the introduced auxiliary losses are needed. In the supplementary material, we show the impact of the proposed pre-training stages and investigate the effect of changing the number of learners $N$ .

![](images/997d9a4a82a690b9164c5917f88e28ca001648fad79a28e2bba7a257a16675b9.jpg)  
Figure 5: Color-coded errors of a 4-learner ACM plotted after performing module-wise representation distillation for modules from eight block of a ViT-B pre-trained model. Tokens are sorted along the Y-axis of the plot by their average error. For most input tokens, the same transformation can be performed by a considerably smaller module consisting of only two or three learners, thus justifying the use of ACMs.

# Computational Inefficiency of Static Modules

To justify the use of ACMs, we make the following experiment. First, we perform module-wise distillation from a selected static module of an ImageNet-1k pre-trained ViT-B into 4 learners, just as we describe in the Method section. After convergence, we randomly select 5000 sample images from the validation dataset and forward them through the original model to save every input token $z _ { i , j }$ and output token $o _ { i , j }$ . For every possible number of learners used $n \in \{ 1 , . . . , \bar { N } \}$ , we are interested in how well the learners imitate the output of the original module. For this, as in training, we use MSE: $\left\| h ( z _ { i , j } ^ { l } , n ; \phi ^ { l } ) - o _ { i } ^ { l } \right\| ^ { 2 }$ . The resulting tensor of MSE values h
as size $( 5 0 0 0 , 1 9 7 , 4 )$ , where 197 is the sequence length specific to ViT-B with patch size set to 16. Since ACMs process each token independently, we flatten the first two dimensions and then sort the tokens by the average error for readability. Figure 5 shows the resulting color-coded error visualizations for selected modules. We emphasize that the four learners have approximately the same cost and number of parameters as the original static module being substituted.

The results show that 1) only a small subset of tokens require the full processing power of the entire module, and for a majority of tokens, and 2) tokens usually exhibit varying levels of difficulty, thus warranting the use of conditional computation on a per-token basis.

# Qualitative Analysis

A dynamical model should take advantage of the fact that images exhibit different difficulty levels for classification. By extracting the gating choices being done by every ACM for each patch, we can plot a heatmap that indicates which regions the model was focused the most on. This map should correlate with the meaningful regions of the image. Figure 6 shows that the model indeed learns to allocate its computational budget to those regions. We show that this effect is not exclusive to vision by performing a similar analysis for audio in Figure 7. We provide additional examples in the supplementary material.

![](images/1212bbb5e2f56a767fd749fb7631c2d75ad59d6f69b8e2d7758a66089b35ecb7.jpg)  
Figure 6: Computational load heatmaps for the model trained with $\beta _ { \mathrm { t a r g e t } } = 0 . 6$ . The model allocates its computational budget to semantically meaningful patches.

![](images/e198932896527bb3d0c9a29e64204e19abb3cf92cdeac15878f6ae5807e2b095.jpg)  
Figure 7: For each input audio token (red waveforms), we show the average number of learners that were activated in the ACMized model for $\beta _ { \mathrm { t a r g e t } } = 0 . 2 5$ (blue bars). We can see that this model also learned to allocate its computational budget to important regions of the input.

# Ablation Study

Being able to dynamically allocate computation when solving a task is the core feature of ACMized models. The auxiliary losses are necessary for the dynamism to emerge. We empirically demonstrate this in Figure 8 by training multiple runs differing only in the loss terms applied during the end-to-end training phase. For each run, we visually analyze their intra-sample computational load distribution and generate a histogram of the total computation spent on every sample.

![](images/afacd8883c94b91a4a14beab9f28c21198892df913b3d38d749e50227df7e90e.jpg)  
Figure 8: Effects of training with different combinations of enabled auxiliary losses. The computational load heatmap is rendered over the original image (top row). The histograms (bottom row) show the distribution of the total computational cost spent by the model on images from the validation set. Only combining all the proposed terms provides the desired diversity of the allocated computational budget.   
Figure 9: Latency of a 128-sample batch processed by a single ACM layer on an A100 GPU. The gating network is included in the measurements, but we replace the gating choices with samples from a random distribution to achieve the desired average number of executed learners. ACMs have negligible overhead, and latency scales linearly with the average number of executed learners.

As expected, ACMs may converge to a solution in which always the same learners are used when only $\mathcal { L } _ { \mathrm { b } }$ is applied. The inclusion of $\mathcal { L } _ { \mathrm { e } }$ helps in diversifying between different regions of the input, but overall the computation spent on each image is mostly the same. Applying $\mathcal { L } _ { \mathrm { d } }$ instead of $\mathcal { L } _ { \mathrm { e } }$ diversifies the distribution of compute spent on every image, but the gating choices for different regions of the input are not diverse on its own. Only by enabling all the proposed losses can we achieve the desired effect of the model focusing on semantically meaningful patches and having a high variability of computational budget allocation.

# Hardware speedup

Due to their design, ACMs are inherently well-suited for parallel execution on GPUs. Specifically: (1) once gating decisions are determined, the execution of learners can occur in parallel; (2) tokens can be reordered without any additional copying when they are loaded from or stored to the main GPU memory by the kernel; (3) when tokens are sorted by the number of selected learners, the computation for each group is standard matrix multiplication for which GPUs are highly optimized for; (4) the hidden dimensionality of each learner is deliberately large to maximize GPU utilization. We implement the ACM forward pass with GPU kernels written in Triton (Tillet, Kung, and $\mathrm { C o x } 2 0 1 9 )$ and employ several optimizations including configuration autotuning and kernel fusion.

In Figure 9 we evaluate our implementation by measuring the wall-clock time of execution of a single ACM module from the ViT-B-based model. The overhead in respect to a static MLP is negligible. Moreover, the wall clock time decreases linearly with the number of executed learners.

0.00.20.40.60.81.01.2Wall time (relative to MLP) M MLP ACM (naive) ACM (Triton) Linear scaling 0.0 0.2 0.4 0.6 0.8 1.0 FLOPs (relative to MLP)

# Conclusion

In this work, we have demonstrated that large static modules lead to ineffective allocation of computational budget. The introduced ACM, a generic module that facilitates granular conditional computation, is designed for computational effectiveness, and models based on it achieve state-of-the-art results among the tested dynamic inference methods. Our training procedure distills a static network into an adaptive one, forcing it to allocate its computational budget to semantically meaningful input regions. Future work might explore the relationship between ACMs and pruning methods. Since our training procedure replaces individual modules with ACMs, one could also consider using quantized learners for further efficiency gains.
# TEncDM: Understanding the Properties of the Diffusion Model in the Space of Language Model Encodings

Alexander Shabalin1,5\*, Viacheslav Meshchaninov1\*, Egor Chimbulatov1, Vladislav Lapikov1, Roman $\mathbf { K i m } ^ { 1 }$ , Grigory Bartosh2, Dmitry Molchanov3, Sergey Markov4, Dmitry Vetrov5

1HSE University 2University of Amsterdam 3Independent Researcher 4SberDevices 5Constructor University {amshabalin, vmeshchaninov, echimbulatov, vlapikov, $\mathrm { r k i m } \}$ @hse.ru, dmolch $1 1 1 @$ gmail.com, g.bartosh@uva.nl, sergei.markoff $@$ gmail.com, dvetrov $@$ constructor.university

# Abstract

This paper presents the Text Encoding Diffusion Model (TEncDM), a novel approach to diffusion modeling that operates in the space of pre-trained language model encodings. In contrast to traditionally used embeddings, encodings integrate contextual information. In our approach, we also employ a transformer-based decoder, specifically designed to incorporate context in the token prediction process. We conduct a comprehensive examination of the influence of the encoder, decoder, noise scheduler, and self-conditioning on zero-shot generation. Furthermore, we compare TEncDM with previous approaches on three conditional text generation tasks: QQP, XSum, and Wiki-Auto. The results show that TEncDM exhibits superior performance compared to existing non-autoregressive diffusion models.

Code — https://github.com/M0RJIQUE/tencdm Extended version — https://arxiv.org/abs/2402.19097

# 1 Introduction

Autoregressive (AR) large language models such as GPT4 (OpenAI 2023) or Llama 3 (Dubey et al. 2024) are the current gold standard in the text generation problem. They are capable of creating high-quality and coherent texts that are practically indistinguishable from the human ones. However, the disadvantage of this approach is the inability of the model to correct its own mistakes made during left-to-right generation. If a mistake occurs, it may spoil the subsequent text. In addition, the autoregressive method of token generation slows down the inference process as it requires performing a single model evaluation for each new token.

Diffusion modeling is currently the state-of-the-art approach for data generation in image (Rombach et al. 2022; Podell et al. 2023), audio (Evans et al. 2024) and video (Blattmann et al. 2023) domains. Researchers now attempt to adapt it also for text generation (Li et al. 2022; Gong et al. 2023; Karimi Mahabadi et al. 2024). Diffusion models are a class of probabilistic generative models that are able to iteratively transfer noise to a representative sample of data. While some of the proposed text diffusion models are autoregressive (Lovelace et al. 2023; Zhang et al. 2023), the majority of them are not and, by design, they have several advantages over AR language models. First, being non-autoregressive (NAR) models, they generate all the tokens simultaneously and can adjust any part of the sequence during the generation process. They also can be faster than AR models because the number of neural function evaluations for diffusion models depends on the number of denoising iterations rather than the length of the sequence. And given the possibility of distillation of diffusion models (Meng et al. 2023), the number of iterations can be greatly reduced.

To date, a number of text diffusion models have been proposed, each based on substantially new ideas with little overlap with other methods. Some works replace Gaussian noise with categorical noise (Hoogeboom et al. 2021; Austin et al. 2021), exploiting the discreteness of the text domain. Others train continuous diffusion on token embeddings (Li et al. 2022; Lin et al. 2023; Gong et al. 2023), on text latent representations reduced in size (Lovelace et al. 2023; Zhang et al. 2023) or on the probability simplex (Karimi Mahabadi et al. 2024; Han, Kumar, and Tsvetkov 2023). There are also differences in the way diffusion outputs are decoded back into text. Diffusion models trained on embeddings round their predictions to the nearest embeddings, while those that utilize small latent spaces decode the predictions with an AR model. This suggests that the scientific community has not yet found the most robust design of a diffusion model for text.

In this paper, we attempt to better understand the specifics of continuous text diffusion models trained in the latent space of token embeddings and identify best practices for their development. We argue that the use of raw embeddings as latents is suboptimal, and that this approach can be enhanced by first extracting context information from embeddings with a language model encoder and training a diffusion model in this latent space. We also investigate several diffusion components in detail: text decoding methods, diffusion model architecture, noise schedule, and selfconditioning (Chen, Zhang, and Hinton 2023). As a result, we combine all our findings in a method called Text Encoding Diffusion Model (TEncDM).

We compare our approach with other non-autoregressive diffusion models trained in the space of embeddings or encodings on three conditional text generation problems – paraphrasing, summarization, and text simplification – and show its superiority over other methods. The main contributions of this work are as follows:

• We propose a new text diffusion framework TEncDM, which trains the diffusion model in the latent space constructed by the outputs of pre-trained Transformer-based encoder.   
• We evaluate the importance of the text decoder and conclude that its robustness to inaccuracies in the generated latents directly affects the generation quality. We propose a Transformer-based decoder and its training method that boosts the model performance.   
• We analyse in detail the effect of self-conditioning and noise schedules on the denoising process and show their effect on the model quality.

# 2 Problem Statement and Background

Text generation problem. In the field of natural language processing, unconditional text generation is a task of sampling $y$ from the unknown distribution $p ( y )$ , where $y \ =$ $[ y _ { 1 } , \ldots , y _ { n } ]$ is a sequence of tokens with variable length $n$ . In conditional text generation the distribution of texts changes to $p ( y | x )$ , where $x$ is a condition variable. The goal is to generate a text, that satisfies this condition.

Gaussian diffusion models. Gaussian diffusion models (Song et al. 2021) learn to sample data from an unknown distribution by gradually denoising random Gaussian noise. The training procedure is defined through a forward diffusion process that satisfies $q ( z _ { t } | z _ { 0 } ) = \mathcal { N } ( \sqrt { \alpha _ { t } } z _ { 0 } , ( 1 - \alpha _ { t } ) \mathbf { I } ) ,$ where $\alpha _ { t } \in [ 0 , 1 ]$ is a predefined noise schedule, $t \in [ 0 , 1 ]$ and $\alpha _ { t } > \alpha _ { t + \Delta t }$ . The denoising network (parameterized by $\theta$ ) is trained to reconstruct the original latent $z _ { 0 }$ given the noisy latent $z _ { t }$ , as expressed in equation 1.

$$
\mathcal { L } ( \theta ) = \underset { \varepsilon \sim \mathcal { N } ( 0 , \mathbf { I } ) , t \sim U [ 0 ; 1 ] } { \mathbb { E } } [ \Vert z _ { 0 } - \hat { z } _ { \theta } ( z _ { t } , t ) \Vert ^ { 2 } ]
$$

Sampling procedure starts from a pure Gaussian noise $z _ { T } \sim$ $\mathcal { N } ( 0 , { \bf { I } } )$ and utilizes the denoising network to iteratively generate latents $z _ { t _ { T - 1 } } , . . . , z _ { t _ { 1 } }$ , where $1 = t _ { T } > . . . > t _ { 1 } = 0$ .

Diffusion models for text generation. The primary feature of the text domain is the discreteness of its samples. In order to train a diffusion model on them, they must first be translated into continuous space. Consequently, alongside the denoising model, the diffusion framework incorporates an encoder that maps tokens into the continuous latents and a decoder that performs the reverse operation, converting the generated latents into text.

# 3 Related Work

Embedding-based diffusion models. The majority of proposed text diffusion models use embeddings of tokens to construct the continuous latent space (Li et al. 2022; Lin et al. 2023; Strudel et al. 2022; Gong et al. 2023; Wu et al. 2023). At the inference stage, to convert the latent predictions into text, they map each latent vector to a token corresponding to the nearest embedding.

Self-Conditioning. Self-conditioning is a technique that significantly increases the performance of the text diffusion model (Chen, Zhang, and Hinton 2023; Strudel et al. 2022; Lovelace et al. 2023). Usually the model is conditioned only on the latent variable $z _ { t }$ and the current timestep $t$ as $\hat { z } _ { 0 } ^ { t } = \hat { z } _ { \theta } ( z _ { t } , t )$ . Self-conditioning proposes to also condition the model on the estimation of data sample from the previous timestep during generation in order to improve the prediction at the current timestep, $\hat { z } _ { 0 } ^ { t } = \hat { z } _ { \theta } ( z _ { t } , t , \hat { z } _ { 0 } ^ { t - 1 } )$ .

Although widely used, no analysis has been conducted to determine why this method is effective or how the generation process is altered by its application.

Noise scheduler. Noise scheduler is a key component of a diffusion model that controls the amount of noise added on each timestep. Previous research (Li et al. 2022; Gao et al. 2024; Ye et al. 2024) has highlighted that the standard noise schedulers used for image diffusion models are unsuitable for the textual domain. Due to the discrete nature of the texts, it is unlikely that an addition of a small amount of noise to a latent will change its nearest text in the latent space. Therefore, to increase the difficulty of the denoising task for the model, the mentioned works recommend adding more noise on iterations that are close to 0.

# 4 Understanding Text Diffusion

In this section, we present our findings on the components of the diffusion model, discuss their weaknesses and propose ways to enhance them.

Encodings are better than embeddings. Most diffusion models utilize token embeddings to map text into a continuous latent space. However, this approach is not optimal because the embeddings do not convey contextual information. This requires the diffusion model to independently search for it to retrieve ambiguous tokens. To simplify the task, instead of embeddings, we can use the final layer outputs of a pre-trained language model (e.g. BERT). They contain contextual information and, thus, should be more suitable for training the diffusion model. We refer to these outputs as encodings.

Experimental results confirming our intuition are presented in Section 7.2. It is worth noting that the use of encodings does not slow down the generation process, as we need to compute them only during the training. To improve the quality even further, it is possible to fine-tune the encoder, but we choose not to in order to avoid overcomplicating the approach. Investigation into fine-tuning is left for the future work.

Decoder is important. The purpose of the decoder in the diffusion model is to map the generated latents into text. Approaches that train diffusion in the space of token embeddings decode latents by rounding them to the nearest embeddings and selecting a corresponding token. However, the diffusion model may produce inaccurate latent samples due to accumulation of errors during the denoising process. Such inaccuracy might significantly spoil the text quality, so it would be wise to train a decoder that could improve it.

In the Section 7.2, we compare different decoder designs and conclude that an advanced context-dependent decoder indeed improves the generation quality.

Self-conditioning affects denoising dynamics. Selfconditioning improves sampling quality by conditioning the model on its previous prediction. However, the mechanics of self-conditioning are not fully understood yet. Our research demonstrates that the addition of self-conditioning increases the model’s prediction confidence at each denoising timestep, resulting in a reduction in the required number of generation steps. Furthermore, the sample quality diminishes as the number of steps increases. We believe that a reason for this behaviour lies in a mismatch between the latents used at the training stage and those at the generation stage. We provide the evidence supporting our conclusions in Section 7.2, along with a comprehensive analysis of the model’s behaviour with and without self-conditioning.

Diffusion needs even more noise. Following the recommendations of previous works (Li et al. 2022; Wu et al. 2023; Ye et al. 2024), we used sqrt noise scheduler that increases the amount of noise added to the diffusion model inputs during training beyond the amount of typically used cosine noise scheduler (Han, Kumar, and Tsvetkov 2023; Lovelace et al. 2023; Strudel et al. 2022; Zhang et al. 2023). However, our experiments led us to conclusion that encoding-based diffusion model requires even more noise for successful training. We hypothesize that this is due to the presence of contextual information in the encodings, which simplifies the denoising task.

In Section 7.2 of this study, we demonstrate that both commonly used cosine and sqrt noise schedules do not introduce a significant level of noise to the latent variables over a wide range of timesteps. As a result, the denoising task becomes too simple for the model, leading to a reduction in the effectiveness of the training signal.

# 5 Methodology

The design of TEncDM is depicted on Figure 1. It consists of three parts – diffusion encoder $E _ { d i f f }$ , diffusion model $\hat { z } _ { \theta }$ and decoder $D$ . For the conditional generation, we also add condition encoder $E _ { c o n d }$ , which encodes an input text. Its output is provided to the diffusion model and decoder through cross-attention.

This section exclusively focuses on the topic of unconditional text generation. The details of the conditional model can be found in Section 5.4.

# 5.1 Diffusion encoder, $E _ { d i f f }$

We use pre-trained Transformer-based (Vaswani et al. 2017b) language model $E _ { d i f f }$ , which we call diffusion encoder, to encode text $y$ into the latent variable $z$ . Encoding of text does not change the length of the sequence. In order to align all texts in length, we add paddings to the end of short texts. After encoding the text, the encodings of all special tokens are replaced by their corresponding embeddings and padding encodings are replaced with zeros. This is necessary because diffusion model does not use an attention mask during training, which means that the reconstruction loss is calculated for both text and special tokens. However, special token encodings usually contain meaningless for diffusion model values. Therefore, minimization of reconstruction loss for these encodings only harms the training process. Embeddings of special tokens, on the other hand, only contain information about the token itself and the diffusion model recovers them much easier. During training we do not update the weights of the encoder in order to keep the approach simple.

![](images/5352911829110f4aa0bcd15c538b1032c9b515d2bffe39389f8ce4fdef549109.jpg)  
Figure 1: Overview of our framework design for conditional generation. Top is the training process, bottom is the generation process.

# 5.2 Decoder, $D$

The decoder $D$ is required to convert latent variables generated by diffusion model into textual output. Although a basic linear decoder can effectively reconstruct tokens with high accuracy, we employ the BERT-type (Devlin et al. 2019) architecture for the decoder to provide it with the ability to capture context information and rectify potential mistakes originating from the diffusion model. Note that we do not use an AR decoder on purpose so as not to transfer the limitations of AR language models to the diffusion model.

We train the decoder independently of the diffusion model using the following objective

$$
- \mathbb { E } \log p _ { D } ( y \mid C o r ( z _ { 0 } ) )  \operatorname* { m i n } _ { D } ,
$$

where $C o r ( z _ { 0 } )$ is a corrupted latent variable extracted from the diffusion encoder. Corruption is needed to expand the decoder training data domain and make it robust to distribution mismatch between text encodings $z _ { 0 }$ and latents $\hat { z } _ { 0 }$ generated by the diffusion model. This mismatch might arise due to the accumulation of errors during the denoising process. Its presence is especially evident for special tokens, which always have the same fixed representations in $z _ { 0 }$ . By default, we take $C o r ( z _ { 0 } )$ to be $z _ { t }$ with randomly sampled $t \in [ 0 , 0 . 1 5 ]$ . We use the diffusion’s noise scheduler to calculate zt.

# 5.3 Diffusion model, $\hat { z } _ { \theta }$

The diffusion model consists of 12 BERT layers and it is trained to predict the original latent $z _ { 0 }$ given its noisy version $z _ { t }$ and a timestep $t$ by minimizing the objective (1). We provide the model with timestep value by adding timestep embedding to the hidden state vectors on each layer.

We train the diffusion model using the variance preserving scheme, discussed in (Song et al. 2021). To achieve zero mean and unit variance we normalize the latent variables $z _ { 0 }$ coordinate-wise, using the statistics from the training set.

Noise scheduler We adopt the noise scheduler from (Hoogeboom, Heek, and Salimans 2023) and use the following equation for $\alpha _ { t }$ :

$$
\alpha _ { t } = { \frac { 1 } { 1 + \tan ( t \pi / 2 ) ^ { 2 } \cdot d ^ { 2 } } } ,
$$

where $d$ is a hyperparameter controlling the rate at which noise is introduced into the system. We set $d = 9$ by default, which corresponds to a significantly higher noise addition rate than what is used in all common noise schedulers. We further refer to our scheduler as tan- $d$ noise scheduler.

Self-condition Following the previous approaches (Lovelace et al. 2023; Strudel et al. 2022) we incorporate self-conditioning into the diffusion model. In order to make the model utilize the data sample estimation from the previous generation step, we modify the training procedure.

According to (Chen, Zhang, and Hinton 2023) we design the training process to emulate the inference behavior. On each training iteration with the probability $p = 0 . 5$ the prediction is computed with the self-conditioning set to zero $\hat { z } _ { 0 } ^ { t } = z _ { \theta } \big ( z _ { t } , t , 0 \big ) .$ . And, with probability $( 1 - p ) = 0 . 5$ we first calculate $\bar { z } _ { 0 } ^ { t } ~ = ~ z _ { \theta } ( z _ { t } , t , 0 )$ and then use it as an estimation of the data sample to obtain a second prediction $\tilde { z } _ { 0 } ^ { t } = z _ { \theta } ( z _ { t } , t , \mathbf { S G } ( \bar { z } _ { 0 } ^ { t } ) )$ , where SG is the stop-gradient function that does not allow the gradient to flow through $\bar { z } _ { 0 } ^ { t }$ . The diffusion model is optimized using the output $\bar { z } _ { 0 } ^ { t }$ in the former scenario and $\tilde { z } _ { 0 } ^ { t }$ in the latter. This training strategy allows the model to accurately approximate $z _ { 0 }$ both with and without self-conditioning. We implement self-conditioning in a same manner as conditioning on timestep. For each diffusion model layer we pass the data estimation through a single linear layer and add it to the hidden state vectors.

# 5.4 Generation process

The generation process is illustrated on the Figure 1 (bottom). To generate text in the inference phase, we start with a random Gaussian sample and denoise it in $T$ steps using the Euler solver. At each step, we apply self-conditioning and, because of it, use a small number of steps – 50 by default.

For the conditional generation we keep the framework design similar to unconditional generation. The only difference is that we add condition encoder to process the input text and provide both diffusion model and decoder with its output via cross-attention. We also add classifier-free guidance with coefficient 0.5, because it slightly improves quality. Implementation details can be found in Appendix E.

# 6 Datasets

To evaluate the performance of our diffusion models we use five datasets in English language. Two of them are unconditional: ROCStories and Wikipedia, and three are conditional: QQP, XSum and Wiki-Auto. The ROCStories (Mostafazadeh et al. 2016) dataset contains 98k fivesentence commonsense fictional stories, that capture causal and temporal relations between daily events. The Wikipedia dataset is obtained from the ROOTS corpus (Lauren¸con et al. 2023), it is a collection of over 2 million cleaned articles from the Wikipedia platform. The subset of QQP (Chen et al. 2017) dataset, proposed in (Gong et al. 2023), consists of $1 4 4 \mathrm { k \Omega }$ question pairs from the Quora platform that are paraphrases of each other. The XSum (Narayan, Cohen, and Lapata 2018) dataset is used for summarization problem and it contains 204k BBC articles, which are provided as document and summary pairs. The Wiki-Auto (Jiang et al. 2020) dataset consists of aligned sentences from complex and simplified Wikipedia1. The detailed statistics for each dataset can be found in Appendix F.

# 7 Empirical Analysis

In this section, we evaluate the components of our framework on the ROCStories and Wikipedia datasets. To simplify the setup, we only consider unconditional generation. In Section 8, we demonstrate that our findings can be successfully transferred to the conditional generation problems. In this section, we do not compare our method with others. The comparison with the GPT-2 (Radford et al. 2019) on unconditional generation is presented in Appendix J.

# 7.1 Evaluation Metrics

We follow the model evaluation scheme from the (Lovelace et al. 2023). To evaluate the quality of our model we use Perplexity (ppl), calculated with GPT-2 Large (Radford et al. 2019). To measure the diversity of the generated text we utilize the diversity metric proposed in (Su et al. 2022). We calculate it as div(y) = 4n=2 # of n-grams in y |# of unique n-grams in y| , where $y$ is a set of generated te ts. To ensure that the model does not reproduce the training dataset during the generation we evaluate the Memorization (mem). We calculate it as the proportion of generated 4-grams that are found in the training set. As Perplexity tends to be small for the texts with repetitions, we also measure MAUVE Score (Pillutla et al. 2021) to estimate the quality of text. MAUVE is a language model-based metric that measures the distance between the distributions of generated and reference texts using divergence frontiers. We leave all MAUVE hyperparameters at the default values presented in the original paper.

Table 1: Quality of diffusion models trained with different diffusion encoders.   

<html><body><table><tr><td>Encoder</td><td>ppl↓</td><td>mem↓</td><td>div个</td><td>mauve 个</td></tr><tr><td colspan="5">ROCStories</td></tr><tr><td>BERTemb BERT</td><td>48.9.36 29.1.89</td><td>.371.003 .453.003</td><td>.324.002 .295.002</td><td>.600.016 .762.043</td></tr><tr><td>RoBERTa T5</td><td>31.3.54 28.3.33</td><td>.443.003 .427.003</td><td>.302.002 .312.004</td><td>.647.019 .706.024</td></tr><tr><td>BART</td><td>34.1.52</td><td>.441.006</td><td>.299.005</td><td>.705.030</td></tr><tr><td>Source text</td><td>21.7</td><td>.365</td><td>.403</td><td>.876</td></tr><tr><td></td><td></td><td>Wikipedia</td><td></td><td></td></tr><tr><td>BERTemb</td><td>156.11.8</td><td>.263.004</td><td>.517.002</td><td>.378.055</td></tr><tr><td>BERT</td><td>104.42.1</td><td>.286.002</td><td>.504.003</td><td>.874.011</td></tr><tr><td>Source text</td><td>37.3</td><td>.122</td><td>.615</td><td>.957</td></tr></table></body></html>

To calculate all the metrics, we generate 1000 texts. For MAUVE, we sample 1000 reference texts from the test set. We repeat this procedure 5 times and report the mean and standard deviation of the results in the mean $\mathrm { \ s t d }$ notation.

# 7.2 Model setup

Unless otherwise stated, the training of TEncDM is performed within the latent space of BERT encodings. A threelayer transformer is employed for the decoder, which is trained to reconstruct $z _ { 0 }$ from $z _ { t }$ , where $t ~ \in ~ U [ 0 , 0 . 1 5 ]$ . A comprehensive analysis of various decoder modifications is presented in this section and Appendix B. The diffusion model is a 12-layer transformer with a dimensionality of 768. We train it with tan-9 noise scheduler.

Effect of Diffusion Encoder We compare latent spaces of BERT (Devlin et al. 2019), RoBERTa (Liu et al. 2019), BART (Lewis et al. 2020) and T5 (Raffel et al. 2020) encoders, as well as BERT embeddings, to ascertain the optimal choice for the diffusion model. All encoders have approximately the same size of 100M parameters. In this experiment, we train diffusion models with the same set of hyperparameters across all diffusion encoders. We train the decoders according to the scheme described in Section 7.2. The results of this comparison are presented in Table 1 and they show a clear advantage of the latent space derived from BERT encodings on ROCStories dataset. Furthermore, the quality of all encoders is superior to that of BERT embeddings. A better div and mem for embeddings can be explained by the presence of words in the corpus that do not align with the context. The text samples are presented in Appendix K. This confirms our hypothesis that encodings are better suited for the training of a diffusion model. We discuss the drop in mauve for RoBERTa in Appendix G.

Effect of Decoder To confirm the hypothesis about the importance of the decoder architecture and its training scheme, we compare an MLP decoder consisting of two linear layers with a 3-layer transformer. The latter is able to extract contextual information for each token. We corrupt the decoder input $z _ { 0 }$ by transforming it into $z _ { t }$ , using the diffusion forward process with $t \in U [ 0 , 0 . 1 5 ]$ . We choose this method, because it brings the decoder input closer to the diffusion output. A more detailed analysis of corruption techniques is presented in the Appendix B. To keep the experiment fair, we apply all decoders to the same generated latents. The results of the experiment are shown in Table 2. The MLP decoder without corruption achieves the lowest text quality in terms of perplexity, but comparable by mauve with Transformer without corruption. However, it is challenging to make meaningful comparisons between decoders by mauve due to the significant variance. From this experiment, we can conclude that corruption of the latent helps to improve the quality for both datasets. At the same time, the incorporation of contextual information into the decoder lead to the best result.

Table 2: Comparison of decoders for encoding-based diffusion model.   

<html><body><table><tr><td>Decoder</td><td>ppl↓</td><td>mem↓</td><td>div个</td><td>mauve 个</td></tr><tr><td colspan="5">ROCStories</td></tr><tr><td>MLP + Cor(zo) Transformer</td><td>39.73.38 31.2.33 34.2.29</td><td>.444.002 .448.002 .445.001</td><td>.297.004 .293.003 .295.003</td><td>.716.074 .739.051 .714.037 .762.043</td></tr><tr><td colspan="5">+ Cor(zo) 29.1.89 .295.002 .453.003</td></tr><tr><td>Transformer + Cor(zo)</td><td>180.63.2 104.42.1</td><td>Wikipedia .261.001 .286.002</td><td>.511.001 .504.003</td><td>.526.025 .874.011</td></tr></table></body></html>

Effect of self-conditioning We conduct a series of experiments to understand how self-conditioning (SC) affects the denoising process. In Figure 2, we compare the quality of the models with and without SC for different number of denoising steps on the ROCStories dataset. The results show that while the quality of the model without SC increases as the number of steps increases, the quality of the model with SC reaches a maximum at a value of 50 steps in terms of mauve, and then it starts to drop. Nevertheless, at the highest point the model with SC surpasses the model without it according to both mauve and perplexity.

We explain this drop in generation quality with mismatch between diffusion model inputs at train and inference stages. To confirm our hypothesis, we calculated the mean-squared norm (magnitude) of the values of each latent $\hat { z } _ { 0 } ^ { t }$ in a minibatch predicted by the diffusion model during generation (i.e. $\frac { 1 } { N \cdot d \cdot m } \| \hat { z } _ { 0 } ^ { t } \| _ { 2 } ^ { 2 }$ , where $N$ is a batch size, $d$ is a dimension and $m$ is a sequence length). We plot this magnitude with respect to timestep for generations with different number of steps as well as for the predictions $\bar { z } _ { 0 } ^ { t }$ from the training stage. The results for the ROCStories dataset are presented in Figure 3. They indicate that self-conditioning significantly increases the prediction magnitude as the number of steps increases. This can be explained by the following: during training, the model learns to use self-conditioning to approximate $z _ { 0 }$ more accurately. Consequently, self-conditioning increases the model’s confidence, which is directly related to prediction magnitude. During the generation process, the model takes its own prediction, which has an increased magnitude, as an input at each step and increases it further. Therefore, the increase in magnitude depends directly on the number of generation steps. Eventually, this leads to a mismatch between the predictions fed into the model during training and generation.

![](images/b8d1063d282fa77968bd66c49826456dd61d6934f5f1694a5e47a8b136035049.jpg)  
Figure 2: Generation quality of diffusion models with and without self-conditioning on ROCStories dataset.

![](images/c5b6a338ddd54b0efaf7aa7273e83103caa7450dceef9a5cbb2d8de3572ad3e3.jpg)  
Figure 3: Prediction magnitudes for generation processes with different amount of steps on ROCStories dataset.

In the Appendix C, we provide a more detailed discussion of this phenomenon and show that the same behavior is observed in the larger Wikipedia dataset. It is worth noting that the smallest mismatch is observed for the trajectory of 50 generation steps, which corresponds to the best quality.

Effect of Noise scheduler We compare our noise scheduler tan- $d$ with previously used cosine and sqrt (visualized in Appendix D) and present the quantitative results in Table 3. We use the same decoder and optimal amount of generation steps for each scheduler. In Figure 4, we evaluate the difficulty of recovering a data sample from noised latent $z _ { t }$ for diffusion model trained with different noise schedulers. We measure the reconstruction loss $\frac { 1 } { N \cdot d \cdot m } \| z _ { 0 } - \bar { z } _ { 0 } ^ { t } \| _ { 2 } ^ { 2 }$ and accuracy of token prediction for every timestep.

While the sqrt noise scheduler adds significantly larger amount of noise in the initial timesteps than cosine one, the rate of noise addition decreases for the subsequent timesteps. For both schedulers, the denoising task becomes insufficiently hard for the most timesteps, which leads to a decrease in their contribution to the generation process. This can be seen from the reconstruction accuracy. In contrast, tan- $d$ noise scheduler adds more noise consistently across all timesteps, leading to a more challenging training task and improved generation performance.

![](images/dac88754b4cd2cde367edc475edc7b4cfc12865ef88094b2e804a6ef3e716fbb.jpg)  
Figure 4: Reconstruction loss and reconstruction accuracy of diffusion models trained with different noise schedulers on ROCStories dataset.

Table 3: Quality of diffusion models trained with different noise schedulers.   

<html><body><table><tr><td>Noise Scheduler</td><td>ppl↓</td><td>mem↓</td><td>div个</td><td>mauve 个</td></tr><tr><td colspan="5">ROCStories</td></tr><tr><td>cosine</td><td>393.2127.6</td><td>.262.004</td><td>.474.006</td><td>.098.011</td></tr><tr><td>sqrt tan-7</td><td>127.229.3 27.1.31</td><td>.264.004 .455.004</td><td>.434.004 .286.001</td><td>.364.041 .730.026</td></tr><tr><td>tan-9</td><td>29.1.89</td><td>.453.003</td><td>.295.002</td><td>.762.043</td></tr><tr><td>tan-11</td><td>33.42.4</td><td>.464.006</td><td>.279.004</td><td>.730.038</td></tr><tr><td></td><td></td><td>Wikipedia</td><td></td><td></td></tr><tr><td>sqrt tan-9</td><td>364.06.5 104.42.1</td><td>.139.001 .286.002</td><td>.664.004 .504.003</td><td>.325.037 .874.011</td></tr></table></body></html>

Based on these observations, we conclude that in order to improve the efficiently of the denoising process, it is essential to increase the amount of added noise within all timesteps. However, it is important to strike a balance as adding excessive noise can negatively impact performance. In our experiments, tan-9 yielded results that were marginally superior to all other schedulers in terms of mauve, with slightly better memorization and diversity, while lagging behind tan-7 in terms of perplexity.

As a rule of thumb, the noise schedule should be such that the diffusion model recovers approximately the same amount of information at each timestep. Otherwise, some of them will not contribute to the denoising process enough.

# 8 Seq2Seq Experiments

We are conducting experiments to evaluate the effectiveness of the proposed components for text diffusion generation on language model encodings on three different tasks: paraphrasing (QQP), summarization (XSum), and text simplification (Wiki-Auto).

Table 4: Seq2Seq evaluation results of Diffusion and AR methods on QQP, XSum and Wiki-Auto datasets. We calculate ROUGE-1/2/L (R-1/2/L), BERTScore (BS) and BLEU-4 (B-4). All results taken from other papers are marked with $\star$ . DiffuSeq and SeqDiffuSeq results were taken from their respective publications. Results for AR models were taken from (Gong et al. 2023; Wu et al. 2023; Lovelace et al. 2023). Additionally, we trained AR-Diffusion and SeqDiffuSeq on previously unreported datasets, using the code from the corresponding papers (marked as $\dagger .$ ).   

<html><body><table><tr><td rowspan="2">Method</td><td colspan="3">QQP</td><td rowspan="2">XSum</td><td colspan="4">Wiki-Auto</td></tr><tr><td>R-L ↑</td><td>BS↑</td><td>B-4个</td><td>R-1/2/L ↑</td><td>BS ↑ R-L 个</td><td>BS↑</td><td>B-4个</td></tr><tr><td>DiffuSeq*</td><td>52.7</td><td>82.4</td><td></td><td>18.9 / 1.3/13.6</td><td>46.8</td><td></td><td>79.1</td><td>26.1</td></tr><tr><td>SeqDiffuSeq*+</td><td></td><td>82.9</td><td>23.3</td><td>14.1 /1.1/11.4</td><td>58.4</td><td></td><td>82.1</td><td>37.1</td></tr><tr><td>GENIE*</td><td></td><td></td><td></td><td>29.3 / 8.3 /21.9</td><td></td><td></td><td></td><td></td></tr><tr><td>AR-Diffusion†</td><td>54.9</td><td>81.4</td><td>31.2</td><td>27.1/6.4/20.8</td><td>59.7</td><td>54.9</td><td>81.3</td><td>32.7</td></tr><tr><td>TEncDM (BERT)</td><td>56.4</td><td>82.4</td><td>30.2</td><td>31.5 /10.0 / 24.9</td><td>68.2</td><td>58.1</td><td>80.5</td><td>41.6</td></tr><tr><td>TEncDM(T5)</td><td>57.3</td><td>83.8</td><td>30.7</td><td>33.4/11.4/26.8</td><td>70.1</td><td>57.7</td><td>81.2</td><td>41.6</td></tr><tr><td>TEncDM (RoBERTa)</td><td>55.8</td><td>82.4</td><td>30.0</td><td>33.7/11.9 /27.1</td><td>69.8</td><td>57.9</td><td>81.0</td><td>40.5</td></tr><tr><td>GPT2-small FT*</td><td>52.1</td><td>82.5</td><td>19.8</td><td></td><td></td><td>54.6</td><td>80.2</td><td>30.8</td></tr><tr><td>Transformer-base*</td><td>57.5</td><td>83.8</td><td>27.2</td><td>30.5/10.4/24.2</td><td></td><td>49.1</td><td>73.8</td><td>26.9</td></tr><tr><td>FLAN-T5-base*</td><td>52.3</td><td>83.2</td><td></td><td>34.6 /12.9 /27.2</td><td>72.7</td><td></td><td></td><td></td></tr></table></body></html>

Baselines We include two groups of baselines in comparison. The first group comprises of popular diffusion methods: DiffuSeq (Gong et al. 2023), SeqDiffuSeq (Yuan et al. 2024), GENIE (Lin et al. 2023), AR-diffusion (Wu et al. 2023). We focus only on non-autoregressive diffusion models trained in the latent space of embeddings or encodings. Besides, we compare TEncDM to classical AR baselines: Transformer (Vaswani et al. 2017a), FLAN-T5-base (Chung et al. 2024) and finetuned GPT-2-small (Radford et al. 2019).

Metrics For evaluation of paraphrasing and simplification tasks, we adopt the setting of SeqDiffuSeq (Yuan et al. 2024) and calculate ROUGE-L (Lin 2004), BERTScore (Zhang et al. 2020) and BLEU-4. In addition, we follow the approach of (Wu et al. 2023) and report ROUGE-1/2/L for summarization task.

Results Table 4 presents a comprehensive comparison of our approach against existing methods across three datasets. Results for DiffuSeq, and SeqDiffuSeq were sourced from their respective papers (Wu et al. 2023; Yuan et al. 2024). Results for GENIE were taken from the (Wu et al. 2023), as the original paper used ground truth labels to select the best generated text, which introduces unfairness. Additionally, we trained both AR-Diffusion and SeqDiffuSeq on previously unreported datasets, using the code from the corresponding papers. Results for AR models were taken from (Gong et al. 2023; Wu et al. 2023; Lovelace et al. 2023).

We experiment with three encoders: BERT-base, T5-base, RoBERTa-base to investigate their efficacy on conditional tasks. We use the same encoder for $E _ { d i f f }$ and $E _ { c o n d }$ . Our findings indicate that all three encoders demonstrate effectiveness across the tasks, achieving comparable performance levels. However, no single encoder outperforms the others across all tasks. T5-base excels in question paraphrasing (QQP), while RoBERTa-base demonstrates superior performance on summarization (XSum) and on text simplification (Wiki-Auto) all encoders exhibit similar performance.

A comparison with other methods clearly shows that TEncDM, using any of the tested encoder models, outperforms popular diffusion embedding-based approaches. Furthermore, TEncDM with an optimal encoder achieves comparable performance to three AR baselines on QQP and XSum and outperforms them on Wiki-Auto.

# 9 Conclusion

In this work, we explore key details of the diffusion pipeline for text generation. We propose TEncDM which trains the diffusion model in the latent space of the language model encoding. Unlike embeddings, they contain contextual information which helps diffusion model to recover latents. To improve text generation performance, we analyse the effect of self-conditioning and conclude that it increases the magnitude of the model’s predictions, which allows to reduce the number of generation steps. In addition, we propose an efficient decoder and noise scheduler that boost the generation quality. Extensive ablation on ROCStories and Wikipedia datasets demonstrates the impact of the proposed design choices. Finally, TEncDM outperforms recent diffusion models and some classical autoregressive methods in downstream task experiments.

# 10 Limitations

There are three limitations that warrant further investigation. First, the quality of the model can be improved by training diffusion encoder, decoder and denoising model simultaneously. However, we avoid doing so in order to avoid overcomplicating the approach. Second, the samples from the latent space have a high dimensionality that depends on the sequence length, so the training of our method slows down significantly as the length increases. This problem can probably be overcome by training the autoencoder to map a text into a fixed-dimensional latent space, which is a great direction for further research. Third, as different diffusion encoders works better for different tasks, it is necessary to find the best one for each task.
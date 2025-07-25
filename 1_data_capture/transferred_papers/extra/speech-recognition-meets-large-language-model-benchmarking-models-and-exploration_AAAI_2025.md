# Speech Recognition Meets Large Language Model: Benchmarking, Models, and Exploration

Ziyang $\mathbf { M } \mathbf { a } ^ { 1 }$ , Guanrou Yang1, Yifan Yang1, Zhifu Gao2, Jiaming Wang2, Zhihao $ { \mathbf { D } }  { \mathbf { u } } ^ { 2 }$ , Fan $\mathbf { Y } \mathbf { u } ^ { 2 }$ , Qian Chen2, Siqi Zheng2, Shiliang Zhang2, Xie Chen1\* 1MoE Key Lab of Artificial Intelligence, X-LANCE Lab, Shanghai Jiao Tong University 2Alibaba Group {zym.22, chenxie95}@sjtu.edu.cn

# Abstract

In this paper, we focus on prompting one of the most important tasks in the field of speech processing, i.e., automatic speech recognition (ASR), with speech foundation encoders and large language models (LLM). Despite the growing body of research in this area, we find that many crucial design decisions in LLM-based ASR systems are often inadequately justified. This lack of clarity impedes the field’s progress, making it challenging to pinpoint which design choices truly improve model performance. To address these challenges, we conduct a comprehensive series of experiments that explore various aspects, leading to the optimal LLM-based ASR system. We found that delicate designs are not necessary, while a clean setup with little task-specific design is competent. The models achieve strong performance on the Librispeech and Gigaspeech datasets, compared to both LLM-based models and non-LLM-based models. Finally, we explore the capability emergence of LLM-based ASR in the process of modal alignment. We hope that our study can facilitate the research on extending LLM with cross-modality capacity and shed light on the LLM-based ASR community.

Codes & Checkpoints — https://github.com/X-LANCE/SLAM-LLM

# 1 Introduction

Automatic speech recognition (ASR) stands as a cornerstone in the realm of intelligent speech technology, enabling machines to understand and transcribe human speech. The significance of ASR in enhancing human-computer interaction and accessibility makes it a crucial area of research and applications in the field of speech processing.

The evolution of ASR technology has been marked by the adoption of various paradigms, each representing a leap forward in terms of accuracy, efficiency, and applicability (Li 2022). Among these, supervised methods including connectionist temporal classification (CTC) (Graves et al. 2006), attention-based encoder-decoder (AED) (Chan et al. 2016), recurrent neural network transducer (RNN-T) (Graves, Mohamed, and Hinton 2013) and their variants have been pivotal. In addition, employing self-supervised methods for pretraining followed by supervised methods for fine-tuning has

also proven to be effective (Baevski et al. 2020; Hsu et al.   
2021; Chen et al. 2022; Ma et al. 2023; Yang et al. 2024).

The evolution of the ASR paradigm from previous NNbased ASR models to LLM-based ASR models, stresses differences across loss and criterion design, text prior knowledge, and model scale. The architecture of LLM-based ASR can be conceptualized as consisting of three primary components: a speech encoder, a projector, and an LLM. Recent works in LLM-based ASR often venture into diverse designs, such as compressing the output temporally from the speech encoder (Wu et al. 2023; Fathullah et al. 2024), tackling modal alignment with the projector (Tang et al. 2024; Yu et al. 2024), and fine-tuning the LLM partly or fully (Wu et al. 2023; Li et al. 2023b; Tang et al. 2024; Wang et al. 2023). This paradigm harnesses pre-existing linguistic knowledge, enabling a more holistic understanding of language, which in turn, translates to significant improvements in the speech recognition task.

Despite advancements in the field, existing papers show a variety of design choices that are frequently either inadequately justified through experiments or only briefly addressed. This situation makes it challenging to discern which decisions genuinely contribute to model performance, thereby hindering meaningful and grounded progress in the field. Thus, we pose the question: What matters when building LLM-based ASR models?

In this work, we aim to provide experimental clarity on these key design decisions, identify optimal model configurations, and explore interesting phenomena. We first benchmark the performance of LLM-based ASR with different combinations of well-known speech encoders and the latest released large language models and get a bunch of empirical conclusions. For example, LLMs with supervised finetuning (SFT, a.k.a. chat model) perform better than raw pre-trained LLMs for the ASR task. Building upon these insights, we get optimal model configurations and achieve exciting performance on the Librispeech (Panayotov et al. 2015) and Gigaspeech (Chen et al. 2021) corpus without delicate designs, compared with the previous best-performing NN-based ASR models and other LLM-based ASR models. Further, our work embarks on an in-depth exploration of the ability of LLM-based ASR models. Interestingly, we observe the capability emergence phenomenon during LLMbased ASR training. The benchmark, models, and exploration show how we harvest the result step by step with a clean setup and little task-specific design. Our streamlined design provides a new baseline for LLM-based ASR.

2 Speech Recognition Meets Large Language Model   
Table 1: ASR Paradigm with representative models. QF means variants of Q-Former (Li et al. 2023a). Both QF and MLP are projector modules used to align the speech encoder and the LLM.   

<html><body><table><tr><td>Model</td><td>Loss</td><td>Learnable</td></tr><tr><td colspan="3">PreviousNN-basedASR</td></tr><tr><td>Quartznet (Kriman et al.2020)</td><td>CTC</td><td>All</td></tr><tr><td>Whisper (Radford etal.2023)</td><td>AED</td><td>All</td></tr><tr><td>Branchformer (Peng et al.2022)</td><td>CTC +AED</td><td>All</td></tr><tr><td>Conformer (Gulati et al. 2020)</td><td>RNN-T</td><td>All</td></tr><tr><td>Zipformer (Yao etal.2024b)</td><td>Pruned RNN-T</td><td>All</td></tr><tr><td>Paraformer (Gao et al. 2022)</td><td>CIF</td><td>All</td></tr><tr><td colspan="3">LLM-basedASR</td></tr><tr><td>LauraGPT(Wang et al.2023)</td><td></td><td>All</td></tr><tr><td>SpeechGPT (Zhang et al.2023) Li et al. (2023b)</td><td>Decoder-</td><td>LLM</td></tr><tr><td>SpeechLLaMA (Wu et al.2023)</td><td>Only,</td><td>Encoder,LLM Adapter</td></tr><tr><td></td><td>Cross</td><td>Encoder,LLMLoRA</td></tr><tr><td>Qwen-Audio (Chu et al. 2023)</td><td></td><td>Encoder,MLP</td></tr><tr><td>SALMONN (Tang et al. 2024)</td><td>Entropy</td><td>QF,LLMLoRA</td></tr><tr><td>Fathullah et al. (2024)</td><td></td><td>MLP,LLMLoRA</td></tr><tr><td>Yu et al. (2024)</td><td></td><td>MLP/QF</td></tr></table></body></html>

# 2.1 Previous NN-based ASR

Previous NN-based ASR systems are designed to align the speech signal with the label sequence accurately. As shown in Table 1, different paradigms are carried out with a series of representative models. Quartznet (Kriman et al. 2020) leverages CTC (Graves et al. 2006), the first E2E technology widely adopted in ASR, yet facing performance limitations due to its frame-independent assumption. Whisper (Radford et al. 2023) utilizes massive pair speech-text data to train the attention-based encoder-decoder (Chan et al. 2016) (AED, a.k.a. LAS in ASR) architecture, empowering the model with the ability to recognize and translate speech in multiple languages. Branchformer (Peng et al. 2022) employs a hybrid architecture that combines CTC and AED (Chan et al. 2016), the integration of the attention mechanism addresses this limitation by introducing implicit language modeling across speech frames. Conformer (Gulati et al. 2020) utilizes neural transducer (Graves, Mohamed, and Hinton 2013), which directly discards the frameindependent assumption by incorporating a label decoder and a joint network, resulting in superior performance. Zipformer (Yao et al. 2024b) adopts Pruned RNN-T (Kuang et al. 2022), which is a memory-efficient variant of the transducer loss, utilizing the pruned paths with minor posterior probabilities. Paraformer (Gao et al. 2022) uses Continuous Integrate-and-Fire (CIF) (Dong and $\mathtt { X u } 2 0 2 0 \rVert$ ), which offers a soft and monotonic alignment mechanism, estimating the number of tokens and generating hidden variables.

# 2.2 Existing LLM-based ASR

LLM-based ASR models adopt decoder-only architectures based on a pre-trained LLM as a new paradigm.

LauraGPT (Wang et al. 2023) connects a modified Conformer (Gulati et al. 2020) encoder with Qwen-2B (Bai et al. 2023) for end-to-end training for multiple speech and audio tasks, with full parameter fine-tuning performed. SpeechGPT (Zhang et al. 2023) discretizes speech tokens with HuBERT (Hsu et al. 2021) and fine-tunes the LLaMA13B (Touvron et al. 2023a) with multiple stages. Although both models are computationally expensive, their performance is limited. (Li et al. 2023b) and (Wu et al. 2023) propose to use inserted Gated-XATT-FFN (Alayrac et al. 2022) or side-branched LoRA (Hu et al. 2022) to fine-tune the LLM partially for conducting ASR task, along with a trainable speech encoder. Qwen-Audio (Chu et al. 2023) is an audio-universal model, which uses massive pair data to finetune the encoder initialized from the Whisper-large (Radford et al. 2023) model, optimized using the loss of the frozen Qwen-7B (Bai et al. 2023) output for backpropagation. All these models require finetuning the encoder. SALMONN (Tang et al. 2024) uses Whisper-large (Radford et al. 2023) and BEATs (Chen et al. 2023) to encode speech and audio, respectively, along with a window-level Q-Former (win-QF), can perform a variety of audio tasks. (Fathullah et al. 2024) connects Conformer with LLaMA7B to conduct monolingual and multilingual ASR successfully. These models require the use of LoRA to be effective. Some work (Radhakrishnan et al. 2023; Chen et al. 2024; Li et al. 2024) directly utilize LLMs for generative error correction in a cascade manner. A recent work (Yu et al. 2024) achieves good results on ASR using the only trainable Q-Former or MLP as the projector. The random concatenation training strategy is designed to alleviate the natural problem of Whisper (Radford et al. 2023) requiring an input speech of 30 seconds. These models construct effective models from different aspects; however, the question of what matters when building LLM-based ASR models is unanswered, and a comprehensive exploration is urgent.

# 2.3 Benckamrking System

Our experimental procedure obeys the KISS (Dalzell 2008) (Keep It Simple, Stupid!) principle to investigate what matters when building LLM-based ASR models. We construct a concise framework to train a benchmarking system, which contains an off-the-shelf speech encoder, a large language model, and the only trainable MLP projector. There are multiple reasons why MLP is chosen in the benchmarking system. On the one hand, previous work on speech (Yu et al. 2024) and vision (McKinzie et al. 2024) found that different projectors have similar effects under similar parameter scales. On the other hand, our preliminary experiments show that training with a Q-Former-based projector is not as stable and efficient as an MLP-based projector, which has also been demonstrated by recent work (Yao et al. 2024a) in the field of Vision-Language Model (VLM).

Given speech $\mathbf { X } ^ { \tilde { \mathbf { S } } }$ , the corresponding transcript $\mathbf { X ^ { T } }$ , and the prompt $\mathbf { X ^ { P } }$ , we first convert the speech into speech features through the speech encoder, which can be written as:

$$
\mathbf { H ^ { S } } = E n c o d e r ( \mathbf { X ^ { S } } ) ,
$$

where $\mathbf { H ^ { S } } \ = \ [ h _ { 1 } ^ { S } , \cdot \cdot \cdot \ , h _ { T } ^ { S } ]$ has $T$ frames in the temporal dimension. Due to the sparsity of speech representation, the speech features sequence $\mathbf { H ^ { S } }$ is still very long for the LLM to tackle1, we downsample the speech with a downsampler. More explicitly, we concatenate every $k$ consecutive frames in the feature dimension to perform a $k$ times downsampling, leading to $\mathbf { Z ^ { S } } = [ z _ { 1 } ^ { S } , \cdot \cdot \cdot , \dot { z } _ { N } ^ { S } ]$ , where

$$
z _ { i } ^ { S } = h _ { k * i } ^ { S } \oplus h _ { k * i + 1 } ^ { S } \oplus \cdots \oplus h _ { k * i + k - 1 } ^ { S } ,
$$

and

$$
N = T / / k .
$$

Next, a projector is applied to transform the speech features $\mathbf { Z ^ { s } }$ into $\mathbf { \dot { E } ^ { S } }$ with the same dimension as the LLM input embedding. In our experiments, we use a single hidden layer followed by a ReLU activation and a regression layer as the projector, donated as:

$$
{ \bf E ^ { S } } = L i n e a r ( R e L U ( L i n e a r ( { \bf Z ^ { S } } ) ) ) .
$$

Finally, we feed the speech embedding $\mathbf { E ^ { S } }$ , transcript embedding $\mathbf { E ^ { T } }$ , and prompt embedding $\bar { \mathbf { E } } ^ { \mathbf { P } }$ into the template to compose the final input $\mathbf { E }$ of LLM, donated as:

$$
\begin{array} { r } { \mathbf { E ^ { T } } = T o k e n i z e r ( \mathbf { X ^ { T } } ) , } \\ { \mathbf { E ^ { P } } = T o k e n i z e r ( \mathbf { X ^ { P } } ) , } \end{array}
$$

$$
\begin{array} { r } { \mathbf { E } = \left\{ \begin{array} { l l } { T e m p l a t e ( \mathbf { E ^ { S } } , \mathbf { E ^ { P } } , \mathbf { E ^ { T } } ) } & { { \mathrm { i f ~ t r a i n i n g } } , } \\ { T e m p l a t e ( \mathbf { E ^ { S } } , \mathbf { E ^ { P } } ) } & { { \mathrm { i f ~ i n f e r e n c e } } } \end{array} \right. } \end{array}
$$

# 3 Experiment Setup

# 3.1 Models and Modules

Speech Encoder Two types of speech encoders are investigated in this paper, which are supervised speech encoders trained on massive speech-text pair data and self-supervised speech encoders trained on large-scale unlabeled speech data. For supervised foundation models, we mainly survey the well-known Whisper (Radford et al. 2023) family of models2 ranging from tiny to large, including whisper-tiny, whisper-base, whisper-small, whisper-medium and whisperlarge- $\cdot \nu 2$ . We discard the decoder of each Whisper model and only use the encoder as a feature extractor. We also investigate Qwen-Audio Encoder3, the encoder fine-tuned from whisper-large- $\cdot \nu 2$ checkpoint on large-scale speech, audio and music data, released along with Qwen-Audio (Chu et al. 2023) model. For self-supervised models, we investigate $H u B E R T ^ { 4 }$ and $W a \nu L M ^ { 5 }$ in different scales, either raw pre-trained or further fine-tuned. For the base-size models, both HuBERT (Hsu et al. 2021) and WavLM (Chen et al. 2022) perform self-supervised pre-training on LibriSpeech (Panayotov et al. 2015) corpus with 960 hours. For the large-size models, HuBERT is trained on LibriLight (Kahn et al. 2020) corpus with 60, 000 hours, while

WavLM is trained on the much larger 94, 000 hours data including LibriLight (Kahn et al. 2020), VoxPopuli (Wang et al. 2021), and GigaSpeech (Chen et al. 2021). Furthermore, HuBERT provides pre-trained models of X-Large size, which is the largest publicly available self-supervised speech encoder. All the models mentioned in this section are obtained from their official repositories.

LLM Two types of large language models are investigated in this paper, which are raw pre-trained LLMs without supervised fine-tuning and chat LLMs with SFT (along with RLHF if conducted). For the pre-trained LLMs, we try TinyLLaMA (Zhang et al. 2024)6 of the 1B-magnitude and LLaMA-2 (Touvron et al. 2023b)7 of the 7B-magnitude. For the chat LLMs, TinyLLaMA-Chat8 of the 1B-magnitude, $P h i { - } 2 ^ { 9 }$ of the 2B-magnitude, LLaMA-2-Chat10 and Vicuna (Chiang et al. $2 0 \hat { 2 } 3 ) ^ { 1 1 }$ of the 7B-magnitude are considered.

Projector The projector can be viewed as an adaptor for other modalities to perform alignment with LLM. In all our experiments, the output of the speech encoder is $5 0 \mathrm { H z }$ , and the downsampling rate $k = 5$ , leading to the input speech features $\mathbf { E ^ { S } }$ of the large model being $1 0 ~ \mathrm { H z }$ . The hidden layer dimension is set to 2048, while the dimension of the speech encoder output ${ \bf { H } ^ { S } }$ and the LLM input dimension vary depending on the model used, respectively.

# 3.2 Datasets

To evaluate the capabilities of the LLM-based ASR models, we use the most widely used benchmark for the ASR task, the standard Librispeech (Panayotov et al. 2015) benchmark with 960 hours of training data without any data augmentation or splicing. We use the dev-other subset as the validation set and test-clean/test-other as the test sets, each of which contains 10 hours of speech. We also test our findings on a more diverse, noisy, and challenging dataset, the Gigaspeech (Chen et al. 2021) dataset. We train the model with Gigaspeech-M with 1, 000 hours, select on the DEV set with 10 hours, and test on the TEST set with 40 hours.

# 3.3 Training Detail

During training, the data is organized in the following format: “USER: $< S > < P > A S S I S T A N T ; < T > ^ { , }$ , where ${ < } S >$ represents speech embedding, ${ < } P >$ represents the prompt, and ${ < } T >$ represents the corresponding transcribed text. We only compute the loss on ${ < } T >$ , as is common practice. For the optimizing strategy, we use AdamW (Loshchilov and Hutter 2019) with a max learning rate of $1 \times 1 0 ^ { - 4 }$ without a weight decay. For the learning rate scheduler, we conduct warmup at the first 1, 000 steps and then keep the maximum learning rate for training all the time. The max training step is set to 100, 000, but we will stop early if the loss on the validation set does not decrease. For the audio embedding provided by the Whisper family of models, we found that not padding would affect the performance. As a result, we pad the speech to 30 seconds for all Whisper models and the batch size is set to 4. For other models, the length of the input audio remains consistent with the original length in the temporal dimension, and the batch is set to 6, which greatly improves the efficiency of training and inference, compared to Whisper models.

<html><body><table><tr><td rowspan="3">Speech Encoder</td><td colspan="4">Pre-trainedModel</td><td colspan="4">ChatModel</td></tr><tr><td colspan="2">TinyLLaMA</td><td colspan="2">LLaMA-2</td><td colspan="2">TinyLLaMA-Chat</td><td colspan="2">LLaMA-2-Chat</td></tr><tr><td>test-clean</td><td>test-other</td><td>test-clean</td><td>test-other</td><td>test-clean</td><td>test-other</td><td>test-clean</td><td>test-other</td></tr><tr><td>Whisper-tiny</td><td>12.72</td><td>21.64</td><td>16.16</td><td>25.17</td><td>9.55</td><td>21.01</td><td>8.97</td><td>18.77</td></tr><tr><td>Whisper-base</td><td>7.35</td><td>15.89</td><td>17.46</td><td>21.84</td><td>7.03</td><td>15.92</td><td>6.37</td><td>12.98</td></tr><tr><td>Whisper-small</td><td>6.61</td><td>11.81</td><td>6.41</td><td>10.88</td><td>5.94</td><td>11.5</td><td>4.51</td><td>8.94</td></tr><tr><td>Whisper-medium</td><td>4.65</td><td>8.95</td><td>3.35</td><td>6.10</td><td>5.01</td><td>8.67</td><td>2.71</td><td>6.37</td></tr><tr><td>Whisper-large</td><td>4.39</td><td>8.22</td><td>3.01</td><td>7.15</td><td>4.33</td><td>8.62</td><td>2.72</td><td>6.79</td></tr></table></body></html>

Table 2: A benchmark with different combinations of speech encoders and LLMs to conduct LLM-based ASR. We benchmark Whisper models with different sizes on pre-trained models and chat models with different scales.

# 3.4 Inference Detail

During inference, the data is organized in the following format: “USER: ${ < } S >$ $\mathrm { < } P \mathrm { > }$ ASSISTANT:”, where large language models answer autoregressively. Typically, LLMs utilize sampling algorithms to generate diverse textual outputs. Since speech recognition is a sequence-to-sequence task with deterministic outputs, we use beam search with $b e a m \textbf { = } 4$ to output the hypothesis corresponding to the speech.

<html><body><table><tr><td rowspan="2">LLM</td><td colspan="2">PPL↓</td><td colspan="2">WER(%)↓</td></tr><tr><td>clean</td><td>other</td><td>clean</td><td>other</td></tr><tr><td>LLaMA-2</td><td>53.74</td><td>58.78</td><td>3.01</td><td>7.15</td></tr><tr><td>LLaMA-2-Chat</td><td>77.60</td><td>85.74</td><td>2.72</td><td>6.79</td></tr><tr><td>Vicuna</td><td>76.44</td><td>84.95</td><td>2.58</td><td>6.47</td></tr></table></body></html>

Table 3: Word-level text perplexity (PPL) and word error rate (WER) of different LLMs on Librispeech test-clean (clean) and test-other (other) subsets. Among the listed models, the LLM-based ASR model with Vicuna has the best word error rate, while LLaMA performs the worst.

# 4 Insights From Benchmarking

In this section, we give benchmarks of combinations of different LLMs and speech encoders and obtain a series of conclusions. We find that chat models perform better than raw pre-trained LLMs on the ASR task and verify that this superiority does not come from the transcribed text data leakage in LLM. We further find that fine-tuned versions of selfsupervised speech encoders with limited data outperform supervised foundation ASR encoders.

# 4.1 Is The Chat Model Better Than The Pre-trained Model in LLM-based ASR?

To answer this question, we benchmark Whisper models with different sizes on pre-trained LLMs and supervised fine-tuned LLMs. We pick TinyLLaMA of the 1Bmagnitude and LLaMA-2 of the 7B-magnitude to make a preliminary assessment. As shown in Table 2, the performance of the ASR task improves as the speech encoder parameter size increases, but the improvement is of diminishing marginal benefit for the Whisper family of models. For the choice of LLMs, the chat models work better than the pre-trained models, regardless of the size. One possible explanation is that the chat models take speech embedding as a form of “language” and perform a machine translation task, which is activated during the SFT process.

# 4.2 Is There Transcribed Text Leakage in The Chat LLM?

Another possible reason for the chat LLM being better for LLM-based ASR is that it introduces the transcribed text information to LLM in the SFT stage, resulting in the model easily outputting the corresponding text after obtaining the speech signal. Thus, word-level text perplexity (PPL) of different LLMs is measured to investigate if the better performance of the chat model is related to domain agreement, rather than supervised fine-tuning.

As shown in Table 3, we measure perplexity on test-clean and test-other subsets. Surprisingly, LLaMA-2 without SFT achieves the lowest perplexity by a large margin compared with chat models, while performing the worst on the word error rate. This proves that the better results of chat models are not due to domain agreement with the transcripts.

# 4.3 What Matters When Choosing A Chat LLM?

We fix the speech encoder as Whisper-large and then explore a better large language model. As shown in Table 4, the Phi-2 chat model with 2.78B parameters has a comparable word error rate with LLaMA-2 with 6.74B parameters on test-other. Vicuna is an open-source chat LLM fine-tuned on user-shared conversational data collected from ShareGPT12, utilizing LLaMA as a pre-trained LLM. The LLM-based ASR model shows better results when Vicuna is used as the LLM compared with LLaMA-2 and LLaMA-2-Chat. All the above experimental results confirm larger sizes and better chat models contribute to the performance of LLM-based ASR systems.

<html><body><table><tr><td>LLM</td><td>#LLMParams</td><td>Hidden Size</td><td>#Projector Params</td><td colspan="2">test-CleaER(%est-other</td></tr><tr><td>Pre-trainedModel</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>2048</td><td>17.31M</td><td>4.39</td><td>8.22</td></tr><tr><td>TinyLLaMA LLaMA-2</td><td>1.10B 6.74B</td><td>4096</td><td>21.50M</td><td>3.01</td><td>7.15</td></tr><tr><td>Chat Model</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>TinyLLaMA-Chat</td><td>1.10B</td><td>2048</td><td>17.31M</td><td>4.33</td><td>8.62</td></tr><tr><td>Phi-2</td><td>2.78B</td><td>2560</td><td>18.35M</td><td>3.88</td><td>7.19</td></tr><tr><td>LLaMA-2-Chat</td><td>6.74B</td><td>4096</td><td>21.50M</td><td>2.72</td><td>6.79</td></tr><tr><td>Vicuna</td><td>6.74B</td><td>4096</td><td>21.50M</td><td>2.58</td><td>6.47</td></tr></table></body></html>

Table 4: Explore the performance with different LLMs for LLM-based ASR. The projector is fixed with linear layers and the speech encoder is fixed with Whisper-large-v2.

# 4.4 What Matters When Choosing A Speech Encoder?

We fix Vicuna as the LLM and benchmark the performance of different speech encoders. As shown in Table 5, for the supervised speech encoders, the performance gets better gradually as the parameter size of the speech encoder increases, which is consistent with the conclusion on the exploration of LLMs. When the Qwen-Audio Encoder is used as the speech encoder, the ASR performance is further improved compared with Whisper-large, which indicates that the encoder fine-tuned on other LLM (i.e. Qwen-7B) with gradient backpropagation, can be transferred to another LLM (i.e. Vicuna-7B), and maintain a certain degree of performance.

For the self-supervised learning speech encoders, HuBERT Base and WavLM Base have about 95M parameters, with 768 dimensions of hidden size. In this configuration, the ASR performance is similar compared with Whisper-small with the same scale, where self-supervised learning does not play a role. When scaling the selfsupervised speech encoders to 0.3B, WavLM Large outperforms all listed supervised speech encoders, including Whisper-medium with 0.3B parameters and Whisper-large with 0.6B parameters, while the improvement from HuBERT Base to HuBERT Large is not obvious. However, if the HuBERT Large encoder is first fine-tuned on Librispeech 960 hours of training data, and used as the speech encoder to train the projector in LLM-based ASR model, the model achieves a WER of $2 . 1 0 \%$ on test-clean and $4 . 2 6 \%$ on testother, exceeding the performance with WavLM Large as the speech encoder. With Librispeech-960 fine-tuned WavLM Large as the speech encoder, our LLM-based ASR model gets a word error rate of $1 . 9 6 \%$ on test-clean and $4 . 1 8 \%$ on test-other, achieving $2 7 . 9 \%$ and $3 8 . 4 \%$ relative WER reduction over the model whose encoder is Whisper-medium with similar parameters, respectively. Additionally, inspired by Fuyu (Bavishi et al. 2024), we also try to drop the speech encoder and directly feed the 80-dimensional FBank features into the projector, which lags far behind utilizing welltrained speech encoders, as shown in the first row of Table 5. The experimental results show the effectiveness of using self-supervised speech encoders and scaling the size of

speech encoders.

# 5 Models

In this section, we integrate a bunch of conclusions above together and compare our models with state-of-the-art NNbased ASR models either trained on specific datasets or trained with massive speech, as well as other LLM-based ASR models with delicate designs.

# 5.1 Compared with NN-based ASR Models

We compare our best recipe with state-of-the-art NN-based models. For specialist models trained on Librispeech-960, we compare with ContextNet (Han et al. 2020), Conformer(Gulati et al. 2020), Branchformer (Peng et al. 2022), and Zipformer (Yao et al. 2024b). All models are of large size, and the results from their papers are demonstrated. These ASR models employ sophisticated system engineering, including SpecAugment and speed perturbation for data augmentation, and the exponential moving average technique for model averaging. To further improve performance, in-domain language models trained on the LibriSpeech language model corpus along with the LibriSpeech-960 transcripts are added for fusing or rescoring. Our LLM-based ASR model achieves a better ASR performance than the best-performing models without using complex system engineering. Compared with general-propose models trained on massive data, Our LLM-based ASR model outperforms Whisper-large-v2 (Radford et al. 2023) in industry, and OWSM-v3.1 (Peng et al. 2024) in the academic community.

For a more challenging and noisy dataset, Gigaspeech, we also compare our model with other well-known models. For specialist models trained on $1 , 0 0 0$ hours GigaspeechM, we compare with the model of the original paper train with Kaldi13, Conformer-Transducer trained with Fairseq14, and Zipformer-Pruned RNN-T trained with ${ \mathrm { K } } 2 ^ { 1 5 }$ . Our LLMbased ASR model also achieves better performance than theirs without using sophisticated systems and data engineering. However, our model does not perform as well as the universal Whisper-large-v2 (Radford et al. 2023), which indicates that the LLM-based ASR model still has limited

<html><body><table><tr><td>Speech Encoder</td><td>#Encoder Params</td><td>Hidden Size</td><td>#Projector Params</td><td>WER(%) ↓ test-clean</td><td>test-other</td></tr><tr><td>Acoustic Feature</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>FBank</td><td>-</td><td>80</td><td>10.03M</td><td>68.95</td><td>99.37</td></tr><tr><td>Supervised Speech Encoder</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Whisper-tiny</td><td>7.63M</td><td>394</td><td>12.33M</td><td>7.07</td><td>16.01</td></tr><tr><td>Whisper-base</td><td>19.82M</td><td>512</td><td>13.64M</td><td>5.07</td><td>13.07</td></tr><tr><td>Whisper-small</td><td>87.00M</td><td>768</td><td>16.26M</td><td>4.19</td><td>9.50</td></tr><tr><td>Whisper-medium</td><td>305.68M</td><td>1024</td><td>18.88M</td><td>2.72</td><td>6.79</td></tr><tr><td>Whisper-large</td><td>634.86M</td><td>1280</td><td>21.50M</td><td>2.58</td><td>6.47</td></tr><tr><td>+ Qwen-Audio Fine-tuning</td><td>634.86M</td><td>1280</td><td>21.50M</td><td>2.52</td><td>6.35</td></tr><tr><td>Self-supervised SpeechEncoder</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HuBERT Base</td><td>94.70M</td><td>768</td><td>16.26M</td><td>4.43</td><td>10.72</td></tr><tr><td>WavLMBase</td><td>94.38M</td><td>768</td><td>16.26M</td><td>4.14</td><td>9.66</td></tr><tr><td>HuBERT Large</td><td>316.61M</td><td>1024</td><td>18.88M</td><td>4.53</td><td>8.74</td></tr><tr><td>+ LS-960 Fine-tuning</td><td>316.61M</td><td>1024</td><td>18.88M</td><td>2.10</td><td>4.26</td></tr><tr><td>WavLM Large</td><td>315.45M</td><td>1024</td><td>18.88M</td><td>2.13</td><td>4.73</td></tr><tr><td>+ LS-960 Fine-tuning</td><td>315.45M</td><td>1024</td><td>18.88M</td><td>1.96</td><td>4.18</td></tr></table></body></html>

Table 5: Explore the performance with different speech encoders for LLM-based ASR. The projector is fixed with linear layer and LLM is fixed with Vicuna-7B-v1.5. LS-960 means the Librispeech 960 hours dataset.

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">WER(%)↓</td></tr><tr><td>test-clean</td><td>test-other</td></tr><tr><td>SpecialistModels</td><td></td><td></td></tr><tr><td>ContextNet-large (Han etal.2020)</td><td>2.1</td><td>4.6</td></tr><tr><td>+ in-domain LM</td><td>1.9</td><td>4.1</td></tr><tr><td>Conformer-large (Gulati et al.2020)</td><td>2.1</td><td>4.3</td></tr><tr><td>+ in-domain LM</td><td>1.9</td><td>3.9</td></tr><tr><td>Branchformer-large (Peng et al.2022)</td><td>2.4</td><td>5.5</td></tr><tr><td>+ in-domain LM</td><td>2.1</td><td>4.5</td></tr><tr><td>Zipformer-large (Yao etal.2024b)</td><td>2.0</td><td>4.4</td></tr><tr><td>+ in-domain LM</td><td>1.9</td><td>3.9</td></tr><tr><td>Ours</td><td>1.8</td><td>3.4</td></tr><tr><td>UniversalModels</td><td></td><td></td></tr><tr><td>Whisper-large-v2 (Radford etal.2023)</td><td>2.7</td><td>5.2</td></tr><tr><td>OWSM-v3.1 (Peng et al. 2024)</td><td>2.4</td><td>5.0</td></tr></table></body></html>

Table 6: Compared with previous NN-based models. Specialist Models means models trained on Librispeech960, and in-domain LM means language models trained on the LibriSpeech language model corpus along with LibriSpeech-960 transcripts. Universal Models means general-propose models trained on massive pair data.

Table 7: Compared with SOTA NN-based models from popular code repositories. Specialist Models means models trained on Gigaspeech-M, and Universal Models means general-propose models trained on massive pair data.   

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">Implementation</td><td colspan="2">WER(%)↓</td></tr><tr><td>DEV</td><td>TEST</td></tr><tr><td colspan="2">SpecialistModels</td><td></td><td></td></tr><tr><td>Gigaspeech</td><td>Kaldi</td><td>17.96</td><td>17.53</td></tr><tr><td>Conformer-Transducer</td><td>Fairseq</td><td>14.30</td><td>14.20</td></tr><tr><td>Zipformer-Pruned RNN-T</td><td>K2</td><td>12.24</td><td>12.19</td></tr><tr><td>Ours</td><td>Ours</td><td>10.6</td><td>11.1</td></tr><tr><td colspan="2">UniversalModels</td><td></td><td></td></tr><tr><td>Whisper-large-v2 (Radford etal. 2023)</td><td>OpenAI/Whisper</td><td>10.5</td><td>10.2</td></tr></table></body></html>

ability to handle more difficult data with limited training data. All in all, experimental results demonstrate the effectiveness of our exploration and the great potential of LLMbased ASR.

# 5.2 Compared with Other LLM-based ASR Models

As shown in Table 8, we exhibit different LLM-based ASR models from concurrent work, either ASR-specific or audiouniversal. A contemporary work (Yu et al. 2024) employs Whisper-large as the speech encoder and Vicuna-13B as the LLM. The segment-level Q-Former (seg-QF) is utilized as the projector to tackle the compatibility between speech sequences and the LLM. Compared with their method, our LLM-based ASR model with WavLM Large as the encoder yields $1 3 . 0 / 1 9 . 2 \%$ relative WER reductions on testclean/other subsets trained with the same 960 hours of Librispeech data, and both encoder and LLM are smaller than their solution. When their model is trained on a larger amount of speech over 4, 000 hours, our model still performs better. Further, we scale the speech encoder to 1B parameters using HuBERT X-Large as the speech encoder, and our model yields $2 1 . 7 / 3 4 . 6 \%$ relative WER reductions on testclean/other subsets compared to their solution.

We also compare our model with the latest LLM-based audio-universal models, SALMONN (Tang et al. 2024) and Qwen-Audio (Chu et al. 2023), which provide results on the Librispeech benchmark. Compared with these audio-based multimodal LLMs, our model still achieves better performance despite the large margin in training data. This shows that a concise model combination with limited data can still work well.

# 5.3 Compared with Different Language Models

LLM-based ASR models can be viewed as connecting large language models with acoustic models and training them end-to-end. Therefore, we fix the encoder as WavLM Large and test the performance without LM, with the indomain LM officially provided by Fairseq and speechcolab , and with LLM. We present results on the Librispeech and Gigaspeech-M datasets, respectively. Experimental results show that the LLM-based ASR model performs very well on clean data while losing some performance on noisy data on the Librispeech dataset. Therefore, there is great potential for robust LLM-based ASR.

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2"> Modech EncodeLearnable</td><td colspan="2">ModuleMLearnable</td><td colspan="2">Moduleo Lernable</td><td rowspan="2">ASR Data(h)</td><td colspan="2">test-clWER( %)t-other</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">LLM-based ASR-specific Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Yu et al. (2024)</td><td>Whisper-large</td><td>×</td><td>Vicuna-13B</td><td>×</td><td>seg-QF</td><td>√</td><td>4.00+</td><td>21</td><td>5.</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="2">Ours</td><td></td><td>×</td><td>Vicuna-7B</td><td>×</td><td>MLP</td><td></td><td></td><td></td></tr><tr><td></td><td>HWBERTX-Lag</td><td></td><td></td><td></td><td></td><td>√</td><td>960</td><td>18</td><td>3</td></tr><tr><td colspan="2">LLM-based Audio-universal Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>SALMONN (Tang et al. 2024)</td><td>Whisper-large,BEATs</td><td>X √</td><td>Vicuna-13B</td><td>LoRA</td><td>win-QF</td><td></td><td>1960</td><td>2.1</td><td>4.9</td></tr><tr><td>Qwen-Audio (Chu et al.2023)</td><td>Whisper-large</td><td></td><td>Qwen-7B</td><td>×</td><td>MLP</td><td></td><td>30.000+</td><td>2.0</td><td>4.2</td></tr></table></body></html>

Table 8: Compared with other LLM-based speech models. The specific information of the different modules is given in the table, and all the numbers are obtained from their paper.

Table 9: Ablation on Librispeech and Gigaspeech.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">test-Librispeshother</td><td colspan="2">DGigaspeechT</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>WavLMLarge (pre-trained) w/LLM</td><td>2.13</td><td>4.73</td><td>11.14</td><td>11.88</td></tr><tr><td>WavLMLarge (fine-tuned)</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o LM (CTC)</td><td>2.66</td><td>4.97</td><td>12.77</td><td>12.77</td></tr><tr><td>w/ in-domain LM</td><td>2.14</td><td>4.00</td><td>11.01</td><td>11.45</td></tr><tr><td>w/LLM</td><td>1.96</td><td>4.18</td><td>10.63</td><td>11.05</td></tr></table></body></html>

![](images/bb569912547ea52da3e4b735a2088dda1ad8ddbbecbfc4e0eb969bb7fe22cb11.jpg)  
Figure 1: Training accuracy with the LLM fixed.

# 6 Capability Emergence

We observe that there is capability emergence for LLMbased ASR during training within 1 epoch (around 12k steps).

Figure 1 demonstrates the training accuracy of the next token prediction with the training steps, where the LLM is kept as Vicuna-7B and the speech encoders vary. As can be seen from the figure, the speech encoders with better performance, in this case, Whisper Large and WavLM Large, will emerge earlier. A possible explanation is that our task is essentially to align speech representations with LLMs, while a powerful speech encoder can provide representations that are easier for the projector to align with LLMs.

![](images/43f5df48babbe29ab621f0c3254544a223e53a12d2697583044f91a16a9f317b.jpg)  
Figure 2: Training accuracy with the speech encoder fixed.

We keep the speech encoder as Whisper Large, change different LLMs, and plot the training accuracy, as shown in Figure 2. Experiments show that LLM-based ASR models with smaller LLMs such as TinyLLaMA-Chat and Phi-2 emerge earlier, however, they are not as effective as larger LLMs such as LLaMA-2-7B-Chat and Vicuna-7B. This shows that the larger language models are harder to align with speech features than the smaller ones.

Less training costs will be spent if the model can emerge early, which is yet to be explored. More explorations can be found in supplementary materials.

# 7 Conclusion and Limitation

In this paper, we systematically explore LLM-based ASR systems with a clean framework. A bunch of conclusions are drawn from benchmarking and optimal configurations are used to train the models with prominent performance. Exploratory experiments show that there is a capability emergence in LLM-based ASR systems. Although there is some progress made in LLM-based ASR, the inference speed is still a bottleneck problem that needs to be solved urgently. We aspire for our research to serve as a step forward in the exploration of LLM-based ASR, offering assistance and insights to the broader community.
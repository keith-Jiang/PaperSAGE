# MathSpeech: Leveraging Small LMs for Accurate Conversion in Mathematical Speech-to-Formula

Sieun Hyeon1\*, Kyudan $\mathbf { J u n g ^ { 2 * } }$ , Jaehee $\mathbf { W o n } ^ { 3 }$ , Nam-Joon $\mathbf { K i m } ^ { 1 \dagger }$ , Hyun Gon Ryu4, Hyuk-Jae Lee1, 5, Jaeyoung Do1, 5†

1Department of Electrical and Computer Engineering, Seoul National University 2Department of Mathematics, Chung-Ang University 3College of Liberal Studies, Seoul National University 4NVIDIA 5Interdisciplinary Program in Artificial Intelligence, Seoul National University zxc2692@snu.ac.kr , wjdrbeks1021@cau.ac.kr , wonjaehe e $@$ snu.ac.kr , knj01 $@$ snu.ac.kr , hryu@nvidia.com , hjlee@capp.snu.ac.kr , jaeyoung.do $@$ snu.ac.kr

# Abstract

In various academic and professional settings, such as mathematics lectures or research presentations, it is often necessary to convey mathematical expressions orally. However, reading mathematical expressions aloud without accompanying visuals can significantly hinder comprehension, especially for those who are hearing-impaired or rely on subtitles due to language barriers. For instance, when a presenter reads Euler’s Formula, current Automatic Speech Recognition (ASR) models often produce a verbose and error-prone textual description (e.g., e to the power of $\mathrm { ~ i ~ x ~ }$ equals cosine of $\mathbf { x }$ plus i side of $\mathbf { \boldsymbol { x } } ^ { \mathrm { { \prime } } }$ ), instead of the concise $\mathrm { \Delta E I E X }$ format (i.e., $\begin{array} { r } { \dot { e } ^ { i x } = \cos ( x ) + i \sin ( x ) ) } \end{array}$ , which hampers clear understanding and communication. To address this issue, we introduce MathSpeech, a novel pipeline that integrates ASR models with small Language Models (sLMs) to correct errors in mathematical expressions and accurately convert spoken expressions into structured $\mathrm { \Delta I A I { _ { E } X } }$ representations. Evaluated on a new dataset derived from lecture recordings, MathSpeech demonstrates $\mathrm { \Delta I ^ { \mathrm { A T } } \mathrm { \Sigma E } ^ { X } }$ generation capabilities comparable to leading commercial Large Language Models (LLMs), while leveraging fine-tuned small language models of only 120M parameters. Specifically, in terms of CER, BLEU, and ROUGE scores for $\mathrm { \Delta I A I { _ { E } X } }$ translation, MathSpeech demonstrated significantly superior capabilities compared to GPT4o. We observed a decrease in CER from 0.390 to 0.298, and higher ROUGE/BLEU scores compared to GPT-4o.

# Code — https://github.com/hyeonsieun/MathSpeech

# Introduction

Taking lectures that cover mathematical content through videos and watching academic presentations via video recordings or online streaming is no longer something out of the ordinary. Online video platforms have revolutionized mathematical education by making high-quality lectures and resources accessible to a global audience, breaking down geographical and financial barriers. However, these platforms face a critical challenge: accurately generating subtitles for mathematics lectures. Although platforms like YouTube offer automatic subtitle services, their performance deteriorates markedly when handling mathematical content, particularly equations and formulas. As a result, some providers (such as MIT OpenCourseWare1) are compelled to manually create subtitles, but this requires a tremendous amount of effort and human labeling resources.

The crux of this issue lies in the limitations of current Automated Speech Recognition (ASR) models when confronted with mathematical expressions. Despite advancements in ASR technology (Radford et al. 2023; NVIDIA 2024a,b; Gandhi, von Platen, and Rush 2023), these models significantly underperform in recognizing and transcribing mathematical speech. This shortcoming became apparent during our initial investigations, highlighting the absence of a benchmark dataset specifically for evaluating ASR models’ proficiency in mathematical contexts.

To address this gap, we first developed a novel benchmark dataset comprising 1,101 audio samples from real mathematics lectures available on YouTube. This dataset2 serves as a crucial tool for assessing the capabilities of various ASR models in mathematical speech recognition. Our evaluations using this dataset revealed not only poor performance in equation transcription but also a critical lack of LATEX generation ability, which is the standard for typesetting mathematical equations, in existing ASR models. This limitation significantly impedes learners’ comprehension, especially when dealing with complex mathematical expressions. An example of this case is presented in Table 1.

To address these challenges, we introduce MathSpeech, an innovative ASR pipeline specifically designed to transcribe mathematical speech directly into $\mathrm { \Delta I A T _ { E } X }$ code instead of plain text. This approach enhances the learning experience by enabling the accurate rendering of mathematical expressions in subtitles, thereby supporting learners in their math education. Rather than incurring the significant costs associated with fine-tuning ASR models on domain-specific mathematical speech data—especially given the lack of publicly available datasets—we developed a novel method that converts mathematical speech into $\mathrm { \Delta I A T _ { E } X }$ using small Language Models (LMs). Our methodology corrects ASR outputs containing mathematical expressions (even if they contain errors) and transforms the corrected output into $\mathrm { \Delta I A T _ { E } X }$ using fine-tuned small LMs. By employing effective finetuning techniques that account for ASR errors and the nuances of spoken English in mathematical contexts, our pipeline has demonstrated superior $\mathtt { I A T _ { E } X }$ generation capabilities compared to commercial large language models like GPT-4o (OpenAI 2024a) and Gemini-Pro (Google Deepmind 2024), despite using a relatively small language model with only 120M parameters.

Table 1: Comparison between the actual equation and the ASR result   

<html><body><table><tr><td>Formula</td><td>ex=cos(x)+isin(x)</td></tr><tr><td>Spoken En- glish(SE)</td><td>e to the power of i x equals cosine of x plus i sineofx</td></tr><tr><td>ASR result with error</td><td>e to the power of i x equals cosine of x plus i side of x</td></tr></table></body></html>

In summary, our contributions are as follows:

• We constructed and released the first benchmark dataset for evaluating ASR models’ ability to transcribe mathematical equations.   
• We identified and demonstrated the poor performance of existing ASR models in reading mathematical equations.   
• We proposed a pipeline that corrects ASR errors and converts the output into $\mathrm { I A T _ { E } X }$   
• We confirmed that our pipeline, despite being significantly smaller (120M parameters) than commercial LLMs, outperformed GPT-4o (OpenAI 2024a) and Gemini-Pro (Google Deepmind 2024).

# Related Works

# ASR correction with LM

As ASR (Radford et al. 2023; NVIDIA 2024a,b; Gandhi, von Platen, and Rush 2023) systems have advanced, the need to correct ASR errors has become increasingly important. To enhance the quality of ASR output, there has been a significant amount of prior research focused on post-processing using language models. Since ASR outputs are in text form, many studies have employed sequence-to-sequence techniques.

In the past, statistical machine translation (Cucu et al. 2013; D’Haro and Banchs 2016) was used for this purpose. With the development of neural network-based language models, autoregressive sequence-to-sequence models are used for error correction (Tanaka et al. 2018; Liao et al. 2023), like neural machine translation. Moreover, with the advancement of attention mechanisms (Chan et al. 2016; Inaguma and Kawahara 2023), research utilizing the Transformer architecture (Vaswani et al. 2017) for error correction has demonstrated strong performance (Mani et al. 2020; Leng et al. 2021b, 2023). Additionally, research on ASR error correction has been conducted using various language models, such as BERT (Yang, Li, and Peng 2022), BART (Zhao et al. 2021), ELECTRA (Futami et al. 2021; Yeen, Kim, and Koo 2023), and T5 (Ma et al. 2023a; Yeen, Kim, and Koo 2023).

Moreover, with the emergence of various Large Language Models (LLMs), which have shown remarkable performance across diverse domains, research on post-processing for ASR correction has also been actively conducted (Hu et al. 2024b; Sachdev, Wang, and Yang 2024; Ma et al. 2023c; Hu et al. 2024a). Several studies have shown that LLMs can be effectively used for ASR correction. However, LLMs have certain drawbacks, such as their large size and slow inference speed.

When using LLMs for ASR error correction, some research has demonstrated that utilizing multiple candidates generated during beam search can result in a voting effect (Leng et al. 2021a), leading to improved performance. This method, known as N-best, allows the N candidates obtained from the ASR output to provide clues to the language model regarding potential errors. Many studies (Imamura and Sumita 2017; Zhu et al. 2021; Ganesan et al. 2021; Ma et al. 2023b,a; Leng et al. 2023; Weng et al. 2020) on ASR correction have adopted this N-best approach.

# $\mathbf { I A I I _ { E } X }$ translation and generation

Research related to LATEX has mainly focused on converting formula images to $\mathrm { { \Delta } B T _ { E } X }$ (Blecher et al. 2023; Blecher, L 2024) using Optical Character Recognition(OCR). Recently, studies have also been conducted on generating $\mathsf { I A T } _ { \mathrm { E } } \mathbf { X }$ from spoken English that describes formulas (Jung et al. 2024a). However, this research has not explored correcting ASR results obtained from actual speech before translating them into $\mathsf { I A T } _ { \mathrm { E } } \mathsf { X }$ . The focus of this study has been on translating clean, error-free spoken English into $\mathsf { I A T } _ { \mathrm { E } } \mathsf { X }$ , without considering ASR errors

Additionally, research has been conducted on using Large Language Models (LLMs) to enhance efficiency and quality in popular academic writing tools like Overleaf (Wen et al. 2024), which includes generating $\mathrm { \Delta \mathrm { { E I E } X } }$ with LLMs. In fact, well-known state-of-the-art commercial LLMs such as GPT series (OpenAI 2024b,a), Gemini-Pro (Google Deepmind 2024) have demonstrated remarkable abilities in generating LATEX. However, the previous study (Jung et al. 2024a,b; Hyeon et al. 2025) indicates that comparing the $\mathrm { \Delta I A T _ { E } X }$ generation and translation capabilities of different LLMs presents a significant challenge, as there is no suitable metric to measure $\mathrm { I A T _ { E } X }$ generation performance.

# Motivation

Subtitle services are often used when individuals watch academic videos or lectures. For the general public, subtitles serve as an auxiliary tool to help them understand video content. However, for individuals with hearing impairments or students who speak a different language than the lecturer, subtitles are essential. Inaccurate subtitle services can severely hinder content comprehension, leading to a significant decrease in learning effectiveness. With recent advancements in Automatic Speech Recognition (ASR) models, the ability to convert speech into text has become highly accurate, greatly benefiting these users. However, the accuracy of ASR models remains significantly lower for academic videos in fields such as mathematics and physics than for other subjects.

Table 2: The WER for Leaderboard was from the HuggingFace Open ASR Leaderboard, while the WER for Formula was measured using our MathSpeech Benchmark.   

<html><body><table><tr><td></td><td>Models</td><td>Params</td><td>WER(%) (Leader- board)</td><td>WER(%) (For- mula)</td></tr><tr><td>OpenAI</td><td>Whisper-base Whisper-small Whisper-largeV2</td><td>74M 244M 1550M</td><td>10.3 8.59 7.83</td><td>34.7 29.5 31.0</td></tr><tr><td>NVIDIA</td><td>Whisper-largeV3 Canary-1B</td><td>1550M 1B</td><td>7.44 6.5</td><td>33.3 35.2</td></tr></table></body></html>

Table 2 presents the results of measuring the WER (Word Error Rate) of ASR models using mathematical speech collected from actual lecture videos.

When comparing the WER results of Whisper (Radford et al. 2023) and Canary-1b (NVIDIA 2024a) on the HuggingFace Open ASR Leaderboard34 with the MathSpeech benchmark dataset WER results, we observed that the WER for formula speech was significantly higher. The reasons for the elevated error rates are as follows.

(1) Severe Noise: Our benchmark dataset consists of audio from lecture videos recorded 10 to 20 years ago, resulting in relatively poor audio quality and higher levels of noise. Additionally, since the audio is taken from real lectures, it includes sounds such as chalk writing on the blackboard and students’ chatter mixed into the speech.

(2) Non-native Accent: When the speaker is not a native English speaker, the model often misinterprets some words as other words due to the speaker’s accent. Our benchmark dataset includes speakers with distinctive accents.

(3) Label Ambiguity Problem: Mathematical expressions read aloud can be ASR-transcribed in multiple ways, which increases the WER. In other words, while the meaning is correct, the text output differs, leading it to be counted as an error. For example, in the case of speech reading number 1, we transcribed it as 1, but the ASR model outputs it as one. Although the ASR output for mathematical speech differed from the labels we assigned, it was semantically equivalent, which contributed to the higher WER measurements.

To accurately convert the spoken English of mathematical expressions into $\mathrm { I A T _ { E } X }$ , it is necessary to carefully consider these factors and adjust the ASR errors accordingly. Therefore, we propose a method for training small language models that considers both the LATEX output and ASR transcription results. This will be discussed in the following section.

# Methodology

# Datasets

Our goal is to implement a lightweight, fast, and highperformance pipeline that outputs $\mathrm { \Delta I A T _ { E } X }$ . We determined that the most effective approach is fine-tuning small LMs, which involves collecting appropriate datasets. To achieve this, we collected two types of data.

# (1) (Spoken English, $\mathbf { \Delta } \mathbf { { B I } } _ { \mathbf { E } } \mathbf { X }$ ) pairs

As mentioned earlier, ASR models convert spoken mathematical expressions into plain English text. Since our goal is to translate such Spoken English(SE) into $\mathrm { \Delta \mathrm { E T } E X }$ , we decided to use a dataset of (Spoken English, $\mathrm { I A T _ { E } X ) }$ pairs. We were able to obtain a publicly available dataset (Jung et al. 2024a) on HuggingFace and used it in our work. This data was collected by web crawling and OCR, extracting only the LATEX portions from arxiv papers and textbooks. This is a large dataset containing 23 million (SE, LATEX) data pairs. We fine-tuned the T5-small (Raffel et al. 2023) with this dataset to convert Spoken English(SE) to $\mathrm { \Delta I A T _ { E } X }$ .

# (2) ASR Error Correction Dataset

In our initial experiments, we attempted to fine-tune T5 (Raffel et al. 2023) only using the (Spoken English(SE), $\mathrm { I A T _ { E } X ) }$ pair dataset and connect it to the ASR as a post-LM. However, the performance was not satisfactory because the ASR model itself made significant errors when converting spoken mathematical expressions into plain text (Table 3).

Table 3: The examples of ASR error results   

<html><body><table><tr><td>Spoken English(SE)</td><td>ASR error result</td></tr><tr><td>X plus 5y plus 1Oz equals O</td><td>X plus 5yplus10zécole 0</td></tr><tr><td>cosine of psi sub i, psi sub j</td><td>Posing of psi sub i, psi sub j</td></tr></table></body></html>

Since such erroneous texts did not exist in the dataset (Jung et al. 2024a), the fine-tuned T5 also produced incorrect $\mathrm { \Delta I A T _ { E } X }$ outputs. Therefore, we determined that adding a process to correct errors that occur in the ASR system would significantly improve performance. Thus, we built another dataset for fine-tuning the ASR error correction model. As shown in Figure 1, we converted Spoken English (SE) into speech using TTS and then fed the speech into the ASR model to obtain the erroneous Spoken English ASR outputs.

In other words, let $y _ { s e }$ represent Spoken English and ylatex represent the $\mathrm { I A T _ { E } X }$ expression for $y _ { s e }$ . Then, we can denote the data pair (Jung et al. 2024a) as $( y _ { s e } , y _ { l a t e x } )$ .

Using TTS, we generate audio for $y _ { s e }$ . If we denote this audio as $x _ { t t s }$ , then we can denote the text obtained by inputting $x _ { t t s }$ into an ASR model as $y _ { a s r ^ { i } }$ , where $i$ is used to distinguish between different models. Therefore, using the method shown in Figure 1, we transform $( y _ { s e } , y _ { l a t e x } )$ into $( y _ { a s r ^ { i } } , y _ { s e } , y _ { l a t e x } )$ .

Spoken English Audio ASR Error Text X plus 5y plus 10z equals 0 TTS ASR X plus 5y plus 10z école 0 yse Xtts yasri

![](images/dd95a65fddbdcc3e88b880ad10f02ae2a6edcf60cfaa3e1af34849ea2c980532.jpg)  
Figure 1: Method for Collecting ASR Error Results.   
Figure 2: This figure compares 2-beam search and our method. The left shows top-2 beam search by a single ASR model, while the right shows top-1 beam search by two ASR models.

At this stage, we used the VITS (Kim, Kong, and Son 2021) model for TTS and collected the voices using 2- speaker. For the ASR model, we used Whisper-base, small, largeV2 (Radford et al. 2023) and Canary-1b (NVIDIA 2024a) to collect the Error ASR results. We collected 6M ASR error results using Whisper-base and small, and 1M ASR error results using Whisper-largeV2 and Canary-1b.

# Models

Our MathSpeech pipeline can be seen in Figure 3. We used two models configured in two stages.

# (1) Error Corrector

The purpose of the Error Corrector is to fix the errors that occur in the ASR model. In other words, the goal of the Error Corrector is to make the ASR error results similar to the input data of the $\mathrm { \Delta I A T _ { E } X }$ Translator, which is called Spoken English. In previous studies, T5 (Ma et al. 2023a,b; Yeen, Kim, and Koo 2023) has demonstrated good performance in ASR error correction. To minimize the model size, we used T5-small for the Error Corrector.

Additionally, recent research has shown that inputting multiple candidates (Imamura and Sumita 2017; Zhu et al. 2021; Ganesan et al. 2021; Ma et al. 2023b,a; Leng et al. 2023; Weng et al. 2020) generated by ASR beam search can yield good performance in various ASR Corrector models due to the voting effect (Leng et al. 2021a). We have further developed this idea. Instead of using multiple candidates generated by ASR beam search, we use multiple candidates from the top-1 results of different models (Figure 2).

An advantage of this method is its versatility. We believe that training the Error Corrector with error information from various models will make it universally applicable to different ASR models. For example, if the corrector is trained with two ASR beam search results from Whisper-small, it may perform well on Whisper-small, but not on Whisper-base. However, if the corrector is trained with the ASR results from both Whisper-small and Whisper-base, we believe that it will effectively correct the errors for both models. Experimental results are presented in the following section.

# (2) $\mathbf { \Delta } \mathbf { { u } } \mathbf { { I } } _ { \mathbf { E } } \mathbf { X }$ Translator

According to a previous study (Jung et al. 2024a), the T5 model achieved the best performance for the $\mathrm { \Delta \mathrm { { B T } _ { E } X } }$ translator. To minimize the model size, we used the smallest version, T5-small. We fine-tuned T5-small using the (SE, $\mathrm { I A T } _ { \mathrm { E } } \mathrm { X } )$ pair dataset so that when Spoken English is input, LATEX is output.

# Training

We implemented a pipeline connecting two T5-small models. Instead of simply chaining the fine-tuned Error Corrector and $\mathrm { \Delta \mathrm { { B I } _ { E } X } }$ Translator, we performed end-to-end training (Figure 4). Considering the characteristics of ASR error results and $\mathrm { I A T _ { E } X }$ , we constructed the loss function as follows.

For a given input audio $x _ { t t s }$ , let the inference results of two ASR models be $y _ { a s r ^ { 1 } }$ and $y _ { a s r ^ { 2 } }$ . Here, $y _ { a s r ^ { 1 } }$ and $y _ { a s r ^ { 2 } }$ are the outputs from the ASR models, which may contain errors. We then provide these two ASR outputs to the Error Corrector $F$ . The resulting text can be denoted as $F ( y _ { a s r ^ { 1 } } , y _ { a s r ^ { 2 } } )$ . Next, we take $\hat { y } _ { s e } = F ( y _ { a s r ^ { 1 } } , y _ { a s r ^ { 2 } } )$ and feed it into the $\mathrm { \Delta \mathrm { { E I E } X } }$ Translator $G$ to produce the corresponding $\mathrm { I A T _ { E } X }$ , denoted as $\hat { y } _ { l a t e x } = \bar { G } ( F ( y _ { a s r ^ { 1 } } , y _ { a s r ^ { 2 } } ) )$ . The loss function is then defined as follows:

$$
\mathcal { L } = \lambda _ { 1 } \mathcal { L } _ { s e } + \lambda _ { 2 } \mathcal { L } _ { l a t e x }
$$

And $\mathcal { L } _ { s e }$ and $\mathcal { L } _ { l a t e x }$ are calculated as cross-entropy losses of the tokenized outputs $\hat { y } _ { s e }$ and $y _ { s e }$ , and $\hat { y } _ { l a t e x }$ and $y _ { l a t e x }$ , respectively.

$$
\mathcal { L } _ { s e } = - \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { V } y _ { s e _ { t } ^ { i } } \log ( \hat { y } _ { s e _ { t } ^ { i } } )
$$

$$
\mathcal { L } _ { l a t e x } = - \sum _ { t = 1 } ^ { T } \sum _ { i = 1 } ^ { V } y _ { l a t e x _ { t } ^ { i } } \log ( \hat { y } _ { l a t e x _ { t } ^ { i } } )
$$

The cross-entropy loss is calculated as a negative sum over all time steps $t$ from 1 to $T$ (the length of the sequence) and over all possible words $i$ in the vocabulary $V$ . At each time step $t$ , $y _ { s e _ { t } }$ is a one-hot encoded vector over the vocabulary, where $y _ { s e _ { t } ^ { i } }$ is its $i$ -th element. Specifically, $y _ { s e _ { t } ^ { i } } = 1$ if the correct ground-truth word at time step $t$ is the $i$ -th word in the vocabulary, and 0 otherwise. Correspondingly, $\hat { y } _ { s e _ { t } }$ is the predicted probability distribution vector over the vocabulary at time $t$ , and $\hat { y } _ { s e _ { t } ^ { i } }$ is the predicted probability that the correct word at time $t$ is the $i$ -th word in the vocabulary.

![](images/dd252cf960dacdc4843f152b8f3401f0f17a363a525e66e1882fc1fa77a2ff26.jpg)  
Figure 3: Our pipeline that converts the lecturer’s voice into $\mathrm { \Delta I A T _ { E } X }$ .

![](images/cd7a91c56c01bcc735f6b084c7077013650292b87a0b0001dc63b2b6e67ce6ed.jpg)  
Figure 4: The method of training MathSpeech in an end-to-end manner

We calculated the final loss $\mathcal { L }$ by assigning different weights to the two cross entropy loss functions. The weight assigned to the $\hat { y } _ { l a t e x }$ was set higher than the weight for the loss on $\hat { y } _ { s e }$ because there can be different SE for the same $\mathrm { I A T _ { E } X }$ result (Table 4).

<html><body><table><tr><td>1st SE 2nd SE</td><td>X plus five y plus ten z equals zero X plus 5yplus 10 z equals 0</td></tr><tr><td>LTgX</td><td>$x+5y+10z=10$</td></tr></table></body></html>

Table 4: The reason why errors between $\mathrm { I A T _ { E } X }$ are more critical than errors between SE.

In other words, even if the SEs are slightly different, the $\mathrm { I A T _ { E } X }$ can still produce the correct answer because they are semantically equivalent. Therefore, our goal is to obtain the correct $\mathrm { \Delta I A T _ { E } X }$ , so we assigned a higher weight to $\mathcal { L } _ { l a t e x }$ when constructing the loss function. Our experimental results showed that setting the weight $\lambda _ { 1 }$ for SE to 0.3 and the weight for $\mathrm { I A T _ { E } X }$ to 0.7 yielded the best performance, so we used these values.

# Evaluation Metrics

In previous studies (Jung et al. 2024a), when evaluating $\mathsf { I A T } _ { \mathrm { E } } \mathsf { X }$ , metrics commonly used in translation tasks, such as ROUGE (Lin 2004) and BLEU (Papineni et al. 2002), were employed. Based on this idea, we used ROUGE-1, BLEU, and ROUGE-L. Furthermore, we employed CER, a traditional ASR metric. However, WER was not measured in our experiments. This is because the spaces are often ignored in $\mathsf { I A T } _ { \mathrm { E } } \mathsf { X }$ . For example, $\$ 89$ and $\$ 4 B9$ result in the same $\mathsf { I A T } _ { \mathrm { E } } \mathbf { X }$ compilation output. So the evaluation was conducted after removing all spaces.

# Experiments

To evaluate the $\mathrm { \Delta I A T _ { E } X }$ translation capabilities of our pipeline, we compared its performance against existing commercial large language models.

# Setup

We fixed the same hyperparameters for the fine-tuned models. The maximum number of training epochs was set to 20, and the model with the lowest validation loss was selected. The learning rate was set to a maximum of 1e-4 and a minimum of 1e-6, adjusted using a linear learning rate scheduler. For the Error Corrector, which requires two ASR outputs as input, the maximum input sequence length was set to 540, with an output length of 275. For the LATEX translator, both input and output sequence lengths were set to 275. T5-small was trained with a batch size of 48 on an NVIDIA A100, and T5-base with a batch size of 84 on an NVIDIA H100.

As a comparison group for our pipeline, we selected GPT3.5 (OpenAI 2024b), GPT-4o (OpenAI 2024a), and GeminiPro (Google Deepmind 2024), using 1-shot prompting with one example for all. To observe LATEX translation results across various ASR models, we used five ASR models.

# Result

The experimental results are presented in Tables 5 and 6, respectively. Table 5 lists the outcomes of inputting two ASR prediction candidates that require correction into the LM. In this regard, MathSpeech achieved the best scores for the CER, ROUGE-1, ROUGE-L, and BLEU. The LLM with the second-best performance was GPT-4o, while GPT-3.5 and Gemini-Pro showed similar results.

Table 5: Having only one model in the ASR column means that the top-2 beam search ASR outputs of a single model are input into the LLM. The bold text indicates the result with the best score for the same ASR output, while an asterisk $( * )$ indicates the second-best score for the same ASR output. The double plus sign $( + + )$ indicates the case with the best score.   

<html><body><table><tr><td>ASR</td><td>LM</td><td>CER↓</td><td>ROUGE-1个</td><td>ROUGE-L↑</td><td>BLEU↑</td></tr><tr><td>whisper-base</td><td>GPT-3.5 GPT-40</td><td>0.443 0.410*</td><td>0.782 0.813*</td><td>0.775 0.808*</td><td>0.483 0.487*</td></tr><tr><td></td><td>Gemini-Pro MathSpeech</td><td>0.424 0.336</td><td>0.756 0.824</td><td>0.749 0.819</td><td>0.418 0.662</td></tr><tr><td>whisper-small</td><td>GPT-3.5</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>0.391</td><td>0.815</td><td>0.809</td><td>0.519*</td></tr><tr><td></td><td>GPT-40</td><td>0.384*</td><td>0.840*</td><td>0.835*</td><td>0.516</td></tr><tr><td></td><td>Gemini-Pro</td><td>0.390</td><td>0.799</td><td>0.792</td><td>0.455</td></tr><tr><td></td><td>MathSpeech</td><td>0.309</td><td>0.852++</td><td>0.847++</td><td>0.689++</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>whisper-largeV2</td><td>GPT-3.5</td><td>0.401</td><td>0.820</td><td>0.813</td><td>0.519</td></tr><tr><td></td><td>GPT-40</td><td>0.380*</td><td>0.844*</td><td>0.839*</td><td>0.520*</td></tr><tr><td></td><td>Gemini-Pro</td><td>0.393</td><td>0.800</td><td>0.792</td><td>0.458</td></tr><tr><td></td><td>MathSpeech</td><td>0.298++</td><td>0.848</td><td>0.844</td><td>0.683</td></tr><tr><td>whisper-largeV3</td><td>GPT-3.5</td><td>0.404</td><td>0.797</td><td>0.792</td><td>0.493</td></tr><tr><td></td><td>GPT-40</td><td>0.398*</td><td>0.826</td><td>0.822</td><td>0.495*</td></tr><tr><td></td><td>Gemini-Pro</td><td>0.409</td><td>0.772</td><td>0.767</td><td>0.431</td></tr><tr><td></td><td>MathSpeech</td><td>0.317</td><td>0.817*</td><td>0.812*</td><td>0.673</td></tr><tr><td>canary-1b</td><td>GPT-3.5</td><td>0.458</td><td>0.795</td><td>0.787</td><td>0.434</td></tr><tr><td></td><td>GPT-40</td><td>0.422*</td><td>0.824*</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>0.819*</td><td>0.445*</td></tr><tr><td></td><td>Gemini-Pro MathSpeech</td><td>0.442 0.325</td><td>0.781 0.832</td><td>0.774 0.828</td><td>0.392 0.674</td></tr></table></body></html>

The main factor that lowered the performance scores of the commercial large language models was hallucination. When the ASR results were unusual or ambiguous, the baseline LLMs would either output completely different LATEX formulas or produce non-LATEX texts (e.g., Sorry, I can’t understand). Additionally, whisper-small and whisper-largeV2 outperformed whisper-base, whisper-largeV3, and canary1b in overall metric scores. This result aligns with the WER measurements of Spoken English for whisper-small and whisper-largeV2, which were relatively better, as observed in Table 2. In other words, better ASR results lead to improved $\mathrm { I A T _ { E } X }$ translation performance.

Table 6 lists the outcomes when the top-1 results from different ASR models were used as inputs. The highest performance scores were achieved when the ASR results from Whisper-base & small were used as inputs. This can be attributed to the fact that our ASR error results dataset contains relatively more information from Whisper-base & small. The second highest performance was observed when using the ASR results from Whisper-small & largev2, which can be attributed to the relatively lower WER of these two models on our benchmark dataset. Furthermore, the strong performance on Whisper-largeV3, for which we did not collect ASR error results, demonstrates that MathSpeech can perform well even on ASR models that were not used during training. Moreover, our MathSpeech model, being a smallsized model with 120M parameters, has low latency. When inference latency was measured on an NVIDIA V100 GPU, it took 0.45 seconds to convert the ASR result of 5 seconds of speech into $\mathsf { I A T } _ { \mathbb { E } } \mathsf { X }$ .

# Ablation Study

To demonstrate the effectiveness of the MathSpeech structure, we conducted ablation studies (Table 7).

To show that correcting ASR outputs is crucial for $\mathrm { \Delta I A T _ { E } X }$ translation, we removed the corrector and conducted an experiment in which ASR outputs were directly translated into LATEX. As a result, both T5-small and T5-base showed significant performance degradation.

To demonstrate that implementing the corrector and translator in a 2-stage structure is effective, we trained a single T5 model to perform both correction and translation in a 1-stage process and observed its performance. The experiment, where we trained the model to translate two ASR error outputs into LATEX with the same setup, showed lower performance compared to our pipeline.

To validate the effectiveness of our end-to-end training method, we compared it with a method where the corrector and translator were trained separately and simply concatenated. The results confirm that the proposed training method is more effective.

Since a single T5-base (220M) is larger than two T5-small models (120M), we did not apply our end-to-end training method to the 2-stage T5-base pipeline. However, we can infer that if we were to apply our training method to the simple concatenation of two T5-base models, the performance could potentially improve further.

Table 6: Having two models in the ASR column means that the top-1 beam search ASR output from each of the two different models is obtained and both are input into the LLM. Total Average refers to the average of all the results from Table 5 and 6.   

<html><body><table><tr><td>ASR</td><td>LM</td><td>CER↓</td><td>ROUGE-1↑</td><td>ROUGE-L↑</td><td>BLEU个</td></tr><tr><td rowspan="4">whisper-base & whisper-small</td><td>GPT-3.5</td><td>0.407</td><td>0.801</td><td>0.792</td><td>0.502</td></tr><tr><td>GPT-40</td><td>0.379*</td><td>0.843*</td><td>0.839*</td><td>0.518*</td></tr><tr><td>Gemini-Pro</td><td>0.382</td><td>0.802</td><td>0.796</td><td>0.457</td></tr><tr><td>MathSpeech</td><td>0.243++</td><td>0.870++</td><td>0.864++</td><td>0.718++</td></tr><tr><td rowspan="4">whisper-small & whisper-largeV2</td><td>GPT-3.5 GPT-40</td><td>0.388 0.373*</td><td>0.825 0.852*</td><td>0.818</td><td>0.523 0.532*</td></tr><tr><td>Gemini-Pro</td><td></td><td></td><td>0.848*</td><td></td></tr><tr><td>MathSpeech</td><td>0.374</td><td>0.806</td><td>0.800</td><td>0.469</td></tr><tr><td></td><td>0.269</td><td>0.864</td><td>0.859</td><td>0.708</td></tr><tr><td rowspan="3">canary-1b & whisper-largeV2</td><td>GPT-3.5 GPT-40</td><td>0.399 0.386*</td><td>0.826 0.849</td><td>0.820 0.844</td><td>0.524* 0.516</td></tr><tr><td>Gemini-Pro</td><td>0.393</td><td>0.800</td><td>0.793</td><td>0.458</td></tr><tr><td>MathSpeech</td><td>0.294</td><td>0.848*</td><td>0.843*</td><td>0.694</td></tr><tr><td rowspan="4">canary-1b & whisper-largeV3</td><td>GPT-3.5</td><td>0.394</td><td>0.813</td><td>0.806</td><td>0.504</td></tr><tr><td>GPT-40</td><td>0.375*</td><td>0.849*</td><td>0.844*</td><td>0.514</td></tr><tr><td>Gemini-Pro</td><td>0.399</td><td>0.810</td><td>0.804</td><td>0.445</td></tr><tr><td>MathSpeech</td><td>0.292</td><td>0.853</td><td>0.846</td><td>0.698</td></tr><tr><td rowspan="4">Total Average</td><td>GPT-3.5</td><td>0.409</td><td>0.808</td><td>0.801</td><td>0.500</td></tr><tr><td>GPT-40</td><td>0.390*</td><td>0.838*</td><td>0.833*</td><td>0.505*</td></tr><tr><td>Gemini-Pro</td><td>0.400</td><td>0.792</td><td>0.785</td><td>0.442</td></tr><tr><td>MathSpeech</td><td>0.298</td><td>0.845</td><td>0.840</td><td>0.689</td></tr></table></body></html>

<html><body><table><tr><td>ASR</td><td>LM</td><td>CER↓</td><td>ROUGE-1↑</td><td>ROUGE-L ↑</td><td>BLEU↑</td></tr><tr><td>whisper-base</td><td>T5-small (1 stage,w/o corrector) T5-base (1 stage, w/o corrector) MathSpeech</td><td>0.693 0.530* 0.336</td><td>0.734 0.749* 0.824</td><td>0.729 0.744* 0.819</td><td>0.483 0.554* 0.662</td></tr><tr><td>whisper-small</td><td>T5-small (1 stage, w/o corrector) T5-base (1 stage,w/o corrector) MathSpeech</td><td>0.630 0.423* 0.309</td><td>0.769 0.783* 0.852</td><td>0.765 0.780* 0.847</td><td>0.521 0.618* 0.689</td></tr><tr><td>whisper-base & whisper-small</td><td>T5-small (1 stage, fine-tuned with errors) T5-base (1 stage,fine-tuned with errors) T5-small (2 stage, Just connect) T5-base (2 stage, Just connect) MathSpeech</td><td>0.403 0.358 0.357 0.343* 0.243</td><td>0.824* 0.813 0.820 0.817 0.870</td><td>0.819* 0.809 0.815 0.814 0.864</td><td>0.635 0.650 0.656* 0.651 0.718</td></tr></table></body></html>

Table 7: Ablation Study Results. 1 stage refers to using a single T5 model, while 2 stage refers to using two T5 models

# Future Works

# (1) Defining a Metric to Solve the Label Ambiguity

Since LATEX can represent the same formula in multiple ways, it is necessary to consider various possible cases when evaluating the performance of $\mathtt { I A T _ { E } X }$ translation. $\mathrm { \Delta I A T _ { E } X }$ is closer to a computer language, like SQL, than to natural language, so metrics like BLEU or CER are not perfect for evaluating $\mathsf { I A T } _ { \mathrm { E } } \mathsf { X }$ . Therefore, it is necessary to implement a metric that is more suitable for evaluating $\mathrm { \Delta I A T _ { E } X }$ .

(2) Formula Detection in Practice and $\mathbf { \Delta } \mathbf { { B I } } _ { \mathbf { E } } \mathbf { X }$ Conversion This research focuses only on the ability of ASR models and LMs to generate $\mathrm { I A T _ { E } X }$ . However, to apply this to actual subtitle services, it is essential to develop the ability to detect and separate formulaic parts from speech. This can likely be achieved by training the LM not only on $\mathrm { \Delta I A T _ { E } X }$ but also on mixed general text. Additionally, in real-world situations, the speaker may not finish verbally expressing the entire formula. It must be possible to complete such interrupted formulas into full formulas. This could be implemented through the inference capabilities of large language models (LLMs).

# Conclusion

In this paper, we confirmed through a self-constructed benchmark dataset that existing ASR models lack the ability to read mathematical formulas and are unable to generate $\mathrm { I A T _ { E } X }$ . To address this, we propose MathSpeech, a pipeline that connects ASR models with small LMs to generate $\mathsf { I A T } _ { \mathrm { E } } \mathsf { X }$ . By effectively connecting two T5-small models and training them end-to-end, our approach demonstrated superior $\mathrm { I A T _ { E } X }$ translation capabilities compared to existing commercial large language models. Our research opens up the possibility of more accurate subtitles in the field of math.
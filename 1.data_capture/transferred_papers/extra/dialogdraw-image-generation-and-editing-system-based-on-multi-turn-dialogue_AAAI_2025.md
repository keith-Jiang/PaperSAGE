# DialogDraw: Image Generation and Editing System Based on Multi-Turn Dialogue

Shichao Ma\*†, Xinfeng Zhang†, Zeng Zao‡, Bai Liu, Changjie Fan, Zhipeng Hu

Fuxi AI Lab, NetEase Inc. mashichao@mail.ustc.edu.cn, {zhangxinfeng01, hzzhaozeng, hzliubai, fanchangjie, zphu} $@$ corp.netease.com

# Abstract

In recent years, diffusion modeling has shown great potential for image generation and editing. Beyond single-model approaches, various drawing workflows now exist to handle diverse drawing tasks. However, few solutions effectively identify user intentions through dialogue and progressively complete drawings. We introduce DialogDraw, which facilitates image generation and editing through continuous dialogue interaction. DialogDraw enables users to create and refine drawings using natural language and integrates with numerous open-source drawing workflows and models. The system accurately recognizes intentions and extracts user inputs via parameterization, adapts to various drawing function parameters, and provides an intuitive interaction mode. It effectively executes user instructions, supports dozens of image generation and editing methods, and offers robust scalability. Moreover, we employ SFT and RLHF to iterate the Intention Recognition and Parameter Extraction Model (IRPEM). To evaluate DialogDraw’s functionality, we propose DrawnConvos, a dataset rich in drawing functions and command dialogue data collected from the open-source community. Our evaluation demonstrates that DialogDraw excels in command compliance, identifying and adapting to user drawing intentions, thereby proving the effectiveness of our method.

artistic creation but also unveils the vast potential of these technologies.

Simultaneously, there is a strong demand for image generation and continuous editing capabilities. Creating exceptional art often requires repeated adjustments. Currently, users typically need to manually download various models and workflows to iteratively refine their creations, a tedious process. The advent of conversational large language models (LLMs) has introduced a more streamlined and intuitive approach. For example, DALL-E 3 (Betker et al. 2023) demonstrates how multi-round dialogue systems facilitate concise and clear image generation and editing. This method has gained public acceptance and is driving research in multiround dialogue-based drawing techniques.

# 1 Introduction

Recently, significant advancements have been achieved in the field of image generation using diffusion models (Ho, Jain, and Abbeel 2020; Song and Ermon 2020; Podell et al. 2023). These large-scale text-to-image models can synthesize high-quality, diverse images from concise textual prompts. The adoption of various diffusion models for a wide array of applications, including image generation, editing, and creative work, is on the rise. Furthermore, the open-source community centered around diffusion models is thriving. Platforms like Civitai (Civitai 2022) and OpenArt (OpenArt 2022) are particularly noteworthy, where numerous expert users share models and workflows. This collaborative environment not only fosters the advancement of

Most current drawing workflows that incorporate Large Language Models (LLMs) primarily rewrite and expand the user’s initial input before feeding it into the text model to generate images. This approach often fails to capture the user’s intent, such as making subtle adjustments or converting styles. Works like InstructPix2Pix (Brooks, Holynski, and Efros 2023), InstructDiffusion (Geng et al. 2024), and DialogPaint (Wei et al. 2023) rely on instruction editing through dialogue but often depend on a single model, limiting their ability to understand instructions and achieve effective edits. Moreover, there is limited research on generating and editing images through multi-round dialogues. For instance, DialogPaint (Wei et al. 2023) can only edit the original image, while DialogGen (Huang et al. 2024) regenerates prompts for dialogues, compromising image consistency. User needs are diverse, as evidenced by open-source communities like OpenArt (OpenArt 2022) and Civitai (Civitai 2022). Many existing instruction editing methods and single-model solutions struggle to meet these varied needs.

Therefore, continuous development and expansion of pipelines are crucial. Currently, there are hundreds of mature pipelines available on OpenArt (OpenArt 2022). The challenge is integrating these pipelines and models with LLM methods to accurately transform natural language descriptions into clear intents, accommodating diverse inputs to create rich and varied artworks. In this paper, we have developed a multi-turn dialogue-based drawing generation and editing system called DialogDraw. This conversational system is designed to generate, understand, and continuously edit images. Our system primarily comprises the In

#1 Can you draw me a #3 Okay, can I have this #5 Convert to anime   
， · cute cat? cat at the beach? style. Intention: SDXL Intention: Change Background++ Intention: animal 2 anime Param: (A cute cat) Param: (at the beach,<image>) Param: (<image>)   
岛 #2 Describe the cat. 8 #4 Put sunglasses on #6 Oh, I like it! A yellow and white cat sitting on the ground Intention: InstructPix2Pix Thanks for the compliment Intention: VQA Param: (Put sunglasses on Param: (Describe the cat, <image>) Intention: Chat cat, <image>) Param: (Oh, I like it!)

tent Recognition and Parameter Extraction Model (IRPEM) and drawing pipelines. Utilizing a multi-turn dialogue multimodal model (MLLM), we created a comprehensive simulated multi-modal dialogue dataset to train our model. By leveraging diverse multi-modal inputs from users and our trained multi-modal intent recognition model, we can accurately infer users’ true intentions—whether they are making queries, generating images, or performing various editing operations and extracting the necessary input parameters for different models and pipelines.

Additionally, we introduce a dataset for training and evaluating multimodal generation and editing systems, named DrawnConvos. Utilizing ChatGPT (OpenAI 2022) and SDXL (Podell et al. 2023), we generated a dataset of multiround drawings through a set of automated processes encompassing image generation, editing, and visual question answering (VQA). We also developed more comprehensive metrics to evaluate the adherence to multi-round drawing instructions and the accuracy of multi-round intent switching. In our evaluation, we compare our approach with recent work in the industry to demonstrate its effectiveness and practicality.

In summary, our contributions are primarily as follows:

• We propose DialogDraw, the first system to combine multiple pipelines for drawing and editing images in multi-turn dialogues, composed of the Intent Recognition and Parameter Extraction Model (IRPEM) and various drawing abilities.   
• We create a new dataset named DrawnConvos, a dataset of multi-round dialogues including image generation and editing, which incorporates numerous open-source workflows and models. Using this dataset, we apply SFT and RLHF methods for IRPEM training.   
• We introduce a benchmark for DialogDraw in multiround drawing scenarios, including metrics like Multiturn VQA Score, Multi-turn CLIP Similarity, and Instruct Edit Coherence. Extensive experiments validate the effectiveness of our approach.

# 2 Related Work

# 2.1 Diffusion Models and Community

Diffusion-based models (Ho, Jain, and Abbeel 2020; Song and Ermon 2020) have demonstrated outstanding performance in image generation, offering enhanced stability and controllability. These models employ a forward process that involves adding Gaussian noise to the input image, followed by an inverse process that generates high-quality images with intricate details and diversity from random Gaussian noise. The latent diffusion model (LDM) (Rombach et al. 2022) has been introduced to shift the diffusion process from the pixel space to the latent space, significantly improving both efficiency and image quality. There are already several diffusion model-based generation methods (GLIDE(Nichol et al. 2022), Imagen(Saharia et al. 2022), Stable Diffusion(Rombach et al. 2022), SDXL(Podell et al. 2023), Controlnet(Zhang, Rao, and Agrawala 2023), T2I-Adapter(Mou et al. 2024)) and editing (InstructPix2Pix (Brooks, Holynski, and Efros 2023), InstructDiffusion (Geng et al. 2024), etc.), many of which also have been integrated in open-source communities.

The open-sourcing of stable diffusion has led to the emergence of numerous derivative models and workflow communities. Notably, Stable Diffusion WebUI and ComfyUI have become popular frameworks for image generation. Additionally, communities like Civitai (Civitai 2022) and OpenArt (OpenArt 2022) have flourished, allowing users to customize and edit images based on the open-source platform.

# 2.2 Large Language Models

Large language models(Brown et al. 2020; Shuster et al. 2022; Wei et al. 2022; Zhang et al. 2019) have been widely studied in recent years, with the capability to chat with humans fluently. Models such as GPT-3 (Brown et al. 2020) can generate simulated data according to given samples, which is a convenient way to gather language data in a specific format and finetune other language models. Conversation-oriented language models like DialoGPT (Zhang et al. 2019), Meena (Adiwardana et al. 2020),

(a) Pipeline Database (b) DrawnConvos   
[ld,Type,Name,Description,Input] Pipeline GPT-4 Database   
[0001, Model, SDXL, generate pictures from texts,(prompt)) generated GPT-4 prompts [0020, Model, InstructPix2Pix, Dialog SDXL Edit pictures based on single round text,(prompt,<image>)] Text-lmage 2-3 rounds Data structuring pairs 3+ rounds Workflow 8 Model 日 DrawnConvos

BlenderBot (Roller et al. 2020), and ChatGPT (OpenAI 2022) have shown exceptional performance in various conversational tasks.

Multimodal large models (Chen et al. 2020; Ramesh et al. 2021; Jia et al. 2021; Singh et al. 2022; Alayrac et al. 2022; Li et al. 2022; Bai et al. 2023) integrate text, image, and audio data, enabling cross-modal information processing and understanding. CLIP (Radford et al. 2021) and DALL-E (Ramesh et al. 2021) achieved significant advancements in image generation and cross-modal retrieval by training on large-scale image and text data. Recent multimodal models like FLAVA (Singh et al. 2022), MUM (Jia et al. 2021), Qwen-VL(Bai et al. 2023) and Hunyuan (Li et al. 2024) exhibit excellent performance across more modalities and tasks, pushing the boundaries of artificial intelligence in handling complex information.

# 2.3 Dialog-based for Drawing

Research on generating and editing images through multiturn dialogue has garnered increasing attention in recent years. Early works (Chen et al. 2018) explored modifying image attributes via natural language instructions and dialogue interactions. Subsequently, a GAN-based model utilized sequential attention mechanisms to edit images based on conversational inputs (Cheng et al. 2020), laying the groundwork for interactive image editing tools.

Advancements such as ChatEdit (Cui et al. 2023), DialogPaint (Wei et al. 2023), and DialogGen (Huang et al. 2024) have further propelled the field. ChatEdit focuses on facial image editing through dialogue and includes a constructed dataset for its studies. DialogPaint bridges conversational interactions with image editing, allowing users to modify images through natural dialogue. DialogGen, a multi-modal interactive dialogue system, addresses multi-turn, multi-modal image generation tasks, showcasing the potential of crossmodal interaction in image generation.

# 3 Methodology

The core of DialogDraw encompasses the structured data construction of the drawing workflows and models, the dialogue data construction, and the development of image generation and editing based on continuous dialogue.

# 3.1 Construction of DrawnConvos

First, we need to obtain the drawing pipelines (including workflows and models), their corresponding functional descriptions, and input parameters. We collected approximately 20 pipelines from OpenArt (OpenArt 2022) and Civitai (Civitai 2022), classifying them based on the number of downloads and varying functions. This classification ensures broad coverage of different drawing functionalities and promotes higher user engagement.

Fig. 2(a) illustrates how to build structured data for each pipeline. Initially, we assign each drawing pipeline an Id, Name, and Type. For the Description, we use GPT4 (Achiam et al. 2023) to expand the original title based on the title of each pipeline and the existing page information. Finally, we define the Input parameters for each pipeline according to the downloaded pipeline. For instance, the ”animal 2 anima” pipeline requires 1 input parameter $( < i m a g e > )$ , while the ”InstructPix2Pix” pipeline requires 2 input parameters $( p r o m p t , < ~ i m a g e ~ > )$ . At present, In our skill pipeline library, aside from the 20 mentioned, there are two more for non-image tasks: VQA for describing images, and Chat for conversing with users.

The construction of our dialogue dataset is primarily depicted in Fig. 2(b). Initially, following specific guidelines, we utilized GPT-4 to generate 500 prompts covering various categories such as people, animals, and landscapes. Then, using SDXL, we created corresponding images for these prompts, yielding 500 (prompt, image) pairs. Subsequently, based on these image-text pairs, we generated dialogues with GPT-4, crafting 3,000 single-turn, 6,000 twoto-three-turn dialogues, and 1,000 dialogues with more than three turns, totaling 10,000 dialogues. It should be noted that each multi-turn dialogue was constructed based on a single (prompt, image) pair.

This dataset is named “DrawnConvos.” Our approach to dataset construction differs from others in that our model’s primary output includes pipeline names and their corresponding parameters; image generation and editing occur within these pipelines. We then randomly divided DrawnConvos into DrawnConvos(SFT), DrawnConvos(HF), and DrawnConvos(TEST) in a 6:3:1 ratio.

DrawnConvos(SFT) In DrawnConvos(SFT), the distribution of dialogue turns still roughly approximates $30 \%$ single-turn dialogues, $60 \%$ $) \% 2 \mathrm { - } 3 \ \mathrm { t u r n }$ dialogues, and $10 \%$ dialogues exceeding three turns. It is used for Supervised Fine-Tuning (SFT) as a dataset for intent recognition and parameter identification to train IRPEM(SFT).

DrawnConvos(RLHF) In the second phase, $\mathbf { I R P E M } _ { ( \mathrm { S F T } ) }$ is prompted with prompts $x$ to generate pairs of answers $( y _ { 1 } , y _ { 2 } ) \sim \pi _ { \mathrm { S F T } } ( y | x )$ . These pairs are then presented to human labelers who express a preference for one answer over the other, denoted as $y _ { w } \succ y _ { l }$ for a given $x$ , where $y _ { w }$ and $y _ { l }$ represent the preferred and less preferred completions, respectively, among the set $( y _ { 1 } , y _ { 2 } )$ . Notably, deviating from the previous strategy (Rafailov et al. 2024), when neither of the answer pairs $( y _ { 1 } , y _ { 2 } )$ meets expectations, we modify them to better align with human preferences. Subsequently, we construct an offline dataset of preferences $D$ consisting of elements $\{ ( x ^ { ( i ) } , y _ { w } ^ { ( i ) } , y _ { l } ^ { ( i ) } ) \}$ for $i = 1 , \ldots , N$ .

![](images/e9d8d4e96c3a749f5a5c5ef84c445c6809eb4334c9b114e4cfbc304ac9965c21.jpg)  
Figure 3: Overview of DialogDraw (a) DialogDraw’s inference procedure: we extract user intent and parameters from dialogues and invoke the corresponding pipeline to edit the images. (b) DialogDraw’s training procedure: using a two-stage training process and multiple pipelines to construct the system.

DrawnConvos(TEST) Similar to DrawnConvos(RLHF), DrawnConvos(TEST) has also been annotated by humans. It is used to test the effectiveness of various models.

# 3.2 Construction of DialogDraw

As shown in Fig. 3(b), DialogDraw is a system for image generation and editing, composed of a model obtained through Reinforcement Learning from Human Feedback (Christiano et al. 2017; Stiennon et al. 2020; Bai et al. 2022) and various pipelines. Specifically, the process is divided into two steps: The first step involves supervised fine-tuning (Gunel et al. 2020; Yu et al. 2020) using the DrawnConvos(SFT) dataset to train the model’s basic intent and parameter recognition capabilities, resulting in the $I R P E M _ { ( S F T ) }$ . Utilizing $I R P E M _ { ( S F T ) }$ , the model’s behavior is continuously adjusted through the Direct Proximal Optimization reinforcement learning framework (DPO) (Rafailov et al. 2024), culminating in the $I R P E M _ { ( R L H F ) }$ . Our system recognizes the user’s intent and parameters by $I R P E M _ { ( R L H F ) }$ and then calls the appropriate pipeline and passes the appropriate parameters to edit and generate the image.

IRPEM(SFT) In the first phase of fine-tuning, the finetuning method is LoRA (Hu et al. 2021), and the dataset is

DrawnConvos(SFT). The loss function is a cross-entropy loss function for predicting the next word:

$$
L = - \sum _ { t = 1 } ^ { T } \log \left( p ( w _ { t } | w _ { 1 : t - 1 } ) \right)
$$

where $p \big ( w _ { t } | w _ { 1 : t - 1 } \big )$ is the probability of word $\boldsymbol { w } _ { t }$ given the previous words $w _ { 1 : t - 1 }$ , and $T$ is the sequence length.

After the first step, we obtain the $I R P E M _ { ( S F T ) }$ . This model analyzes user questions to determine intent and corresponding parameters, providing a foundation for optimization.

$\mathbf { I R P E M _ { ( R L H F ) } }$ In this phase, the model builds upon the previous $I R P E M _ { ( S F T ) }$ , integrating DrawnConvos(RLHF), and continuously refines the model’s behavior through the DPO reinforcement learning framework.

DrawnConvos(RLHF) can be represented as $\begin{array} { r l } { D } & { { } = } \end{array}$ i(N), y(wi , yl( )}iN=1, where we can parameterize a reward model $r _ { \phi } ( x , y )$ and estimate the parameters via maximum likelihood. In reinforcement learning, we frame the problem as a binary classification task, where the negative loglikelihood loss is given by

$$
\begin{array} { r } { \mathcal { L } _ { R } ( r _ { \phi } , D ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim D } \left[ \log \sigma ( r _ { \phi } ( x , y _ { w } ) - r _ { \phi } ( x , y _ { l } ) ) \right] } \end{array}
$$

where $\sigma$ is the logistic function.

In the DPO strategy, we have the probability of human preference data in terms of the optimal policy rather than the reward model, we can formulate a maximum likelihood objective for a parametrized policy $\pi _ { \boldsymbol { \theta } }$ . Analogous to Equation 2, our policy objective becomes:

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { D P O } } ( \pi _ { \theta } ; \pi _ { \mathrm { r e f } } ) = - \mathbb { E } _ { ( x , y _ { w } , y _ { l } ) \sim D } } \\ & { \qquad \biggl [ \log \sigma \left( \beta \log \frac { \pi _ { \theta } \left( y _ { w } | x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { w } | x \right) } - \beta \log \frac { \pi _ { \theta } \left( y _ { l } | x \right) } { \pi _ { \mathrm { r e f } } \left( y _ { l } | x \right) } \right) \biggr ] } \end{array}
$$

Table 1: Comparison of quantitative analysis indicators for different models. ✓indicates that the model used this strategy during training, while $\times$ indicates that it did not. $\uparrow$ indicates that a higher score is better, and bold indicates the best results. The Instruct Edit Coherence score fully aligns with human evaluation.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">Task</td><td colspan="4">Quantitative Metrics</td></tr><tr><td>Image</td><td>Image</td><td>VQA</td><td>Chat</td><td>Multi-turn VQA Multi-turn CLIP Instruct Edit Human</td><td></td><td></td><td></td></tr><tr><td></td><td>Generate</td><td>Edit</td><td></td><td></td><td>Score ↑</td><td>Similarity ↑</td><td>Coherence ↑ Score↑</td><td></td></tr><tr><td>SEED-LLaMA-8B(Ge et al. 2023)</td><td>7</td><td>×</td><td>√</td><td>√</td><td>0.7646</td><td>0.6719</td><td>0.7229</td><td>0.7117</td></tr><tr><td>SEED-LLaMA-14B(Ge et al. 2023)</td><td>√</td><td>X</td><td>√</td><td>√</td><td>0.7781</td><td>0.6865</td><td>0.7369</td><td>0.7432</td></tr><tr><td>GPT-4(Achiam et al. 2023)</td><td>√</td><td>X</td><td>√</td><td>√ √</td><td>0.8523</td><td>0.7412 0.7217</td><td>0.8023 0.7765</td><td>0.8218</td></tr><tr><td>DialogGen(Huang et al. 2024)</td><td>√</td><td></td><td>√</td><td>√</td><td>0.8213</td><td>0.7835</td><td></td><td>0.8081</td></tr><tr><td>DialogDraw(Ours)</td><td>√</td><td>√</td><td>√</td><td></td><td>0.8491</td><td></td><td>0.8196</td><td>0.8494</td></tr></table></body></html>

By employing this method, we establish an implicit reward through a different parameter setup, with the optimal policy being $\pi _ { \boldsymbol { \theta } }$ . After this step, we obtain the $I R P E M _ { ( R L H F ) }$ .

DialogDraw The model for parameter and intent recognition, denoted as $I R P E M _ { ( R L H F ) }$ , is represented by $M _ { D }$ . Subsequently, the various pipelines are represented by $P$ . Thereupon, our DialogDraw system can be expressed as the synthesis of $M _ { D }$ and $P$ , represented mathematically as:

$$
\mathrm { D i a l o g D r a w } = M _ { D } \oplus P
$$

# 3.3 The Benchmark of DialogDraw

Multi-turn VQA Score. The VQA Score(Lin et al. 2024) serves as a robust metric for assessing the alignment between model-generated images and their corresponding prompt texts. Its approach is straightforward: it measures the generative likelihood of responses to simple questions in an end-to-end manner.

While effective for single-turn dialogues, this method falls short in evaluating the alignment across multiple dialogue turns. To overcome this limitation, we introduce the multi-turn VQA Score. The concept is as follows: for a dialogue represented as $( t e x t _ { i } , i m a g e _ { i } )$ , taking $i = 3$ for illustration, the calculation proceeds as follows: - The initial dialogue round uses the VQA Score for $( t e x t _ { 1 } , i m a g e _ { 1 } )$ . - The second round calculates the score for the integrated text $F ( t e x t _ { 1 } , t e x t _ { 2 } )$ with image2, where ”F” signifies the concatenation of $t e x t _ { 1 }$ and $t e x t _ { 2 }$ using GPT-4 to form a comprehensive prompt for multiturn dialogues. - The third round extends this approach to $( ( F ( t e x t _ { 1 } , t e x t _ { 2 } , t e x t _ { 3 } ) , i m a g e _ { 3 } )$ ).

The multi-turn VQA Score $S _ { \mathrm { M } _ { \mathrm { - } } \mathrm { V Q A } }$ is encapsulated by the formula:

$$
S _ { \mathrm { M } . \mathrm { { v Q A } } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } V Q A ( F ( t e x t _ { 1 } , t e x t _ { 2 } , \ldots , t e x t _ { i } ) , i m a g e _ { i } )
$$

Here, $n$ denotes the total number of dialogue turns, $F$ is the function for integrating multi-turn text, and $V Q A$ is the scoring mechanism.

Multi-turn CLIP Similarity. To gauge the consistency of image generation and editing throughout multi-turn dialogues, we propose an additional metric utilizing the CLIP

model (Radford et al. 2021). This metric, $S _ { \mathrm { M } \_ \mathrm { C L I P } }$ , assesses the similarity between each pair of consecutive images:

$$
S _ { \mathrm { M . C L I P } } = \frac { 2 } { n ( n - 1 ) } \sum _ { i = 2 } ^ { n } \sum _ { j = 1 } ^ { i - 1 } \mathbf { C L I P } \left( \mathrm { i m a g e } _ { j } , \mathrm { i m a g e } _ { i } \right)
$$

This formula accounts for all possible pairs of images across the dialogue turns, with $n$ being the total number of turns and CLIP the function that computes the similarity between two images.

Instruct Edit Coherence. It’s important to clarify that this metric is predicated on the iterative editing of the same image. If a new image is generated in a particular round, the aforementioned formula would not be applicable.

We further refine our assessment by combining the multiturn VQA Score and the CLIP Similarity, assigning weights $q _ { 1 }$ and $q _ { 2 }$ respectively. The weighting is adjusted based on the nature of the dialogue round: - For image editing rounds, the weights are balanced at $q _ { 1 } : q _ { 2 } = 0 . 5 : 0 . 5$ . - For rounds involving new image generation, the weights are set to $q _ { 1 }$ : $q _ { 2 } = 1 : 0$ .

The final metric to measure the multi-turn dialogue’s command understanding capability and the consistency of image generation and editing, Instruct Edit Coherence (IEC), is defined as:

$$
\begin{array} { r } { I E C = q _ { 1 } \times S _ { \mathrm { M _ { \mathrm { - } C L I P } } } + q _ { 2 } \times S _ { \mathrm { M _ { \mathrm { - } C L I P } } } } \end{array}
$$

It should be noted that our system includes ”Chat” and ”VQA” skills, designed for interactions not related to image generation or editing. In such cases, these instances are excluded from the scoring calculation.

# 4 Experiments 4.1 Experimental Setup

All our experiments are performed on four NVIDIA A100 GPUs using the PyTorch framework. During the training phase, we initialized our model with a pre-trained QwenVL (Bai et al. 2023) model. In the first phase, we trained the model for 50 epochs using DrawnConvos(SFT) to obtain $I R P E M _ { ( S F T ) }$ . The second phase was built upon the first, where we further trained the model for another 50 epochs using DrawnConvos(RLHF) to achieve IRPEM(RLHF). Both phases utilized the AdamW optimizer with weight decay set to 0.1 and 0.05, respectively. The initial learning rate for both stages is initialized as 1e-5.

![](images/18137189075501b832122ec7e77d66366fccfdeade09a5ed6b739f5bc8cdfdeb.jpg)  
Figure 4: Visualization of Dialog Outputs. The above are three multi-round dialogues. In dialogue1, first, a painting of autumn scenery is wanted, then it’s asked to be in Van Gogh’s style. Dialogue2 starts with a request for a running dog image, followed by asking to draw it, changing the background to a spring garden, and finally converting to anime style. For dialogue3, initially, a rabbit on the moon is requested, then sunglasses are added, and the background is changed to the beach.

# 4.2 Analysis and Comparisons

Table 1 shows the results of different models on the test set, and we can analyze their performance using the following evaluation metrics. In Fig. 4, we present images generated by different models.

Multi-turn VQA Score. In the Multi-turn VQA Score metric, as shown in Table 1, DialogDraw ranks second with a score of 0.8491, just behind GPT-4. This metric assesses the model’s ability to transform text into images. DialogDraw utilizes SDXL for drawing and does not particularly pursue drawing capabilities, whereas GPT-4 excels in this area, hence its higher score. However, unlike the previous single-turn text-to-image generation, this metric considers the integration of previous text rounds, emphasizing the significance of contextual relationships—an area where DialogDraw excels. Consequently, DialogDraw scores higher than DialogGen and the other two models. In Fig. 4, DialogGen’s lower score results from inaccuracies in understanding specific texts. For example, in dialogue 2, where the user intended to receive an image of a specific object, DialogGen incorrectly provides only the background without the object.

Multi-turn CLIP Similarity. DialogDraw secures the top position with a score of 0.7835, which is $5 . 7 1 \%$ higher than the second place. As shown in Fig. 4, other models such as DialogGen and seed-llama essentially generate new images for editing tasks rather than editing the original image; GPT4 behaves similarly, and its inability to perform certain image editing instructions, like a cutout, leads to a lower score. DialogDraw, with its capability to directly edit the original image, achieves the highest score in this category, indicating the best consistency in image editing.

Instruct Edit Coherence. DialogDraw leads with a score of 0.8196. This score is a composite of the previous two metrics and reflects the model’s understanding of instructions and the consistency of image generation and editing in multi-turn dialogues. DialogDraw’s superior performance here is attributed to its excellent consistency in multi-turn image generation and editing; moreover, GPT-4 and DialogGen rank second and third, respectively. It is noteworthy that SEED-LLaMA performs poorly across these metrics.

Hunan Score. We invited users to rate image generation and editing in dialogues from different models. In the evaluation by users, two criteria are employed to assess the performance: the model’s ability to comprehend instructions within multi-turn dialogues, and the consistency between image generation and editing. Both metrics are given equal weight. Randomly picking 100 dialogues, we received feedback from 50 users. As shown in Table 1, DialogDraw is favored over the Baseline, ranking higher in relevance and user preference. It scores 0.8350 for understanding dialogue commands, placing it just below GPT-4. For image consistency, it achieves a 0.8638 score. Overall, it tops the Human Score category with 0.8494, validating the Instruct Edit Coherence(IEC) metric.

# 4.3 Ablation Experiments

In this section, we have constructed a series of ablation studies to further analyze IRPEM and the intent and parameter recognition framework that we have proposed.

Table 2: Results of ablation study to analyze different components.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">Training Method</td><td colspan="3">Quantitative Metrics</td></tr><tr><td>SFT</td><td>DPO</td><td>Intent Switch Accuracy ↑</td><td>Parameter Accuracy ↑</td><td>Structural Conformity ↑</td></tr><tr><td>Qwen-VL-zero-shot</td><td>×</td><td>×</td><td>0.702</td><td>0.754</td><td>0.738</td></tr><tr><td>w/ {Supervised Fine-Tuning }</td><td>×</td><td>√</td><td>0.902</td><td>0.870</td><td>0.932</td></tr><tr><td>w/ {Direct Preference Optimization }</td><td>√</td><td>×</td><td>0.874</td><td>0.857</td><td>0.865</td></tr><tr><td>IRPEM(RLHF)</td><td>√</td><td>√</td><td>0.924</td><td>0.892</td><td>0.953</td></tr></table></body></html>

Table 3: Ablation study on the effects of different strategies for multi-turn dialogue image generation and editing.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="4">Quantitative Metrics</td></tr><tr><td></td><td>Multi-turn VQA Multi-turn CLIP Instruct Edit Human</td><td></td><td></td></tr><tr><td></td><td>Score↑</td><td>Similarity ↑ 0.7317</td><td>Coherence ↑ Score ↑</td><td></td></tr><tr><td>Qwen-VL-zero-shot+Pipelines w/ {Supervised Fine-Tuning} + Pipelines</td><td>0.7613 0.8287</td><td>0.7707</td><td>0.7480 0.8026</td><td>0.7542 0.8228</td></tr><tr><td>w/{Direct Preference Optimization}+Pipelines</td><td>0.8082</td><td>0.7584</td><td>0.7858</td><td>0.8041</td></tr><tr><td>IRPEM(RLHF) + Pipelines</td><td>0.8491</td><td>0.7835</td><td>0.8196</td><td>0.8494</td></tr></table></body></html>

Quantitative Metrics. To present the results more intuitively, we have quantified the outputs of various models on DrawnConvos(TEST), focusing on three key metrics:

• Intent Switch Accuracy (ISA). ISA measures the match between the model’s identified intents and the test set’s standard intents:

$$
\mathrm { I S A } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } 1 _ { \mathrm { c o r r e c t \mathrm { - } i n t e n t } _ { i } }
$$

Here, ISA represents the Intent Switch Accuracy, $n$ is the total number of dialogues in the test set. The indicator 1correct intent is 1 for a match and 0 for a mismatch.   
• Parameter Accuracy. This metric’s evaluation by fifty users focuses on three criteria: key term inclusion, condition adherence, and negative vocabulary check, weighted as 4:3:3.   
• Structural Conformity. Similarly, this metric is evaluated by human users, focusing on three primary aspects: whether the output conforms to the specified JSON format and the absence of repetition or garbled text, with weights allocated as 6:4.

Component Analysis. We conduct an ablation study on the component, as detailed in Table 2. Our findings indicate: (i) Compared to Qwen-VL-zero-shot, the model’s performance after training has greatly improved in all metrics, with over $10 \%$ increase, especially in Structural Conformity, highlighting the fine-tuned model’s strength in recognizing intent and parameters. (ii) The standalone SFT strategy occasionally generates outputs with uninterpretable repetitions, garbled text, or negative sentiments. Models trained with DPO regard such responses as suboptimal and suppress them (Line 3 vs. Line 4). (iii) For DPO strategy, IRPEM(SFT) is a better base model than Qwen-VL-zero-shot. As it minimizes the gap between model outputs and preference data, leading to responses more attuned to human preferences (Line 2 vs. Line 4).

Effectiveness Analysis. In the context of multi-turn dialogue for image generation and editing, we analyze these strategies, with the results presented in Table 3. This highlights the significance of these strategies in enhancing image editing during multi-turn conversations. Notably, the $I R P E M _ { ( R L H F ) }$ integrated with Pipelines outperforms standalone DPO and SFT strategies by $2 . 1 2 \%$ and $4 . 3 0 \%$ in the IEC metric, showing significant gains over zero-shot strategies. This indicates our method has higher-quality images in the context of multi-turn dialogues.

# 5 Conclusion

In this paper, we introduce DialogDraw, a conversational image generation and editing system that also supports VQA and text dialogue functionality. Our approach stands out by accurately understanding and structuring users’ natural language descriptions, enabling seamless integration with various mainstream open-source drawing models and pipelines. The system’s high scalability is driven by continuous iterations of our proposed intent understanding model. We developed a multimodal dataset featuring multi-turn dialogues with rich drawing instructions, including image generation and editing. This dataset, curated from top pipelines in the open-source community, is used to train our intent recognition model. Experimental results show that DialogDraw outperforms current mainstream dialogue-based drawing models in intent recognition accuracy, drawing continuity, and consistency, as evidenced by both qualitative and quantitative results. However, our system has some limitations. Currently, it integrates only the 20 most popular pipelines and models, and we have made limited progress in enhancing VQA and text dialogue capabilities. In the future, we plan to expand the system’s drawing capabilities and improve intent accuracy through human feedback to better align with user preferences.
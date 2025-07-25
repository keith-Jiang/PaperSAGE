# CoMT: A Novel Benchmark for Chain of Multi-modal Thought on Large Vision-Language Models

Zihui Cheng1,2\*, Qiguang Chen3\*, Jin Zhang3, Hao Fei4, Xiaocheng Feng3, Wanxiang Che3, Min $\mathbf { L i } ^ { 1 }$ , Libo $\mathbf { Q i n } ^ { 1 , 2 \dag }$ ,

1School of Computer Science and Engineering, Central South University, China 2Key Laboratory of Data Intelligence and Advanced Computing in Provincial Universities, Soochow University, China 3Research Center for SCIR, Harbin Institute of Technology, Harbin, China 4National University of Singapore, Singapore czh up@csu.edu.cn, qgchen $@$ ir.hit.edu.cn

# Abstract

Large Vision-Language Models (LVLMs) have recently demonstrated amazing success in multi-modal tasks, including advancements in Multi-modal Chain-of-Thought (MCoT) reasoning. Despite these successes, current benchmarks still follow a traditional paradigm with multi-modal input and text-modal output, which leads to significant drawbacks such as missing visual operations and vague expressions. Motivated by this, we introduce a novel Chain of Multi-modal Thought (CoMT) benchmark to address these limitations. Different from the traditional MCoT benchmark, CoMT requires both multi-modal input and multi-modal reasoning output, aiming to mimic human-like reasoning that inherently integrates visual operations. Specifically, CoMT consists of four categories: (1) Visual Creation, (2) Visual Deletion, (3) Visual Update, and (4) Visual Selection to comprehensively explore complex visual operations and concise expression in real scenarios. We evaluate various LVLMs and strategies on CoMT, revealing some key insights into the capabilities and limitations of the current approaches. We hope that CoMT can inspire more breakthroughs on introducing multi-modal generation into the reasoning process.

# Code — https://github.com/czhhzc/CoMT Extended version — https://arxiv.org/abs/2412.12932

Question: Rationale:   
Prove that the sum of Missing Step 1: First, label the angles   
the interior angles of 1 Visual as ∠1, ∠2 and ∠3 separately.   
the triangle below is Operation Step 2: Secondly, observe the   
180 degrees. 2 Vague, $\angle 1 = 4 0 ^ { \circ }$ ， $\angle 2 = 7 0 ^ { \circ }$ ， $\angle 3 = 2 5 ^ { \circ }$ 180°? Expression Step 3: The sum of the interior 。 angles is $4 0 ^ { \circ } + 7 0 ^ { \circ } + 2 5 ^ { \circ } = 1 3 5 ^ { \circ }$ A Answer: I can't prove it. LVLMs (a) Traditional Multi-modal Chain-of-Thought   
Question: Rationale:   
Prove that the sum of Step 1: Draw auxiliary lines   
the interior angles of based on the original image.   
the triangle below is   
180 degrees. 180°3 Expression Step 2: Within the new image, 。 Cv ∠2=∠4,∠3=∠5,therefore,∠1 福 +∠2+∠3=∠1+∠4+∠5=180 A Answer: Thesum is180 LVLMs (b) Chain of Multi-modal Thought

# 1 Introduction

Recently, large vision-language models (LVLMs) have achieved remarkable success across various multi-modal tasks (Liu et al. 2024b; Zhu et al. 2023; Qin et al. 2024b; Zhang et al. 2024b; Fei et al. 2024b). In addition, LVLMs have also emerged with amazing capabilities, especially the capability of chain-of-thought (CoT) reasoning, which can perform step-by-step reasoning (Lu et al. 2022; Chen et al. 2024b; Xu et al. 2024; Fei et al. 2023). Specifically, Zhang et al. (2023) first formally introduce the concept of Multimodal-CoT (MCoT) and extend it into a rationalizinganswering stages paradigm. Wang et al. (2024a) propose TSciQ to distill the advanced large language models (LLMs)

to smaller models for better MCoT reasoning. Building on this foundation, Zheng et al. (2024) propose DDCoT, utilizing advanced LLMs to split questions into a series of subquestions and then answer them by LVLMs. Mondal et al. (2024) further inject the knowledge graph information into the MCoT reasoning process, reducing the hallucinations of LLMs. He et al. (2024) devise a novel latent space learning approach to acquire image features through diffusion processes, achieving more complex CoT reasoning capabilities.

While remarkable success has been witnessed in MCoT, current MCoT benchmarks still follow a traditional paradigm that reads multi-modal input but can only produce single-modal reasoning output. Such a paradigm lacks integrated multi-modal reasoning output, leading to the following issues:

Table 1: Comparison of CoMT and multi-modal related datasets. $\mathbf { \Psi } ^ { 1 } \# \mathbf { X }$ : the size of X; VO: supported visual operations; MMCoT: the ratio of samples with multi-step MCoT (MMCoT) in the datasets; MT: Multi-modal Thought. Avg. MT Step: The average step of Multi-modal Thought. Our benchmark has the following two advantages: (1) abundant rationale containing multi-modal thought, (2) more comprehensive and fine-grained visual operation.   

<html><body><table><tr><td>Benchmark</td><td>#Question</td><td>#Image</td><td>#VO</td><td>MMCoT</td><td>MT</td><td>Avg. MT Step</td><td>Rationale</td></tr><tr><td>VCR (Zellers et al. 2019)</td><td>290k</td><td>99,904</td><td>×</td><td>~4%</td><td>×</td><td>×</td><td></td></tr><tr><td>A-OKVQA (Schwenk et al. 2022)</td><td>24,903</td><td>23.692</td><td>×</td><td>~21%</td><td>×</td><td>×</td><td></td></tr><tr><td>KI-VQA (Li et al. 2023b)</td><td>4,290</td><td>4,189</td><td>×</td><td>~17%</td><td>×</td><td>×</td><td>√</td></tr><tr><td>ScienceQA (Lu et al. 2022)</td><td>21,208</td><td>10,332</td><td>×</td><td>~8%</td><td>×</td><td>×</td><td><</td></tr><tr><td>MMMU(Yue etal.2023)</td><td>11,550</td><td>11,264</td><td>×</td><td>~8%</td><td>×</td><td>×</td><td><18%</td></tr><tr><td>MCoT (Chen et al. 2024b)</td><td>11,459</td><td>11,293</td><td>×</td><td>100%</td><td>×</td><td>×</td><td>√</td></tr><tr><td>CoMT (ours)</td><td>3853</td><td>14,801</td><td>4</td><td>100%</td><td></td><td>3.11</td><td>√</td></tr></table></body></html>

(1) Missing Visual Operations. Effective multi-modal reasoning often requires visual operations. However, traditional MCoT paradigms produce only textual reasoning outputs, which greatly hinders the multi-modal reasoning. As shown in Figure 1 (a), traditional methods can express operations in language, such as “label the angles”, but they fail to execute visual operations, omitting the actual image-processing procedure.

(2) Vague Expressions. The adage “a picture is worth a thousand words” highlights the limitations of text in conveying visual reasoning conditions. As shown in Figure 1 (a), phrases like “ $\angle 1 = 4 0 ^ { \circ } { } ^ { , }$ are imprecise in the absence of actual annotations, failing to accurately reflect the mapping relationship between angles and measures, thus leading to ambiguity and loss of visual information.

Actually, when humans perform reasoning, they naturally integrate images into the process: using visual thought for concrete, detailed reasoning while using textual thought for abstract, logical reasoning (Lehmann et al. 2010; Lin et al. 2024; Wu et al. 2024b). Take Figure 1 (b) as an example, LVLMs can accurately locate the specific angle by generating an annotated image. By labeling the angles and drawing auxiliary lines, LVLMs can perform clearer expressions and better multi-modal reasoning. Inspired by this, in this paper, we aim to explore a new MCoT paradigm that requires generating multi-modal reasoning outputs.

To fill this gap, we introduce a novel Chain of Multimodal Thought benchmark (CoMT). Unlike the traditional MCoT benchmarks, CoMT requires both multi-modal input and multi-modal reasoning output, aiming to enhance LVLMs’ performance in concise expression and complex visual operations in real-world scenarios. Specifically, CoMT contains four categories to comprehensively assess the ability of LVLMs to use multi-modal thought processes: (1) Visual Creation assesses the ability to generate images from scratch, thereby visualizing abstract problems; (2) Visual Deletion evaluates the removal of irrelevant information from given images; (3) Visual Update examines the integration of updated images while retaining prior information; (4) Visual Selection tests the selection of specific visual features for improved image comparison. The detailed comparisons and analyses are shown in Table 1.

We evaluate abundant representative LVLMs and prompting strategies on CoMT in extensive scenarios, yielding several key takeaways: (1) CoMT presents a significant challenge; nearly all zero-shot methods perform only marginally better than random, which demonstrates huge gaps compared with human performance. (2) In-context learning (ICL) has better hope on triggering LVLMs for better multimodal thought in CoMT. (3) Future advancements in CoMT should focus on integrating multi-modal generation, logical reasoning and visual operations into MCoT more effectively. Our main contributions are as follows:

• To our knowledge, this is the first work to establish a benchmark for chain of multi-modal thought (CoMT) in LVLMs, which encompasses four fundamental operations for comprehensive evaluation. • We evaluate various representative LVLMs and prompting strategies, revealing a huge performance gap between LVLMs and humans. Except for Gemini, nearly all LVLMs perform at random chance levels. • We explore in-context learning to enhance performance and highlight some future directions for integrating multi-modality into MCoT reasoning, hoping to provide insights for further research.

# 2 Benchmark Construction

We introduce $\mathrm { C o M T } ^ { 2 }$ , which aims to assess the ability of multi-modal thought, consisting of four types: Visual Creation (§2.1), Visual Deletion $( \ S 2 . 2 )$ , Visual Update (§2.3), and Visual Selection (§2.4). Specially, we design a specified question-answering template, which involves question, options, image, rationale, and answer, to standardize the format for all tasks within CoMT. More annotation details are shown in Technical Appendix C.

# 2.1 Visual Creation

An image is worth a thousand words. As shown in Figure 2 (a), visual creation tasks emphasize generating images from textual descriptions to improve multi-modal reasoning.

# Original Sample

Question: In △ABC, line BD is perpendicular AD …

Options: (A) $2 0 ^ { \circ }$ ; (B) 30°; (C) 60°; (D) 70° Answer: Step 1: Line BD is perpendicular AD … Step N: Consequently, ∠CBD = 90° - ∠A $= 7 0 ^ { \circ }$ . Therefore, the answer is D.

# Template-based Modification

# Modified Sample

Question: In △ABC, line BD is perpendicular AD,   
AD=DC, $\angle \mathbf { A } = 2 0 ^ { \circ }$ . The size of ∠CBD is?   
Options: (A) 20°; (B) 30°; (C) 60°; (D) 70°   
Rationale:   
Step 1: According to the given instructions, the   
image is depicted as follows:

![](images/aaeb0dcb37773fd392e1900f4ba3949f407d14bb0f0d2ab27865d3549504f621.jpg)

Step 2: Therefore, because line BD is … Step N+1: Consequently, $\angle C \mathbf { B } \mathbf { D } = 9 0 ^ { \circ }$ - $\angle \mathbf { A } = 7 0 ^ { \circ }$ . Answer: (D) $7 0 ^ { \circ }$

# Human Recheck & Visual Creation Assurance

# Final Sample

![](images/0b4b4cd53b73889f393e285c59e23efa91b3c48217366f1e3d2296e85e640d6a.jpg)  
Figure 2: The overall annotation process for four tasks of CoMT, which consists of (a)visual creation, (b)visual deletion, (c)visual update, and (d)visual selection.

Question: In $\bigtriangleup$ ABC, line BD is perpendicular and bisects AC, $\angle \mathbf { A } = 2 0 ^ { \circ }$ . The size of $\angle C \mathrm { B D }$ is? Options: (A) $2 0 ^ { \circ }$ ; (B) $3 0 ^ { \circ }$ ; (C) $6 0 ^ { \circ }$ ; (D) $7 0 ^ { \circ }$   
Rationale:   
Step 1: According to the given instructions…   
Step $\mathbf { \delta } N { + } I { : }$ Consequently, $\angle C \mathrm { B D } = 9 0 ^ { \circ }$ - ∠A $\mathbf { \Sigma } = \mathbf { \Sigma }$ $7 0 ^ { \circ }$ .   
Answer: (D) 70° (a) Visual Creation ipmeoapgle are visible in the Options: oStnetphe1:lef…t seildiemionfatthe tihmeafgaec.es for faces to the right … (A) 14 It reveals other 5 individuals (B) 24 … Step N: The count is 24.   
Bounding Box It reveals 5 individuals … Answer: (B) 24 (b) Visual Deletion Tangram D "blades tQanugersatimoni:n iWmhagtedsohesowthae SRtaetpio1n:al…e:analyze the part … oSftetphe2i:m…agetothwatrdesxhtihbei a…reas closer likeness to? that contains any given color.   
Annotation (OAp) oDnosg: ?blades "blades building blad (B) Windmill … Step N: … is Windmill. 4:building (C) a resemblance to blades. Answer: (B) Windmill   
whole: Windmill (c) Visual Update   
□ 1 cbQoeutuewnsteteionnto:h  iWmhdaigtff ir?s tchees Options: (A) 12; (B) 15 sSRetaevtpeir l1a:d :Ffeirsetn, pwaer ca…n get oStheepr2d:ifAfeftrernthpatr,tswewictahning…et ETe 电 POPOnO · We can identify □ □ other 3 i + differences… Difference We can identify 3 differences. Answer: (A) 12 Annotation (d) Visual Selection

• Original Dataset: We develop visual creation tasks based on the ${ \mathrm { G e o Q A } } +$ dataset (Cao and Xiao 2022), which includes geometric images and textual questions as input, with textual rationales as output.

• Template-based Modification: We first follow the template to modify the visual creation data. Specifically, we maintain the original question and option part from ${ \mathrm { G e o Q A } } +$ and split the whole response into rationale and the final answer. Furthermore, we reposition the image from question to the output rationale as visual thought, with step information supplemented.

• Human Recheck: To ensure the accurate reproduction of images, we manually augment the geometric description within the question by aligning with the image details.

# 2.2 Visual Deletion

In logical reasoning, it is crucial to eliminate redundant information and clarify the logical chain. By progressively removing visual features, LVLMs experience reduced confusion, enabling step-by-step reasoning for the final answer, as illustrated in Figure 2 (b).

• Original Dataset: We utilize the crowd-counting task from the JHU-CROWD $^ { + + }$ dataset (Sindagi, Yasarla, and Patel 2020), which includes images with numerous faces and corresponding boxing.   
• Step-by-Step Boxing: The most crucial aspect of crowdcounting is identifying human individuals where faces serve as a significant visual feature. To demonstrate the marking and removal of redundant visual features, we batch-mask faces based on the boxing provided, preparing for the next operation.

• Template-based Modification: We construct the com

Visual Creation Visual Deletion 5% Step Number of Multi-modal Thought Visual Update Visual Selection 5% Culture & Art Step Number of Rationale   
1,100 11% Mathematical Geometry 1500 1020 1008 Abstract Graph 1000   
1,000 932 28% 44% Human Activity 500   
900 893 Everyday Objects & Items 0 7% Landscape & Architecture 1 3 5 9 11 13 15 17 19 800 (c) Distribution of the number of steps (a) Distribution of different visual (b) Distribution of different image categories, in multi-modal thought and the number operations. all of which are classified based on CLIP. of steps in complete rationale.

Figure 3: Distribution of CoMT tasks across four types of image processing.

plete sample by following the CoMT template, involving inquiries about the people count in the image (question) and clarifications of the identified count (rationale), etc. The prepared images serve as the visual thought within the rationale.

# 2.3 Visual Update

Marking can help sort out the logic. LVLMs often make mistakes in reasoning due to forgetting visual features, while humans mitigate this by annotating images. Inspired by this, as illustrated in Figure 2 (c), we propose the Visual Update task to annotate the images step-by-step.

• Original Dataset: We leverage the KILOGRAM (Ji et al. 2022) dataset to implement tangram recognition, including the tangram image and labels of both individual pieces and the whole shape.   
• Tangram Annotation: For accurate assessments, we enhance the original tangram by applying different colors to each label category which consists of multiple individual pieces. After coloring, we explicitly annotate each category with label texts.   
• Template-based Modification: Finally, we follow the CoMT template to construct the whole sample and combine the enhanced images with the textual rationales to represent the multi-modal thoughts.

# 2.4 Visual Selection

Text cannot indicate the location intuitively. Accurately selecting among similar objects using text alone is challenging due to the inherent difficulty in precise location and difference descriptions. Following this intuition, we construct the Visual Selection task, as shown in Figure 2 (d).

• Original Dataset: We construct the task from the spotdiff3 dataset. This dataset provides pairs of similar images and corresponding difference annotations, requiring precise identification of differences between two images. • Step-by-Step Annotation: According to the annotations, we extract the distinct areas of image pairs in batches, keeping the same position and size as the original images. • Template-based Modification: We then supplement the textual section within the template and integrate corresponding images to construct a multi-modal rationale.

Table 2: Basic statistics of CoMT, including sample numbers, steps of rationale, length of rationale, and image number generated in CoT.   

<html><body><table><tr><td>Statistics</td><td>Number</td></tr><tr><td>Total Sample</td><td>3.853</td></tr><tr><td>Total Image</td><td>14,801</td></tr><tr><td>Average Question Length</td><td>22.66</td></tr><tr><td>Average Choice Length Average RationaleLength</td><td>1.33</td></tr><tr><td>Average Multi-modal Thought Step</td><td>104.74</td></tr><tr><td>Average Rationale Step</td><td>3.11 7.71</td></tr></table></body></html>

# 3 Benchmark Analysis

Basic statistics As shown in Table 2, CoMT comprises 3,853 samples and 14,801 images. CoMT encompasses two primary domains within $\mathrm { M ^ { 3 } C o { \bar { T } } }$ (Chen et al. 2024b) and four visual operations (illustrated in Figure 3 (a)) for comprehensive evaluation. Additionally, CoMT requires more intricate reasoning, with an average length of 104.7 words and 7.7 steps per sample, significantly higher than ScienceQA’s 48 words and 2.5 steps.

Multi-modal diversity CoMT includes a diverse array of multi-modal tasks (visual creation, visual deletion, visual update and visual selection), ranging from mathematical problems to commonsense challenges, such as geometry and recognition. Furthermore, as depicted in Figure 3 (b), CoMT features a wide range of image types encompassing “Culture & Art”, and “Abstract Graph”, etc, classified by CLIP (Radford et al. 2021).

Rationale diversity As illustrated in Figure 3 (c), CoMT exhibits a broad range in the number of reasoning steps. Additionally, the multi-modal thought steps also show both diversity and sufficient volume. This allows for a comprehensive evaluation across different steps within CoMT.

# 4 Experiments

# 4.1 Experiments Setting

In our experiments, we select a range of LVLMs as backbones, including those trained on image generation tasks as well as those that are not, including Gemini-Pro (Team et al. 2023), Qwen-VL (Bai et al. 2023), LLaVA-NeXT (Liu et al. 2024a), GILL (Koh, Fried, and Salakhutdinov 2023), NExT-GPT (Wu et al. 2024a), AnyGPT (Zhan et al. 2024). Additionally, we explore various prompting strategies: (1) Direct prompts the model to directly generate the answer. (2) CoT (Kojima et al. 2022) is a widely used prompt method to stimulate LLMs to generate steps with “Let’s think stepby-step!”. (3) Desp-CoT (Wu et al. 2023) enhances reasoning quality by instructing the model to generate a description before answering. (4) VoT (Wu et al. 2024b) utilizes “Visualize the state after each reasoning step.” to imagine the reasoning path with text-modal. Following Qin et al. (2023) and Chen et al. (2024b), we extract the final generated answers using regular expressions. See Technical Appendix D for further experimental details.

Table 3: Main results on various LVLMs. The bold content indicates the best performance across all models and all prompting methods, while the underlined content signifies the best performance within a single model across all methods. See Table 4 in Technical Appendix F for complete results.   

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">Visual Creation</td><td colspan="2">Visual Deletion</td><td colspan="2">Visual Update</td><td colspan="2">Visual Selection</td><td colspan="2">Average</td></tr><tr><td>Acc</td><td>Macro-F1</td><td>Acc</td><td>Macro-F1</td><td>Acc</td><td>Macro-F1</td><td>Acc</td><td>Macro-F1</td><td>Acc</td><td>Macro-F1</td></tr><tr><td></td><td>27.10</td><td>26.75</td><td>25.17</td><td>25.15</td><td>24.06</td><td>24.05</td><td>25.59</td><td>25.55</td><td>25.48</td><td>25.37</td></tr><tr><td colspan="9">Qwen-VL-7B (Bai et al. 2023)</td><td></td><td></td></tr><tr><td>Direct (Bai et al. 2023)</td><td>21.49</td><td>12.78</td><td>26.35</td><td>18.29</td><td>37.64</td><td>30.34</td><td>22.08</td><td>13.80</td><td>26.89</td><td>18.80</td></tr><tr><td>CoT (Kojima et al. 2022)</td><td>23.96</td><td>19.22</td><td>12.63</td><td>11.81</td><td>33.62</td><td>26.13</td><td>23.22</td><td>18.00</td><td>23.26</td><td>18.79</td></tr><tr><td>Desp-CoT(Wu et al.2023)</td><td>19.90</td><td>13.23</td><td>20.94</td><td>7.73</td><td>30.59</td><td>23.85</td><td>26.05</td><td>10.48</td><td>24.37</td><td>13.82</td></tr><tr><td>VoT (Wu et al. 2024b)</td><td>22.08</td><td>17.51</td><td>14.43</td><td>11.71</td><td>28.52</td><td>21.02</td><td>22.08</td><td>12.47</td><td>21.78</td><td>15.68</td></tr><tr><td colspan="9">LLaVA-NeXT-13B (Liu et al. 2024a)</td><td></td></tr><tr><td>Direct (Liu et al. 2024a)</td><td>26.34</td><td>19.72</td><td>20.64</td><td>20.06</td><td>35.47</td><td>34.26</td><td>22.76</td><td>19.60</td><td>26.30</td><td>23.41</td></tr><tr><td>CoT (Kojima et al. 2022)</td><td>22.18</td><td>12.33</td><td>21.44</td><td>15.21</td><td>26.36</td><td>18.99</td><td>24.92</td><td>19.91</td><td>23.73</td><td>16.61</td></tr><tr><td>Desp-CoT (Wu et al.2023)</td><td>19.90</td><td>12.82</td><td>23.45</td><td>17.47</td><td>27.01</td><td>18.82</td><td>25.59</td><td>20.77</td><td>23.99</td><td>17.47</td></tr><tr><td>VoT (Wu et al. 2024b)</td><td>20.79</td><td>15.58</td><td>25.55</td><td>18.55</td><td>27.55</td><td>18.95</td><td>26.61</td><td>17.23</td><td>25.13</td><td>17.58</td></tr><tr><td colspan="9"> GILL (Koh, Fried, and Salakhutdinov 2023)</td><td></td><td></td></tr><tr><td>Direct (Koh,Fried,and Salakhutdinov 2023)</td><td>16.93</td><td>15.75</td><td>22.65</td><td>13.90</td><td>23.43</td><td>12.62</td><td>18.12</td><td>10.16</td><td>20.28</td><td>13.11</td></tr><tr><td>CoT (Kojima et al.2022)</td><td>8.61</td><td>9.96</td><td>12.63</td><td>8.62</td><td>18.11</td><td>8.20</td><td>17.21</td><td>8.34</td><td>14.14</td><td>8.78</td></tr><tr><td>Desp-CoT(Wu et al.2023) VoT (Wu et al. 2024b)</td><td>6.83</td><td>7.93</td><td>20.74</td><td>9.60</td><td>21.69</td><td>10.90</td><td>20.95</td><td>9.12</td><td>17.55</td><td>9.39</td></tr><tr><td></td><td>5.94</td><td>7.01</td><td>17.94</td><td>11.81</td><td>21.04</td><td>11.51</td><td>14.27</td><td>9.23</td><td>14.80</td><td>9.89</td></tr><tr><td colspan="9">NExT-GPT (Wu et al. 2024a)</td><td></td></tr><tr><td>Direct (Wu et al. 2024a)</td><td>24.26</td><td>19.00</td><td>25.75</td><td>19.15</td><td>24.30</td><td>18.04</td><td>22.42</td><td>16.24</td><td>24.18</td><td>18.11</td></tr><tr><td>CoT (Kojima et al. 2022)</td><td>20.20</td><td>13.88</td><td>23.85</td><td>17.25</td><td>22.78</td><td>17.95</td><td>21.52</td><td>18.39</td><td>22.09</td><td>16.87</td></tr><tr><td>Desp-CoT(Wu et al.2023)</td><td>17.52</td><td>13.93</td><td>23.95</td><td>14.13</td><td>25.38</td><td>17.91</td><td>22.99</td><td>16.90</td><td>22.46</td><td>15.72</td></tr><tr><td>VoT (Wu et al. 2024b)</td><td>13.17</td><td>12.91</td><td>22.85</td><td>14.38</td><td>25.05</td><td>16.28</td><td>22.88</td><td>18.32</td><td>20.99</td><td>15.47</td></tr><tr><td colspan="9">AnyGPT (Zhan et al. 2024)</td><td></td></tr><tr><td>Direct (Zhan et al. 2024)</td><td>19.11</td><td>12.18</td><td></td><td></td><td>23.10</td><td>17.85</td><td>27.63</td><td>16.91</td><td>21.82</td><td>14.72</td></tr><tr><td>CoT (Kojima et al. 2022)</td><td>10.10</td><td>10.36</td><td>17.43 21.74</td><td>11.92 11.96</td><td>24.08</td><td>18.37</td><td>22.20</td><td>15.77</td><td>19.53</td><td>14.12</td></tr><tr><td>Desp-CoT (Wu et al.2023)</td><td>19.31</td><td>14.15</td><td>22.75</td><td>12.22</td><td>24.84</td><td>18.72</td><td>25.59</td><td>16.63</td><td>23.12</td><td>15.43</td></tr><tr><td>VoT (Wu et al. 2024b)</td><td>11.78</td><td>10.22</td><td>23.45</td><td>11.45</td><td>26.36</td><td>19.44</td><td>25.59</td><td>18.43</td><td>21.80</td><td>14.89</td></tr><tr><td colspan="9">Gemini (Team et al. 2023)</td><td></td></tr><tr><td>Direct (Team et al.2023)</td><td>28.91</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CoT (Kojima et al. 2022)</td><td>27.92</td><td>25.43</td><td>30.86</td><td>22.28</td><td>46.36 40.24</td><td>46.26 40.02</td><td>27.63 27.39</td><td>20.69 23.60</td><td>33.44 31.08</td><td>28.67 27.36</td></tr><tr><td>Desp-CoT(Wu et al.2023)</td><td>18.04</td><td>23.07</td><td>28.76</td><td>22.73</td><td>31.05</td><td>23.20</td><td>25.14</td><td>11.32</td><td>25.90</td><td>17.64</td></tr><tr><td>VoT (Wu et al. 2024b)</td><td>33.27</td><td>14.61 26.48</td><td>29.36 27.05</td><td>21.43 20.79</td><td>35.36</td><td>27.83</td><td>24.92</td><td>19.38</td><td>30.15</td><td>23.62</td></tr></table></body></html>

# 4.2 Main Results

Table 3 presents the main results, from which we derive the following key findings:

All LVLMs perform poorly on the CoMT. Despite Gemini achieving a $2 8 . 6 7 \%$ F1 score across four tasks, this performance is marginally better than the random baseline by

$3 . 3 \%$ , indicating significant room for improvement. Additionally, except for Gemini, most models perform at or below random levels. We attribute these to the lack of multimodal reasoning in current LVLMs.

Traditional Multimodal CoT almost completely fails on CoMT. We observe that pure text-modal CoT does not attain improvement in addressing the CoMT problem and even degrades the performance of most models to near-random levels. We attribute it to the fact that the inability of the model to execute specific visual logic expressions and operations results in poor performance.

All models fail to visualize thought in textual words. As demonstrated in Table 3, all LVLMs fail to utilize VoT effectively to improve performance. Specifically, $\scriptstyle \mathtt { V o T }$ prompts LVLMs to visualize states through textual representation and results in an average accuracy decrease of $1 2 . 2 8 \%$ . This finding suggests that although textual representation can convey visual features, the inherent differences between modalities still constrain the expression of multi-modal thought.

![](images/d5aebde4a2ab85cde93db55fc53b1a1795778db4712463ace0a61696ffa6c098.jpg)  
Figure 4: Analysis of the correlation between the model performance and the quality of rationale for different LVLMs based on ROSCOE (Golovneva et al. 2023).

![](images/a297d87d6a6cca05e34557763c55c271b4b106319e12128e6805e6f34714a415.jpg)  
Figure 5: CLIPScore of LVLMs on 4 tasks within CoMT. The $\mathbf { \boldsymbol { x } }$ -axis represents the CLIPScore, and the y-axis represents the accuracy.

# 4.3 Analysis

This section will conduct a further analysis on CoMT. See Technical Appendix E for more implementation details.

Improving the quality of rationale is essential for CoMT. As illustrated in Figure 4, the quality of CoT rationale significantly impacts the CoMT performance. Poor rationale quality constrains the logical coherence of LVLMs, limiting their reasoning capacities, which aligns with Chen et al. (2024b). Consequently, enhancing reasoning quality in LVLMs is a crucial area for further exploration.

CoMT benefits from improved multi-modal thought. To assess the impact of multi-modal thought on performance within CoMT, we calculate the CLIPScore (Hessel et al. 2021) to reflect the similarity between model output and each image within the ideal rationale pre-defined. Averaging these scores yields a multi-modal alignment score for each reasoning chain generated. As shown in Figure 5, there is a significant positive correlation between performance and multi-modal alignment scores across four tasks, which indicates that CoMT benefits from more multi-modal thought.

The performance relies more on the quality of multi-modal alignment than on parameter size. As shown in Table 4 in Technical Appendix F, the IDEFICS2-8B, with finegrained multi-modal alignment, surpasses the 13B models, even approaching the performance of the Gemini-Pro $\mathrm { ( > 1 0 0 B }$ , Team et al. (2023)). We think that CoMT performance depends more on multi-modal alignment quality rather than parameter size.

![](images/e57c1201a7cc28d91c772d108cf09f0a3ca1cb23acabfd4649d4dbb73d2918f6.jpg)  
Figure 6: Analysis on In-context Learning of Gemini$P r o$ (Team et al. 2023) in CoMT.

# 4.4 In-context Learning Explorations

In-context Learning with multi-modal input and output can effectively promote the performance in CoMT. As shown in Figure 6, using in-context learning (ICL) (Li et al. 2023a; Qin et al. 2024a) with multi-modal input and multimodal output demonstrations significantly improves performance. It not only surpasses zero-shot prompting but also outperforms ICL with text-modal output. This approach can be successful due to the fact that LVLMs can learn to effectively facilitate multi-modal thought through such demonstrations, even though Gemini is limited to producing rationales in the textual modality alone.

Not more demonstrations means better performance in CoMT. As shown in Figure 6, the model exhibits a significant downward trend in performance when the number of demonstrations exceeds four. It shows that more demonstrations are not necessarily better, as multimodal demonstrations often require the consumption of a substantial number of tokens, which can also lead to more complex challenges associated with longer contexts.

# 4.5 Error Analysis

Insufficient Multi-modal Thought. When dealing with multi-modal problems, models struggle to integrate multimodal thought most of the time. As illustrated in Figure 7, we observe that despite certain models (e.g., GILL, NExTGPT, AnyGPT) being trained on image generation tasks, at least $48 \%$ of their reasoning processes do not incorporate image generation. This occurs even when image generation is crucial for accurate outcomes, indicating a disjunction between image generation and text processing.

Inaccurate Textual Reasoning. When logical errors occur in textual reasoning, they hinder the advancement towards the correct answer. For example, Figure 10 in Technical Appendix reveals that the model demonstrates poor reasoning logic, with significant logical errors, such as calculation mistakes (like ${ } ^ { \cdot } 2 ^ { \ast } 5 ^ { \ast } 5 { = } 2 ^ { \ast } 1 0 ^ { \cdot \cdot }$ ). These inaccurate textual reasoning significantly impedes progress in this field.

![](images/35280c9585ab67da3ba1ccf4bd5750944e9dc7c7a689e4e9abf4163479c42a92.jpg)  
Figure 7: Image generation frequency during reasoning

Incoherent Visual Reasoning. Although certain models generate images when reasoning, not all image contents align with the reasoning path, revealing an immature interaction between modalities. We manually evaluate the generated images, with results shown in Figure 8. The distribution reveals that current LVLMs often generate irrelevant images during reasoning (an average of $43 \%$ , represented by score $O$ ) and fail to perform effective visual logic (on average $45 \%$ of images exhibit logical mistake, represented by score 1,2). The judgment criteria can be found in Technical Appendix C.3. To be specific, Figure 11 in Technical Appendix G shows instances with irrelevant text and image logic.

# 4.6 Future Directions

Based on the above analysis, we summarize the future directions for current LVLMs tackling CoMT.

How can we effectively integrate multi-modal thought reasoning? The absence of visual thought significantly increases the difficulty when addressing certain multi-modal tasks, such as CoMT. How to enable models to integrate multi-modal reasoning is an intriguing research topic. Furthermore, given the inherent differences between textual and visual modalities, exploring how to align these two modalities during reasoning presents another valuable challenge.

How can we enhance logical reasoning capabilities for textual reasoning? The inadequacies in textual reasoning logic lead to inaccurate conclusions during inference, such as calculation mistakes. Therefore, how to enable models with better textual logic to perform effective text reasoning is a critical topic to explore.

How can we achieve effective vision logic for visual reasoning? Since some generated images fail to perform effective visual logic or even be irrelevant, not all visual thoughts generated have a positive influence on the reasoning. How to enable models to develop better visual logic to produce images that are relevant and consistent with the progression of rationale is a topic worth exploring.

# 5 Related Work

The emergence of Multi-modal Chain-of-Thought (MCoT) techniques elicits the step-by-step zero-shot and fewshot multi-modal reasoning capabilities of Large VisionLanguage Models (LVLMs) (Wang et al. 2024c,b; Chen et al. 2024a,c; Liu et al. 2023; He et al. 2024; Qin et al. 2024a; Fei et al. 2024a,c). Pioneering work introduces the ScienceQA benchmark (Lu et al. 2022), involving multimodal scientific questions. Subsequently, Zhang et al. (2023) formally propose the concept of MCoT and introduce a two-stage framework encompassing both reasoning and answering. Additionally, Tan et al. (2024); Wang et al. (2024a); Zhang et al. (2024a); Mondal et al. (2024); Lee et al. (2024) introduce more knowledge to improve the performance and reduce hallucinations in MCoT reasoning. Following this, Zheng et al. (2024) propose DDCoT, which breaks down the question into a series of sub-questions and solves them using LVLMs. Building upon this, Chen et al. (2024b) further introduce a multi-domain multi-step multimodal benchmark to fully evaluate the complex MCoT capabilities. Based on traditional MCoT, some works begin preliminary exploration integrating the diffusion model or retriever model as a tool for better MCoT. Meng et al. (2023) propose CoI to generate images as intermediate reasoning steps in single modal tasks, outperforming purely textual CoT. Wu et al. (2024b) propose VoT, requiring text-only LLMs to imagine their vision reasoning paths, which increases the spatial reasoning abilities.

![](images/49fdd5f8647c0bfec9363bb186b13095e928d656f1477f9229f260abb7d89086.jpg)  
Figure 8: Distribution of human-evaluated image quality scores $( \uparrow )$ which are mainly determined based on Relevance and Logical Correctness. See Technical Appendix C.3 for evaluation details.

In contrast to our work, their strategies rely solely on textual modalities for reasoning, lacking visual operation or detailed visual expression in reasoning. To fill this gap, we propose CoMT to comprehensively reveal diverse multi-modal thought capabilities. We hope CoMT will inspire research on promoting better multi-modal reasoning.

# 6 Conclusion

In this work, we introduce a Chain of Multi-modal Thought (CoMT) benchmark to evaluate and improve the multi-modal reasoning capabilities of Large Vision-Language Models (LVLMs). Through extensive experiments, our findings reveal a significant performance gap between LVLMs and human, with models generally not outperforming random chance in zero-shot scenarios. In-context Learning with multi-modal rationale emerges as a promising approach to better integrate visual and textual reasoning in LVLMs. We hope this research lays the groundwork for future enhancements in multi-modal reasoning technologies.
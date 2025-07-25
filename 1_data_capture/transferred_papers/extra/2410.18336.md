# Assessing the Creativity of LLMs in Proposing Novel Solutions to Mathematical Problems

Junyi $\mathbf { Y } \mathbf { e } ^ { 1 }$ , Jingyi $\mathbf { G } \mathbf { u } ^ { 1 }$ , Xinyun Zhao1, Wenpeng $\mathbf { Y } \mathbf { i n } ^ { 2 }$ , Guiling Wang

1New Jersey Institute of Technology, Newark, USA 2The Pennsylvania State University, State College, PA, USA jy394, jg95, xz43, gwang $@$ njit.edu, wenpeng $@$ psu.edu

# Abstract

The mathematical capabilities of AI systems are complex and multifaceted. Most existing research has predominantly focused on the correctness of AI-generated solutions to mathematical problems. In this work, we argue that beyond producing correct answers, AI systems should also be capable of, or assist humans in, developing novel solutions to mathematical challenges. This study explores the creative potential of Large Language Models (LLMs) in mathematical reasoning, an aspect that has received limited attention in prior research. We introduce a novel framework and benchmark, CREATIVEMATH, which encompasses problems ranging from middle school curricula to Olympic-level competitions, designed to assess LLMs’ ability to propose innovative solutions after some known solutions have been provided. Our experiments demonstrate that, while LLMs perform well on standard mathematical tasks, their capacity for creative problem-solving varies considerably. Notably, the Gemini1.5-Pro model outperformed other LLMs in generating novel solutions. This research opens a new frontier in evaluating AI creativity, shedding light on both the strengths and limitations of LLMs in fostering mathematical innovation, and setting the stage for future developments in AI-assisted mathematical discovery.

Code — https://github.com/NJIT-AI-Center/CreativeMath

# Introduction

In recent years, artificial intelligence has made significant strides, particularly in the development of Large Language Models (LLMs) capable of tackling complex problemsolving tasks. Models like GPT-4 and Gemini-1.5-Pro have demonstrated impressive proficiency on rigorous mathematical benchmarks (Ahn et al. 2024) such as GSM8K (Cobbe et al. 2021) and MATH (Hendrycks et al. 2021a), underscoring the evolving role of LLMs from simple text generators to sophisticated tools capable of engaging with high-level mathematical challenges. Beyond solving student-oriented math problems, leading mathematicians have begun exploring the use of LLMs to assist in tackling unresolved mathematical challenges (Romera-Paredes et al. 2024; Trinh et al.

2024). Despite these models’ success in achieving high accuracy on existing mathematical datasets, their potential for creative problem-solving remains largely underexplored.

The standard definition of creativity, as articulated by (Runco and Jaeger 2012), emphasizes two essential criteria: novelty and usefulness. While correctness aligns with usefulness, evaluating novelty remains a challenge, especially in the domain of mathematics. Mathematical creativity goes beyond solving problems correctly; it involves generating novel solutions, applying unconventional techniques, and offering deep insights—areas traditionally associated with human ingenuity. Yet, most studies have focused primarily on correctness and efficiency, paying little attention to the innovative approaches LLMs might employ. Furthermore, creativity in mathematical problem-solving is rarely integrated into existing benchmarks, limiting our understanding of LLMs’ full potential. The current research landscape lacks a comprehensive framework that evaluates both the accuracy and the creative capacity of LLMs. This gap highlights the need for new methodologies and benchmarks specifically designed to assess and cultivate the creative problem-solving abilities of LLMs in mathematics, which is the focus of this paper.

We created the dataset CREATIVEMATH, a comprehensive math benchmark that includes problems from middle school to Olympic-level competitions, each accompanied by multiple high-quality solutions ranging from straightforward to highly innovative approaches. Additionally, we designed a multi-stage framework to rigorously evaluate the creativity of LLMs in generating novel math solutions. This evaluation spans closed-source, open-source, and math-specialized LLMs, assessing both the correctness and novelty of their solutions based on different reference prior solutions.

Our evaluation revealed several interesting key insights: (1) Gemini-1.5-Pro excelled in generating unique solutions, with most correct answers being distinct from the provided references, while smaller and math-specialized models struggled with novelty. (2) Providing more reference solutions generally improved accuracy, with Gemini-1.5-Pro achieving perfect accuracy with four prior solutions. However, increased references made it harder for models to generate unique solutions, indicating a trade-off between leveraging existing knowledge and fostering creativity. (3) As math problem difficulty increased, LLM accuracy declined, but successful solutions were more likely to be innovative, suggesting that tougher problems encourage creativity. (4) Analysis of solution similarity among different LLMs showed that models like Llama-3-70B and Yi-1.5-34B explored diverse approaches, while others like Mixtral- $\mathbf { 8 } \mathbf { x } 2 2 \mathbf { B }$ produced more similar solutions, highlighting the value of using a diverse set of LLMs to enhance originality.

This study lays the groundwork for future advancements in LLM math creativity. The major contributions include: (1) Introducing a new task—evaluating LLMs’ mathematical creativity, (2) Creating the CREATIVEMATH dataset, (3) Developing a framework for assessing mathematical creativity in LLMs, and (4) Evaluating state-of-the-art LLMs, revealing key insights into their strengths and limitations.

# Related Work

LLMs have demonstrated significant advancements in both mathematical reasoning and creative capabilities, making them increasingly powerful tools in a variety of domains. In the realm of mathematical reasoning, techniques such as prompt engineering, Chain-of-Thought (CoT) prompting, and program-aided language modeling have notably enhanced LLMs’ abilities to solve complex problems (Brown 2020; Wei et al. 2022; Zhou et al. 2023). These approaches enable models to break down problems into more manageable steps, thereby improving their accuracy and reasoning depth. Moreover, specialized models like MathVerse (Zhang et al. 2024) and Internlm-Math (Ying et al. 2024), which are trained on extensive mathematical corpora, have achieved significant improvements in mathematical problem-solving performance (Lewkowycz et al. 2022; Ying et al. 2024). Benchmarks such as GSM8K and MATH further provide a structured means to evaluate and compare these advancements, highlighting the continuous progress in this area (Cobbe et al. 2021; Hendrycks et al. 2021b).

In terms of creativity, LLMs have shown remarkable prowess across diverse fields. They have excelled in generating high-quality, human-like content, ranging from code generation (Ni et al. 2023; Liu et al. 2024a) and music composition (Yuan et al. 2024) to literature (Go´mez-Rodr´ıguez and Williams 2023; Liu et al. 2024b) and educational tools (Lan and Chen 2024; Orenstrakh et al. 2023). Creativity in LLMs is often evaluated using frameworks like Margaret Boden’s taxonomy (Boden 2004), which categorizes creativity into combinational, exploratory, and transformational types. While LLMs perform well in combinational creativity, achieving true transformational creativity remains a significant challenge (Franceschelli and Musolesi 2023). Psychological metrics such as the Torrance Tests of Creative Thinking (TTCT) (Torrance 1966), where LLMs have demonstrated high fluency, originality, and flexibility. However, the applicability of these traditional creativity metrics to AI systems is still a topic of debate, as they were originally designed to assess human creativity (Zhao et al. 2024).

Techniques such as associative thinking have been employed to enhance the creative output of LLMs further, although challenges remain in ensuring that these models can meaningfully integrate unrelated concepts (Mehrotra, Parab, and Gulwani 2024). The ethical and legal implications of AIgenerated creativity continue to be a significant area of concern, underscoring the need for ongoing research to refine evaluation methods and address societal impacts (Lofstead 2023).

# CREATIVEMATH Curation

This section details the creation, collection, and processing of our dataset CreativeMath, which comprises high-quality mathematical problems from various competitions and their numerous solutions. The dataset is diverse, encompassing a broad range of mathematical topics and problem types, and covers difficulty levels from middle school to Olympiad level. It includes problems from eight major US competitions: AMC 8, AMC 10, AMC 12, AHSME, AIME, USAJMO, USAMO, and $\mathrm { I M O ^ { 1 } }$ .

Data Collection. The dataset was sourced from the Art of Problem Solving $( \mathrm { A o P S } ) ^ { 2 }$ , a platform offering the most comprehensive collection of problems from various math competitions, along with multiple solutions contributed by participants over the years. As the most popular and soughtafter resource for math competitors, AoPS effectively functions as a natural crowdsourcing platform. It uniquely approximates the complete set of viable human solutions for each problem, with later contributors often building on earlier ones.

We meticulously scraped data from eight competitions, ranging from middle school level to Olympic-level, to capture the breadth of mathematical challenges and the depth of solution strategies available.

Data Cleaning. To ensure the integrity and reliability of the dataset, we conducted a rigorous data cleaning procedure. We accurately extracted LaTeX-formatted problems and solutions from HTML, ensuring their correct representation. Irrelevant comments were removed to make each problem and solution clear and self-sufficient. Samples with images, problems without solutions, or incomplete entries were manually removed from the dataset. After this process, the dataset comprises 6,469 mathematical problems and 14,223 solutions. Each problem in the dataset is tagged with detailed metadata, including difficulty level, math category, and problem type. Difficulty levels and problem types were assigned based on official competition data, while the math category were determined using the Llama-3-70B model.

Dataset Analysis. As shown in Figure 1, the problem distribution inside CreativeMath reveals that Algebra and Geometry are the most represented categories across all com

Algebra 216 386 437 853 273 16 75 113 -800 Arithmetic 220 80 54 66 4 0 0 1 -600. Counting 82 100 84 36 104 10 18 15 Geometry 253 326 323 530 222 34 87 133 400 Number 99 144 128 104 171 20 63 68 Probability 51 94 83 36 73 0 6 2 -200 Others 39 21 26 41 15 3 20 22 -0 名° 名 石 名 2 VSM ZME NOAJM GAMC 名 Competition

![](images/1bbe654b573c5bb2bee9e32780c6b181042de6892696d922ade52bc383385c8f.jpg)  
Figure 1: Distribution of problems across different math categories and competitions in the CreativeMath dataset.   
Figure 2: Distribution of the number of solutions per problem across different competitions.

petitions. The number of solutions across different competitions, as depicted in Figure 2, reflects the varying complexity of the problems. Medium-difficulty competitions like AMC 10, AMC 12, and AIME typically have a larger number of solutions, as these problems allow for a variety of approaches. In contrast, simpler competitions like AMC 8 tend to have fewer solutions due to the straightforward nature of the problems, which often have limited methods of solving. Olympic-level competitions such as USAJMO, USAMO, and IMO also see fewer solutions, likely due to the high complexity of the problems, which limits the number of viable solving strategies.

# Methods

Our approach consists of a multi-stage pipeline designed to evaluate the novelty of mathematical solutions generated by an LLM. The methodology is structured into four key stages: Novel Solution Generation, Correctness Evaluation, Coarse-Grained Novelty Assessment, and FineGrained Novelty Assessment. This comprehensive pipeline illustrated in Figure 3 ensures that the generated solutions are not only correct but also exhibit a meaningful degree of novelty relative to the reference solutions. The sample prompts and LLMs’ responses are provided in the Appendix.

# Novel Solution Generation

The first stage of the methodology aims to generate novel solutions for the given mathematical problem using LLM. For each problem, a subset of $k$ reference solutions (where $k$ ranges from 1 to $n$ , with $n$ representing the total number of available reference solutions) is sequentially selected based on the order in which competitors uploaded their solutions on the website. Earlier solutions are often the most common and intuitive, while later ones may build on previous methods, offer improvements, or introduce entirely novel algorithms. Consequently, as $k$ increases, the difficulty in generating new and innovative solutions also increases.

To ensure clarity and consistency in both prompting and evaluating the novelty of generated solutions, we define a set of criteria agreed upon in consultation with several mathematicians. These criteria guide both the generation and the evaluation process and are used to assess the distinctiveness of the solutions. The criteria are as follows:

• Methodological Differences: If the methods used to arrive at the solutions are fundamentally different (e.g., algebraic manipulation versus geometric reasoning), the solutions are considered distinct. • Intermediate Step Variation: Even if the final results are identical, if the intermediate steps or processes involved in reaching those solutions differ significantly, the solutions are considered novel. • Assumptions and Conditions: Solutions that rely on different assumptions, initial conditions, or constraints are treated as distinct. • Generalization: A solution that generalizes to a broader class of problems is considered novel compared to one that is specific to certain conditions. • Complexity: If one solution is notably simpler or more complex than another, they are regarded as different, even if they lead to the same final result.

These criteria, also illustrated in Figure 4, are embedded into the prompt used to guide the LLM in generating novel solutions. The reference solutions provided to the model aim to capture a variety of approaches, and the LLM is instructed to output a new solution that is distinct according to the defined criteria. The prompt emphasizes generating solutions that use different problem-solving methods, distinct intermediate steps, and variations in assumptions or generalizability.

As part of this process, to avoid influencing the judgment of evaluators during the subsequent evaluation stage, transition sentences and justifications explaining why the new solution is distinct from the reference solutions are manually removed. Only the newly generated solution is presented for evaluation.

# Correctness and Novelty Evaluation

To rigorously evaluate the correctness and novelty of the generated solutions, we employ three leading LLMs—GPT

![](images/0bf99f7e7f7835027319640bd76caea6859df556a0345532dcd89b18656d8e77.jpg)  
Figure 3: The framework includes solution generation (left) and the evaluation pipeline (middle). The flowchart of the detailed evaluation pipeline is illustrated on the right.   
Figure 4: The prompt template for generating novel solution.

# Criteria for evaluating the difference between two mathematical solutions include:

1.If the methodsused to arriveat the solutionsare fundamentally different, suchasalgebraic manipulation versus geometric reasoning theycan be considered distinct;   
2.Even if the final resultsare the same, if the intermediatesteps or processes involved in reaching those solutions vary significantly, the solutionscan be considered different;   
3.If two solutions rely on different assumptions or conditions, they are likely to be distinct;   
4.Asolution might generalize toa broaderclassof problems,while another solution might be specific to certain conditions.In such cases,theyareconsidereddistinct;   
5.If one solution is significantly simpler or more complex than the other, theycanberegardedasessentiallydifferent, evenif theylead to the same result.

Given the following mathematical problem: {problem}

And some typical solutions: {solutions}

Please output a novel solution distinct from the given ones for this math problem.

4, Claude 3.5 Sonnet, and Gemini 1.5 Pro—as LLM Evaluators, recognized among the strongest models available. These LLM Evaluators collaboratively assess the solutions following the framework illustrated in Figure 3 (middle). Each LLM Evaluator adheres to the flowchart depicted in Figure 3 (right) to systematically evaluate the generated solutions across three dimensions:

• Correctness: The solution must first be validated for correctness, ensuring it produces the correct result for the problem. Only correct solutions proceed to the novelty

# assessment stages.

• Coarse-Grained Novelty: If the solution is correct, it is then evaluated for novelty against a subset of $k$ reference solutions. A solution is deemed novel if it is distinct from these $k$ solutions.

• Fine-Grained Novelty: A solution deemed novel in the coarse-grained assessment undergoes further evaluation against the entire set of $n$ human-provided solutions. This stage distinguishes between:

– Novel-Unknown: A solution that is distinct from all $n$ human-generated solutions, representing a truly original contribution.   
– Novel-Known: A solution that is distinct from the $k$ reference solutions but similar to others in the remaining $n - k$ solutions.

Evaluation Strategy We apply different strategies for correctness and novelty evaluation to ensure both rigor and practicality. For correctness, only solutions unanimously deemed correct by all LLM Evaluators proceed to the novelty assessment, ensuring that only fully reliable solutions are considered. Given the subjective nature of assessing novelty, we use a majority voting strategy, which balances diverse perspectives and effectively identifies genuinely innovative solutions without being overly restrictive.

Correctness Evaluation Once a solution is generated, the first essential step is to verify its correctness. The newly generated solution, along with the original problem and a set of reference solutions, is evaluated by the LLM Evaluators using the prompt shown in Figure 5, top. The LLM Evaluators determine if the solution leads to the correct outcome, with responses of “YES” indicating correctness and “NO” indicating otherwise. Only solutions unanimously validated as correct by all LLM Evaluators advance to the novelty assessment stages.

Given the following mathematical problem: {problem}
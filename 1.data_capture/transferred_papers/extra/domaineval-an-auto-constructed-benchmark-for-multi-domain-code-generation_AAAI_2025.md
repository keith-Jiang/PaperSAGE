# DOMAINEVAL: An Auto-Constructed Benchmark for Multi-Domain Code Generation

Qiming $\mathbf { Z } \mathbf { h } \mathbf { u } ^ { 1 , 2 * }$ , Jialun $\mathbf { C a o } ^ { 3 * }$ , Yaojie $\mathbf { L u } ^ { 1 \dagger }$ , Hongyu Lin1, Xianpei $\mathbf { H a n } ^ { 1 }$ , Le $\mathbf { S u n } ^ { 1 }$ , Shing-Chi Cheung

1Chinese Information Processing Laboratory, Institute of Software, Chinese Academy of Sciences, Beijing, China 2University of Chinese Academy of Sciences, Beijing, China 3The Hong Kong University of Science and Technology, Hong Kong, China zhuqiming2022, luyaojie, hongyu, xianpei, sunle $@$ iscas.ac.cn jcaoap, scc @cse.ust.hk

# Abstract

Code benchmarks such as HumanEval are widely adopted to evaluate capabilities of Large Language Models (LLMs), providing insights into their strengths and weaknesses. However, current benchmarks primarily exercise LLMs’ capability on common coding tasks (e.g., bubble sort, greatest common divisor), leaving domain-specific coding tasks (e.g., computation, system, cryptography) unexplored. To fill this gap, we propose a multi-domain code benchmark, DOMAINEVAL, designed to evaluate LLMs’ coding capabilities thoroughly. Our pipeline works in a fully automated manner, enabling a push-button pipeline from code repositories into formatted subjects under study. Interesting findings are observed by evaluating 12 representative LLMs against DOMAINEVAL. We notice that LLMs are generally good at computation tasks while falling short on cryptography and system coding tasks. The performance gap can be as much as $6 8 . 9 4 \%$ $8 0 . 9 4 \%$ - $1 2 . 0 \%$ ) in some LLMs. We also observe that generating more samples can increase the overall performance of LLMs, while the domain bias may even increase. The contributions of this study include a code generation benchmark dataset DOMAINEVAL, encompassing six popular domains, a fully automated pipeline for constructing code benchmarks, and an identification of the limitations of LLMs in code generation tasks based on their performance on DOMAINEVAL, providing directions for future research improvements.

Code — https://github.com/domaineval

# Network

# Computation

def calculate_mean(lst): total $\mathbf { \sigma } = \mathbf { \sigma }$ sum(lst) mean $\mathbf { \tau } = \mathbf { \tau }$ total / len(lst) return mean

import ipaddress   
def is_valid_ipv6_address(ip_str): try: ipaddress.IPv6Address(ip_str) except ValueError: return False return True

# Basic

# Visualization

import datetime as dt def get_weekday_occurrence( d: dt.date): xthday $= 1 +$ (d.day-1)//7 return d.weekday(),xthday ef color_xy_to_temperature $( \mathsf { x } , \mathsf { y } )$ : $\mathsf { n } = \mathsf { \left( x - \theta . 3 3 2 \theta \right) }$ / (0.1858 - y) cct $\mathbf { \tau } = \mathbf { \tau }$ 437\* n\*\*3 + 3601\* n\*\*2 + 6861 \* n + 5517 return int(cct)

# Cryptography

# System

def derive_key( master: bytes

def free_physmem(): ) $$ bytes: out $\mathbf { \sigma } = \mathbf { \sigma }$ sh( derived=HKDF( ['free', '-b'], master,32,b'',SHA256 env={'LANG': 'C.UTF-8'} ) ) return derived lines $\mathbf { \tau } = \mathbf { \tau }$ out.split('\n') for line in lines: if line.startswith('Mem'): total,used,free,shared=(int(x) for x in line.split()[1:5]) nt $\mathbf { \sigma } = \mathbf { \sigma }$ collections.namedtuple( 'free’,'total used free shared output' ) return nt(total, used, free, shared, out) raise ValueError("can't find 'Mem'")

# Introduction

Large Language Models (LLMs) have revolutionized various areas such as question answering (Rogers, Gardner, and Augenstein 2023), math reasoning (Imani, Du, and Shrivastava 2023), and especially software development (Lozhkov et al. 2024). Stakeholders are eager to know whether and to what extent LLMs can improve development efficiency.

To this end, a variety of code generation benchmarks such as HumanEval (Chen et al. 2021) and MBPP (Austin et al.

2021) have been introduced and intensively used to evaluate LLMs’ coding capability. They primarily consist of common coding tasks such as sorting an array or computing the greatest common divisor. Furthermore, to meet more realistic needs, more benchmarks are emerging, expanding mainly along with two perspectives. First, linguistic diversity. This line of work (Wang et al. 2024; Cassano et al. 2022; Chai et al. 2024) mitigates linguistic bias in both natural languages (e.g., English, Chinese) and programming languages (e.g., Python, Java, $\mathrm { C / C + + }$ ). Second, code scale diversity. This line of work aims to scale up the code granularity from

Step I. Domain Repository Collection Step II. Test-Method Matching & Selection Step III. Instruction H 1 Generation 山 DomainEval For each Extract r each For each Filter Generate ASTs </>   
DoRmepaions-istopreiceisfic Repo Tests Test Methods Method Execute LLDMiscard Instruction </> Dependencies Subjects Domain Repositories Collected from GitHub Methods Collected NSumbjbecrtosf   
Computation (Total 15) numpy, pandas, scikit-learn, librosa, nltk, obspy, scikit- get_datevalue, is_float_dtype, median, power, sort, 1705 image,gensim, geopandas, statsmodels, sympy, tensorflow, … calculate_smoothing_matrix, collect_sqrt, …   
Basic (Total 12) setuptools, prettytable, charset_normalizer, arrow, max_height, total_time, join_dict_keys, get_path, get_dates, 107 chinese-calendar, khal, vdirsyncer, workalendar, … get_weekday_occurrence, format_timestamp,   
Network (Total 11) django, flask-restful, geopyi, mechanize, mitmproxy, format_html, get_password_validators, parse_cookie, 256 requests, scrapy, sendgrid-python, werkzeug, wtforms, yt-dlp filepath_to_uri, get_callable, is_valid_ipv6_address, … (Total 18) bokeh, datashader, gradio, matplotlib, Pillow, vaeX, get_colormap, create_sphere, plotting_context, histplot, Visualization 186 pygwalker, redashi, seaborn, tensorboardX, word_cloud, … husl_palette, light_palette, meshgrid_triangles, hsl_to_rgb, … (Total 16) anaconda, billiard, core, glances, loguru, mpire, is_admin, chunk_tasks, free_swap, serialize_response, sysctl, System openpyxl, pandapower, poetry, psutil, sentry, xlrd, xlwt, … free_physmem, parse_environ_block, get_mac_address, … 100 (Total 19) asn1crypto, badsecrets, blake3-py, crypto-attacks, BinaryMatrixRank, derive_key, encode_bitstring, attack,   
Cryptography cryptography, detect-secrets, firmware, freqtrade, UniversalDistribution, solve_right, load_cryptrec_vectors, 100 paranoid_crypto, pycryptodome, python-hdwallet, python-rsa, … load_rsa_nist_vectors, gen_keys, socket_any_family, …

function-level (Chen et al. 2021; Yu et al. 2024; Austin et al. 2021) to class-level (Du et al. 2023) and repo-level (Zhang et al. 2023; Cao et al. 2024a; Li et al. 2024b).

However, if LLMs are to be applied in real-world industrial scenarios where different product lines prioritize different domains of code, it becomes essential to further understand these LLMs’ coding capabilities across various domains. In other words, while existing benchmarks make significant efforts in varying natural/programming languages and code granularity, the LLMs’ ability to generate domainspecific code remains under-explored.

To fill this gap, we introduce DOMAINEVAL, a multidomain code generation benchmark. It consists of 2454 subjects (i.e., instruction, reference solution, test cases), along with 5892 test cases, covering six domains of code, i.e., computation, network, basic operation, system, visualization, and cryptography. Figure 1 shows representative examples from each domain. We can see clear functional distinctions among codes from different domains. For example, code in computation involves computational tasks such as mean calculation. Code in network handles network requests/communications and remote connections. Code in cryptography incorporates cryptographic algorithms, encrypting plain-text with a key or conducting key recovery attacks.

It is noteworthy that in order to facilitate the benchmark construction and welcome future contributions to DOMAINEVAL, we provide a fully automated test-guided construction pipeline. The pipeline works in a push-button manner, i.e., given a code repository, it outputs a set of formatted subjects (i.e., instruction, reference solution, test cases). Empowered by this pipeline, DOMAINEVAL is with exceptional scalability, capable of incorporating the everevolving code corpus into it. Moreover, the continuous influx of updated code through this pipeline fortifies DOMAINEVAL against the data contamination threat (Cao et al. 2024b), thereby maintaining its integrity and novelty.

Our extensive experiments on $^ { 1 0 + }$ LLMs indicate that LLMs are generally good at computation, with an average of $8 2 . 4 4 \%$ Pass $\ @ 1$ , while falling short on cryptography and system domains, with an average of $3 3 . 0 8 \%$ Pass $@ 1$ and $3 7 . 5 0 \%$ Pass $\ @ 1$ , respectively. The performance gap among domains can be as much as $6 8 \% +$ from Llama-2-13B-Chat model, which is observed to have the largest performance variance compared with other LLMs. We also observe that generating more samples can increase the overall performance of LLMs, while the domain bias may even increase.

The contributions can be summarized as follows.

• We introduce DOMAINEVAL, a multi-domain code generation benchmark that consists of $2 \mathrm { k } +$ subjects (i.e., description, reference code and tests) covering six domains. • We provide a fully automated test-guided construction pipeline to facilitate the benchmark construction.

# Benchmark Construction

In this section, we introduce the whole construction pipeline of DOMAINEVAL, as shown in Figure 2. We provide a fully automated, test-guided construction algorithm that transforms code repositories into a collection of formatted subjects for LLM evaluation. Each subject consists of three components: instruction for LLM evaluation, reference solution, and a series of test cases, as illustrated in Figure 3.

The pipeline begins with the collection of raw code snippets from specific domains, which are then systematically transformed into suitable benchmark data. This process involves three key steps: (1) Domain Repository Collection: The first step involves collecting raw code data $( \mathbf { R } )$ from

Instruction Reference Solution Test Cases   
[instruction] [method_name]  eval Test 0   
Functionality: The Commutator function is [full_method_name] Commutator.eval [test_path]   
designed to calculate the commutator of  ... [method_path] …/sympy/physics/tests/test_secondquant.py   
I-nap(uEtsx:pr): The first argument for ... sympy/sympy/physics/secondquant.py [test_code]   
- b (Expr): The second argument for ... [method_code] from sympy.core.symbol import symbols   
Outputs: from sympy.core.singleton import S ...   
- Expr: The result of the commutator  ... def test_commutation():   
[method_code_mask] class Commutator(Function):   
from sympy.core.singleton import S is_commutative $\mathbf { \sigma } = \mathbf { \sigma }$ False a, b, c, d $\mathbf { \sigma } = \mathbf { \sigma }$ symbols('a,b,c,d', @classmethod above_fermi $\ c =$ True)   
class Commutator(Function): def eval(cls, a, b): """built-in comments""" """built-in docstring""" assert Commutator.eval(c1, c1) $\scriptstyle = = 0$ is_commutative $\mathbf { \tau } = \mathbf { \tau }$ False if not (a and b): test_commutation() @classmethod return S.Zero def eval(cls, a, b): [MASK] ： Test N

selected domains. These snippets are categorized into two groups: $\bf { C _ { 0 } }$ , containing function code, and $\mathbf { T _ { 0 } }$ , including tests designed to validate the correctness of the code in $\bf { C _ { 0 } }$ . (2) Test-Method Matching & Selection: This step aims to create a set of candidate subjects $\mathbf { S } ^ { \prime } = ( c , { \hat { \mathbf { T } } } )$ from the collected domain code, where each pair consists of a function code $c$ and its corresponding test cases $\hat { \mathbf { T } }$ . To achieve this, we first match each function code $c$ with its corresponding set of test cases $\hat { \mathbf { T } }$ , ensuring they are compatible with the execution environments $E$ . We then filter $\bf { C _ { 0 } }$ to retain only those code snippets that are executable and accompanied by valid test functions, resulting in a refined set $\bf { C _ { 1 } }$ and its associated test cases, which together form the candidate subject set $\mathbf { S } ^ { \prime }$ . (3) Instruction Generation: For each code snippet $c$ in $\bf C _ { 1 }$ , we employ a LLM to generate corresponding instruction $i = \mathrm { L L M } ( c )$ . The instruction $i$ , along with the associated function code $c$ and test cases $\hat { \mathbf { T } }$ , are combined to create the final subjects in benchmark dataset, with each entry represented as $( i , c , \hat { \mathbf { T } } )$ .

# Domain Repository Collection

Referring to the work (Zhuo et al. 2024), we select six domains for code generation: computation, network, basic operation, system, visualization, and cryptography, to serve as the domain divisions for DOMAINEVAL.

To ensure that DOMAINEVAL closely aligns with the realworld code requirements of human engineers, we chose GitHub repositories as the source of raw data. We cloned a selection of representative code repositories from GitHub, particularly those with at least 100 stars, as these are considered high-quality code data and reflect the actual needs of engineers. The code repositories used in each domain are shown in Figure 2. Given that each GitHub repository represents a specific real-world application scenario, we classify the code snippets based on the repository’s domain. In practice, we use the repository’s topic labels and README files to accurately assign the code to the appropriate domain.

# Test-Method Matching & Selection

This section describes how to obtain reference solutions and their corresponding tests from code repositories to construct candidate subjects for LLM evaluation and the automated construction pipeline.

Given that code snippets from GitHub repositories are written by human engineers in real-world production environments, they tend to be more complex, with more dependencies and higher levels of encapsulation than standalone functions in some code datasets. This complexity makes it challenging to select function code that can serve as benchmark data while simultaneously acquiring corresponding test cases to form complete candidate subjects.

To address this challenge, we propose a test-method tracing strategy aimed at constructing candidate subjects for benchmarking. Our approach has two main steps: (1) Search the code repository for tests related to the reference code and perform Test-Method matching, where each function code $c$ is automatically paired with its corresponding tests Tˆ . (2) To ensure the final candidates are suitable for LLM evaluation and can smoothly pass through the automated construction pipeline, we filter the matched results using three criteria: executable, significant, and appropriate difficulty.

Test-Method Matching To match the function code with corresponding tests and package them into candidate subjects, we start with the test code and trace it back to the associated function code. Specifically, we search for Python code files in repositories and use the library ast to parse Python code into an abstract syntax tree. Then, we sequentially traverse the nodes of the syntax tree, extracting all functions and their class context (if present). Next, we select test code snippets based on two heuristic rules: First, the function or class name of a test should contain test or Test. Second, the test code snippet should contain an assert statement.

To match the selected test code with the corresponding function code, we identify ast.Call nodes as function calls. To retrieve the implementation of the called function, we consider two scenarios: If the function and test are in the same file, we traverse the abstract syntax tree to locate and unparse the function node. If the function call spans files, we use recursive path matching by analyzing Python’s import behaviors until the specific file is found. Specifically, we continuously retrieve and replace the name of the next level path based on the content of init .py and the import statement in from... import... as... format, until the specific Python file is located. Once located, the process is the same as in the first scenario, using ast to parse and identify the function node.

In real-world code repositories, the correlation between tests and function code is often not one-to-one. To address this, we identify all functions within the test code and pair them with the corresponding function code. The function code is then used as the reference solution, and we group all related tests into a test suite. This process allows us to package function code snippets from GitHub repositories with their corresponding test cases, creating candidate subjects.

Test-Method Selection After packaging the candidate subjects, we continue to construct our benchmark dataset. We notice that not every function can directly convert to benchmark data suitable for LLM evaluation. Therefore, to facilitate the automatic construction of code benchmarks, we impose three criteria on the candidate subject.

First, Executable. To use $\mathrm { P a s s } @ \mathrm { k }$ (Chen et al. 2021) for evaluation, reference solution code must be executable to verify semantic consistency with generated code against test cases. To ensure the security of our benchmark data, we utilize a sandbox to isolate code execution and maintain a list of banned keywords (Xie et al. 2024). If any of these prohibited keywords are detected, we consider the execution and evaluation of such code to be potentially risky and consequently discard it. The test environment first consists of a basic Python environment and necessary packages. To execute a piece of code, it needs adequate dependencies, so we concatenate the required context (e.g., import statements, class context, static variables) for the function. After preparing the context, we run the function code along with its tests.

Second, Significant. The code used for LLM evaluation should be important and meaningful, crucial in real-world production scenarios. For example, init functions primarily involve repetitive variable assignments, which are mechanical and lack significance, failing to reflect the capabilities of LLMs. In contrast, human engineers tend to write tests for code that implements critical functionality. Therefore, we extract the code with tests as candidates.

## System Message   
You are preparing an interview for software   
engineers. The interviewees are going to complete the \${FUNCTION NAME} function. Write a clear instruction describing this function in detail, which includes the functionality, the input   
arguments (if any), and the outputs (if any). Do not reveal test cases. Generate the instruction with the following format:   
‘‘   
Functionality: .   
Inputs: ..   
Outputs:   
## Instruction   
\${CODE}

![](images/a225c8662981fbb6ac443a5440c6b9aba27829b53a12726bb5b2c111058cc915.jpg)  
Figure 4: Dialogue prompt template for guiding the LLM to generate detailed instruction fields.   
Figure 5: Distribution of line counts for the reference code (context included) across domains in DOMAINEVAL. Box plots provide a visualization of data distribution, showing the interquartile range, median, normal range, and outliers.

Third, Appropriate Difficulty. The number of lines of reference code is one of the direct indicators of the complexity of the code generation task. We set a limit on the function implementation of reference code, i.e., the standard answer, used in the task. We restrict them to between 3 and 100 lines. On the one hand, functions with fewer than three lines typically have overly simple logic (Yu et al. 2024) and do not effectively reveal the shortcomings of LLMs. On the other hand, functions exceeding 100 lines may contain overly complex logic, which can present significant challenges for evaluation. These challenges include exceeding LLMs’ context limitations or overwhelming their information capacity, making it difficult to generate precise instructions in subsequent steps and thus hindering the automatic construction pipeline.

# Instruction Generation

After filtering the candidate subjects, our subjects still lack the instruction field. To ensure the reproducibility of the dataset construction process, we employ open-source LLM, i.e.Qwen2-72B-Instruct-GPTQ-Int4, for data generation. We utilize a dialogue prompt template, as depicted in Figure 4, to guide the LLM in generating detailed instruction fields for each validated subject. These instruction fields comprehensively describe the function, including its purpose, input arguments, and expected outputs, which serve as the input for evaluating the code generation capabilities of LLMs. By leveraging LLMs, we can generate natural language descriptions that outline the desired functionality, inputs, and outputs of the code (Xie et al. 2024; Li et al. 2024a).

Finally, to construct the template for code generation, we mask the function code, following prior work (Xie et al. 2024), by [MASK] as shown in the method code mask field in Figure 3, and instruct LLMs to complete the masked segments during the evaluation.

# Benchmark Statistics

The above pipeline generates instructions, reference code, and a series of tests shown in Figure 3, which combine to form complete subjects. As a result, DOMAINEVAL consists of 2454 code subjects. Figure 2 shows the number of subjects constructed from repositories across different domains. Overall, the number of subjects from each domain is at least 100. The computation domain encompasses the greatest number of subjects, with 1705 subjects, compared to $1 0 0 \sim 2 5 6$ subjects in other domains. This is because the computation-related repositories offer more function code and also include a larger number of test cases to ensure the accuracy of each computational operation.

Figure 5 illustrates the distribution of lines of reference code within DOMAINEVAL. Overall, the lines of reference code (context included) across six domains are similar, ranging from 4 to 198, with an average of 55.69. In particular, code in computation has slightly more lines of code than that in other domains, with an average of 63.20 lines of code compared with $3 3 . 9 5 \sim 4 2 . 0 3$ in other domains.

# Experiments

# Experiment Setup

Studied LLMs. We assess 12 representative instructiontuned LLMs against DOMAINEVAL, including GPT-3.5- turbo, GPT-4o-mini (Brown et al. 2020; Achiam et al. 2023), Qwen2 (Yang et al. 2024), Phi-3 (Abdin et al. 2024)), DeepSeek-Coder series (Zhu et al. 2024; Guo et al. 2024), Llama2 (Touvron et al. 2023) and CodeLlama series (Roziere et al. 2023), CodeQwen1.5 (Bai et al. 2023) with size of open-source models from 6.7B to 72B. These models exhibit proficiency in following instructions and delivering appropriately formatted responses.

Evaluation Metrics. Our evaluation uses the unbiased version of $\mathrm { P a s s } @ \mathrm { k }$ (Chen et al. 2021) to accurately assess the functional correctness of code snippets generated by LLMs. Following prior work (Zhuo et al. 2024), we report $\mathrm { P a s s } @ 1$ and Pass $\textcircled { a } 5$ for the experiment in zero-shot setting and use macro-average as scores. For $\mathrm { P a s s } @ 1$ metric, we use greedy decoding, i.e.set temperature to 0.0. For Pass $\textcircled { a } 5$ metric, we opt for the minimum sample size $N = 5$ and maintain temperature at 0.2 and top- $\cdot \mathbf { p }$ at 0.95. For code generation tasks, we use torch.bfloat16 when loading LLMs.

Evaluation Process. During the evaluation, to prevent LLMs from failing execution due to omitted import statements, which is a tolerable flaw but could potentially distort assessment results, we implement a corrective measure, i.e., completing the missing dependencies based on the import scenario mentioned in the instruction.

# Overall Result

Figure 6 shows LLMs’ $\mathrm { P a s s } @ 1$ and $\mathrm { P a s s } @ 5$ against DOMAINEVAL. Columns plotted in blue show the Pass $\textcircled { \omega } 1 / 5$ values; the bluer, the larger. The columns in orange and green highlight the average (“Mean”) and standard deviation (“Std”) of the corresponding rows, respectively. Overall, the average performance across studied LLMs is similar, ranging from $4 9 . 8 9 \% \sim 6 7 . 1 3 \%$ . At the same time, the performance across domains varies, i.e., the performance in Computation reaches the top among all LLMs, with an average of $8 2 . 4 4 \%$ Pass $\ @ 1$ and $8 8 . 5 7 \%$ Pass $\textcircled { a } 5$ , while the worst scores are observed in Cryptography domain, with $3 3 . 0 8 \%$ Pass $\ @ 1$ and $4 0 . 2 5 \%$ Pass $\textcircled { a } 5$ . In other words, the performance gaps between different domains cannot be ignored.

# Domain Biases

From Figure 6, we can see significant gaps across six domains. Horizontally, LLMs are generally good at computation tasks while falling short on cryptography and system coding tasks. In particular, LLMs excel in computation domain, where $\mathrm { P a s s } @ \mathrm { k }$ metrics all exceed $7 5 \%$ , with some reaching over $90 \%$ . When it comes to cryptography and system domains, LLMs exhibit significantly lower performance, with average $\mathrm { P a s s } @ 1$ of $3 3 . 0 8 \%$ and $3 7 . 5 0 \%$ , respectively. The performance gap can be as much as $6 8 . 9 4 \%$ $( 8 0 . 9 4 \% - 1 2 . 0 \% )$ Pass $\ @ 1$ in Llama-2-13B-Chat. Vertically, all LLMs exhibit similar domain biases, i.e., LLMs universally show consistent performance gaps with a shared trend of strengths and weaknesses.

In addition, a recent work (Zhuo et al. 2024) also explored the coding capability (i.e., using APIs to implement domain-specific code) across various domains and concluded that LLMs are good at cryptography domain. Our conclusion does not conflict with theirs because we generate the domain-specific code while they invoke the domainspecific APIs. In other words, being good at calling APIs in a domain does not mean being good at implementing code in the domain. Therefore, our finding serves as a supplement to previous work (Zhuo et al. 2024).

# LLMs Biases

Among 12 studied LLMs, the closed-source model GPT4o-mini exhibits the average highest performance, with a $6 7 . 1 3 \%$ Pass $\textcircled { a } 5$ . Qwen2-72B-Instruct-GPTQ-Int4 has the best overall performance among open-source models, with a $6 4 . 2 5 \%$ Pass $\textcircled { a } 5$ . Moreover, considering the variation across domains, GPT-4o-mini exhibits the most stable performance, with a 14.75 standard deviation in Pass $\textcircled { a } 5$ , compared with $1 5 . 4 5 \sim 2 4 . 1 0$ of other LLMs.

Figure 6: Pass $\ @ 1$ and $\mathrm { P a s s } @ 5$ of LLMs against DOMAINEVAL. We use Comp for computation, Visual for visualization, and Crypt for cryptography. Mean represents the macro-average of $\operatorname* { P a s s } ( \varnothing \mathrm { k }$ across different domains, which is used to reflect the overall performance of LLMs. Std indicates the standard deviation of Pass $@ \mathbf { k }$ across different domains, which is used to reflect the different degrees of performance of LLM across various domains. To highlight the differences, we use color scales.   

<html><body><table><tr><td>Pass@1 (Greedy Search N=1)</td><td>Size</td><td>Comp</td><td>Network</td><td>Visual</td><td>Basic</td><td>System</td><td>Crypt</td><td>Mean</td><td>Std</td></tr><tr><td>GPT-4o-mini</td><td>一</td><td>90.38</td><td>70.31</td><td>59.68</td><td>69.16</td><td>51.00</td><td>43.00</td><td>63.92</td><td>16.68</td></tr><tr><td>GPT-3.5-turbo</td><td>一</td><td>83.40</td><td>58.98</td><td>48.92</td><td>56.07</td><td>32.00</td><td>31.00</td><td>51.73</td><td>19.50</td></tr><tr><td>Qwen2-72B-Instruct-GPTQ-Int4</td><td>72B</td><td>86.86</td><td>66.80</td><td>49.46</td><td>69.16</td><td>41.00</td><td>36.00</td><td>58.21</td><td>19.39</td></tr><tr><td>DeepSeek-Coder-33b-instruct</td><td>33B</td><td>83.93</td><td>64.45</td><td>50.54</td><td>59.81</td><td>46.00</td><td>35.00</td><td>56.62</td><td>16.94</td></tr><tr><td>DeepSeek-Coder-V2-Lite-Instruct</td><td>16B</td><td>86.04</td><td>62.11</td><td>50.00</td><td>65.42</td><td>41.00</td><td>38.00</td><td>57.10</td><td>17.92</td></tr><tr><td>DeepSeek-Coder-6.7b-instruct</td><td>6.7B</td><td>83.52</td><td>58.98</td><td>45.70</td><td>57.94</td><td>36.00</td><td>40.00</td><td>53.69</td><td>17.32</td></tr><tr><td>CodeLlama-34b-Instruct</td><td>34B</td><td>76.07</td><td>60.16</td><td>41.94</td><td>55.14</td><td>35.00</td><td>31.00</td><td>49.89</td><td>17.09</td></tr><tr><td>CodeLlama-13b-Instruct</td><td>13B</td><td>80.29</td><td>62.11</td><td>42.47</td><td>58.88</td><td>34.00</td><td>27.00</td><td>50.79</td><td>19.90</td></tr><tr><td>CodeLlama-7b-Instruct</td><td>7B</td><td>77.13</td><td>60.55</td><td>43.55</td><td>52.34</td><td>36.00</td><td>32.00</td><td>50.26</td><td>16.82</td></tr><tr><td>CodeQwen1.5-7B-Chat</td><td>7B</td><td>85.16</td><td>60.94</td><td>47.85</td><td>60.75</td><td>37.00</td><td>37.00</td><td>54.78</td><td>18.31</td></tr><tr><td>Phi-3-medium-4k-instruct</td><td>14B</td><td>75.54</td><td>60.16</td><td>45.16</td><td>61.68</td><td>42.00</td><td>35.00</td><td>53.26</td><td>15.10</td></tr><tr><td>Llama-2-13b-chat</td><td>13B</td><td>80.94</td><td>53.12</td><td>34.95</td><td>44.86</td><td>19.00</td><td>12.00</td><td>40.81</td><td>24.97</td></tr><tr><td>Average</td><td></td><td>82.44</td><td>61.56</td><td>46.69</td><td>59.27</td><td>37.50</td><td>33.08</td><td>53.42</td><td>18.33</td></tr><tr><td>Pass@5 (Sampling Search N=5)</td><td>Size</td><td>Comp</td><td>Network</td><td>Visual</td><td>Basic</td><td>System</td><td>Crypt</td><td>Mean</td><td>Std</td></tr><tr><td>GPT-4o-mini</td><td>一</td><td>91.26</td><td>72.66</td><td>61.83</td><td>71.03</td><td>57.00</td><td>49.00</td><td>67.13</td><td>14.75</td></tr><tr><td>GPT-3.5-turbo</td><td>一</td><td>87.33</td><td>62.89</td><td>52.15</td><td>60.75</td><td>36.00</td><td>34.00</td><td>55.52</td><td>19.74</td></tr><tr><td>Qwen2-72B-Instruct-GPTQ-Int4</td><td>72B</td><td>90.15</td><td>70.70</td><td>54.84</td><td>73.83</td><td>50.00</td><td>46.00</td><td>64.25</td><td>16.90</td></tr><tr><td>DeepSeek-Coder-33b-instruct</td><td>33B</td><td>89.79</td><td>70.70</td><td>55.38</td><td>68.22</td><td>57.00</td><td>42.00</td><td>63.85</td><td>16.34</td></tr><tr><td>DeepSeek-Coder-V2-Lite-Instruct</td><td>16B</td><td>88.91</td><td>65.62</td><td>53.76</td><td>68.22</td><td>49.00</td><td>44.00</td><td>61.59</td><td>16.35</td></tr><tr><td>DeepSeek-Coder-6.7b-instruct</td><td>6.7B</td><td>89.79</td><td>63.67</td><td>55.38</td><td>67.29</td><td>49.00</td><td>44.00</td><td>61.52</td><td>16.36</td></tr><tr><td>CodeLlama-34b-Instruct</td><td>34B</td><td>85.10</td><td>63.28</td><td>48.39</td><td>62.62</td><td>41.00</td><td>42.00</td><td>57.07</td><td>16.83</td></tr><tr><td>CodeLlama-13b-Instruct</td><td>13B</td><td>89.85</td><td>65.62</td><td>51.61</td><td>66.36</td><td>38.00</td><td>35.00</td><td>57.74</td><td>20.55</td></tr><tr><td>CodeLlama-7b-Instruct</td><td>7B</td><td>86.80</td><td>63.67</td><td>51.61</td><td>64.49</td><td>43.00</td><td>40.00</td><td>58.26</td><td>17.28</td></tr><tr><td>CodeQwen1.5-7B-Chat</td><td>7B</td><td>91.03</td><td>64.06</td><td>55.38</td><td>68.22</td><td>45.00</td><td>42.00</td><td>60.95</td><td>17.95</td></tr><tr><td>Phi-3-medium-4k-instruct</td><td>14B</td><td>85.10</td><td>67.58</td><td>54.30</td><td>67.29</td><td>47.00</td><td>44.00</td><td>60.88</td><td>15.45</td></tr><tr><td>Llama-2-13b-chat</td><td>13B</td><td>87.68</td><td>55.86</td><td>39.78</td><td>48.60</td><td>26.00</td><td>21.00</td><td>46.49</td><td>24.10</td></tr><tr><td>Average</td><td>1</td><td>88.57</td><td>65.53</td><td>52.87</td><td>65.58</td><td>44.83</td><td>40.25</td><td>59.60</td><td>17.72</td></tr></table></body></html>

Notably, CodeLlama-13B, which fine-tuned from Llama2-13B, achieves an $1 1 . 2 5 \%$ $( 5 7 . 7 4 \% - 4 6 . 4 9 \% )$ average improvement, while the deviation across domains still remains. It indicates that although fine-tuning can bring about overall improvement, while the domain gaps still exist.

# Impact of Generated Samples

Finally, we analyze the impact of generated samples (i.e., (Pass@1 Greedy Search $N { = } I$ ) with sub-table $( P a s s @ 5$ Sampling Search $N { = } 5$ )). From Figure 6, we can see that after increasing the number of samples from 1 to 5, the average performance increases from $5 3 . 4 2 \%$ to $5 9 . 6 0 \%$ , with consistent improvements on all six domains. Yet, in terms of standard deviation (the smaller, the less bias, the better), there is little improvement, from an average 18.33 to 17.72. What is worse, CodeLlama-13B-instruct even observes an increased deviation, from 19.90 to 20.55, indicating a more bias as generation goes on. In other words, generating more sam

# ples can increase the overall performance, while the domain bias may even increase.

# Case Study

Despite impressive performance exhibited by LLMs in computation domain, their shortcomings in other domains cannot be overlooked. This claim is supported by two indicative cases from cryptography and system domains, which highlight the challenges faced by LLMs. LLMs need to acquire more background knowledge to enhance their code generation capabilities in specific vertical domains.

The context of this subject is a classical cryptography scenario, the attack method targeting the RSA encryption algorithm. This involves the recovery of the two prime factors, $p$ and $q$ , of the RSA modulus $N$ , given the public key (consisting of the modulus $N$ and the public exponent e) and the private key (the private exponent $d$ ). When encrypting, in order to enhance the security and impede decryption attempts, the selected $p$ and $q$ by human are both extremely large prime numbers. Although the instruction mentioned keywords such as RSA, the model does not realize that in

import math   
from random import randrange   
def attack(N, e, d): if discriminant $> = ~ 0$ : sqrt_discriminant $\mathbf { \Sigma } = \mathbf { \Sigma }$ int(discriminant\*\*0.5) sqrt_discriminant $\mathbf { \Sigma } = \mathbf { \Sigma }$ math.isqrt(discriminant ))   
import collections   
import os   
import re   
from psutil.tests import sh   
def free_physmem():   
- output $\mathbf { \Sigma } = \mathbf { \Sigma }$ os.popen(‘free $- \mathbf { b } ^ { \prime }$ ).read()   
+ output $\mathbf { \tau } = \mathbf { \tau }$ sh(['free','-b'],env={'LANG':'C.UTF-8'}) mem_values $\mathbf { \tau } = \mathbf { \tau }$ re.split(r'\s+',mem_line.strip())   
- shared $\mathbf { \Psi } = \mathbf { \Psi }$ int(mem_values[5]) shared $\mathbf { \tau } = \mathbf { \tau }$ int(mem_values[4])

the context of cryptography, rounding the square root of a large number cannot be directly converted to $* * 0 . 5$ , and instead math.isqrt should be used to avoid OverflowError.

The response presented in Figure 7 is from GPT-4o-mini, which is considered as the premier model within cryptography domain. The error in its response is failed: int too large to convert to float. Similar errors are also observed in some responses from DeepSeek and Qwen series models.

# Related Work

titions that emphasize logical thinking. MBPP (Austin et al. 2021) and HumanEval (Chen et al. 2021) are datasets crafted manually for testing. These datasets are designed for evaluation, featuring highly independent functions and simplistic, idealized problem scenarios. Standalone functions are predominantly focused on by these benchmarks; however, nonstandalone functions are commonly encountered in pragmatic code generation scenarios (Yu et al. 2024).

In the same way, although GPT-4o-mini exhibits the best performance in system domain, Figure 8 shows a flawed response from it. The function is designed to parse the output of free command (run with - $\_ b$ option) to determine the physical memory state on a Linux system. However, the model fails to ensure that the output format of free command remains consistent across different language and regional settings, resulting in incorrect string matching during subsequent parsing. Similar errors are observed in models such as CodeQwen1.5-7B-Chat, Qwen2-72B-InstructGPTQ-Int4, Phi-3-medium-4k-instruct, the DeepSeek series, and the CodeLlama series. Additionally, GPT-4o-mini lacks sufficient understanding of free command, leading to misjudgments regarding the position of shared. Ultimately, none of LLMs are able to pass this subject.

Some other benchmarks attached importance on realworld problems by sourcing data from real scenarios such as StackOverflow, GitHub, and curating data manually to form benchmark datasets. For instance, PandasEval, NumpyEval (Zan et al. 2022), and SecurityEval (Siddiq and Santos 2022) are tailored benchmarks for specific scenarios (Zan et al. 2023). Focusing on specific scenarios is their limitation.

To assess the code generation capabilities of models, numerous code benchmarks were introduced such as APPS (Hendrycks et al. 2021) and CodeContests (Li et al. 2022), which are datasets sourced from algorithmic design compe

CoderEval (Yu et al. 2024), ClassEval (Du et al. 2023), DevEval (Li et al. 2024b), and ODEX (Wang et al. 2023) were constructed in open domains, but they required significant investment of human labor in the stages of data curating, filtering, or annotation.

RepoEval (Zhang et al. 2023), MultiPL-E (Cassano et al. 2022), and Exec-CSN (Xie et al. 2024) were constructed with little human involvement. However, RepoEval is focused on repository level task, which differs from ours. MultiPL-E obtained its data by translating other code datasets, and thus, it was not oriented towards open domains. Exec-CSN employed LLMs to curate the CodeSearchNet dataset and generate test cases, resulting in a final dataset that has a gap from the real-world code on GitHub.

Moreover, previous benchmarks including (Chen et al. 2024; Li et al. 2024a; Ding et al. 2023; Liu et al. 2024) have not explored the difference of code generation capability of LLMs across multiple domains. Though a recent work (Zhuo et al. 2024) exercised capability of LLMs in using APIs from several domains, but its benchmark need human-LLM collaboration to construct. Our DOMAINEVAL benchmark is designed not for tool utilization, but for concrete implementation of APIs and functions. Furthermore, DOMAINEVAL can be constructed through a fully automated pipeline across open domains.

# Conclusion

We introduce DOMAINEVAL, a function code generation benchmark across multiple programming domains. Our research underscores the need for a comprehensive benchmark to evaluate LLMs’ code generation capabilities, both broadly and within specific verticals. The automated pipeline we introduce ensures dataset diversity and realtime updates, while enabling the creation of custom domain benchmarks for others. Preliminary results show that while LLMs excel in computation, their performance in cryptography and system requires enhancement.

For the future, several research directions merit exploration. First, developing specialized training strategies and data augmentation techniques to improve LLMs’ code generation capabilities for specific domains like cryptography and system. Second, leveraging our automated pipeline to create benchmarks for a broader range of private and domain-specific code, aiming to assess and enhance LLMs.
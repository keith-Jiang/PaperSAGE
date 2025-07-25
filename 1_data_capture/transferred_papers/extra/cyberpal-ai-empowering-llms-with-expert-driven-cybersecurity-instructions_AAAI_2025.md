# CyberPal.AI: Empowering LLMs with Expert-Driven Cybersecurity Instructions

Matan Levi1,2, Yair Allouche1, Daniel Ohayon1, Anton Puzanov1

1IBM Research 2Ben-Gurion University   
matanle@{il.ibm.com, post.bgu.ac.il}   
daniel.ohayon,antonp,yair @il.ibm.com

# Abstract

Large Language Models (LLMs) have significantly advanced natural language processing (NLP), providing versatile capabilities across various applications. However, their application to complex, domain-specific tasks, such as cybersecurity, often faces substantial challenges. In this study, we introduce SecKnowledge and CyberPal.AI to address these challenges and train security-expert LLMs. SecKnowledge is a domain-knowledge-driven cyber-security instruction dataset, meticulously designed using years of accumulated expert knowledge in the domain through a multi-phase generation process. CyberPal.AI refers to a family of LLMs fine-tuned using SecKnowledge, aimed at building securityspecialized LLMs capable of answering and following complex security-related instructions. Additionally, we introduce SecKnowledge-Eval, a comprehensive and diverse cybersecurity evaluation benchmark, composed of an extensive set of cyber-security tasks we specifically developed to assess LLMs in the field of cyber-security, along with other publicly available security benchmarks. Extensive evaluations demonstrate a significant average improvement of up to $24 \%$ over the baseline models, underscoring the benefits of our expertdriven instruction dataset generation process. These findings contribute to the advancement of AI-based cyber-security applications, paving the way for robust security-expert LLMs that can enhance threat-hunting and investigation processes.

# 1 Introduction

The rapid progress of LLMs offers a wide range of new capabilities that would have been considered unrealistic only a few years ago. LLMs have emerged as disruptive technology in domains ranging from healthcare to finance, changing the way we consume information and perform our daily tasks. As LLMs are trained on trillions of tokens, they should have fundamental knowledge of most domains available online.

One such domain is cyber-security. Yet, cyber-security is also a very complex domain. It requires deep understanding in multiple areas of expertise, such as operating systems, network and communication protocols, malware analysis, threat management, and many others. Furthermore, as cybersecurity practice spans from security at the physical layer to security at the application layer, navigating this diverse landscape requires a comprehensive understanding and the ability to connect disparate elements effectively. Therefore, traditional data generation methods will not be effective (Mitra et al. 2023). As cyber-security is complex and highly domain-expert-driven, it is required to present LLMs with domain-specific data generated from expert knowledge to unlock and harness the potential of LLMs in the field.

Over the past decades, security experts have invested considerable time and resources into monitoring cyber activities, investigating incidents, and producing high-quality reports and comprehensive knowledge bases, which include detection rules for identifying and mitigating threats, among other crucial activities. This study seeks to utilize this extensive domain knowledge and, by integrating it with the capabilities of LLMs, create a highly valuable instruction-tuning dataset that can unlock the potential of LLMs in cybersecurity. Overall, we make the following contributions:

• We construct SecKnowledge, a reasoning instruction tuning dataset generated using an expert-driven process on a wide range of security-related datasets. The dataset construction involves two main steps. In the first step, instructions are created based on schemas established through domain expertise. These schemas define templates that are filled with domain-expert knowledge and supplemented with LLM-generated content when necessary. In the second step, the initial dataset is expanded through a novel hybrid synthetic content-based data generation process.   
• We train CyberPal.AI, a family of cyber-security expert LLMs designed to understand and reason about complex security concepts. CyberPal.AI demonstrates the advantages of enhancing LLMs with our domain-knowledge instruction dataset, SecKnowledge.   
• We developed SecKnowledge-Eval, a suite of evaluation datasets specifically designed to assess LLMs in the cyber-security domain. SecKnowledge-Eval consists of evaluation datasets we constructed to assess LLMs’ capabilities on complex cyber-security tasks, alongside public benchmarks, intending to generate a comprehensive and diverse evaluation dataset for assessing both knowledge and understanding of models in the field of cyber-security. CyberPal.AI demonstrated superior performance over its baseline models, showing a substantial average improvement of up to $24 \%$ in training-aligned

tasks and up to $10 \%$ in public cyber-security benchmarks. CyberPal.AI also demonstrated better robustness to adversarial benchmarks, showing more than double the robustness of the baseline models.

# 2 Related Work

General instruction tuning (Wei et al. 2021; Longpre et al. 2023; Raffel et al. 2020; Xu et al. 2022; Sanh et al. 2021; Chung et al. 2022; Ouyang et al. 2022; Chiang et al. 2023) demonstrates how fine-tuning Language Models (LMs) with NLP instructions enhances base models’ performance in following general instructions.

Our work falls within the line of research focuses on developing expert LLMs through instruction tuning in specific domains, such as writing assistants (Zhang et al. 2023), arithmetic (Liu and Low 2023), translation (Jiao et al. 2023), medicine (Thawkar et al. 2023), code (Chaudhary 2023; Luo et al. 2023), and many others.

Specifically for the domain of cyber-security, there have been several studies that aimed at training security models. Although not directly related, a line of works trains Encoderonly architecture on security data (Bayer et al. 2022; Park and You 2023; Ranade et al. 2021; Aghaei et al. 2022). However, these models are neither generative nor were trained to follow instructions. Specifically for fine-tuning generative models for cyber-security, VulDetect (Omar and Shiaeles 2023) fine-tuned GPT-2 on a dataset containing both vulnerable and non-vulnerable code. The model is fine-tuned to detect anomalies that represent regular behavior. CyberBench was introduced by Liu, Shi, and Buford (2024) as a cybersecurity evaluation dataset that was collected from different works and combined into one security benchmark that includes Name Entity Recognition (NER) tasks for cybersecurity corpus, summarization of security blogs, multichoice Q&A and Classification tasks. SecureFalcon (Ferrag et al. 2023) was trained to differentiate between vulnerable and non-vulnerable C code samples, and is specialized in detecting software vulnerabilities. In contrast to previous efforts, we do not focus on one or more predefined set of tasks. We generate a highly complex and diverse dataset of security instructions spanning a broad spectrum of topics and skills using a domain-expert-driven instruction generation process. As will be described below, we use both domainexpert knowledge alongside LLM generation capabilities to populate our security instruction dataset. This comprehensive dataset enables us to train general-purpose security models.

# 3 SecKnowledge: Domain-knowledge Driven Cyber-security Instruction Dataset

This section details the construction of SecKnowledge, a novel instruction tuning dataset tailored for cyber-security. We leverage expert knowledge and employ a two-step process to build a comprehensive and diverse dataset capable of supporting instruction tuning for various security-related tasks. The two-step process is defined as follows:

1. The first generation step focuses on creating highquality instructions based on predefined schemas. These schemas are established through experts-driven in-depth analysis of the diverse set of security datasets, their individual characteristics, and the relationships between different entities within and between datasets. This ensures that the instructions are relevant, accurate, and capture the nuances of various security concepts and tasks. More specifically, each predefined schema consists of rules by which the data-source should be processed into instructions using parsers we developed, ensuring that the generated instructions focus on the important and unique characteristics of the data-source, and are representative of real-world security scenarios. Our method can be considered as an extension to methods such as Wei et al. (2021); Longpre et al. (2023), where templates are simply assigned with predefined questions and answers. In Section 3.1 we break down the generation process.

2. The second generation step expands the generated initial dataset and improves its diversity and complexity. To do so, we employ a hybrid synthetic content-grounded data generation process. More specifically, we fused EvolInstruct (Xu et al. 2023) and Self-Instruct (Wang et al. 2022) and combined them with content-grounded generation and evaluation pipelines. Additionally, we implemented a routing mechanism between the two generation methods that helps to reduce hallucinations. This process leverages the initial set of instructions from the first generation step to generate additional instructions that follow the established schemas but increase the model’s overall generalizability. By incorporating content-grounded synthetic data, we increase the diversity and volume of the final dataset, ultimately leading to more robust and capable security models. In Section 3.2, we further elaborate on the specifics of the generation process.

Our final SecKnowledge dataset consists of various instruction types, among which are: open/closed book question answering, yes/no questions, multi-choice Q&A, CoT (Wei et al. 2022), summarization, logic validation, odd/leave one out multi-choice Q&A, question generation, query/rule explanation and generation, TTP mapping, and others.

Figure 1 illustrates our SecKnowledge generation flow, and Table 1 summarizes the security instruction datasets we composed in the first generation step. We selected these initial data sources because they encompass fundamental security concepts and common, practical scenarios such as rule creation and summarization. We detail the generation process for the primary datasets below. Unless otherwise specified, we use the open-source Mixtral (Jiang et al. 2024) and Mistral-Large-Instruct models for both data generation and evaluation processes.

# 3.1 First Generation Step: Domain Knowledge-Driven Instruction Generation

Leveraging domain expertise, we first parse and enrich each one of the various security data sources using their unique characteristics and structure, derive connections between the documents in each data-source, and even derive connections between different data sources, as we will describe in the upcoming sections.

MITRESchema Example Tactics Structure-driven   
Techniques Use-cases Generation MITRE Interdata A   
data source sources Grounded   
relationships relationships Hybrid SecEach data source is SDG Knowledge   
defined by a schema Expert-driven Parsing Structure & LLMObjects Augmented Generation Data Sources

We establish a set of predefined, domain-knowledgedriven, schemas that capture the essential elements of different security tasks. Each schema consists of a series of pre-defined, domain-expertise-driven rules. Each rule is then translated into a parsing object. The parsing object will then generate and fill customized instruction templates with the parsed data. The schemas capture each dataset’s unique objectives and characteristics. This approach ensures that the generated instructions accurately reflect the desired model behavior. The subsequent paragraphs provide a detailed description of the data sources and methodologies employed in the first step of the SecKnowledge data generation process.

Table 1: Overview of the initial instructions constructed by the first step as described in 3.1. These instructions will be used as the seed for the second generation step.   

<html><body><table><tr><td>Dataset</td><td>#of generated instruction</td></tr><tr><td>MITREATT&CK CWE CVE</td><td>45,901 4,080 8,447</td></tr><tr><td>CAPEC Security Wiki Security interview Q&A Threat reports BRON</td><td>3,917 11,000 500 4,500 62,227</td></tr><tr><td>SIEM rules (TTP mapping) Sigmarules</td><td>400 9,329</td></tr><tr><td>Security Stack Exchange Total</td><td>2,573 152,874</td></tr></table></body></html>

Structure-driven instruction generation The straightforward method for creating an instruction dataset from the documents is to provide a teacher model with raw documents and instruct it to generate instructions based on the content. However, relying on models to produce instructions that simultaneously capture the unique characteristics of datasets while maintaining complexity and diversity proves to be a difficult task. One reason is that models tend to focus on specific or localized sections of a document when generating instructions. More significantly, models struggle to capture and exploit the relationships between different components within each dataset and the relationships between different datasets.

To utilize the structured nature of the different security datasets and frameworks, we built unique parsers that are fed with the relationships between different entities in the datasets, alongside a schema of required instructions.

Our approach exploits the structured nature of the various cyber-security documents to create a high-quality, diverse, and complex instruction dataset. We demonstrate the efficiency of our method using the MITRE frameworks, among which we can find: MITRE ATT&CK (comprehensive knowledge base of adversary tactics, techniques, detections, and mitigations), CWE (Common Weakness Enumeration), CVE (Common Vulnerabilities and Exposures), CAPEC (Common Attack Pattern Enumeration and Classification), and more. Compiling these frameworks encompasses a vast repository of cyber-security domain knowledge and offers extensive coverage of the security field, making it an excellent resource for fine-tuning our model to the specific requirements and nuances of cyber-security.

Each MITRE framework comprises a structured format that categorizes different aspects of the subject matter, enabling organized analysis of the different security aspects. As such, we create a schema for each MITRE framework, where each framework’s schema defines the following:

1. An instruction set aimed to teach the model the specific characteristics of each object (i.e., tactic, technique, mitigation, detection, attack pattern, etc.) For example, an instruction might detail the relationships between an attack pattern and its corresponding severity, prerequisites, or consequences.   
2. An instruction set aimed to teach the model the relationships between different objects within each dataset

![](images/ee584d0681cbba767b9aab3f7363807385346cdc8c53646d6ede4f5f3c3db0d1.jpg)  
Figure 1: High level overview of our SecKnowledge generation pipeline.   
Figure 2: MITRE ATT&CK components relationship.

Next, we provide an example of chain-of-thoughts instruction generated by utilizing the MITRE ATT&CK structure. Figure 2 demonstrates the relationships between different objects within the MITRE ATT&CK framework. Using these relationships, complex instructions are constructed on the wide range of the attack landspace. For instance, see Figure 3, where CoT training example is created from a malicious software to the exploited tactic.

Note that no language model is used during the construction; the connections and relevant text are derived based on our knowledge of the dataset’s structure.

Structured LLM-Augmented Instruction Generation Previously, we demonstrated how to use raw data alongside domain expertise to populate our schema templates. Here, we combine the same abilities of robust predefined schemas and domain knowledge with the flexibility and rea

Instruction: Instruction:   
Which tactic is related to the following Which tactic is related to the following application of software{software_name): application of software "Kwampirs":Kwampirs (software_usage) copies itself overnetworksharestomove Answer: laterallyonavictimnetwork   
{software_description} Answer:   
Kwampirs is a backdoor Trojan used by {software_name) is identified by ID Orangeworm.Ithasbeenfoundonmachines {software_id}. which had software installed for   
The provided usage of {software_name} ..Kwampirs is identified byMITRE ID S0236. ({software_id}) pertainsto the   
MITRE sub-technique The provided usage of Kwampirs software {sub_technique_name) ({sub_technique_id}). (S0236) pertains to the MITRE sub-technique: This is since using the given sub-technique, "Remote Services:SMB/WindowsAdmin {sub technique description}. Shares"(T1021.002). This is sinc.. sub-technique {sub_technique_name} sub-technique "Remote Services:   
({sub_technique_id}) isa type of technique SMB/Windows Admin Shares" (T1021.002) is {technique_name) ({technique_id) atypeoftechnigueRemoteServices(T1021) {technique_name) ({technique_id}) isa type Technique Remote Services (T1021) isa type of{tactic_name} ({tactic_id}). of tactic Lateral Movement   
Therefore,theanswer is:tactic (TAO0o8).Therefore,theanswer is: tactic {tactic_name_and_id}. Lateral Movement (TA0008).

soning abilities of LLMs to create comprehensive instructions. This approach leverages structured templates for consistency while utilizing a teacher model to dynamically generate and fill specific content, ensuring both accuracy and adaptability in instruction creation. More specifically, our main goal of using the teacher model is not to generate general content but rather to harness its reasoning capabilities to guide our models to reason on complex security concepts.

We use this approach on the following datasets: BRON, SIEM Rules to TTP Mapping, and Sigma Rules.

BRON: BRON (Hemberg et al. 2021) is a knowledge graph that interconnects threat data sourced from MITRE ATT&CK, CAPEC, CWE, CVE, MITRE Engage, MITRE D3FEND, MITRE CAR, and exploitdb. This interconnected graph enhances the capabilities of security researchers in conducting advanced threat hunting.

After demonstrating how the MITRE frameworks can be utilized (individually) to generate instructions on the specific characteristics of each MITRE object and the relationships between different objects within each framework, we will leverage BRON to generate instructions on the relationships between different objects across frameworks.

With hundreds of thousands of nodes and millions of edges interconnecting them, BRON’s sheer scale makes it impractical to feed directly to an LLM with the expectation of comprehensively learning all relationships. Therefore, our objective is to generate an instruction set that teaches the model to reason if and how different entities are related.

Specifically, using BRON, we have two main goals: 1) construct instructions that will guide LLMs on how to reason if two consecutive entities are related to each other (e.g., CWE and CVE nodes), and 2) showcase the reasoning process for LLMs to derive the path from a specific entity of interest to any other entity in the graph , to accommodate user instruction. This reasoning process will enable a more comprehensive understanding of the relationships between different entities, such as the connection between a platform and its relevant weaknesses, which are not directly related.

To meet the stated goals, the graph should be processed and traversed into paths, which we will later enrich with domain knowledge from the different resources. These paths and explanations will become CoT examples that can guide LLMs to perform effective, complex threat-hunting reasoning. Our processing consists of the following four steps:

Paths Extraction First, we gather all one-step paths between nodes of different types that are directly connected in the graph (e.g., all connections between tactic nodes and technique nodes). Next, for all non-direct paths, we perform a random walk on the graph and construct up to 5000 paths between each pair of node types that are not directly connected (e.g., paths between tactic nodes and CVE nodes).

Derive the Connection Between Direct Nodes After extracting one-step paths between nodes of different types that are directly connected in the graph, we take these direct links and use the reasoning process of a teacher model to explain the connection between each pair of nodes. This involves sending the teacher model instructions that include the descriptions, alongside other information about each node, for each pair of nodes, requiring the teacher model to examine the information and decide if and how the nodes are connected. Additionally, we incorporate negative sampling to illustrate that not all nodes in the graph are connected, compelling the model to make decisions based on the nodes’ information. The negative sampling stage is pivotal as it’s impractical to present all existing paths (amounting to millions) to the models we fine-tune. Instead, as our models already see these data sources (e.g., CWE) separately during fine-tuning, we aim to equip them with the ability to ascertain whether two nodes are linked based on their information, in the expectation that our models will generalize to paths they haven’t encountered during training.

Constructing CoT on Paths Additionally, we present the model with longer paths that involve multiple nodes, between non-direct nodes. For each edge in the path, similar to the direct node processing, we use a teacher model to explain the connection between nodes based on the relationship status (e.g., CVE is a specific implementation of CWE) and the nodes’ information (e.g., description). The explanations of all the edges in the path is then chained together to construct a unified CoT explaining that path.

Multi-path CoT Lastly, we construct more complex instructions, i.e., instructions that can be answered only by involving multiple paths from the graph. One example can be a two-stage/paths instruction where we can first ask what the relevant attack patterns for a given weakness are and how to detect/mitigate them in the second stage. See Figure 4 for an example.

Sigma Rules: Sigma is a structured and open signature format that allows to define and describe detection logic. The rule format is flexible and platform-agnostic. The main purpose of Sigma is to provide a structured form in which researchers and analysts can describe their developed detection methods and make them shareable. SigmaHQ1 is the main rule repository where detection engineers, threat

Instruction:   
Given software epolicy_orchestrator (by mcafee) version 5.10.0,identify a CWE that might be connected to it.Then,suggest some strategies to mitigate it.Explain the rationale behind your choicesand how theyare relevant to the software.   
Answer:   
The conceivable link between application epolicy_orchestrator (by mcafee) version 5.10.0 and Authentication Bypass by Alternate Name (CWE-289) is illuminated by the sequence:   
application epolicy_orchestrator (by mcafee) version ${ \bf 5 . 1 0 . 0 } \Rightarrow$ CVE-2022-0845 $\Rightarrow$   
Authentication BypassbyAlternateName (CWE-289)   
The relationship between application epolicy_orchestrator (by mcafee) version 5.10.0 and Authentication Bypass by Alternate Name (CWE-289)is clarified by the reasoning outlined below: Step 1:   
The CVE-2022-0845affects theapplicationepolicy_orchestrator version 5.10.0by McAfee,as it is vulnerabletocode injectionduetoavulnerabilityin thepytorchlightning/pytorch-lightning GitHub repository prior to version 1.6.0.   
Step 2:   
The CWE-289 Authentication Bypass by Alternate Name is being used in CVE-2022-0845 by exploiting thecode injection vulnerabilityin pytorchlightning/pytorch-lightning prior to1.6.0,where theauthentication is performed based on the name of the resource being accessed,but itdoes not properly check all possible names for that resource.   
You can mitigate Authentication Bypass by Alternate Name (CWE-289) by:   
1)Avoid making decisions based on names of resources (e.g.files)if those resources can have alternatenames.   
Sigma rule goal: Instruction:   
DetectsObfuscated Howcan ldetect Obfuscated Powershell via use MSHTA in   
Powershell viause MSHTA in Scripts?   
Scripts Answer:   
logsource: To Detect Obfuscated Powershell via use MSHTA in   
product:windows Scripts,you should:   
category:driver_load   
detection: Monitor Windows drivers loading forany instances where   
selection: thelmagePathcontainsaspecificsequenceofelements ImagePathlcontainslall: indicatingapotentialattemptatobfuscation. 'set','&&','mshta', '(window.close)','run', Specifically,lookoutforthesimultaneouspresenceof'set', 'vbscript:createobject', &&','mshta','vbscript:createobject','run',and (window.close)'within the ImagePath.   
condition: selection   
falsepositives:Unknown This detection is critical due to its high severity level .   
level: high   
tags: The relevant tacticsand techniques from the MITRE attack.defense_evasion, ATT&CKframeworkassociatedwiththisactivityinclude attack.t1027, Defense Evasionand Execution,specifically identified as attack.execution, Obfuscated Files or Information (T1027) and Command attack.t1059.001 andScripting Interpreter:PowerShell (T1059.001).

hunters, and all defensive security practitioners collaborate on detection rules. Here, we leverage the repository’s dataset, which contains over 3000 diverse and reliable detection rules as a baseline for our rule instruction set.

Sigma rules contain multiple fields, among which are: the ”logsource” field which specifies the type of log data the rule applies to, and the ”detection” field which defines the specific conditions that trigger the rule, including event attributes, expected values, and filters for accurate detection. The ”level” field indicates the severity of the detected event. Each Sigma rule is connected to the attack it tried to detect.

We take advantage of the Sigma rules structure, feed the relevant fields to a teacher model, and construct the following types of reasoning instruction tasks:

• Step-by-step attack detection explanation using log source and rule detection filters within the detection field.   
• Step-by-step reasoning for attack type mapping via detection indicators.   
• Sigma rule generation from attack type and/or detection indicators.

We define a schema for each task, ensuring it contains the necessary information. In Figure 5, we showcase the process of converting an entry into step-by-step attack detection instructions. The process works as follows: First, we extract the log source, product, category, detection options, severity level, and any possible rule tags from the given rule. Because Sigma rules are well-structured, we provide the teacher model with an explanation of each field and its role. Next, the teacher model is given the specific values for each field. Finally, it is tasked with explaining the purpose of the rule based on its understanding of these fields and their values. This entire process is fully automated, utilizing a pipeline guided by predefined prompts and a structured flow of actions, making it applicable to any rule. As with other generation processes, we also apply an evaluator that tests the correctness of the generated text.

SIEM Rules to TTP Mapping: SIEM (Security Information and Event Management) is a security platform that monitors and correlates threat intelligence, network, and user behavior anomalies to prioritize high-fidelity alerts. We have collected a list of 400 rules from IBM’s SIEM, QRadar, along with their corresponding Tactics, Techniques, and Procedures (TTP) mappings.

TTP mapping of detection rules is critical in cybersecurity as it enables organizations to systematically identify and counteract specific adversary behaviors, thereby enhancing the precision and effectiveness of threat detection.

QRadar’s rules are well-structured and include fields such as rule ID, description, pattern, relevant MITRE tactic/technique ID and name, rule risk level, and more. In the following, we will demonstrate how we leverage this structure to develop a series of instructions for educating CyberPal.AI on mapping rules to Tactics, Techniques, and Procedures (TTPs). Our goal is not merely to create a simple mapping task but to teach the model to reason about TTP mapping. To achieve this, we combine expert knowledge with LLMs to generate a comprehensive TTP reasoning instruction dataset, as we describe below.

The process of creating the TTP mapping instruction set involves retrieving the rule description, tactic/technique ID, and name for each rule and its corresponding TTP mapping. Using this information, we access the description and additional relevant data of the tactic/technique from the MITRE ATT&CK framework. We tailor a specific schema, that leverages the required information, and guides the teacher model to reason and clarify the relationship between the rule and the provided TTP based on the descriptions and additional relevant data. Subsequently, the model generates an explanation, which undergoes evaluation for correctness by another model (evaluator). Upon acceptance by the evaluator, a set of instructions is formulated based on the rule, the TTP mapping, and the explanation. The goal of the generated set is to teach CyberPal.AI models the expected reasoning process the models need to perform when mapping between rules and TTPs.

Additional Datasets We provided a detailed explanation of the process used to create our primary data sources for the first generation step. Alongside these, we also utilized additional datasets for this first generation step, including security interview Q&A, threat reports on various security threats, discussions from security and reverse-engineering Stack Exchange, and Wikipedia pages focused on computer security. For each such dataset, we define a schema based on its structure and build instructions in a similar manner to the previously mentioned datasets.

# 3.2 Second Generation Step: Content-Ground Synthetic Data Generation

In the second step of our security instruction generation process, we expand the generated initial dataset from Section 3.1 and improve its diversity and complexity.

We build upon the ideas of Self-Instruct (Wang et al. 2022) and Evol-Instruct (Xu et al. 2023) and fuse them alongside content-grounded generation and an instruction routing mechanism. The reason for switching between the two methods is due to our empirical observations that using In-depth Evol-Instruct with content-grounded generation tends to diverge and generate inaccurate instructions after several iterations (usually around 3), resulting in instructions that an LLM cannot answer, non-grounded instructions, or instructions that deviate from the relevant topic. Therefore, we incorporate a dynamic mechanism that combines Evol-Instruct and Self-Instruct, where in the early stages of the synthetic data generation, we mainly focus on In-depth vvolving to generate more complex instructions on the same topic, and as the generation process progresses, we shift the focus towards Self-Instruct, which can be thought of as In-breath evolving, where we mainly focus on generating new tasks, while keeping them grounded and in the same domain as the document. More specifically, the probability of Self-Instruct being chosen is doubled every two iterations (See ”Instructions router” in Figure 6). We find that combining Evol and Self-Instruct leads to more grounded instructions due to the difficulty of preserving complex contentgrounded instructions in later stages. Using our dynamic method, we managed maximize quality while minimizing errors and increasing quantity. Statistically, when we attempted to generate the same number of instructions using only content-grounded Evol-Instruct, the ratio of hallucinations and incorrect responses was six times higher compared to our method.

Additionally, we incorporate an internal evaluation mechanism using an LLM evaluator. An instruction will be added to the instruction pool and be used in the next iteration only if it passes the evaluator’s assessment. The evaluator is defined by the following objectives:

• Evaluate if the generated instruction is more challenging/complex/rare (for in-depth evolving) or diverse (for Self-Instruct).   
• Evaluate if the new instruction is of the same domain as the given instruction based on the document, and evaluate that this new instruction can be answered by the document.

Repeatforkiterations In-depth content grounded LLM Add instruction Ageneration Evaluator to instructions pool if all InSurrentn In-breadouient conditions hold K 目 content Eliminateif Router conditions Document grounded arenot met generation 直 Document

• Evaluate if the generated answer correctly answers the new instruction, and that the generated answer is grounded by the document.

We employ our content-grounded SDG process to Security Wiki, Security interview Q&A, Security Stack Exchange, MITRE ATT&CK, CWE, CVE, and CAPEC. In the second generation step, we generate an additional 250,000 instructions. The final dataset comprises approximately 400,000 complex and diverse instructions following the two-generation steps.

# 4 SecKnowledge-Eval: A Comprehensive Security Evaluation Dataset

To assess CyberPal.AI’s performance, we constructed a diverse set of eight new evaluation datasets aimed at testing the model’s capabilities in cyber threat intelligence. To ensure no data contamination between the fine-tuning and testing phases, we partitioned the raw documents into train and test sets, such that the model did not encounter any test-related documents during fine-tuning. After splitting the data, we transformed the documents from the test split into the evaluation tasks described below. Furthermore, we benchmarked CyberPal.AI against another seven public and general cyber-security evaluation datasets to demonstrate its robustness and comprehensive understanding of security concepts. Overall, our evaluation benchmark consists of 15 diverse datasets, with various task types. To the best of our knowledge, this is the most comprehensive cybersecurity evaluation benchmark, comprising a diverse set of cyber-security tasks alongside other publicly available security benchmarks. The following paragraphs provide an overview of the evaluation datasets we developed:

# 4.1 Multiple Choice Tasks

Questions in this section are formatted with four multiplechoice answers, similar to (Hendrycks et al. 2020).

Adversarial MITRE ATT&CK We compiled this dataset using various MITRE ATT&CK sources. This evaluation benchmark is designed to assess the model’s knowledge of malicious software, campaigns, attack tactics, potential detections and mitigations for different attacks, etc. The input consists of information about a given MITRE instance (e.g., description), and the correct answer is the source from which it was derived (e.g., specific sub-technique). To enhance this evaluation dataset’s difficulty and test our models robustness, we developed a novel adversarial attack (Goodfellow, Shlens, and Szegedy 2014; Carlini and Wagner 2017; Levi and Kontorovich 2023) for multi-choice questions on a closed set of options, where the attack chooses the false options (from a closed list of possible options) that will confuse the model with the highest probability, while keeping the original question intact.

Specifically, to increase the difficulty of multiplechoice questions, we developed a novel adversarial attack targeting closed-domain options, where the choices are drawn from a closed list of possible options. Here’s how it works:

1. Given a multiple-choice question where the possible answers are selected from a fixed list of size $k$ , with one of the options being the correct answer.   
2. For each of the $k { - } l$ incorrect options, we create a new classification question. This classification question retains the original question but presents only two options: the correct choice and one of the $k { - } l$ incorrect options.   
3. We then query a language model (LLM) with each of these $k { - } l$ binary classification questions.   
4. From the responses, we identify the incorrect options that the model is most likely to select, given the original question and the correct answer, using the conditional loss on the incorrect option.

The process ensures that the false options selected are those most likely to confuse the model, thereby enhancing the overall difficulty of the dataset without tempering the questions’ content.

Refer to Figure 7 for an example where we posed a question related to one of the MITRE ATT&CK tactics. In this case, one option represents the correct answer, while our method selects the other three options from the list of all possible tactics, prioritizing those most likely to fool an LLM. The attack is an adversarial transfer attack, as we use Phi-3- small as the reference model (the model we attack), and test the attack results using the adversarial generated dataset on the other CyberPal.AI models.

Note that attacking questions related to MITRE ATT&CK tactics requires less computational resources since the list contains only 14 possible options, but on other types of tasks, i.e., technique/software-related tasks, there are hundreds of possible options. Therefore, this attack is time and resource-consuming, but it is done only once, during the generation of the adversarial evaluation dataset.

Regarding the performance degradation of the models on the adversarial dataset: when tested on the Adversarial MITRE evaluation dataset and its non-adversarial version, our fine-tuned CyberPal.AI models demonstrated greater robustness compared to the baseline models, showing more than double the robustness of the baseline models.

Specifically, in Table 2, our models exhibited smaller degradation in performance between the adversarial and non-adversarial versions of the MITRE ATT&CK dataset: for Mistral, our fine-tuned model showed a degradation of $6 \%$ in accuracy when tested with the adversarial version of the dataset, compared to the original Mistral model, which experienced a degradation of $14 \%$ . Similarly, for Llama, the fine-tuned model exhibited a $9 \%$ degradation in accuracy with the adversarial dataset, in contrast to Llama’s baseline model, which showed a $19 \%$ degradation. Finally, for Phi, the fine-tuned model demonstrated a degradation of $7 \%$ in accuracy with the adversarial dataset, compared to the Phi base model’s $13 \%$ degradation. These results highlight the robust knowledge CyberPal.AI acquired during our fine-tuning process and suggest that CyberPal.AI is more resilient and has successfully generalized to the domain of cyber-security.

<html><body><table><tr><td rowspan="2">Model</td><td colspan="2">ATT&CKEval.Dataset</td></tr><tr><td>Original</td><td>Adversarial</td></tr><tr><td>Mistral-7B-Instruct-v0.3 Sec-Mistral (Ours)</td><td>73.24</td><td>59.57 (-13.67)</td></tr><tr><td rowspan="2">Meta-Llama-3-8B-Instruct Sec-Llama (Ours)</td><td>98.87</td><td>92.54 (-6.3)</td></tr><tr><td>78.59 97.04</td><td>59.57 (-19.0)</td></tr><tr><td rowspan="2">Phi-3-medium-4k-instruct Sec-Phi-3-medium (Ours)</td><td>77.32</td><td>87.74 (-9.3)</td></tr><tr><td>96.76</td><td>64.50 (-12.8) 89.57 (-7.1)</td></tr></table></body></html>

Table 2: Models’ results before and after applying our adversarial attack to generate the adversarial multiple-choice dataset. The Original column presents the evaluation dataset results on MITRE ATT&CK prior to the application of our adversarial method. The Adversarial column shows the results on MITRE ATT&CK after applying our adversarial technique. It is evident that CyberPal.AI models exhibit greater robustness to adversarial changes, with their results showing less drastic variation compared to those of the nonsecurity models.

SIEM Rule TTP Mapping SIEM solutions usually include rules that detect a wide range of activities, including excessive firewall denies, multiple failed login attempts, and potential botnet activity. We developed a dataset comprising IBM’s QRadar rules, aiming to classify each rule according to the appropriate tactic or technique.

CTI Detection and Mitigation Mapping As outlined, BRON captures the interrelationships between different Cyber Threat Intelligence (CTI) frameworks. We created a dataset designed to assess model’s proficiency in mapping from tactics, techniques, attack patterns, weaknesses, and vulnerabilities to potential detections and mitigations.

CWE Technical Impact Mapping In CWE, each weakness, if successfully exploited, can lead to one or more technical impacts out of eight options: modify data, read data, DoS: unreliable execution, DoS: resource consumption, execute unauthorized code or commands, gain privileges / assume identity, bypass protection mechanism, and hide activities. This evaluation set presents the model with CWEs and their descriptions, where the goal is to map each CWE to its related technical impact.

# 4.2 Classification Tasks

CTI Relationship Prediction A major role of our model is to comprehend the relationships between different CTI frameworks. To test this ability, we have built a dataset that

$\mathsf { A } = - 2 . 5$ $\mathsf { B } = - 3 . 1$ Identify the MITRE tactic described here: objectives,or guiding furtheractivities.   
All possible answers (14 tactics): conditional   
Reconasance Perseneasion Dislever pr $\mathsf { A } = - 1 . 1$ Optixnitration   
Pratl P Empteato $\mathsf { B } = - 2 . 9$ D.consae A.Reconnaissance B.Resource Development   
Correct answer: Reconnaissance Overall 13 binary Conditional loss classification questions on incorrect answer

presents the model with two entities (e.g., instances of CVE and CWE) and two possible explanations—one explaining why the entities are related and another explaining why they aren’t. The objective is to determine if the two are related.

CTI Entity Classification This evaluation set consists of various descriptions of CTI entities (such as tactics, malwares, etc.). The objective is to classify whether given description is related to the specified entity.

# 4.3 Summarization Tasks

CWE Description Summarization We have developed a dataset containing weaknesses from the CWE dataset, intending to summarize the extended descriptions of each CWE. The target of the summarization is the short description provided for each CWE, which aims to offer a concise explanation of the CWE’s extended description.

# 4.4 Public and General Cyber-security Benchmarks

We used the following public multi-choice tasks: CISSP Assessment Questions, MMLU Computer Security (SecMMLU) (Hendrycks et al. 2020), Cybersecurity Skill Assessment2 , CyberMetric (Tihanyi et al. 2024), Cyber Threat Intelligence Multiple Choice Questions (CTI-MCQ) (Alam et al. 2024), and SecEval (Li et al. 2023). Additionally, we used the Cyber Threat Intelligence Root Cause Mapping (CTI-RCM) classification task introduced in Alam et al. (2024).

# 5 Experiments 5.1 CyberPal.AI Training Details

Similar to (Mitra et al. 2023), we’ve empirically noticed that presenting the model with instructions of increasing length improves the model’s learning ability. We extend on this idea and employ an incremental training methodology, organized at the dataset level. We structure the datasets into two hierarchical orders: first, we sequence the datasets by their data source category, with instructions from simpler data sources introduced first. For example - BRON-related instructions will be presented only after we present the model with the different MITRE frameworks BRON is composed of. In the second hierarchy, within each category, we arrange the instructions based on the increasing length of their outputs.

To train CyberPal.AI, we use our generated SecKnowledge dataset. As our base models, we used Llama-3 instruct 8B (AI@Meta 2024), Mistral instruct 7B v0.3 (Jiang et al. 2023), and Phi-3-medium- $4 \mathrm { k }$ -instruct (Abdin et al. 2024). We employ a learning rate of $4 e ^ { - 5 }$ for Llama and Phi, and $3 e ^ { - 5 }$ for mistral. Additionally, we employ linear warm-up for 125 steps. The context length is set to 4096, and effective batch size of 2048 is achieved using gradient accumulation. Based on our empirical findings, beyond 2 epochs, we observed that additional epochs have negligible impact on the final loss before the model starts to overfit.

# 5.2 Evaluation Metrics

Assessing LLMs on selected datasets requires appropriate evaluation metrics. We apply suitable metrics for each task as described below. For Multiple-choice Q&A, we employ the common HELM (Liang et al. 2022) evaluation method where the token with maximum probability is chosen. For summarization tasks, we use ROUGE (Lin 2004). For summarization, we ran 3 evaluations, computed the average, and removed one standard deviation. Finally, for classification tasks, we use accuracy as the metric. To calculate the average score for evaluation tasks, a straightforward averaging technique is utilized. For summarization tasks specifically, the mean of ROUGE-1, ROUGE-2, and ROUGE-L scores is first determined before calculating the overall average. All tasks except for summarization were tested in greedy mode. All tasks were done in a zero-shot setting.

# 5.3 CyberPal.AI Results

To demonstrate the effectiveness of our approach, we present SecKnowledge-Eval’s results for baseline models and their fine-tuned versions, trained using SecKnowledge.

Table 3: Evaluation results for CyberPal.AI models compared to the base model on designated datasets constructed to evaluate the models’ performance on training-aligned security tasks. For the MITRE ATT&CK evaluation set, we provide results for both the original evaluation set and its adversarial version, where we can see that CyberPal.AI demonstrates greater robustness.   

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">Original/Adv. MITACK</td><td rowspan="2">SIEM Rule TTP Mapping</td><td rowspan="2">CTI Detection and Mitigation</td><td rowspan="2">CWE R-1/22ao</td><td rowspan="2">Technical Mmapptng</td><td rowspan="2">CTI Relationship</td><td colspan="2">CTI</td></tr><tr><td></td><td>Entsfiainn</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>73.24/59.57</td><td>52.05</td><td>56.22</td><td>28.25/8.16/20.57</td><td>59.59</td><td>52.44</td><td>65.31</td><td>52.02</td></tr><tr><td>CyberPal.AI-Mistral</td><td>98.87/92.54</td><td>67.12</td><td>70.26</td><td>56.78/51.79/54.71</td><td>69.05</td><td>97.81</td><td>83.66</td><td>76.41</td></tr><tr><td>Meta-Llama-3-8B-Instruct</td><td>78.59/59.57</td><td>60.27</td><td>55.77</td><td>26.38/8.16/18.33</td><td>59.02</td><td>59.76</td><td>55.77</td><td>52.54</td></tr><tr><td>CyberPal.AI-Llama</td><td>97.04/87.74</td><td>64.38</td><td>81.95</td><td>46.43/38.45/43.88</td><td>66.18</td><td>97.81</td><td>81.95</td><td>74.70</td></tr><tr><td>Phi-3-medium-4k-instruct</td><td>77.32/64.50</td><td>55.47</td><td>67.92</td><td>27.96/7.83/19.94</td><td>66.76</td><td>63.75</td><td>67.92</td><td>57.84</td></tr><tr><td>CyberPal.AI-Phi</td><td>96.76/89.57</td><td>65.07</td><td>81.24</td><td>48.23/39.24/44.67</td><td>68.20</td><td>96.27</td><td>81.24</td><td>75.10</td></tr></table></body></html>

Table 4: Evaluation results for CyberPal.AI models compared to the base on public and general cyber-security benchmarks datasets. Although our models were not trained on these tasks, they exhibit significant and consistent improvement.   

<html><body><table><tr><td rowspan="2">Model</td><td rowspan="2">CISSP Assessment</td><td rowspan="2">SecMMLU</td><td rowspan="2">Cybersecurity Skill Assessment</td><td rowspan="2">CyberMetric</td><td rowspan="2">CTI-MCQ</td><td rowspan="2">CTI-RCM</td><td rowspan="2">SecEval</td><td rowspan="2">Avg.</td></tr><tr><td></td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>63.63</td><td>67.00</td><td>78.69</td><td>80.80</td><td>58.03</td><td>45.85</td><td>32.98</td><td>60.99</td></tr><tr><td>CyberPal.AI-Mistral</td><td>89.93</td><td>74.00</td><td>78.11</td><td>81.60</td><td>65.33</td><td>58.20</td><td>42.30</td><td>69.92</td></tr><tr><td>Meta-Llama-3-8B-Instruct</td><td>71.71</td><td>74.00</td><td>82.24</td><td>83.20</td><td>63.28</td><td>41.45</td><td>32.61</td><td>64.07</td></tr><tr><td>CyberPal.AI-Llama</td><td>90.40</td><td>77.00</td><td>86.98</td><td>84.80</td><td>66.41</td><td>60.65</td><td>55.04</td><td>74.47</td></tr><tr><td>Phi-3-medium-4k-instruct</td><td>77.27</td><td>78.00</td><td>83.43</td><td>87.20</td><td>65.53</td><td>30.68</td><td>45.36</td><td>66.78</td></tr><tr><td>CyberPal.AI-Phi</td><td>90.40</td><td>80.00</td><td>86.39</td><td>91.00</td><td>72.65</td><td>53.00</td><td>67.47</td><td>77.27</td></tr></table></body></html>

In Table 3, we compare the performance of CyberPal.AI across eight training-aligned tasks. Our fine-tuned CyberPal.AI models demonstrate significant and consistent improvements across various tasks, including multiplechoice question answering, summarization, and classification. Overall, our fine-tuned models achieved a substantial average improvement of $1 8 { - } 2 4 \%$ across all CTI evaluation datasets. A notable example is the Adversarial MITRE ATT&CK evaluation dataset and its non-adversarial counterpart. As shown in Tables 2 and 3, our fine-tuned models exhibit greater robustness compared to the baseline models. These results highlight CyberPal.AI’s robustness and its ability to generalize effectively to the cyber-security domain.

In Table 4, we compare CyberPal.AI with seven general and public cyber-security benchmarks that the models had not encountered during the fine-tuning process. This evaluation tests whether our fine-tuned models can generalize to tasks different from those they were trained on. Overall, our fine-tuned models achieved an average improvement of 9- $10 \%$ across the general cyber-security evaluation datasets. This is impressive, considering that our models were finetuned with different kinds of datasets. See our extended version (Levi et al. 2024) for additional results.

# 6 Conclusion

In this work, we introduced SecKnowledge, SecKnowledgeEval, and CyberPal.AI. SecKnowledge is a domainknowledge-driven cyber-security instruction dataset aimed at fine-tuning LLMs for the security domain. The dataset is built in two steps: first, we generate instructions based on predefined schemas established through domain expertise; second, we expand this dataset using a hybrid synthetic content-grounded data generation process. CyberPal.AI represents a family of LLMs fine-tuned using SecKnowledge, aimed at developing security-specialized models capable of answering and following complex securityrelated instructions. To evaluate CyberPal.AI, we introduced SecKnowledge-Eval, a comprehensive suite of evaluation datasets covering diverse cyber-security tasks we developed, alongside publicly available security benchmarks. Our finetuned models demonstrated impressive performance on various security-related tasks, including threat hunting (e.g., up $26 \%$ in CTI Detection and Mitigation), TTP mapping (up $17 \%$ in SIEM Rule Mapping), summarization (up $3 5 \%$ in CWE Summarization), and impact classification (up $11 \%$ in CWE technical impact). They also effectively captured relationships between different components within various security frameworks (e.g., up $45 \%$ in CTI relationship prediction). CyberPal.AI also showcased enhanced performance on general security knowledge benchmarks such as the security portion of MMLU, skill assessment tests, and analysts’ assessment tests, among others. Overall, CyberPal.AI models outperformed their baseline counterparts, achieving significant average improvement of up to $24 \%$ on trainingaligned tasks and up to $10 \%$ average improvement on public cyber-security benchmarks. These results underscore the extensive knowledge and deep understanding gained through fine-tuning the models with our SecKnowledge dataset.
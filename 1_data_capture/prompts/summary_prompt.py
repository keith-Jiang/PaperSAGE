summary_prompt_v1 = """
你是一个Arxiv学术论文总结专家，你的任务是给定你一篇论文，你需要用中文总结出该论文的核心思想、核心算法方案细节以及该论文的关键词。
# 论文
{paper}
"""

summary_prompt_v2 = """
# Role: Top-Tier Academic Paper Analyst

你是一位顶尖的学术论文分析师与总结专家，擅长以极其清晰、结构化、专业的视角，深入剖析学术论文，并提炼其核心价值。

---
# Core Task

你的任务是接收一篇学术论文，并严格按照以下指定的结构和要求，用【简体中文】生成一份全面的、高质量的分析报告。

---
# Input

- 学术论文全文：`{paper}`

---
# Output Structure & Instructions

请严格按照以下【四大部分】的序号、标题和子项要求进行组织和撰写。

### 1. 核心概要 (Core Summary)
对论文的顶层设计进行精炼总结。
- **问题定义 (Problem Definition)**：清晰描述论文旨在解决的具体学术或工业问题。此问题的重要性体现在哪些方面？
- **方法概述 (Method Overview)**：用一两句话高度概括论文提出的核心方法或模型框架。
- **主要贡献与效果 (Key Contributions & Results)**：列出论文最主要的 1-3 个贡献点。并用关键数据（如准确率提升了 X%，速度快了 Y 倍）来量化说明其达成的效果。

### 2. 算法/方案详解 (Algorithm/Method Details)
深入剖析论文提出的核心方法。
- **核心思想 (Core Idea)**：详细阐述该方法背后的核心原理和直觉。它为什么会有效？
- **创新点 (Innovation Points)**：与先前的工作相比，该方法在哪些方面进行了创新？（例如，新的网络结构、改进的注意力机制、新的优化目标等）。
- **具体实现步骤 (Implementation Steps)**：按照逻辑顺序，分步描述算法或方案的具体执行流程。最好能以列表（1, 2, 3...）的形式呈现。如果论文中包含伪代码或关键公式，请在此处引用和解释。
- **案例解析 (Example Illustration)**：如果论文中提供了具体的例子来说明算法如何工作，请在此处复述该例子，以帮助理解。

### 3. 对比实验分析 (Comparative Analysis)
对论文的实验部分进行结构化总结。
- **基线模型 (Baseline Models)**：列出论文中用于对比的核心基线模型（Baseline）。
- **性能对比 (Performance Comparison)**：请分点、按【关键评估指标】进行组织，用自然语言详细阐述性能对比结果，不要用markdown格式表格。对于每个指标，请明确指出**本文方法所达到的数值**，并与**核心基线模型的数值**进行比较。请使用“优于”、“高出X%”、“降低了Y”等描述性词语来凸显效果差异。

**描述示例：**
*   **在 [准确率/Accuracy] 指标上：** 本文方法在 [数据集A] 上达到了 **95.2%** 的准确率，显著优于基线模型 Baseline-A (92.5%) 和 Baseline-B (93.1%)，相对最佳基线提升了 2.1个百分点。
*   **在 [推理速度/Inference Speed] 指标上：** 本文方法的处理速度为 **120 FPS**，远高于需要大量计算的 Baseline-C (45 FPS)，同时与轻量级模型 Baseline-D (125 FPS) 的速度相当，但在准确率上远超后者。

### 4. 关键词 (Keywords)
提取并格式化论文的核心关键词。
- **提取范围**：关键词应涵盖【研究领域】、【具体问题】、【核心技术/算法】、【应用场景】等。
- **格式要求**：每个关键词必须严格遵循 `中文名称 (English Full Name, Abbreviation)` 的格式。
    - 如果没有官方或通用的英文全称或缩写，请用 `N/A` 填充对应位置。

**格式示例：**
- 目标检测 (Object Detection, OD)
- 知识蒸馏 (Knowledge Distillation, KD)
- 联邦学习 (Federated Learning, FL)
- 医疗影像分析 (Medical Image Analysis, N/A)

---
# General Requirements

- **语言 (Language)**：必须使用专业、严谨的简体中文。
- **严谨性 (Rigor)**：所有总结内容必须源自论文原文，严禁进行任何形式的推测或编造。
- **完整性 (Completeness)**：如果论文中未明确提及某个子项所需的信息（如没有提供案例），请在该子项下明确注明“**论文未明确提供此部分信息**”，而不是直接跳过。
- **一致性 (Consistency)**：严格遵循上述所有结构和格式指令，确保输出报告的规整和统一。
"""

summary_prompt_v3 = """
<System>
# Mission
你是一位世界顶级的学术研究员与算法工程师，拥有海量论文阅读经验。你的核心任务是将输入的学术论文，精准地转化为一个结构化、信息丰富的JSON对象。这个JSON对象需要同时满足机器可读性（严格的JSON格式）和人类可读性（清晰的Markdown内容）。

## Core Workflow: Chain-of-Thought
在生成最终结果前，请在你的“内心”遵循以下思考链，以确保输出的质量和准确性：
1.  **Step 1: Global Comprehension** - 通读并深刻理解论文的每一个部分，包括摘要、引言、方法、实验和结论，构建对论文的整体认知。
2.  **Step 2: Field-by-Field Extraction** - 严格按照下方 `<Output_Specification>` 中定义的四个JSON键 (`core_summary`, `algorithm_details`, `comparative_analysis`, `keywords`)，一次只专注于一个键。对于每一个键，回到论文原文中定位最相关的信息源（如段落、图表、公式）。
3.  **Step 3: Faithful Restatement & Formatting** - 将提取到的原文信息，用专业、客观的中文进行重述。严格按照每个字段要求的Markdown格式进行组织。**绝对忠于原文，不添加任何推测性信息。**
4.  **Step 4: Handling Missing Information** - 如果某个子项（如“案例解析”）在原文中找不到明确对应的内容，必须在该子项下明确标注“**论文未明确提供此部分信息**”。
5.  **Step 5: Final Assembly & Validation** - 将所有处理好的字符串值填入JSON结构中。在输出前，进行最后一次严格检查，确保整个输出是**一个没有任何前后缀文本、可被 `json.loads()` 直接解析的、单一且完整的JSON对象**。
</System>

<Input>
-   **学术论文全文:** `{paper}`
</Input>

<Output_Specification>
# Output Rules

## General Rules
1.  **最终输出必须且只能是一个JSON对象。**
2.  **禁止在JSON对象前后添加任何解释性文字或注释。**
3.  **请再次记住要求，严格按照Core Workflow: Chain-of-Thought进行。**
4.  **所有Markdown内容必须语法正确，尤其注意列表、加粗、换行的使用。确保最终的字符串值在解析后能被Markdown渲染器正确显示。**

## JSON Schema & Content Structure
```json
{{
  "core_summary": "### 🎯 核心概要\n\n> **问题定义 (Problem Definition)**\n> *   清晰、准确地描述论文所要解决的核心学术或工业问题。\n> *   阐述该问题的重要性，例如：它克服了现有方法的哪些瓶颈？在哪些应用场景下具有关键价值？\n\n> **方法概述 (Method Overview)**\n> *   用一到两句高度凝练的语言，概括论文提出的核心方法、模型或框架。\n> *   *把它想象成这篇论文的“电梯演讲”。*\n\n> **主要贡献与效果 (Contributions & Results)**\n> *   以列表形式，列出论文最主要的 **1-3 个创新贡献点**。\n> *   为每个贡献点附上**关键数据**来量化其效果，例如：准确率提升了 `X%`，速度加快了 `Y` 倍，或成本降低了 `Z`。",
  "algorithm_details": "### ⚙️ 算法/方案详解\n\n> **核心思想 (Core Idea)**\n> *   详细阐述该方法背后的核心原理和直觉洞察。\n> *   解释该方法为何有效 (*Why does it work?*)，它的设计哲学是什么？\n\n> **创新点 (Innovations)**\n> *   **与先前工作的对比：** 先前相关工作存在哪些具体局限或问题？\n> *   **本文的改进：** 本文提出的方法是如何针对性地解决这些局限的？体现在哪些方面（如结构、算法、策略等）？\n\n> **具体实现步骤 (Implementation Steps)**\n> *   按照逻辑顺序，以编号列表（1, 2, 3...）的形式，分步描述算法或方案的具体执行流程。\n> *   如果论文中包含关键的**伪代码**或**核心数学公式**，请在此处引用并加以解释说明其含义。\n\n> **案例解析 (Case Study)**\n> *   如果论文中提供了具体的示例（toy example）来说明算法的运作方式，请在此处复述该例子，帮助读者直观地理解算法流程。",
  "comparative_analysis": "### 📊 对比实验分析\n\n> **基线模型 (Baselines)**\n> *   列出论文中用于对比性能的核心基线模型（Baseline Models）。\n\n> **性能对比 (Performance Comparison)**\n> *   **重要提示：** 请分点、按【关键评估指标】进行组织，用自然语言详细阐述性能对比结果，**绝对禁止使用 Markdown 表格**。\n> *   请参考并遵循以下**描述模板**：\n>\n>   ```markdown\n>   *   **在 [指标名称] 上：** 本文方法在 [数据集A] 上达到了 **[本文方法数值]**，显著优于基线模型 [基线A名称] ([基线A数值]) 和 [基线B名称] ([基线B数值])。与表现最佳的基线相比，提升了 [具体差异值，如 X 个百分点或 Y%]。\n>   \n>   *   **在 [另一个指标，如推理速度] 上：** 本文方法的处理速度为 **[本文方法数值]**，远高于 [基线C名称] ([基线C数值])，同时与轻量级模型 [基线D名称] ([基线D数值]) 的速度相当，但在 [某个质量指标] 上远超后者。\n>   ```",
  "keywords": "### 🔑 关键词\n\n> **提取与格式化要求**\n> 1.  **数量与范畴：** 提取 **5-8 个** 核心关键词，应涵盖【研究领域】、【具体问题】、【核心技术/算法】、【应用场景】等。\n> 2.  **格式模板（必须严格遵守）：** 每个关键词**无一例外**都必须遵循以下格式结构：\n>     ```\n>     中文名称 (English Full Name, Abbreviation)\n>     ```\n> 3.  **处理缺失缩写 (Abbreviation)：**\n>     *   如果一个术语**没有**通用或官方的英文缩写，**必须**在该位置使用大写的 `N/A`。\n>     *   **禁止**留空或删除缩写部分。例如，`中文名称 (English Full Name)` 的格式是**错误**的。\n\n> **格式示例（请严格参考）**\n> \n> ✅ **正确示例:**\n> *   目标检测 (Object Detection, OD)\n> *   生成对抗网络 (Generative Adversarial Network, GAN)\n> *   医疗影像分析 (Medical Image Analysis, N/A)  <-- *正确处理无缩写的情况*\n> *   联邦学习 (Federated Learning, FL)\n> \n> ❌ **错误示例:**\n> *   医疗影像分析 (Medical Image Analysis)  <-- *错误：缺少了 N/A*\n> *   Transformer                   <-- *错误：缺少中文和英文全称部分*\n> *   GAN (Generative Adversarial Network)  <-- *错误：顺序和格式不正确*"
}}
</Output_Specification>
"""
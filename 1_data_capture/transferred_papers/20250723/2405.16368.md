# Qsco: A Quantum Scoring Module for Open-Set Supervised Anomaly Detection

Yifeng Peng1, Xinyi $\mathbf { L i } ^ { 1 }$ , Zhiding Liang2, Ying Wang1

1Stevens Institute of Technology, Hoboken, NJ, USA 2Rensselaer Polytechnic Institute, Troy, NY, USA {ypeng21, xli215, ywang6}@stevens.edu, liangz9@rpi.edu

# Abstract

Open set anomaly detection (OSAD) is a crucial task that aims to identify abnormal patterns or behaviors in data sets, especially when the anomalies observed during training do not represent all possible classes of anomalies. The recent advances in quantum computing in handling complex data structures and improving machine learning models herald a paradigm shift in anomaly detection methodologies. This study proposes a Quantum Scoring Module (Qsco), embedding quantum variational circuits into neural networks to enhance the model’s processing capabilities in handling uncertainty and unlabeled data. Extensive experiments conducted across eight real-world anomaly detection datasets demonstrate our model’s superior performance in detecting anomalies across varied settings and reveal that integrating quantum simulators does not result in prohibitive time complexities. At the same time, the experimental results under different noise models also prove that Qsco is a noise-resilient algorithm. Our study validates the feasibility of quantumenhanced anomaly detection methods in practical applications.

![](images/d04aada77234c65b97e7dbcd46060d74483717e37301a7c966ea175378dcde35.jpg)  
Figure 1: Visualization of the anomalies and normal data predicted by DRA (Ding, Pang, and Shen 2022) and Qsco (Ours) in the MVTec AD (Bergmann et al. 2019) dataset (carpet subset). The parameter $\ell$ controls the depth of the variational circuit in Qsco. With the correct $\ell$ level, Qsco enhances DRA’s ability to distinguish boundaries between anomalies and normal data. However, if $\ell$ is too high $\ell = 3 )$ , it results in over-fitting, as shown in the last graph, while if $\ell$ is too low $\ell = 1 \mathrm { \AA }$ ), it leads to under-fitting.

# Introduction

Precisely identifying anomalous patterns within extensive and intricate datasets is paramount, especially when data labels are nonexistent, or the underlying data distributions remain undefined. Such conditions delineate the open set anomaly detection (OSAD) field, characterized by its necessity to function robustly across varied and often unforeseen data landscapes. Historically, the challenges inherent in OSAD have been addressed through conventional machine learning strategies (Ding, Pang, and Shen 2022; Li et al. 2021a; Liznerski et al. 2020; Pang et al. 2021; Pang, Shen, and Van Den Hengel 2019). While these methods have proven effective, they increasingly encounter constraints related to data’s escalating scale and complexity.

The advent of quantum computing has foreboded new prospects for augmenting machine learning architectures (Pan et al. 2023; Caro et al. 2022). Quantum computing offers substantial enhancements in computational efficiency and capacity, particularly adept at managing the high-dimensional data spaces prevalent in anomaly detection tasks. The synergy of quantum computing with machine learning, commonly referred to as quantum machine learning (QML) (Farhi and Neven 2018; Alchieri et al. 2021; Peng et al. 2024), introduces innovative approaches that may surmount some of the fundamental limitations of classical methodologies, including scalability, complexity, and uncertainty. With advancements in quantum hardware, QML is poised for revolutionary applications in AI domains like quantum natural language processing (QNLP) (Abbaszade et al. 2021), quantum computer vision (QCV) (Wei et al. 2023), and quantum anomaly detection (QAD)(Liu and Rebentrost 2018).

This research introduces the Quantum Scoring Module (Qsco), a state-of-the-art quantum model of quantum variational circuits with different rotation and CNOT gates. This model aims to exploit the quantum-enhanced computational power to refine the detection of obscure patterns and anomalies in open-set contexts. With the help of entanglement characteristics in quantum computing, we can discover the distribution of unknown anomalies and solve distribution problems that classical computing cannot simulate. We provide initial findings that indicate the Qsco model’s enhanced efficacy in processing high-dimensional data relative to traditional models.

In this paper, we investigate the application of QML in OSAD and endeavor to serve as a foundational text for subsequent inquiries into the fusion of quantum and classical computing paradigms. Through this scholarly endeavor, we aspire to forge the development of novel anomaly detection frameworks that are more competent, precise, and equipped to handle the multifaceted demands of contemporary data landscapes.

# Related Work

Open Set Anomaly Detection The recent advancements in open set recognition have led to various innovative approaches to detect and classify unknown samples effectively. (Liu et al. 2019) introduced the Margin Learning Embedded Prediction (MLEP) framework, emphasizing the importance of margin learning for enhancing open set recognition. Building on this, (Acsintoae et al. 2022) proposed a general open-set method, GOAD, which uses random affine transformations to extend the applicability of transformationbased methods to non-image data, thus relaxing current generalization assumptions. (Sun et al. 2020) tackled the limitations of variational auto-encoders (VAEs) in providing discriminative representations for known classifications by proposing Conditional Gaussian Distribution Learning (CGDL), which enhances open set recognition. (Kong and Ramanan 2021) demonstrated that a well-chosen GAN discriminator on real outlier data can achieve state-of-theart results and further improved performance by augmenting real open training examples with adversarially synthesized ”fake” data, culminating in the OpenGAN framework. (Yoshihashi et al. 2019)) focused on joint classification and reconstruction of input data through their Classification-Reconstruction learning for Open-Set Recognition (CROSR), which utilizes latent representations to enhance the separation of unknowns from knowns without compromising known-class classification accuracy. Lastly, (Zhang et al. 2020) proposed the OpenHybrid framework, which combines an encoder, a classifier, and a flow-based density estimator to detect unknown samples while classifying inliers accurately and effectively.

Supervised Anomaly Detection Recent anomaly detection advancements have introduced various innovative methods and frameworks to improve how anomalies are identified and classified across various domains. (Go¨rnitz et al. 2013) established a foundation with their supervised approach, paving the way for further research such as (Ruff et al. 2019), who enhanced accuracy through deep semisupervised methods leveraging both labeled and unlabeled data. (Li et al. 2021b) introduced Cutpaste, a self-supervised technique that emphasized the growing relevance of selfsupervision in anomaly detection.

Further innovations include (Akcay, AtapourAbarghouei, and Breckon 2019), who developed Ganomaly using GANs for semi-supervised anomaly detection, and (Han et al. 2022), who created Adbench, a comprehensive benchmark for method evaluation. In video anomaly detection, (Georgescu et al. 2021) and (Tian et al.

Table 1: Overview of Quantum Anomaly Detection Algorithms.   

<html><body><table><tr><td>Model</td><td>Algorithm</td><td>Supervised</td><td>VQC</td></tr><tr><td>(Liu and Rebentrost 2018)</td><td>Quantum PCA</td><td></td><td>X</td></tr><tr><td>(Guo et al. 2022)</td><td>AE</td><td>X</td><td>X</td></tr><tr><td>(Kottmann et al. 2021)</td><td>VQC based</td><td>X</td><td>√</td></tr><tr><td>(Guo et al. 2023)</td><td>Quantum LOF</td><td>×</td><td></td></tr><tr><td>(Wang et al. 2023)</td><td>QHDNN</td><td>√</td><td>√</td></tr><tr><td>(Moro and Prati 2023)</td><td>AQAs</td><td>√</td><td></td></tr><tr><td>Ours (Qsco)</td><td>VQC based</td><td>√</td><td>√</td></tr></table></body></html>

bis semi-supervised and cis both of VQC and quantum mechanics.

2021) demonstrated the effectiveness of self-supervised and weakly-supervised learning for handling complex data scenarios and robust temporal feature extraction, respectively.

Additionally, (Tang et al. 2024) and (Yao et al. 2023) contributed by benchmarking graph-based methods with Gadbench and enhancing supervised detection with boundaryguided contrastive learning. In the realm of weak supervision, (Cho et al. 2023) and (Bozorgtabar and Mahapatra 2023) explored relational and attention-conditioned learning approaches, while (Chen et al. 2023) combined magnitudecontrastive learning with glance-and-focus mechanisms for video anomaly detection. (Sultani, Chen, and Shah 2018) focused on real-time anomaly detection, highlighting the necessity for efficient, real-time solutions.

Quantum Anomaly Detection Quantum anomaly detection utilizes principles of quantum computing to significantly enhance the identification of data outliers, which is crucial for fraud detection, network security, and medical diagnostics applications. This emerging field has seen promising developments, as shown in Table 1: (Liu and Rebentrost 2018) introduced quantum algorithms for kernel principal component analysis (PCA) and one-class support vector machine (SVM), which can operate with resources logarithmic in the dimensionality of quantum states. (Guo et al. 2022) proposed an efficient quantum algorithm for the computationally demanding ADDE algorithm, using Amplitude Estimation (AE) to handle large datasets effectively. Similarly, (Kottmann et al. 2021) developed a variational quantum anomaly detection algorithm capable of extracting phase diagrams from quantum simulations without prior physical knowledge, fully operable on the quantum device used for the simulations. Extending further, (Guo et al. 2023) transformed the Local Outlier Factor (LOF) algorithm into a quantum version, achieving exponential speedup on data dimensions and polynomial speedup on data quantity. Additionally,(Wang et al. 2023) engineered a quantum-classical hybrid DNN (QHDNN) that learns from normal images to isolate anomalies, exploring various quantum layer architectures and implementing a VQC-based solution. Lastly, (Moro and Prati 2023) explored the potential of adiabatic quantum annealers (AQAs) for quantum speedup in anomaly detection.

![](images/3e9dfac0e7235c362cfbd3f0dc4c3e53a8bd576c67b46a49cb53d7491e8578de.jpg)  
Figure 2: Overview of our proposed Qsco module. The left-hand section displays dataset samples from open set data, denoted as the input $\mathcal { X }$ , which are fed into the feature extractor $\mathcal { F } ( \cdot )$ to obtain the feature map $\mathcal { X } ^ { \prime }$ . This feature map $\mathcal { X } ^ { \prime }$ is the input to the quantum variational circuit. In the circuit, the $x$ values are used in the $R Y ( \cdot )$ gates, while the $\theta$ parameters are optimized during the training process in the $R o t ( \cdot )$ operations. The green-shaded area represents the optimized layer in the Qsco module. The Qsco module output score is $\hat { \boldsymbol { S } }$ .

# Proposed Approach

Problem Statement The studied problem, open-set supervised AD, can be formally stated as follows. Given a set ies $\begin{array} { r c l } { \mathcal { X } } & { = } & { \left\{ { \bf x } _ { i } \right\} _ { i = 1 } ^ { N + M } } \end{array}$ l,  s twhaincdh $\begin{array} { r l } { \mathcal { X } _ { n } } & { { } = } \end{array}$ $\left\{ \mathbf { x } _ { 1 } , \mathbf { x } _ { 2 } , \ldots , \mathbf { x } _ { N } \right\}$ $\begin{array} { r l } { \mathcal { X } _ { a } } & { { } = } \end{array}$ $\left\{ { \bf x } _ { N + 1 } , { \bf x } _ { N + 2 } , \ldots , { \bf x } _ { N + M } \right\}$ $( M \ll N )$ is a very small set of annotated anomalies that provide some knowledge about true anomalies, and the $M$ anomalies belong to the seen anomaly classes $\zeta \subset \varpi$ , where $\varpi = \{ c _ { i } \} _ { i = 1 } ^ { | \varpi | }$ denotes the set of all possible anomaly classes, and then the goal is detect both seen and unseen anomaly classes by learning an anomaly scoring function $g : \mathcal { X }  \mathbb { R }$ that assigns larger anomaly scores to both seen and unseen anomalies than normal samples.

Quantum Scoring Module In open-set anomaly detection, the model needs to identify new categories or anomalies not seen during the training phase. This type of anomaly detection is critical for many applications, such as security monitoring, medical diagnostics, and industrial quality control, where new and unknown anomalies often arise.

For an image in the dataset $\begin{array} { r c l } { { \mathcal { X } } } & { { = } } & { { \left\{ { \bf x } _ { i } \right\} _ { i = 1 } ^ { N + M } } } \end{array}$ , ${ \bf x } _ { i } \in  { }$ $\mathbb { R } ^ { H \times W \times C }$ . Usually, this image can be represented as a threedimensional tensor, which for color images usually contains three color channels (red, green, blue). Let the height of the image be $H$ , the width is $W$ , and the color channel is $C$ .

The feature extractor $\mathcal { F } ( \cdot )$ is a function that converts the input image into a feature vector or matrix. It may contain multiple convolution layers, pooling layers, and activation layers and can be expressed as:

$$
\begin{array} { r } { \mathcal { X } ^ { \prime } = \mathcal { F } ( \mathcal { X } ) : \mathbb { R } ^ { H \times W \times C } \longrightarrow \mathbb { R } ^ { H ^ { \prime } \times W ^ { \prime } \times D } , } \end{array}
$$

where $D$ represents feature depth and $( D \gg C )$ .

The hyperbolic tangent function, or tanh, is used to normalize input data to a fixed interval. For quantum computing’s rotational gates (like RY), input typically needs to be in terms of angles, making tanh ideal for scaling data to the $[ - 1 , 1 ]$ range, which helps control the rotation angles of these gates. The next step is to apply the hyperbolic tangent activation function tanh to the feature vector $\mathcal { X } ^ { \prime }$ . The hyperbolic tangent function performs a nonlinear transformation on each element $\mathcal { X } ^ { \prime }$ , and each element of the output is $[ - 1 , 1 ]$ , mathematically expressed as :

$$
\hat { \mathcal { X } } = \operatorname { t a n h } ( \mathcal { X } ^ { \prime } ) = \left[ \operatorname { t a n h } ( \mathbf { x } _ { 1 } ^ { \prime } ) , \operatorname { t a n h } ( \mathbf { x } _ { 2 } ^ { \prime } ) , \dots , \operatorname { t a n h } ( \mathbf { x } _ { N + M } ^ { \prime } ) \right] .
$$

We applied a set of $K$ qubits, denoted as $\tau \in \mathbb { C } ^ { K }$ , where each $\tau _ { m }$ represents the state of the mth qubit, in the Qsco module. A quantum state could be represented as:

$$
\left| \psi _ { m } \right. = \alpha _ { m } \left| 0 \right. + \beta _ { m } \left| 1 \right. ,
$$

where $| \psi _ { m } \rangle$ is the state of qubit, $\alpha _ { m }$ and $\beta _ { m }$ are the complex probability amplitudes, which satisfy:

$$
\left| \alpha _ { m } \right| ^ { 2 } + \left| \beta _ { m } \right| ^ { 2 } = 1 .
$$

According to Eq. 4, Eq. 3 could be represented as:

$$
\left| \psi _ { m } \right. = e ^ { i \gamma _ { m } } \left( \cos \frac { \theta _ { m } } { 2 } \left| 0 \right. + e ^ { i \varphi _ { m } } \sin \frac { \theta _ { m } } { 2 } \left| 1 \right. \right) ,
$$

where $\theta _ { m } , \varphi _ { m }$ , and $\gamma _ { m }$ are real numbers specific to the $m$ th qubit. The factor $e ^ { i \gamma _ { m } }$ can be ignored due to its negligible effect (Nielsen and Chuang 2010a). Therefore, we can rewrite Eq. 5 as:

$$
\left. \psi _ { m } \right. = \cos \frac { \theta _ { m } } { 2 } \left. 0 \right. + e ^ { i \varphi _ { m } } \sin \frac { \theta _ { m } } { 2 } \left. 1 \right. .
$$

The input of the quantum circuits is the $\begin{array} { r l } { \hat { \mathcal { X } } } & { { } = } \end{array}$ $\left\{ \hat { \mathbf { x } } _ { 1 } , \hat { \mathbf { x } } _ { 2 } , \hat { \mathbf { \Xi } } _ { \cdot } . \hat { \mathbf { \Xi } } _ { \cdot } , \hat { \mathbf { x } } _ { N + M } \right\}$ as shown in Fig. 2. We take $\hat { \mathbf { x } } _ { m }$ and utilize the Rotation $\mathrm { Y }$ gate (Ry gate) as below:

$$
\begin{array} { r } { R y ( \hat { \mathbf { x } } _ { m } ) = \left( \begin{array} { c c } { \cos \bigl ( \frac { \hat { \mathbf { x } } _ { m } } { 2 } \bigr ) } & { - \sin \bigl ( \frac { \hat { \mathbf { x } } _ { m } } { 2 } \bigr ) } \\ { \sin \bigl ( \frac { \hat { \mathbf { x } } _ { m } } { 2 } \bigr ) } & { \cos \bigl ( \frac { \hat { \mathbf { x } } _ { m } } { 2 } \bigr ) } \end{array} \right) . } \end{array}
$$

For a qubit in state in Eq. 3, we can calculate the result as below:

$$
\begin{array} { r } { R y ( \hat { \mathbf { x } } _ { m } ) | \psi _ { m } \rangle = \left( { \alpha _ { m } \cos ( \frac { \hat { \mathbf { x } } _ { m } } { 2 } ) - \beta _ { m } \sin ( \frac { \hat { \mathbf { x } } _ { m } } { 2 } ) } \right) . } \end{array}
$$

Then we initialize the parameters for the rotation gates ${ \cal R } o t ( \cdot ) , \theta \in \mathbb { R } ^ { m \times n \times k }$ defined as below:

$$
R o t \left( \cdot \right) = R _ { z } ( \lambda ) R _ { y } ( \phi ) R _ { x } ( \theta ) .
$$

Rx gate (Rotation around the $\mathrm { \Delta X }$ -axis gate) in quantum computing is used to rotate qubits around the $\mathrm { \Delta X }$ -axis of the Bloch ball, and the Ry gate is shown in Eq. 7. The Rotation $Z$ gate $\scriptstyle \mathbf { R } \mathbf { Z }$ gate) is used in quantum computing to rotate the qubit around the $Z$ -axis as shown in Fig. 3.

$$
\begin{array} { r } { R x ( \theta ) = \left( \frac { \cos \left( \frac { \theta } { 2 } \right) } { - i \sin \left( \frac { \theta } { 2 } \right) } - i \sin \left( \frac { \theta } { 2 } \right) \right) , R z ( \lambda ) = \left( \begin{array} { c c } { e ^ { - i \frac { \lambda } { 2 } } } & { 0 } \\ { 0 } & { e ^ { i \frac { \lambda } { 2 } } } \end{array} \right) . } \end{array}
$$

So the quantum state after the $R o t ( \cdot )$ could be expressed in summary as:

$$
| \psi _ { m } ^ { \prime \prime } \rangle = R _ { z } ( \lambda ) R _ { y } ( \phi ) R _ { x } ( \theta ) | \psi _ { m } ^ { \prime } \rangle
$$

The function of the CNOT gate is as follows: if the control bit is in the $| 1 \rangle$ state, it will perform a Pauli-X gate operation on the target bit (flip the state of the target bit). The target bit is unchanged if the control bit is in the $| 0 \rangle$ state.

We utilize the CNOT gate to implement conditional logic between qubits:

$$
| \tilde { \psi } _ { m } \rangle = | \psi _ { m } ^ { \prime \prime } \rangle \otimes | \psi _ { m + 1 } ^ { \prime \prime } \rangle , m \in [ 0 , K - 1 ] .
$$

We implement a Pauli- ${ \bf \nabla } \cdot { \bf Z }$ gate $\mathbf { Z }$ gate for short) on a superposition state that will affect the phase of the qubit without changing its probability amplitude as:

$$
\sigma _ { z } = \bigg [ \frac { 1 } { 0 } \quad \frac { 0 } { - 1 } \bigg ] .
$$

10 0 $| i + \rangle =$ 10p p 1 TpA（+） Ay 1$\cdot$ 10T √ 1 (10)-i|1) +exp(ip)sin )[1j1> |1> (1>

When we apply the Pauli- ${ \bf \nabla } \cdot { \bf Z }$ gate in Eq. 13 to a qubit in Eq. 3, we can get:

$$
\left| \psi \right. = \alpha \left| 0 \right. - \beta \left| 1 \right. .
$$

Physically, the $Z$ -gate acts like a 180 degree rotation of the qubit’s phase. This phase rotation does not change the measured probability distribution of the qubit, but it affects the global phase of the quantum state:

$$
\tilde { \mathbf { s } } _ { m } = \sigma _ { z } \cdot | \tilde { \psi } _ { m } \rangle .
$$

The output from Qsco as the input from the feature extractor $\hat { \mathcal X } _ { n }$ to the output $\hat { S } _ { n }$ is shown as below:

$$
\hat { \mathcal { X } } \longrightarrow \hat { \mathcal { S } } , S = \{ \mathbf { s } _ { 1 } , \mathbf { s } _ { 2 } , \dotsc , \mathbf { s } _ { N + M } \}
$$

$\hat { \boldsymbol { \mathcal { S } } }$ is the output of the Qsco module, which can be added to the scoring part of any model.

Composite Feature Learning   
Raturencep Feature Map Normality Normality Feature √ 1 Feature Learning Learning   
Residual Feature Gap Qsco √ Feature Vector Feature Vector Plain Fmniture Anomaly Anomaly scoring   
DRA: Scoring ? ↓   
Qsco: Anomaly Score + Quantum Anomaly ↓ Score Anomaly Score (DRA $^ +$ Qsco)

We chose DRA to be our backbone model because it is the SOTA method in the OSAD task, and we insert Qsco to DRA as an additional part to participate in the scoring process as shown in Fig. 4.

# Experimental Results

Hyper parameters We employ Adam as the optimizer, initialized with a learning rate of 0.0002 and a weight decay factor of $1 e \mathrm { ~ - ~ } 5$ . The StepLR scheduler is utilized with the learning rate decaying by 0.1 every 10 epoch. The batch size is 48, and steps for each training epoch are 20, and there are 30 epochs for training. Many studies use synthetic anomaly detection datasets derived from well-known image classification benchmarks, like MNIST (LeCun et al. 2010) and CIFAR-10 (Krizhevsky 2009), utilizing either one-vsall or one-vs-one approaches. These methods create anomalies that are significantly different from standard samples. However, the differences between anomalies and standard samples are often subtle and minor in real-world scenarios, such as detecting defects in industrial settings or identifying lesions in medical imaging. Consequently, following previous research (Ding, Pang, and Shen 2022), we prioritize natural anomaly datasets, including five industrial defect inspection datasets: MVTec AD (Bergmann et al. 2019), AITEX (Silvestre-Blanes et al. 2019), SDD (Tabernik et al. 2019), ELPV (Deitsch et al. 2019) and Optical(Wieler, Hahn, and Hamprecht 2023), and three more real-world datasets: BrainMRI (Salehi et al. 2021) and HeadCT (Salehi et al. 2021), Hyperkvasir (Borgli et al. 2020). Experiments are performed on an NVIDIA 3060 GPU, 48 GB RAM, and an NVIDIA RTX A6000, 48 GB RAM. We built the Qsco quantum circuit on PennyLane (Bergholm et al. 2022) for quantum simulation.

Table 2: AUC results (mean $\pm$ std) on 8 real-world AD datasets under general setting (noted by $\heartsuit$ ). The best and second-best results and the third-best are respectively highlighted in red and blue and bold. $\varpi$ is the number of anomaly classes.   

<html><body><table><tr><td>Dataset (α)</td><td>DRA</td><td>DRA+Qsco (l = 2）|DRA+Qsco (l =1) |DRA +Qsco (l = 3)</td><td></td><td></td></tr><tr><td colspan="5">Ten Training Anomaly Examples (Random) </td></tr><tr><td>AITEX (12)</td><td>0.886±0.021</td><td>1 0.875±0.029</td><td>0.832±0.021</td><td>0.880±0.020</td></tr><tr><td>SDD (1)</td><td>0.972±0.010</td><td>0.980±0.007</td><td>0.979±0.006</td><td>0.974±0.010</td></tr><tr><td>ELPV (2)</td><td>0.821±0.028</td><td>0.819±0.020</td><td>0.820±0.017</td><td>0.819±0.018</td></tr><tr><td>Optical (1)</td><td>0.967±0.007</td><td>0.969±0.008</td><td>0.950±0.014</td><td>0.960±0.009</td></tr><tr><td>BrainMRI (1)</td><td>0.959±0.011</td><td>0.972±0.017</td><td>0.978±0.017</td><td>0.961±0.019</td></tr><tr><td>HeadCT (1)</td><td>0.998±0.003</td><td>0.999±0.003</td><td>0.989±0.008</td><td>0.980±0.012</td></tr><tr><td>Hyper-Kvasir (4)| 0.840±0.029</td><td></td><td>0.813±0.019</td><td>一 0.811±0.025</td><td>0.812±0.018</td></tr><tr><td>MVTec AD（-)</td><td>0.960±0.011</td><td>0.968±0.010</td><td>0.966±0.013</td><td>0.961±0.024</td></tr><tr><td colspan="5">One Training Anomaly Example (Random) </td></tr><tr><td>AITEX (12)</td><td>0.703±0.038</td><td>0.708±0.027</td><td>0.685±0.044</td><td>0.728±0.018</td></tr><tr><td>SDD (1)</td><td>0.853±0.056</td><td>0.876±0.032</td><td>0.803±0.064</td><td>0.866±0.067</td></tr><tr><td>ELPV (2)</td><td>0.626±0.074</td><td>0.749±0.048</td><td>0.588±0.022</td><td>0.595±0.038</td></tr><tr><td>Optical (1)</td><td>0.894±0.026</td><td>0.910±0.011</td><td>一 0.859±0.020</td><td>0.879±0.017</td></tr><tr><td>BrainMRI (1)</td><td>0.638±0.045</td><td>0.692±0.051</td><td>0.718±0.043</td><td>0.708±0.039</td></tr><tr><td>HeadCT(1)</td><td>0.804±0.010</td><td>0.781±0.007</td><td>一 0.818±0.006</td><td>0.846±0.003</td></tr><tr><td>Hyper-Kvasir (4)| 0.712±0.010</td><td></td><td>0.768±0.015</td><td>一 0.771±0.008</td><td>0.770±0.014</td></tr><tr><td>MVTec AD (-)</td><td>0.904±0.033</td><td>一 0.914±0.029</td><td>0.905±0.031</td><td>0.908±0.035</td></tr></table></body></html>

One and Ten Anomaly Examples It means the number of anomaly data in the training set is one and ten, respectively. We use the following two experiment protocols:

General setting It mimics a typical open-set anomaly detection situation, where a few anomaly examples are randomly selected from all potential anomaly classes in the test set for each dataset. These selected anomalies are then excluded from the test data. This approach aims to reflect real-world conditions where it is uncertain which anomaly classes are known and the extent of the anomaly classes covered by the given examples. Consequently, the datasets may include both seen and unseen anomaly classes or solely the seen anomaly classes, depending on the complexity of the applications, such as the total number of possible anomaly classes.

Hard setting It aims to specifically assess the models’ ability to detect unseen anomaly classes, which is a critical challenge in open-set AD. To achieve this, anomaly example sampling is restricted to a single anomaly class, and all samples from this class are removed from the test set to ensure it includes only unseen anomaly classes. It is important to note that this setting applies only to datasets with at least two anomaly classes.

In Table 2, we conducted an empirical study on how changing the depth of the Qsco model (the number of layers $\ell$ ) affects its performance across eight different datasets. We categorized the tests into scenarios involving ten training anomalies and one training anomaly. Our results imply that increasing the parameter $\ell$ , which controls the depth of layers in Qsco, leads to better anomaly detection AUC scores, especially in scenarios with only one training anomaly. As $\ell$ increases, Qsco’s variational circuit becomes deeper, enhancing its ability to detect anomalies in more challenging tasks (performance on one training anomaly is better on ten training anomalies). However, increasing the depth can reduce performance in simpler scenarios with ten anomalies in the training set, possibly due to over-fitting. This highlights the importance of selecting an optimal value for $\ell$ , as the model’s complexity needs to be adjusted based on the specific characteristics of the task. For MVTec AD, more class details (noted as $( - ) _ { . } ^ { . }$ ) can be found in Appendix Table. 1 at https://arxiv.org/abs/2405.16368.

Meanwhile, we measured the simulation time of quantum variational circuits using PennyLane 0.35.1, and the computing resource used was an NVIDIA GeForce RTX 3060 and 64GB RAM. While Qsco performance has been improved as indicated in Table 2, the quantum approach also has a relative shortcoming: it will cause a certain amount of time loss. However, from the table. 3, we observe that the additional time consumption of approximately $2 0 \%$ (6 seconds) is acceptable compared to the performance enhancement, attributed to the fewer quantum bits $\tau = 9$ . The reason behind the time loss is the deepening of the quantum variational circuit, which ultimately leads to an increase in the revolving gate operations $R o t ( \cdot )$ . Nevertheless, a too-deep circuit (bigger $\ell$ ) will lead to over-fitting. On the contrary, a too-shallow circuit will be unable to explore the abnormal data space, leading to under-fitting problems. Therefore, we need to consider fewer quantum gate operations while ensuring performance because, in the NISQ era (Preskill 2018), a controllable number of quantum gates has higher implement ability and operability.

In Table. 4, after applying the best-performing parameter $\ell \ = \ 2$ , we compared the performance (AUC score) with more baseline methods, namely SAOE (combining data augmentation-based Synthetic Anomalies (Li et al. 2021a; Liznerski et al. 2020; Tack et al. 2020), FLOS (Lin et al. 2017), MLEP (Liu et al. 2019), and DevNet (Pang et al. 2021; Pang, Shen, and Van Den Hengel 2019). This table indicates that Qsco has better anomaly detection performance than most baseline methods among all datasets.

Table. 5 presents performance quantification following the established general configuration under ten and one anomaly examples. This table computes the average values for each dataset, with additional details available (subset for each dataset) in the Appendix Table. 2 for ten anomaly examples and Appendix Table. 3 for one anomaly example. Table. 5 demonstrates that for the dataset under ”One Training Anomaly Examples (Random),” the introduction of Qsco significantly improves the performance metrics of the DRA. However, for ”Ten Training Anomaly Example (Random),” it is important to recognize that integrating DRA and Qsco does not consistently achieve superior outcomes across the datasets. The proposed Qsco improves the performance of DRA under hard settings for ten training anomaly examples. However, DRA $^ +$ Qsco doesn’t always take the best performance. This discrepancy arises because the foundational DRA often demonstrates diminished effectiveness compared to alternative methods. Therefore, the combined performance of DRA and Qsco falls short. This shortfall is primarily attributable to the inherent limitations of DRA, despite our efforts to enhance its performance.

We present a comprehensive visualization of the performance metrics of Qsco $\ell = 2$ ) across eight datasets, contrasting the results for scenarios involving one anomaly and ten anomaly examples in Fig. 5. Details for more $\ell$ could be found in Appendix Fig.1 and Fig.2. The results reveal that Qsco significantly enhances the effectiveness of the DRA across a broad spectrum of evaluation metrics in situations under one anomaly example. This enhancement is consistent and marked by a notably more substantial improvement than scenarios with ten training anomaly examples. Furthermore, despite the inherently lower task complexity associated with ten training anomaly examples, Qsco demonstrates distinct advantages. The performance improvements, although less dramatic than in single anomaly scenarios, suggest that Qsco’s methodology adapts effectively to varying degrees of complexity within anomaly detection tasks. This adaptability indicates robustness, critical for practical applications where anomaly conditions can vary widely.

# Noise Analysis

In the NISQ era, anticipating the deployment of Qsco on actual quantum computing platforms necessitates addressing several critical issues, among which quantum noise is paramount. Accordingly, this section is dedicated to the application of various noise models to Qsco, enhancing our understanding of quantum noise in practical scenarios.

Depolarizing noise. The Depolarizing channel is denoted as $\Delta _ { \gamma }$ with probability $\gamma$ , replaces the state $\rho$ (density matrix) with the maximally mixed state $\textstyle { \frac { I } { d } }$ , where $I$ is the identity matrix and $d$ is the dimension of the Hilbert space.

The action of the depolarizing channel on a state $\rho$ is given by:

$$
\Delta _ { \gamma } ( \rho ) = ( 1 - \gamma ) \rho + \frac { \gamma } { d } I .
$$

Bit Flip Noise. The bit flip noise flips the state from $| 0 \rangle$ to $| 1 \rangle$ or vice versa with probability $\gamma$ as

$$
\mathcal { E } ( \rho ) = ( 1 - \gamma ) \rho + \gamma X \rho X ,
$$

where $X$ is the Pauli- $\mathbf { X }$ gate.

Phase Flip Noise. Similar to the bit flipping noise. The probability $\gamma$ , this noise applies a Pauli- ${ \bf \nabla } \cdot { \bf Z }$ gate to the quantum state, adding a negative sign to the $| 1 \rangle$ as:

$$
\mathscr { E } ( \rho ) = ( 1 - \gamma ) \rho + \gamma Z \rho Z .
$$

Amplitude Damping. The amplitude damping channel simulates the effect of energy dissipation from the qubit to the environment, decaying from the excited state $| 1 \rangle$ to the ground state $| 0 \rangle$ . The effect can be represented using Kraus operators:

$$
\mathcal { E } ( \rho ) = E _ { 0 } \rho E _ { 0 } ^ { \dagger } + E _ { 1 } \rho E _ { 1 } ^ { \dagger } ,
$$

where $E _ { 0 } = \binom { 1 } { 0 } \binom { 0 } { 1 - \gamma }$ and $E _ { 1 } = { \binom { 0 } { 0 } } \quad { \sqrt { \gamma } } \quad$ are the Kraus operators (Nielsen and Chuang 2010b).

<html><body><table><tr><td>Dataset (ω)</td><td>DevNet</td><td>DRA</td><td></td><td>DRA+Qsco(l=2)|DRA+Qsco(l=1）|</td><td>DRA + Qsco (l = 3)</td></tr><tr><td colspan="6">Ten Training Anomaly Examples (Random) </td></tr><tr><td>AITEX (12)</td><td></td><td></td><td></td><td></td><td>|20.93±0.76|36.75±0.50|42.22±2.01 (14.9% ↑)|41.38±1.02 (12.6% ↑)|43.53±2.66 (18.4% ↑)</td></tr><tr><td>SDD (1)</td><td></td><td></td><td></td><td>|21.28±0.98|36.59±3.51|42.28±2.37 (15.6% ↑)| 40.00±2.72 (9.3% ↑) |45.43±3.15 (24.2% ↑)</td><td></td></tr><tr><td>ELPV (2)</td><td></td><td></td><td></td><td></td><td>|21.41±0.95|34.30±2.30|41.79±2.29 (21.8% ↑)|40.80±3.12 (19.0% ↑) | 44.31±2.93 (29.2% ↑)</td></tr><tr><td>Optical (1)</td><td></td><td></td><td></td><td></td><td>|21.43±1.04|35.40±1.55|42.81±3.06 (20.9% ↑)|40.44±2.26 (14.2% ↑)|43.97±3.43 (24.2% ↑)</td></tr><tr><td>BrainMRI (1)</td><td></td><td></td><td></td><td></td><td>|13.81±0.72|19.12±0.17|24.25±0.44 (26.8% ↑)|22.85±0.50 (19.5% ↑)|26.45±0.52 (38.3% ↑)</td></tr><tr><td>HeadCT (1)</td><td></td><td></td><td></td><td>14.03±0.55|19.18±0.25|24.64±0.37 (28.5% ↑)|22.67±0.27 (18.2% ↑)| 26.26±0.13 (37.0% ↑)</td><td></td></tr><tr><td></td><td></td><td></td><td>Hyper-Kvasir(4)|21.47±1.09|34.06±2.77|41.70±3.21 (22.4% ↑)|41.37±5.5 (21.5% ↑)| 53.65±3.43 (57.5%↑)</td><td></td><td></td></tr><tr><td colspan="6">One Training Anomaly Example (Random) </td></tr><tr><td>AITEX (12)</td><td></td><td></td><td></td><td>20.89±1.07|32.66±2.09|44.12±1.86 (35.1% ↑) | 43.68±4.56 (33.7% ↑)| 44.74±1.5 (37.0% ↑)</td><td></td></tr><tr><td>SDD (1)</td><td></td><td></td><td></td><td>|21.90±1.31|35.14±2.09|40.54±3.20 (15.4% ↑)|38.64±3.03 (10.0% ↑)|42.06±3.40 (19.7% 个)</td><td></td></tr><tr><td>ELPV (2)</td><td></td><td></td><td>|21.39±1.80 |33.01±2.15|46.60±4.03 (41.2% ↑) |43.50±2.85 (31.8% ↑)| 47.33±5.22 (43.4% ↑)</td><td></td><td></td></tr><tr><td>Optical (1)</td><td></td><td></td><td>20.94±1.30|34.01±1.70|42.92±3.50 (26.2% ↑)|40.84±2.12 (20.1% ↑)| 44.18±0.35 (30.0% ↑)</td><td></td><td></td></tr><tr><td>BrainMRI (1)</td><td></td><td></td><td>|14.19±0.87|19.33±0.82|24.46±1.97 (26.5% ↑) | 23.95±2.11 (24.0% ↑) |30.82±4.07 (59.4% ↑)</td><td></td><td></td></tr><tr><td>HeadCT (1)</td><td></td><td></td><td>|14.73±0.71|19.74±0.79|30.20±1.67 (53.0% ↑)|22.73±1.65 (15.1% ↑)|31.09±5.10 (57.5% ↑)</td><td></td><td></td></tr><tr><td></td><td></td><td></td><td>Hyper-Kvasir (4)|23.08±1.22|34.55±2.08|41.75±3.29 (20.8% ↑)|41.15±4.28 (19.1% ↑)|47.70±2.67 (38.1% 个)</td><td></td><td></td></tr></table></body></html>

![](images/974b8acf806baeeeecf69ab70081c4d93d37131648d8353772020bf2410ab888.jpg)  
Table 3: Time complexity (s) per epoch. $\varpi$ is the number of anomaly classes. The testing computing resource is an NVIDIA GeForce RTX 3060 and 64GB RAM. Simulation is processed by PennyLane 0.35.1. $\heartsuit$ means general settings. Red means the biggest time consumption.   
Figure 5: Performance Comparison of DRA and Qsco $( \ell = 2$ ) with different numbers of anomaly examples. This comparison evaluates the performance (AUC score) of DRA and $\operatorname { D R A } + \operatorname { Q s c o } \ \ell = 2$ under the same general settings across eight different datasets. The left side displays results for a single anomaly example, while the right shows results for ten. In the column chart, the $\cdot$ indicates an improvement from the DRA model, and the $\downarrow$ signifies a decrease from the DRA.

Phase Damping. Unlike amplitude damping, phase damping preserves the amplitude of the quantum state but loses phase information, describing a dephasing process of quantum information without energy loss as:

$$
\mathscr { E } ( \rho ) = E _ { 0 } \rho E _ { 0 } ^ { \dagger } + E _ { 1 } \rho E _ { 1 } ^ { \dagger } .
$$

As illustrated in Table. 6, it is evident that various noise models adversely affect the AITEX, SDD, BrainMRI, HeadCT, and Hyper-Kvasir datasets in the context of Ten

Training Anomaly Examples, with the overall maximum fluctuation around $1 0 \%$ . Conversely, performance on the ELPV and Optical datasets has improved. We hypothesize that introducing noise acts as a form of regularization for these two datasets, potentially preventing overfitting. This may allow the models to improve generalization and robustness in noisy real-world environments. For One and Ten Training Anomaly Examples in the Mvtecad dataset could

Table 4: AUC results (mean $\pm$ std) on 8 real-world AD datasets under general setting (noted by $\heartsuit$ ). The best and second-best results and the third-best are respectively highlighted in red and blue and bold. $\varpi$ is the number of anomaly classes. SAOE, FLOS, and MLEP results are from Ding, Pang, and Shen (2022).   

<html><body><table><tr><td>Dataset (ω) 一</td><td>SAOE</td><td>FLOS</td><td>MLEP</td><td>DevNet</td><td>DRA</td><td>DRA + Qsco (l = 2)</td></tr><tr><td colspan="7">Ten Training Anomaly Examples (Random) </td></tr><tr><td>AITEX (12)</td><td>0.874±0.024</td><td>0.841±0.049</td><td>0.867±0.037丨0.881±0.039</td><td></td><td>0.886±0.021</td><td>0.875±0.029</td></tr><tr><td>SDD (1)</td><td>0.955±0.020</td><td></td><td>0.967±0.018丨0.783±0.013丨0.977±0.010</td><td></td><td>0.972±0.010</td><td>0.980±0.001</td></tr><tr><td>ELPV (2)</td><td>0.793±0.047</td><td></td><td>0.818±0.032丨 0.794±0.047丨0.856±0.013</td><td></td><td>0.821±0.028</td><td>0.819±0.020</td></tr><tr><td>Optical (1)</td><td>0.941±0.013</td><td>0.720±0.055</td><td>0.740±0.039</td><td>0.777±0.022</td><td>0.967±0.007</td><td>0.969±0.008</td></tr><tr><td>BrainMRI (1)</td><td>0.900±0.041</td><td>0.955±0.011</td><td>0.959±0.011</td><td>0.959±0.012</td><td>0.959±0.011</td><td>0.972±0.017</td></tr><tr><td>HeadCT(1)</td><td>一 0.935±0.021</td><td>0.971±0.004</td><td>0.972±0.014</td><td>0.993±0.004</td><td>0.998±0.003</td><td>0.999±0.003</td></tr><tr><td>Hyper-Kvasir (4)</td><td>0.666±0.050</td><td></td><td>0.773±0.029丨 0.600±0.069</td><td>0.876±0.012</td><td>0.840±0.029</td><td>0.813±0.019</td></tr><tr><td>MVTec AD (-)</td><td>0.926±0.010</td><td>0.939±0.007丨0.907±0.005</td><td></td><td>0.934±0.019</td><td>0.960±0.011</td><td>0.968±0.010</td></tr><tr><td colspan="7">One Training Anomaly Example (Random) </td></tr><tr><td>AITEX (12)</td><td>0.675±0.094</td><td></td><td>0.538±0.073丨0.564±0.055丨0.627±0.031</td><td></td><td>0.703±0.038</td><td>0.708±0.027</td></tr><tr><td>SDD (1)</td><td>0.781±0.009</td><td></td><td>0.840±0.043丨 0.811±0.045丨0.873±0.055</td><td></td><td>0.853±0.056</td><td>0.876±0.032</td></tr><tr><td>ELPV (2)</td><td>0.635±0.092</td><td></td><td>0.457±0.056丨 0.578±0.062丨0.691±0.069</td><td></td><td>0.626±0.074</td><td>0.749±0.048</td></tr><tr><td>Optical (1)</td><td>0.815±0.014</td><td>0.518±0.003</td><td>0.516±0.009</td><td>0.537±0.016</td><td>0.894±0.026</td><td>0.910±0.011</td></tr><tr><td>BrainMRI (1)</td><td>0.531±0.060</td><td>0.693±0.036</td><td>0.632±0.017</td><td>0.765±0.030</td><td>0.638±0.045</td><td>0.718±0.043</td></tr><tr><td>HeadCT(1)</td><td>0.597±0.022</td><td>0.698±0.092</td><td>0.758±0.038</td><td>0.768±0.031</td><td>0.804±0.010</td><td>0.781±0.007</td></tr><tr><td>Hyper-Kvasir (4)</td><td>1 0.498±0.100</td><td>0.668±0.004</td><td>0.445±0.040</td><td>0.682±0.038</td><td>0.712±0.010</td><td>0.768±0.015</td></tr><tr><td>MVTec AD (-)</td><td>0.834±0.007</td><td>0.792±0.014</td><td>0.744±0.019</td><td>0.812±0.049</td><td>0.904±0.033</td><td>0.914±0.029</td></tr></table></body></html>

<html><body><table><tr><td>Dataset</td><td>SAOE</td><td>FLOS</td><td>MLEP</td><td>DevNet</td><td>DRA</td><td>DRA +Qsco (l = 2)</td></tr><tr><td colspan="7">Ten Training Anomaly Examples (Random) </td></tr><tr><td>Carpet oo</td><td>0.762±0.073</td><td>0.761±0.012</td><td>0.751±0.023</td><td>0.847±0.017</td><td>0.920±0.039</td><td>0.927±0.044</td></tr><tr><td>Metal-nut oo</td><td>0.855±0.016</td><td>0.922±0.014</td><td>0.878±0.058</td><td>0.965±0.011</td><td>0.949±0.021</td><td>0.956±0.020</td></tr><tr><td>AITEX o</td><td>0.724±0.032</td><td>0.635±0.043</td><td>0.626±0.041</td><td>0.683±0.032</td><td>0.700±0.049</td><td>0.709±0.038</td></tr><tr><td>ELPV o</td><td>0.683±0.047</td><td>0.646±0.032</td><td>0.745±0.020</td><td>0.702±0.023</td><td>0.705±0.050</td><td>0.731±0.042</td></tr><tr><td>Hyper-Kvasir o</td><td>0.698±0.021</td><td>0.786±0.021</td><td>0.571±0.014</td><td>0.822±0.019</td><td>0.764±0.047</td><td>0.750±0.031</td></tr><tr><td colspan="7">One Training Anomaly Example (Random) </td></tr><tr><td>Carpet o</td><td>0.753±0.055</td><td>0.678±0.040</td><td>0.679±0.029</td><td>0.767±0.018</td><td>0.878±0.066</td><td>0.881±0.072</td></tr><tr><td>Metal-nut oo</td><td>0.816±0.029</td><td>0.855±0.024</td><td>0.825±0.023</td><td>0.855±0.016</td><td>0.932±0.028</td><td>0.928±0.037</td></tr><tr><td>AITEX oo</td><td>0.674±0.034</td><td>0.624±0.024</td><td>0.466±0.030</td><td>0.646±0.034</td><td>0.678±0.071</td><td>0.676±0.040</td></tr><tr><td>ELPV oo</td><td>0.614±0.048</td><td>0.691±0.008</td><td>0.566±0.111</td><td>0.648±0.057</td><td>0.627±0.056</td><td>0.616±0.032</td></tr><tr><td>Hyper-Kvasir oo|</td><td>0.406±0.018</td><td>0.571±0.004</td><td>0.480±0.044</td><td>0.595±0.023</td><td>0.677±0.065</td><td>0.687±0.037</td></tr></table></body></html>

Table 5: AUC results (mean $\pm$ std) on 8 real-world AD datasets under the hard setting (noted by $\diamondsuit .$ ). The best and second-best results and the third-best are respectively highlighted in red and blue and bold. $\infty$ is the average of anomaly classes. SAOE, FLOS, MLEP, and DevNet results are from Ding, Pang, and Shen (2022).

Table 6: AUC results (mean $\pm$ std) on 7 real-world AD datasets under general setting (noted by $\heartsuit$ ). The increasing and the decreasing (absolute values) is compared with $\ell = 2$ . $\varpi$ is the number of anomaly classes. If the performance under the noise model is still better than the baseline (DRA), it is noted as $\flat$ . The amplitude damping noise model is $\clubsuit$ , the phase flipping noise is $\spadesuit$ , the depolarizing noise is $^ *$ , and the bit flipping noise is $\varnothing$ .   

<html><body><table><tr><td>Dataset (ω)</td><td>[DRA + Qsco (l = 2)|</td><td>DRA +Qsco (l=2)+</td><td>DRA + Qsco (l = 2) +</td><td>DRA +Qsco (l=2)*</td><td>DRA +Qsco (𝑙=2)O</td></tr><tr><td colspan="6">Ten Training Anomaly Examples (Random) </td></tr><tr><td>AITEX (12)</td><td>一 0.875±0.029</td><td></td><td></td><td>|0.869±0.040 (0.69%)↓ |0.778±0.032 (11.09%)↓|0.801±0.0334(8.46%)↓|0.794±0.020 (9.26%)↓</td><td></td></tr><tr><td>SDD (1)</td><td>一 0.980±0.007</td><td></td><td></td><td>0.966±0.020 (1.43%)↓|0.971±0.006 (0.92%)↓|0.968±0.010 (1.22%)↓ |0.972±0.012 (0.82%)↓b</td><td></td></tr><tr><td>ELPV (2)</td><td>1 0.819±0.020</td><td></td><td></td><td>0.832±0.021(1.59%)↑b |0.823±0.018 (0.49%) ↑b|0.832±0.015 (1.59%) ↑b|0.840±0.020 (2.56%)↑ b</td><td></td></tr><tr><td>Optical (1) 1</td><td>0.969±0.008</td><td></td><td></td><td>|0.971±0.012 (0.21%)↑b |0.970±0.009 (0.10%) ↑b|0.973±0.017 (0.41%) ↑b|0.983±0.005 (1.44%)↑ b</td><td></td></tr><tr><td>BrainMRI (1)</td><td>0.972±0.017</td><td></td><td></td><td>0.878±0.018 (9.67%)↓ |0.893±0.019 (8.13%)↓|0.866±0.020 (10.90%)↓|0.871±0.022 (10.39%) ↓</td><td></td></tr><tr><td>HeadCT (1) 1</td><td>0.999±0.003</td><td></td><td></td><td>0.998±0.002 (0.10%)↓b|0.990±0.007 (0.90%)↓ |0.982±0.010 (1.70%)↓|0.982±0.010 (1.70%）↓</td><td></td></tr><tr><td>Hyper-Kvasir (4)|</td><td>0.813±0.019</td><td></td><td></td><td>0.852±0.023 (4.80%) ↑b|0.789±0.041 (2.95%)↓ |0.740±0.039 (9.00%)↓|0.787±0.038 (3.20%) ↓</td><td></td></tr><tr><td colspan="6">One Training Anomaly Example (Random) </td></tr><tr><td>AITEX (12) 一</td><td>0.708±0.027</td><td></td><td></td><td>0.734±0.023 (3.67%)↑b|0.700±0.038(1.13%)↓|0.722±0.019 (2.00%) ↑b| 0.708±0.034 ←→ b</td><td></td></tr><tr><td>SDD (1) 一</td><td>0.876±0.032</td><td>0.848±0.045 (3.20%)↓ |0.908±0.026 (3.65%)↑b|0.839±0.045 (4.22%)↓ |0.904±0.036 (3.20%)↑b</td><td></td><td></td><td></td></tr><tr><td>ELPV (2) 一</td><td>0.749±0.048</td><td></td><td></td><td>|0.621±0.046(17.10%)↓b|0.749±0.020 ←→b|0.770±0.035(2.80%)↑b|0.760±0.031(1.47%)↑b</td><td></td></tr><tr><td>Optical (1) 一</td><td>0.910±0.011</td><td></td><td></td><td>0.893±0.024 (1.87%)↓ |0.904±0.015 (0.66%)↓b|0.890±0.017 (2.20%)↓ |0.909±0.017 (0.11%)↓b</td><td></td></tr><tr><td>BrainMRI (1) 一</td><td>0.718±0.043</td><td></td><td></td><td>0.739±0.046 (2.92%)↑b |0.742±0.040 (3.34%) ↑b|0.729±0.044 (1.53%)↑b|0.703±0.048 (2.09%)↓b</td><td></td></tr><tr><td>HeadCT (1)</td><td>0.781±0.007</td><td></td><td></td><td>[0.871±0.061(11.52%)↑b|0.847±0.048 (8.45%)↑b|0.710±0.100 (9.10%)↓|0.716±0.085 (8.32%)↓</td><td></td></tr><tr><td>Hyper-Kvasir (4)|</td><td>0.768±0.015</td><td></td><td></td><td>0.804±0.057 (4.69%) ↑b |0.771±0.037 (0.39%) ↑b|0.804±0.035 (4.69%) ↑b|0.788±0.066 (2.60%)↑ b</td><td></td></tr></table></body></html>

be found in Appendix.

In the One Training Anomaly Example scenario, we observe an overall improvement, which is likely due to the reduced number of abnormal samples in the training data, which enables the model to focus more on capturing and learning the characteristics of normal data rather than being influenced by the noise associated with abnormal samples. Consequently, when noise is introduced, the model demonstrates a robust capacity to withstand minor data disturbances. This fortifies its adaptability and resilience against unseen noise, enhancing performance outcomes.

Table 7: Time complexity (in seconds) in the scalability analysis of Qsco. Average of 100 tests was performed on an NVIDIA GeForce RTX 3060 with 48 GB of RAM.   

<html><body><table><tr><td>Qubits</td><td>9</td><td>12</td></tr><tr><td>PennyLane</td><td>13.11 ±0.25</td><td>17.26 ± 1.37</td></tr><tr><td>Qubits</td><td>15</td><td>18</td></tr><tr><td>PennyLane</td><td>58.55 ± 1.24</td><td>182.19 ± 2.35</td></tr></table></body></html>

Limitations and Future Work The data presented in Table. 7 shows that classically simulating multiple instances of Qsco results in an exponential increase in time complexity. This exponential growth significantly constrains the scalability of classical simulations of Qsco. However, with the advent of frameworks such as NVIDIA’s CUDA Quantum (Kim et al. 2023), which enables the simulation of quantum algorithms on CUDA-enabled GPUs, there is potential to enhance the scalability and efficiency of our experiments by leveraging GPU-accelerated quantum simulations in the future. Despite these limitations, current findings demonstrate that a single Qsco surpasses the established baseline in the OSAD task. Meanwhile, we will continue to compare with the most SOTA OSAD methods (Zhu et al. 2024), striving to continuously improve Qsco and explore the potential of quantum computing in the OSAD domain.

# Conclusion

This paper proposed a quantum scoring module Qsco for the OSAD task. This module can be embedded in any advanced model and learn the high-dimensional distribution of abnormal data through variational quantum circuits and entanglement between qubits. The proposed approach solves the problem of abnormal data not seen during the training process, which causes model performance degradation. Experimental results indicate that implementing Qsco markedly enhances the predicted AUC score and does not result in significant time loss. In this paper, we explored the application of quantum computing in anomaly detection by implementing Qsco. Drawing on results from a simulated quantum computing environment with noise models, we demonstrate a proof of concept highlighting the potential of integrating quantum computing and deep learning into open-set anomaly detection.
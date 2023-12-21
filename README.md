# Randomised Nyström algorithm

Consider a matrix $A \in \mathbb{R}^{mxn}$ symmetric and psd (positive semi-definite).

For a sketching matrix $\Omega_1 \in \mathbb{R}^{nxl}$, the randomised Nyström approximation of $A$ takes the form:

$$\tilde{A}_{Nyst} = (A \Omega_1)(\Omega_1^T A \Omega_1)^+ (\Omega_1^T A) $$

where $(\Omega_1^T A \Omega_1)^+$ defines the pseudo-inverse of $\Omega_1^T A \Omega$. The randomness in the construction ensures that $\tilde{A}_{Nyst}$ is a good approximation to the original matrix $A$ with high probability.

From Equation \ref{eq:nys} you can observe that the three terms can be defined by just doing the multiplication $C = A \Omega_1$ once. Since, the middle term is given by $\Omega_1^T C$, while the third term is the transpose of the first one.

So, it is important to note that the randomised Nyström required just **one pass** over the original data $A$. 

Two aspects have to be taken into consideration:

- how do you compute the pseudo-inverse of $B = \Omega_1^T A \Omega_1$ ?
- where should I do the rank-k approximation? On the middle term $B$ or directly on $\tilde{A}_{Nyst}$?

Concerning the first question the easiest way to proceed is by applying the **Cholesky factorization** to $B$. Sometimes, the matrix $\Omega_1^T A \Omega_1$ can be rank-deficient, for instance, if $A$ or $B$ have a lower rank than $l$, which will cause a problem for obtaining a Cholesky factorization. In this case, a remedy can be to compute an SVD instead of the Cholesky factorization of $B$ [[1]](#1).

Notice that the SVD factorization has to satisfy the property from Cholesky factorization such that the starting matrix can be expressed as the product of a lower triangular matrix and its conjugate transpose. To have this nice property, since $A$ is symmetric, the SVD coincide with the definition of eigenvalues and eigenvectors $A = U \Sigma U^T$. Then, we can observe that $A = U \sqrt{\Sigma} \sqrt{\Sigma} U^T$.

Concerning the second question, if you do a rank-k truncation directly on $B = \Omega_1^T A \Omega_1$ you lose accuracy in the algorithm. The main idea for the Nyström code can be wrapped in the following algorithm:

**Randomised Nyström with rank-k truncation on $B$.**

Input $A \in \mathcal{R}^{mxn}$, $\Omega_1 \in \mathcal{R}^{nxl}$:

1. Compute $C = A \Omega_1$, $C \in \mathcal{R}^{mxl}$.
2. Compute $B = \Omega_1^T C$, $B \in \mathcal{R}^{lxl}$.
3. Compute truncated rank-k eigenvalue decomposition of $B = U \Lambda U^T$.
4. Compute the pseudo-inverse as: $B_k^+ = U(:, 1:k) \Lambda(:, 1:k)^{+} U(:,1:k)^T$.
5. Compute the QR decomposition of $C = QR$.
6. Compute the eigenvalue decomposition of $R B_k^+ R^T = U_k \Lambda_k U_k^T$.
7. Compute $\hat{U_k} = Q U_k$.
8. Output $\hat{U_k}$ and $\Lambda_k$.

Then, the preferred way is to do the rank-k approximation to $\tilde{A}_{Nyst}$. The algorithm follows these steps:

**Randomised Nyström with rank-k truncation on $\tilde{A}_{Nyst}$.**

Input $A \in \mathcal{R}^{mxn}$, $\Omega_1 \in \mathcal{R}^{nxl}$:

1. Compute $C = A \Omega_1$, $C \in \mathcal{R}^{mxl}$.
2. Compute $B = \Omega_1^T C$, $B \in \mathcal{R}^{lxl}$.
3. Apply Cholesky factorization to $B = LL^T$.
4. Compute $Z = CL^{-T}$ with back substitution (which means solve the system $L^T Z = C$).
5. Compute the QR factorization of $Z = QR$.
6. Compute truncated rank-k SVD of $R$ as $U_k \Sigma_k V_k^T$.
7. Compute $\hat{U_k} = Q U_k$.
8. Output $\hat{U_k}$ and $\Sigma_k^2$.

A small note, you can decide to compute $\hat{U_k}$ in two ways: the more stable way is to write it as $Q U_K$, while the more efficient way is to write it as $ZV_k \Sigma_k^{-1}$.

From an algebraic point of view, the following steps have to be made:

(A Ω₁)(Ω₁ᵀ A Ω₁)⁺ (Ω₁ᵀ A)

= C L⁻ᵀ L⁻¹ Cᵀ

= Z Zᵀ [recall Z = CL⁻ᵀ]

= Q R Rᵀ Qᵀ [QR-factorization of Z]

= Q Uₖ Σₖ Σₖ Uₖᵀ Qᵀ [rank-k SVD of R]

= Ûₖ Σₖ² Ûₖᵀ

and that:

$$\hat{U}_k = Q U_k = C L^{-T} R^{-1} U_k = Z V_k \Sigma_k^{-1}$$

in this case since $\Sigma_k$ is diagonal then $\Sigma_k \Sigma_k^{-1} = I$ and the last step reduce to simply evaluate: $\tilde{A}_{Nyst} = Z V_k (Z V_k)^T$.

The **project** will be developed by using Algorithm 2 (Randomised Nyström with rank-k truncation on $\tilde{A}_{Nyst}$) and the two ways of writing $\hat{U}_k$ can be tested.

## Sketching matrix

Two sketching matrices will be used to test the algorithm. The first one is the **Gaussian embeddings**, the idea is to generate a matrix $\Omega_1 \in \mathbb{R}^{l \times m}$ from a Gaussian distribution with mean zero and unitary variance.

The second sketching matrix is the **block SRHT embeddings**. It has been derived from the sub-sampled randomised Hadamard transform matrix:

$$\Omega_1 = \sqrt{\frac{m}{l}} P H D$$

The three matrices are the following:

- $D \in \mathbb{R}^{m \times m}$: diagonal matrix of independent random sign (plus or minus ones in the diagonal).
- $H \in \mathbb{R}^{m \times m}$: normalized Walsh-Hadamard matrix.
- $P \in \mathbb{R}^{l \times m}$: draws $l$ rows uniformly at random.

In this case, $\Omega_1$ is an OSE(n; $\epsilon$; $\delta$) of size $l = \textit{O}(\epsilon^{-2} (n + \log(\frac{m}{\delta})) \log(\frac{n}{\delta}))$.

Regrettably, the suitability of products featuring SRHT matrices for distributed computing is limited, thereby restricting the advantages of SRHT on contemporary architectures. This limitation primarily arises from the challenge of computing products with $H$ in tensor form.

This justifies the introduction of the new method block-SRHT, which attempts to distribute the workload between the various processors as $\Omega_1=[\Omega_1^{(1)},...,\Omega_1^{(P)}]$ and:

$$\Omega_1^{(i)} = \sqrt{\frac{m}{Pl}} \tilde{D}^{(i)} P H D^{(i)}$$

Each $\Omega_1^{(i)}$ is related to a unique sampling matrix $R$ and different (independent from each other) diagonal matrices $D^{(i)}$ with i.i.d.. The dimensions of the matrices are the following: $D^{(i)} \in \mathbb{R}^{m/p \times m/p}$, $H \in \mathbb{R}^{m/p \times m/p}$, $P \in \mathbb{R}^{l \times m/p}$, and $\tilde{D}^{(i)} \in \mathbb{R}^{l \times l}$.

The global $\Omega_1$ can be seen as:

$$
    \Omega_1 = \sqrt{\frac{m}{Pl}} [\tilde{D}^{(1)} ... \tilde{D}^{(P)}] 
    \begin{bmatrix}
    P H &  &  \\
        & \ddots & \\ 
        &   & P H \\
  \end{bmatrix} 
  \begin{bmatrix}
    D^{(1)} &  &  \\
        & \ddots & \\ 
        &   & D^{(P)}\\
  \end{bmatrix} 
$$

Notice that the $ H $ matrix maintains orthogonality because overall it appears as a block diagonal matrix. Also, the advantage of the sketching matrix written in this way is that it is easy to parallelize the matrix-matrix multiplication:

$$\Omega_1 W = \sqrt{\frac{m}{Pl}} \sum_{i=1}^P \tilde{D}^{(i)} P H D^{(i)} W^{(i)}$$

## Data set

Two datasets were used to test the Nyström algorithm. The first one is the $\textbf{Modified NIST (MNIST)}$ dataset, which contains binary images of handwritten digits and was created by "re-mixing" the samples from NIST's original datasets. 

The original black and white (bi-level) images were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The normalized image is located in a 28×28 plane.

The MNIST database contains 60,000 training images and 10,000 testing images. The MNIST dataset has images with pixel values in the range [0, 255]. It is considered the scaled one, so each feature is divided by 255 and now the range of values is between [0, 1].\\


The second dataset $\textbf{YearPredictionMSD}$ is derived from the $\textbf{Million Song Dataset (MSD)}$, which is an extensive compilation of audio features and metadata for a million contemporary popular music tracks, freely available for exploration. Many tasks can be addressed using the $\textbf{MSD}$, in the dataset $\textbf{YearPredictionMSD}$ the focus is on year prediction. The meaning of 'year prediction' is related to estimating the year in which a song was released based on its audio features.

The dataset contains songs that are mostly Western, commercial tracks ranging from 1922 to 2011, with a peak in the year 2000s. The dimension of the training set is 463,715 and the dimension of the testing set is 51,630. Each element contains 90 features: 12, the timbre average, and 78, the timbre covariance.

The previous dataset was chosen to make a comparison with the results coming from the article [[1]](#1).
The matrix used for testing was built through the following radial basis function:
$e^{\frac{-||x_i - x_j||^2}{\sigma^2}}$

The python class create_matrix.py contains other functions to create other matrices for the analysis of the Nyström algorithm. A description of those matrices can be found in [[2]](#2).

## References
<a id="1">[1]</a> 
Balabanov, Oleg and Beaupère, Matthias and Grigori, Laura and Lederer, Victor (2022). 
'Block subsampled randomised Hadamard transform for low-rank approximation on distributed architectures'. 
$\textit{International Conference for Machine Learning}$.
\\
<a id="2">[2]</a> 
Tropp, J. A. and Yurtsever, A. and Udell, M. and Cevher, V. (2017).
'Fixed-rank approximation of a positive-semidefinite matrix from streaming data'.
$\textit{Advances in Neural Information Processing Systems}$

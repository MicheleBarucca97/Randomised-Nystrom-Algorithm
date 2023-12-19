# Randomized Nyström algorithm

Consider a matrix $A \in \mathbb{R}^{mxn}$ symmetric and psd (positive semi-definite).

For a sketching matrix $\Omega_1 \in \mathbb{R}^{nxl}$, the randomized Nyström approximation of $A$ takes the form:

$$\tilde{A}_{Nyst} = (A \Omega_1)(\Omega_1^T A \Omega_1)^+ (\Omega_1^T A) $$

where $(\Omega_1^T A \Omega_1)^+$ defines the pseudo-inverse of $\Omega_1^T A \Omega$. The randomness in the construction ensures that $\tilde{A}_{Nyst}$ is a good approximation to the original matrix $A$ with high probability.

From Equation \ref{eq:nys} you can observe that the three terms can be defined by just doing the multiplication $C = A \Omega_1$ once. Since, the middle term is given by $\Omega_1^T C$, while the third term is the transpose of the first one.

So, it is important to note that the randomized Nyström required just **one pass** over the original data $A$.

Two aspects have to be taken into consideration:

- how do you compute the pseudo-inverse of $B = \Omega_1^T A \Omega_1$ ?
- where should I do the rank-k approximation? On the middle term $B$ or directly on $\tilde{A}_{Nyst}$?

Concerning the first question the easiest way to proceed is by applying the **Cholesky factorization** to $B$. Sometimes, the matrix $\Omega_1^T A \Omega_1$ can be rank-deficient, for instance, if $A$ or $B$ have a lower rank than $l$, which will cause a problem for obtaining a Cholesky factorization. In this case, a remedy can be to compute an SVD instead of the Cholesky factorization of $B$ (Block subsampled randomized Hadamard transform for low-rank approximation on distributed architectures, Balabanov et al).

Notice that the SVD factorization has to satisfy the property from Cholesky factorization such that the starting matrix can be expressed as the product of a lower triangular matrix and its conjugate transpose. To have this nice property, since $A$ is symmetric, the SVD coincide with the definition of eigenvalues and eigenvectors $A = U \Sigma U^T$. Then, we can observe that $A = U \sqrt{\Sigma} \sqrt{\Sigma} U^T$.

Concerning the second question, if you do a rank-k truncation directly on $B = \Omega_1^T A \Omega_1$ you lose accuracy in the algorithm. The main idea for the Nyström code can be wrapped in the following algorithm:

**Randomized Nyström with rank-k truncation on $B$.**

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

**Randomized Nyström with rank-k truncation on $\tilde{A}_{Nyst}$.**

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

1. \( (A \Omega_1)(\Omega_1^T A \Omega_1)^+ (\Omega_1^T A) \)

2. \( = C L^{-T} L^{-1} C^T \)

3. \( = Z Z^T \) \[ \text{recall } Z = CL^{-T} \]

4. \( = Q R R^T Q^T \) \[ \text{QR-factorization of } Z \]

5. \( = Q U_k \Sigma_k \Sigma_k U_k^T Q^T \) \[ \text{rank-k SVD of } R \]

6. \( = \hat{U}_k \Sigma_k^2 \hat{U}_k^T \)


and that:

\begin{align}
    \hat{U}_k = Q U_k = C L^{-T} R^{-1} U_k = Z V_k \Sigma_k^{-1}
\end{align}

in this case since $\Sigma_k$ is diagonal then $\Sigma_k \Sigma_k^{-1} = I$ and the last step reduce to simply evaluate: $\tilde{A}_{Nyst} = Z V_k (Z V_k)^T$.

The **project** will be developed by using Algorithm \ref{alg:A} and the two ways of writing $\hat{U}_k$ will be tested.

## Sketching matrix

Two sketching matrices will be used to test the algorithm. The first one is the **Gaussian embeddings**, the idea is to generate a matrix $\Omega_1 \in \mathbb{

# k-means聚类算法及相关问题详解

## 🧑‍💻 User

8. (16 pts) Consider partitioning a set of data points $\{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n\}$, where each $\mathbf{x}_i$ is a 3-dimensional RGB color vector, into $K$ clusters by using k-means algorithm. (a) Assume we are given a set of initial $K$ mean vectors, $\mathbf{c}_1, \dots, \mathbf{c}_K$. Give the main two-step procedure in the k-mean clustering. (b) How is the K-means++ algorithm used in the k-means clustering? What is its advantage? (c) For a normalized-cut algorithm for image segmentation, how do you define the weights between two connected nodes when considering distance and color similarities in the affinity measure? (d) How do you apply the mean-shift algorithm for this clustering problem? Give the main steps.

---
## 😀 默认助手 (7)

## (a) The two-step k-means (Lloyd's) procedure

Given initial means $\mathbf{c}_1,\dots,\mathbf{c}_K$, the algorithm alternates the following two steps until convergence (cluster assignments stop changing):

**Step 1 — Assignment.** Assign every point to the cluster whose mean is closest (here Euclidean distance in RGB space):

$$
\mathbf{x}_i \in \mathcal{C}_k \quad\text{where}\quad k=\arg\min_{j=1,\dots,K}\;\|\mathbf{x}_i-\mathbf{c}_j\|^2
$$

**Step 2 — Update.** Recompute each mean as the centroid of its assigned points:

$$
\mathbf{c}_k=\frac{1}{|\mathcal{C}_k|}\sum_{\mathbf{x}_i\in\mathcal{C}_k}\mathbf{x}_i
$$

Iterate 1–2 until the assignments/centroids are stable (or a distortion tolerance is met). Each step never increases the objective

$$
J=\sum_{k=1}^{K}\sum_{\mathbf{x}_i\in\mathcal{C}_k}\|\mathbf{x}_i-\mathbf{c}_k\|^2
$$

so the process converges, but only to a **local** minimum that depends on the initialization. (If a cluster becomes empty, re-seed it, e.g. with the farthest point.)

---

## (b) K-means++ initialization

K-means++ only replaces the **initialization**; the two steps in (a) are unchanged. It picks the initial means sequentially:

1. Pick the first center $\mathbf{c}_1$ uniformly at random from $\{\mathbf{x}_i\}$.
2. For each point, compute its squared distance to the nearest chosen center:
   $$
   D(\mathbf{x})^2=\min_{j\in\text{chosen}}\|\mathbf{x}-\mathbf{c}_j\|^2
   $$
3. Choose the next center $\mathbf{x}_i$ with probability proportional to $D(\mathbf{x}_i)^2$ (i.e. $p_i = D(\mathbf{x}_i)^2/\sum_{i'}D(\mathbf{x}_{i'})^2$).
4. Repeat until $K$ centers are obtained, then run standard k-means.

**Advantage:** it spreads the seeds out, giving an $O(\log K)$-approximation to the optimal distortion **in expectation**, so it avoids the poor local minima and empty clusters produced by random seeding — typically faster convergence and more stable, reproducible results.

---

## (c) Affinity weights for normalized cut

Nodes $i,j$ are pixels (or data points); the weight $w_{ij}$ should be large for pairs that are likely in the same region. With intensity/color vector $\mathbf{I}_i=\mathbf{x}_i$ (RGB) and spatial position $\mathbf{X}_i$, the standard choice is a product of two Gaussians, restricted to a local neighborhood:

$$
w_{ij}=
\begin{cases}
\exp\!\left(-\dfrac{\|\mathbf{I}_i-\mathbf{I}_j\|^2}{\sigma_I^2}\right)\exp\!\left(-\dfrac{\|\mathbf{X}_i-\mathbf{X}_j\|^2}{\sigma_X^2}\right), & \|\mathbf{X}_i-\mathbf{X}_j\|<r\\[2mm]
0, & \text{otherwise}
\end{cases}
$$

- $\sigma_I$ controls **color** tolerance, $\sigma_X$ the **spatial** (distance) tolerance, and $r$ keeps $\mathbf{W}$ sparse (only nearby pixels are connected).
- If the data are only RGB points with no spatial coordinates (the setting in this problem), the spatial factor drops out and one uses the color-only kernel
  $$
  w_{ij}=\exp\!\left(-\|\mathbf{x}_i-\mathbf{x}_j\|^2/\sigma^2\right),
  $$
  possibly set to $0$ outside a $k$-NN or radius neighborhood.

Then with $\mathbf{W}=[w_{ij}]$, degree matrix $\mathbf{D}=\mathrm{diag}(d_i)$, $d_i=\sum_j w_{ij}$, the **Ncut** objective for a bipartition $(A,B)$ is

$$
\mathrm{Ncut}(A,B)=\frac{\mathrm{cut}(A,B)}{\mathrm{assoc}(A,V)}+\frac{\mathrm{cut}(A,B)}{\mathrm{assoc}(B,V)}
$$

minimized (relaxed) by the generalized eigenproblem $(\mathbf{D}-\mathbf{W})\mathbf{y}=\lambda\mathbf{D}\mathbf{y}$; the eigenvector of the second-smallest eigenvalue gives the partition (recursive or $K$-way via several eigenvectors).

---

## (d) Mean-shift clustering for this problem

Mean-shift seeks the **modes** of the color density; it needs no $K$ and no initialization, only a bandwidth $h$.

**Density estimate** (Gaussian kernel $K$):

$$
\hat f(\mathbf{x})=\frac{1}{nh^3}\sum_{i=1}^{n}K\!\left(\frac{\mathbf{x}-\mathbf{x}_i}{h}\right),\qquad
K(\mathbf{u})=c\,k(\|\mathbf{u}\|^2)
$$

**Mean-shift vector** (gradient-ascent direction):

$$
\mathbf{m}(\mathbf{x})=\frac{\sum_i \mathbf{x}_i\,k\!\left(\left\|\frac{\mathbf{x}-\mathbf{x}_i}{h}\right\|^2\right)}{\sum_i k\!\left(\left\|\frac{\mathbf{x}-\mathbf{x}_i}{h}\right\|^2\right)}-\mathbf{x}
$$

**Steps:**

1. For each data point $\mathbf{x}_i$, initialize $\mathbf{y}\leftarrow\mathbf{x}_i$.
2. Repeatedly move to the kernel-weighted mean of neighbors:
   $$
   \mathbf{y}\leftarrow\frac{\sum_i \mathbf{x}_i\,g\!\left(\left\|\frac{\mathbf{y}-\mathbf{x}_i}{h}\right\|^2\right)}{\sum_i g\!\left(\left\|\frac{\mathbf{y}-\mathbf{x}_i}{h}\right\|^2\right)},\qquad g=-k'
   $$
   until $\|\mathbf{y}_{\text{new}}-\mathbf{y}\|<\varepsilon$ (converged to a mode).
3. Record the converged mode for each point.
4. **Merge:** points whose trajectories end at the same mode (or whose modes are within $h$) form one cluster; distinct modes are distinct clusters.
5. Optionally merge modes/regions smaller than a minimum size.

The bandwidth $h$ is the single scale parameter: too small gives many tiny clusters, too large merges everything. Because only local neighbor sums are needed, the weighted-mean step is $O(n)$ per iteration per point (or accelerated with a KD-tree).

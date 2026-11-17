# 结构从运动 SfM 问题与算法详解

## 🧑‍💻 User

7. (16 pts) Consider the structure-from-motion (SfM) problem, given the detected 2D image feature points $\mathbf{x}_{ij}$, where $i$ denotes the $i$-th camera (camera project matrix $\mathbf{P}_i$) and $j$ denotes the $j$-th 3D point $\mathbf{X}_j$, $i = 1, \dots, m$ and $j = 1, \dots, n$. (a) Describe the structure-from-motion problem. (b) In the sequential structure from motion algorithm, describe the two main steps for each additional view. (c) What is the reprojection error in bundle adjustment? (d) What is the reconstruction ambiguity in the SfM problem?

---
## 😀 默认助手 (7)

# Structure from Motion (SfM) — Solution

## (a) The SfM problem

**Given:** only 2D feature correspondences (tracks) between images,
$$\mathbf{x}_{ij}\in\mathbb{P}^2,\qquad i=1,\dots,m \ \text{(cameras)},\quad j=1,\dots,n\ \text{(points)},$$
where $\mathbf{x}_{ij}$ is the (homogeneous) image of 3D point $\mathbf{X}_j$ in camera $i$ — possibly not every point is seen in every image (a partly-filled *visibility matrix*).

**Unknowns:** the camera projection matrices and the 3D structure
$$\mathbf{P}_i=\mathbf{K}_i\,[\mathbf{R}_i\,|\,\mathbf{t}_i]\in\mathbb{R}^{3\times 4},\qquad \mathbf{X}_j\in\mathbb{P}^3 .$$

**Model:** the pinhole/central projection
$$\lambda_{ij}\,\mathbf{x}_{ij}=\mathbf{P}_i\mathbf{X}_j \quad\Longleftrightarrow\quad \mathbf{x}_{ij}\times\big(\mathbf{P}_i\mathbf{X}_j\big)=\mathbf{0},$$
the latter giving **2 independent scalar equations** per observation.

**Goal:** recover *both* $\{\mathbf{P}_i\}$ (intrinsics $\mathbf{K}_i$ and extrinsics $\mathbf{R}_i,\mathbf{t}_i$) and $\{\mathbf{X}_j\}$ *simultaneously, from the images alone*, by minimizing a global cost — the **total reprojection error** (bundle adjustment):
$$\min_{\{\mathbf{P}_i\},\{\mathbf{X}_j\}}\ \sum_{(i,j)\in\mathcal{V}}\Big\|\mathbf{x}_{ij}-\pi\big(\mathbf{P}_i\mathbf{X}_j\big)\Big\|^{2},\qquad \pi([u,v,w]^\top)=\Big(\tfrac{u}{w},\tfrac{v}{w}\Big).$$

**Why it is hard (the essence of the problem):**
- It is a **chicken-and-egg / bilinear** problem: to get the 3D points you need the camera poses, and to get the poses you need the 3D points. Neither is known in advance.
- It is **nonlinear and non-convex** (perspective division in $\pi$), so no closed-form global solution; one needs a good initialization plus iterative refinement.
- It is **scale-/gauge-ambiguous and only defined up to a global transformation** (see (d)); the intrinsics may also be unknown.
- It requires **robustness to outliers** (wrong matches) and care with **degenerate configurations** (pure rotation, planar scene, small parallax / short baseline).

**Counting:** $2mn$ scalar equations vs. $11m+3n$ unknowns ($\mathbf{P}_i$: 11 DOF, $\mathbf{X}_j$: 3 DOF); the problem is well-posed only when $2mn \gg 11m+3n$ (minus gauge DOF) and the motion has sufficient parallax. Standard solvers: *factorization* (affine case) or *incremental/sequential SfM* with two-view initialization and bundle adjustment.

---

## (b) Two main steps per additional view (sequential/incremental SfM)

A two-view (epipolar) reconstruction is built first: estimate the essential/fundamental matrix, decompose to get relative pose, triangulate the initial points, then localize via bundle adjustment. Then for each new view:

### Step 1 — Camera resection / pose estimation of the new camera ($2$D–$3$D)
Using the 3D points already reconstructed, find the **2D–3D correspondences** between the new image's features and existing 3D points (by matching against the tracks in already-registered views), and solve for the new pose $\mathbf{P}_{new}$ (or $\mathbf{R},\mathbf{t}$ with known $\mathbf{K}$):
$$\mathbf{x}_{new,j}\ \longleftrightarrow\ \mathbf{X}_j \quad\Longrightarrow\quad \mathbf{P}_{new}.$$
This is the **Perspective-$n$-Point (PnP)** problem: a linear DLT solution
$$\mathbf{x}\times(\mathbf{P}\mathbf{X})=\mathbf{0}$$
followed by nonlinear refinement of $\mathbf{R},\mathbf{t}$ on the reprojection error, wrapped in **RANSAC** for outlier rejection (a minimal 3–4 point solver inside the loop).

### Step 2 — Triangulation / structure extension ($2$D–$2$D $\to$ $3$D)
With the new camera now *registered*, **triangulate new 3D points** for the tracks that are visible in the new view and in at least one (well-conditioned) already-registered view. For views $i,k$:
$$\mathbf{x}_{ij}\times(\mathbf{P}_i\mathbf{X}_j)=\mathbf{0},\qquad \mathbf{x}_{kj}\times(\mathbf{P}_k\mathbf{X}_j)=\mathbf{0}\ \Rightarrow\ \mathbf{A}_j\mathbf{X}_j=\mathbf{0},$$
solved by SVD (or a midpoint/optimal two-view triangulation). New points are added only if they have **sufficient parallax** (large enough baseline angle) — otherwise the depth is ill-conditioned.

### (Optional but standard) Step 3 — Bundle adjustment
Run **local (windowed/sliding-window) BA** on the newly added cameras and points (then a global BA at the end) to jointly refine poses and structure and to distribute accumulated drift.

> Each new view thus = **"localize (resect) → extend (triangulate) → refine (BA)"**, and the loop repeats for views $3,4,\dots,m$.

---

## (c) Reprojection error in bundle adjustment

For observation $\mathbf{x}_{ij}$ and the current estimate $(\mathbf{P}_i,\mathbf{X}_j)$, the **predicted** image point is $\hat{\mathbf{x}}_{ij}=\pi(\mathbf{P}_i\mathbf{X}_j)$, and the **reprojection error** is the geometric distance in the image plane between the measured and predicted point:
$$\mathbf{e}_{ij}=\mathbf{x}_{ij}-\pi(\mathbf{P}_i\mathbf{X}_j)\in\mathbb{R}^2,\qquad e_{ij}=\|\mathbf{e}_{ij}\|^2 .$$

Bundle adjustment minimizes the **sum of squared reprojection errors over all observations** (all cameras and all points, jointly, with $\mathbf{P}_i=\mathbf{K}_i[\mathbf{R}_i|\mathbf{t}_i]$):
$$\min_{\{\mathbf{K}_i,\mathbf{R}_i,\mathbf{t}_i\},\{\mathbf{X}_j\}}\ \sum_{i=1}^{m}\sum_{j\in\mathcal{V}(i)}\rho\!\left(\big\|\mathbf{x}_{ij}-\pi\big(\mathbf{K}_i[\mathbf{R}_i|\mathbf{t}_i]\mathbf{X}_j\big)\big\|^{2}\right),$$
where $\mathcal{V}(i)$ denotes the points visible in view $i$ (missing observations are simply omitted), and $\rho(\cdot)$ is an optional **robust loss** (e.g. Huber) to down-weight mismatches.

Key points:
- The error is measured **in the 2D image (in pixels)** — it is the *geometric* (not algebraic) error, preferable to the algebraic $\|\mathbf{x}\times\mathbf{P}\mathbf{X}\|$ because it is physically meaningful and statistically well behaved.
- It is nonlinear in both $\mathbf{P}_i$ and $\mathbf{X}_j$; minimized by **Levenberg–Marquardt / Gauss–Newton**, exploiting the sparse block structure (each observation couples one camera and one point) via the **Schur complement**.
- Since only image residuals are minimized (no 3D-point residuals), the cost is invariant to the ambiguity in (d); hence the **gauge is fixed** by convention (e.g. $\mathbf{P}_1=[\mathbf{K}_1|\mathbf{0}]$, or $\|\mathbf{t}_1\|=1$ / unit scale) to make the normal equations non-singular.
- It is the *final* joint refinement stage: structure and motion are optimized *together*, so errors are not committed to a bad early guess.

---

## (d) Reconstruction ambiguity in SfM

The reprojection error only constrains $\mathbf{P}_i\mathbf{X}_j$ up to a **global nonsingular $4\times4$ transformation** $\mathbf{H}$. Indeed, for any such $\mathbf{H}$,
$$\mathbf{P}_i'=\mathbf{P}_i\mathbf{H}^{-1},\qquad \mathbf{X}_j'=\mathbf{H}\mathbf{X}_j \quad\Longrightarrow\quad \mathbf{P}_i'\mathbf{X}_j'=\mathbf{P}_i\mathbf{X}_j,$$
so the images (and hence the reprojection error) are **exactly identical**. The reconstruction is therefore **not unique** — only determined up to a group of transformations:

| Assumption | Ambiguity group $\mathbf{H}$ | DOF |
|---|---|---|
| Unknown intrinsics, general cameras | arbitrary nonsingular $4\times4$ $\Rightarrow$ **projective** (homography of $\mathbb{P}^3$) | $15$ |
| Affine cameras (or camera centers at infinity) | affine transform | $12$ |
| **Calibrated** cameras ($\mathbf{K}_i$ known) | **similarity**: rotation (3) + translation (3) + scale (1) | $7$ |

- **Projective ambiguity:** from point correspondences alone with unknown $\mathbf{K}$, the reconstruction is only a *projective* one — straight lines stay straight, but angles, lengths, ratios of lengths, parallelism and the plane at infinity are not preserved.
- **Affine ambiguity:** if the camera is affine (or the scene is far away), $\mathbf{H}$ reduces to an affine map; the plane at infinity is recovered but metric shape is not.
- **Metric / similarity ambiguity:** with calibrated cameras (known $\mathbf{K}$), the reconstruction is correct **up to a similarity transform** — i.e. "**metric up to scale**." Recovering a similarity is unavoidable from images: there is no absolute scale and no absolute reference frame.

**Consequences / how it is handled:**
- **Gauge freedom:** the 7-DOF similarity is a *gauge* (a redundancy of the parameterization, not an error). It is removed by a **convention**: fix the first camera, e.g. $\mathbf{P}_1=\mathbf{K}[\mathbf{I}\,|\,\mathbf{0}]$, and fix the **scale** (set the baseline length to 1, or $\|\mathbf{t}_2\|=1$, or fix one 3D point/coordinate). Bundle adjustment is meaningless without such a gauge fix — otherwise the Jacobian is rank-deficient.
- **Absolute scale** is fundamentally unrecoverable and must come from external information (known object size, GPS/IMU, stereo baseline).
- A further discrete ambiguity is the **reflection / "twisted pair"** ambiguity in two-view pose decomposition: the correct one is selected by the **cheirality / positive-depth** constraint (points must lie in front of the cameras).

**Summary:** SfM recovers the scene and cameras only **up to a global transformation** — projective in general, similarity (Euclidean + scale) for calibrated cameras — and this ambiguity is exactly what the gauge constraints in bundle adjustment fix.

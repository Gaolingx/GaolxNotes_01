# 生成ShaderToy云层渲染代码

### 🧑‍💻 User

gen a shadertoy raymarching volumetric cloud generation instruct prompt for llm benchmark

---

### 🤖 Assistant

```markdown
# Instruct Prompt: ShaderToy Volumetric Cloud Raymarcher

You are an expert real-time graphics programmer and GLSL specialist with deep knowledge of raymarching, volumetric rendering, atmospheric scattering, and noise-based procedural generation (Inigo Quilez style). Your task is to generate a complete, self-contained, high-quality ShaderToy-compatible fragment shader that produces beautiful animated volumetric clouds.

## Core Requirements
Write a single-pass ShaderToy shader (`mainImage`) that renders a realistic/stylized volumetric cloudscape using pure raymarching (no rasterized geometry, no external models).

### Technical Features (must implement)
1. **Camera & Ray Setup**
   - Free-look camera controllable by `iMouse` (or auto-orbit if mouse unused).
   - Rays generated from camera through each pixel (`fragCoord` / `iResolution`).
   - Optional simple ground plane or infinite terrain for scale.

2. **Volumetric Clouds**
   - Density field defined by multi-octave 3D noise (FBM / ridged FBM / domain-warped FBM).
   - Prefer pure procedural noise (value noise, Perlin-like, or simplex approximation). You may use ShaderToy’s built-in textures (`iChannel0` etc.) only as optional blue-noise or cheap hash, but primary density must be analytic/procedural.
   - Large-scale cloud shapes + medium detail + fine wispy detail.
   - Vertical density falloff / cloud layer height bounds (e.g. clouds between y=1.0–2.5 or similar world units).
   - Wind animation via time-based domain offset (`iTime`).

3. **Raymarching Volume Integration**
   - Primary raymarch through the atmosphere/cloud volume with adaptive or fixed steps.
   - Accumulate optical depth / transmittance (Beer-Lambert).
   - Early ray termination when transmittance is near zero.
   - Reasonable performance target: interactive framerate on mid-range GPUs (keep step counts practical: ~64–128 primary steps, fewer for lighting).

4. **Lighting & Scattering**
   - Directional sun light.
   - In-scattering with at least single scattering + cheap multiple-scattering approximation.
   - Phase function (Henyey-Greenstein or Schlick approximation) for forward/backward scattering.
   - Soft self-shadowing via secondary short raymarch or cone/height-based approximation toward the sun.
   - Ambient sky light / multi-scatter bounce approximation so clouds are not pure black in shadow.
   - Optional silver lining / edge lighting.

5. **Atmosphere & Sky**
   - Simple but pleasing sky gradient or basic atmospheric scattering (Rayleigh-ish).
   - Sun disk with glow.
   - Optional god-rays / crepuscular rays if performance allows (cheap version).
   - Clouds correctly composite over the sky with proper alpha/transmittance.

6. **Aesthetics & Polish**
   - Cinematic, soft, fluffy clouds with good silhouette and internal structure.
   - Nice color palette (warm sun, cool shadows, subtle purple/blue ambient).
   - Temporal coherence (no crazy flickering).
   - Mild post-process: tonemapping (ACES or Reinhard), subtle contrast/saturation, optional vignette or grain.
   - Resolution-aware and aspect-correct.

### Code Constraints
- Pure GLSL ES 1.00 / ShaderToy dialect.
- Everything inside `void mainImage(out vec4 fragColor, in vec2 fragCoord)`.
- Helper functions allowed (noise, FBM, raymarch, lighting, etc.).
- Use only standard ShaderToy uniforms: `iTime`, `iResolution`, `iMouse`, `iFrame`, `iChannelResolution`, etc.
- Prefer no mandatory external textures. If you use channels, document what they should be (e.g. “iChannel0 = noise texture”).
- Must compile and run when pasted directly into https://www.shadertoy.com/new.
- Add clear comments explaining major sections and tunable parameters (cloud height, density multiplier, sun direction, scattering coefficients, step counts, etc.).
- Expose a few `#define` or `const` parameters at the top for easy artistic control.

### Quality Bar (for benchmark scoring)
- Visually impressive out-of-the-box result (not just a gray blob).
- Correct volumetric integration (transmittance + scattering energy roughly conserved).
- Stable under camera motion and animation.
- Clean, readable, reasonably optimized code.
- No major artifacts (banding, extreme noise, black holes, energy explosions).

### Output Format
Return **only** the complete GLSL shader code, ready to paste into ShaderToy.
Start with any necessary comments / parameter block, then the full source including `mainImage`.
Do **not** wrap the code in markdown fences in your final answer.
Do **not** add explanations before or after the code — the code itself (with internal comments) is the entire deliverable.

### Optional Stretch Goals (bonus quality)
- Multiple cloud layers or storm system.
- Simple 2D cloud map for large-scale coverage.
- Cheap temporal reprojection or blue-noise dithering to reduce banding.
- Interactive sun position via mouse.
- Day/night cycle.

Generate the shader now.
```

This prompt is designed as a strong LLM benchmark task because it requires:
- Solid understanding of raymarching + volumetric integration
- Procedural noise / FBM expertise
- Physically-plausible lighting & scattering
- Practical real-time constraints
- Clean ShaderToy-specific GLSL
- Artistic judgment for a beautiful result

You can feed the above block directly to an LLM as the user (or system+user) message to evaluate code generation quality.
# Title

### 🧑‍💻 User

```python
import numpy as np
import open3d as o3d

from tqdm import tqdm

def disk_samples_fibonacci(K, radius):
    ga = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(K, dtype=np.float32)
    r = radius * np.sqrt((i + 0.5) / K)
    theta = i * ga
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float32)

def pair_indices(idx_list):
    idx = np.asarray(idx_list)
    i, j = np.triu_indices(len(idx), k=1)  # upper triangle, exclude diagonal
    pair_a = idx[i]  # shape (m,)
    pair_b = idx[j]  # shape (m,)
    return pair_a, pair_b

def process_frames(vertices, faces, idx_list, disk_samples=100, disk_radius=0.1, device='CPU:0'):
    disk = disk_samples_fibonacci(disk_samples, disk_radius)
    pair_a, pair_b = pair_indices(idx_list)
    n = len(vertices)
    result = np.zeros((n, 15, 3))
    for i, vts in tqdm(enumerate(vertices), total=n):
        scene = build_scene(vts, faces, device=device)
        result[i] = np.array(bundle_scores(scene, vts[pair_a], vts[pair_b], disk, dev=device)).T
    return result



def build_scene(vertices, faces, nthreads=0, device="CPU:0"):
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.uint32)

    dev = o3d.core.Device(device)
    scene = o3d.t.geometry.RaycastingScene(nthreads=nthreads, device=dev)
    scene.add_triangles(
        o3d.core.Tensor(v, dtype=o3d.core.Dtype.Float32, device=dev),
        o3d.core.Tensor(f, dtype=o3d.core.Dtype.UInt32,  device=dev),
    )
    return scene

def segment_occlusion_and_thickness(
    scene,
    p0,
    p1,
    endpoint_eps=1e-4,
    t_eps=1e-6,
    dev=o3d.core.Device("CPU:0"),
    *,
    compute_occlusion=True,
    occupancy_nsamples=3,
):
    """
    p0, p1: (M,3) arrays
    Returns:
      occluded: (M,) bool      True if any surface hit occurs before the segment ends
      L_inside: (M,) float32   total length of segment that lies inside the (watertight) mesh
      L_total:  (M,) float32   segment length after endpoint trimming

    Notes:
    - Same algorithm as your original version.
    - Main speedup is eliminating the Python loop over rays for thickness accumulation.
    - If you only need thickness, set compute_occlusion=False to skip cast_rays().
    """
    endpoint_eps = np.float32(endpoint_eps)
    t_eps = np.float32(t_eps)

    p0 = np.ascontiguousarray(p0, dtype=np.float32)
    p1 = np.ascontiguousarray(p1, dtype=np.float32)

    d = p1 - p0
    L = np.linalg.norm(d, axis=1).astype(np.float32)
    M = p0.shape[0]

    occluded = np.zeros(M, dtype=bool)
    L_inside = np.zeros(M, dtype=np.float32)
    L_total  = np.zeros(M, dtype=np.float32)

    good = L > np.float32(1e-12)
    if not np.any(good):
        return occluded, L_inside, L_total

    # Unit direction for original segment
    u = np.zeros_like(d, dtype=np.float32)
    u[good] = d[good] / L[good, None]

    # Trim endpoints to avoid self-hit (points are on the surface)
    shrink = np.minimum(endpoint_eps, np.float32(0.49) * L).astype(np.float32)
    o = p0 + u * shrink[:, None]
    e = p1 - u * shrink[:, None]

    seg = e - o
    L2 = np.linalg.norm(seg, axis=1).astype(np.float32)
    L_total[good] = L2[good]

    good2 = L2 > np.float32(1e-12)
    if not np.any(good2):
        return occluded, L_inside, L_total

    # Work only on valid rays, then scatter back to full-size outputs.
    idx = np.flatnonzero(good2)
    o2 = o[idx]
    seg2 = seg[idx]
    L2_2 = L2[idx]

    u2 = seg2 / L2_2[:, None]  # (R,3), unit direction => t_hit is distance
    R = idx.size

    rays = np.empty((R, 6), dtype=np.float32)
    rays[:, :3] = o2
    rays[:, 3:] = u2
    rays_t = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32, device=o3d.core.Device(dev))

    # ---- 1) Occlusion via first hit (optional) ----
    if compute_occlusion:
        first = scene.cast_rays(rays_t)
        t_first = first["t_hit"].cpu().numpy().astype(np.float32)  # inf if no hit
        occluded2 = np.isfinite(t_first) & (t_first < (L2_2 - t_eps))
        occluded[idx] = occluded2

    # ---- 2) Thickness via all intersections + inside/outside toggling ----
    # occupancy at start tells whether we begin inside the volume (watertight assumption)
    # Reuse the ray tensor slice to avoid another Tensor construction.
    inside0 = scene.compute_occupancy(
        rays_t[:, :3],
        nsamples=int(occupancy_nsamples),
    ).cpu().numpy().astype(np.int8)  # (R,)

    inter = scene.list_intersections(rays_t)
    ray_splits = inter["ray_splits"].cpu().numpy().astype(np.int64)  # (R+1,)
    t_all = inter["t_hit"].cpu().numpy().astype(np.float32)          # (N,)

    # Vectorized equivalent of:
    #   hits = sorted(t in (0,L2))
    #   toggle inside on each hit, accumulate inside segments
    counts = np.diff(ray_splits)  # (R,)
    N = t_all.size

    if N == 0:
        # No intersections for any ray: inside length is either full segment or 0.
        L_inside[idx] = L2_2 * inside0.astype(np.float32)
        return occluded, L_inside, L_total

    # Map each hit to its ray id (0..R-1)
    ray_id = np.repeat(np.arange(R, dtype=np.int64), counts)

    # Filter hits to (0, L2) like your original code
    in_seg = (t_all > 0.0) & (t_all < L2_2[ray_id])
    t = t_all[in_seg]
    rid = ray_id[in_seg]

    if t.size == 0:
        # All hits were outside (0,L2) after filtering
        counts2 = np.zeros(R, dtype=np.int64)
        alt = np.zeros(R, dtype=np.float32)
    else:
        # Sort by (ray_id, t) so we have per-ray sorted hit lists (same as hits.sort()).
        order = np.lexsort((t, rid))  # rid primary, t secondary
        t = t[order]
        rid = rid[order]

        # Number of filtered hits per ray
        counts2 = np.bincount(rid, minlength=R).astype(np.int64)

        # Compute index-within-ray for each hit in the sorted arrays:
        # idx_in = 0,1,2,... per ray
        splits2 = np.empty(R + 1, dtype=np.int64)
        splits2[0] = 0
        np.cumsum(counts2, out=splits2[1:])

        idx_in = np.arange(t.size, dtype=np.int64) - splits2[rid]

        # Alternating-sum sign pattern is exactly equivalent to toggling:
        # inside0=0: signs = - + - + ...
        # inside0=1: signs = + - + - ...
        s = inside0[rid].astype(np.int64)
        parity = (idx_in + (1 - s)) & 1
        sign = (1 - 2 * parity).astype(np.float32)  # 0->+1, 1->-1

        weighted = t.astype(np.float32) * sign
        alt = np.bincount(rid, weights=weighted, minlength=R).astype(np.float32)

    # Tail term corresponds to whether we end inside after last hit:
    # tail_inside = inside0 XOR (n_hits % 2)
    tail_inside = (inside0.astype(np.int64) ^ (counts2 & 1)).astype(np.float32)
    L_inside2 = alt + L2_2 * tail_inside

    L_inside[idx] = L_inside2.astype(np.float32)
    return occluded, L_inside, L_total

def bundle_scores(scene, p0, p1, xy, endpoint_eps=1e-4, t_eps=1e-6, dev="CPU:0"):

    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    xy = np.asarray(xy, np.float32)
    assert xy.ndim == 2 and xy.shape[1] == 2, "xy must be (K,2)"

    K = xy.shape[0]

    d = p1 - p0
    L = np.linalg.norm(d, axis=1)
    M = p0.shape[0]
    good = L > 1e-12

    u = np.zeros_like(d)
    u[good] = d[good] / L[good, None]

    # Per-segment perpendicular basis (v,w)
    a = np.tile(np.array([1, 0, 0], np.float32), (M, 1))
    mask = np.abs(u[:, 0]) > 0.9
    a[mask] = np.array([0, 1, 0], np.float32)

    v = np.cross(u, a)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    w = np.cross(u, v)

    ox = xy[:, 0]  # (K,)
    oy = xy[:, 1]  # (K,)

    # (M,K,3) offsets in 3D
    off = v[:, None, :] * ox[None, :, None] + w[:, None, :] * oy[None, :, None]

    p0k = (p0[:, None, :] + off).reshape(-1, 3)
    p1k = (p1[:, None, :] + off).reshape(-1, 3)

    # Uses your previously defined segment_occlusion_and_thickness(scene, dev, ...)
    _, L_inside_k, _ = segment_occlusion_and_thickness(
        scene, p0k, p1k, endpoint_eps=endpoint_eps, t_eps=t_eps, dev=dev
    )
    L_inside_k = L_inside_k.reshape(M, K)

    blocked = L_inside_k > 0.0
    blocked_fraction = blocked.mean(axis=1)

    blocked_count = np.maximum(blocked.sum(axis=1), 1)
    mean_L_inside = (L_inside_k * blocked).sum(axis=1) / blocked_count

    min_L_inside = L_inside_k.min(axis=1)

    # degenerate segments -> zeros
    blocked_fraction[~good] = 0.0
    mean_L_inside[~good] = 0.0
    min_L_inside[~good] = 0.0

    return blocked_fraction.astype(np.float32), mean_L_inside.astype(np.float32), min_L_inside.astype(np.float32)
```


and in the main program I just call process_frame

it's slow.

After 4 hours trying to deal with open3d GPU support I gave up. Please don’t ask me to work on it anymore.

I am looking for another solution.

Since I have NVIDIA GPU and all cuda stuff ready, write a complete C++ project that done exactly all thing as in the open3d version above, using optiX and pybindings. You may not assume testing point does not begin inside or outside of the model.

---

### 🤖 Assistant

# OptiX-based Ray Caster with Python Bindings

I'll create a complete C++ project using NVIDIA OptiX 7+ with pybind11 bindings that replicates the functionality of your Open3D code.

## Project Structure

```
optix_raycaster/
├── CMakeLists.txt
├── src/
│   ├── launch_params.h
│   ├── optix_kernels.cu
│   ├── optix_raycaster.h
│   ├── optix_raycaster.cpp
│   └── python_bindings.cpp
├── python/
│   └── raycaster.py
└── setup.py
```

---

## 1. `CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.18)
project(optix_raycaster LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)
find_package(pybind11 CONFIG REQUIRED)

# OptiX path - set this to your OptiX installation
set(OptiX_INSTALL_DIR "$ENV{OptiX_INSTALL_DIR}" CACHE PATH "Path to OptiX installation")
if(NOT OptiX_INSTALL_DIR)
    set(OptiX_INSTALL_DIR "/opt/optix" CACHE PATH "Path to OptiX installation" FORCE)
endif()

if(NOT EXISTS "${OptiX_INSTALL_DIR}/include/optix.h")
    message(FATAL_ERROR "Cannot find optix.h in ${OptiX_INSTALL_DIR}/include. "
                        "Please set OptiX_INSTALL_DIR to your OptiX 7+ installation directory.")
endif()

# Compile PTX from CUDA
set(CUDA_NVRTC_FLAGS 
    -I${OptiX_INSTALL_DIR}/include
    -I${CMAKE_CURRENT_SOURCE_DIR}/src
    --std=c++17
    -arch=compute_70
    -use_fast_math
    -lineinfo
    -default-device
    -rdc=true
    --relocatable-device-code=true
)

# Generate PTX at build time
set(PTX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/src/optix_kernels.cu)
set(PTX_OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/optix_kernels.ptx)

add_custom_command(
    OUTPUT ${PTX_OUTPUT}
    COMMAND ${CUDAToolkit_NVCC_EXECUTABLE}
        -ptx
        -I${OptiX_INSTALL_DIR}/include
        -I${CMAKE_CURRENT_SOURCE_DIR}/src
        --std=c++17
        -arch=sm_70
        --use_fast_math
        -lineinfo
        -o ${PTX_OUTPUT}
        ${PTX_SOURCE}
    DEPENDS ${PTX_SOURCE} ${CMAKE_CURRENT_SOURCE_DIR}/src/launch_params.h
    COMMENT "Compiling OptiX kernels to PTX"
    VERBATIM
)

add_custom_target(ptx_target DEPENDS ${PTX_OUTPUT})

# Embed PTX as string
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ptx_embedded.cpp.in
"namespace optix_raycaster {
extern const char* embedded_ptx_code = R\"PTX(@PTX_CONTENT@)PTX\";
}
")

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/ptx_embedded.cpp
    COMMAND ${CMAKE_COMMAND}
        -DPTX_FILE=${PTX_OUTPUT}
        -DOUTPUT_FILE=${CMAKE_CURRENT_BINARY_DIR}/ptx_embedded.cpp
        -DTEMPLATE_FILE=${CMAKE_CURRENT_BINARY_DIR}/ptx_embedded.cpp.in
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake
    DEPENDS ${PTX_OUTPUT} ptx_target
    COMMENT "Embedding PTX code"
    VERBATIM
)

# Create embed script
file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_ptx.cmake
"file(READ \${PTX_FILE} PTX_CONTENT)
string(CONFIGURE \"\@PTX_CONTENT\@\" PTX_ESCAPED)
file(READ \${TEMPLATE_FILE} TEMPLATE_CONTENT)
string(REPLACE \"@PTX_CONTENT@\" \"\${PTX_CONTENT}\" OUTPUT_CONTENT \"\${TEMPLATE_CONTENT}\")
file(WRITE \${OUTPUT_FILE} \"\${OUTPUT_CONTENT}\")
")

# Main library
pybind11_add_module(optix_raycaster_cpp
    src/optix_raycaster.cpp
    src/python_bindings.cpp
    ${CMAKE_CURRENT_BINARY_DIR}/ptx_embedded.cpp
)

target_include_directories(optix_raycaster_cpp PRIVATE
    ${OptiX_INSTALL_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(optix_raycaster_cpp PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)

target_compile_definitions(optix_raycaster_cpp PRIVATE
    PTX_EMBEDDED
)

# Install
install(TARGETS optix_raycaster_cpp
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
```

---

## 2. `src/launch_params.h`

```cpp
#pragma once

#include <cuda_runtime.h>
#include <optix.h>

namespace optix_raycaster {

// Shared data structures between host and device

struct alignas(16) LaunchParams {
    // Traversable handle for the scene
    OptixTraversableHandle traversable;
  
    // Ray data (device pointers)
    float3* ray_origins;
    float3* ray_directions;
    float*  ray_t_max;          // Maximum t for each ray
    int     num_rays;
  
    // Output for closest-hit query
    float*  t_hit_closest;      // t of closest hit (INFINITY if no hit)
  
    // Output for all-hits query
    int*    hit_counts;         // Number of hits per ray
    float*  hit_t_buffer;       // Flat buffer of all t values
    int*    hit_offsets;        // Start offset in hit_t_buffer for each ray
    int     max_hits_per_ray;   // Allocated space per ray
  
    // Output for occupancy query
    int8_t* occupancy;          // 1 if inside, 0 if outside
  
    // Query mode
    int     mode;               // 0: closest, 1: all hits, 2: occupancy
};

// Ray payload for closest hit
struct ClosestHitPayload {
    float t_hit;
};

// Ray payload for all hits (count mode)
struct AllHitsPayload {
    int   ray_idx;
    int   hit_count;
    float t_hits[64];  // Fixed size for register storage
};

// Ray payload for occupancy
struct OccupancyPayload {
    int intersection_count;
};

// SBT record types
template<typename T>
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

struct EmptyData {};
struct HitGroupData {
    float3* vertices;
    uint3*  indices;
};

using RayGenRecord   = SbtRecord<EmptyData>;
using MissRecord     = SbtRecord<EmptyData>;
using HitGroupRecord = SbtRecord<HitGroupData>;

// Constants
constexpr int MODE_CLOSEST   = 0;
constexpr int MODE_ALL_HITS  = 1;
constexpr int MODE_OCCUPANCY = 2;

// Ray types
constexpr int RAY_TYPE_CLOSEST   = 0;
constexpr int RAY_TYPE_ALL_HITS  = 1;
constexpr int RAY_TYPE_OCCUPANCY = 2;
constexpr int NUM_RAY_TYPES      = 3;

} // namespace optix_raycaster
```

---

## 3. `src/optix_kernels.cu`

```cpp
#include <optix.h>
#include <cuda_runtime.h>

#include "launch_params.h"

using namespace optix_raycaster;

extern "C" {
    __constant__ LaunchParams params;
}

//------------------------------------------------------------------------------
// Helper functions
//------------------------------------------------------------------------------

static __forceinline__ __device__ void packPointer(void* ptr, unsigned int& u0, unsigned int& u1) {
    const unsigned long long uptr = reinterpret_cast<unsigned long long>(ptr);
    u0 = uptr >> 32;
    u1 = uptr & 0x00000000ffffffff;
}

static __forceinline__ __device__ void* unpackPointer(unsigned int u0, unsigned int u1) {
    const unsigned long long uptr = static_cast<unsigned long long>(u0) << 32 | u1;
    return reinterpret_cast<void*>(uptr);
}

template<typename T>
static __forceinline__ __device__ T* getPayload() {
    return reinterpret_cast<T*>(unpackPointer(optixGetPayload_0(), optixGetPayload_1()));
}

//------------------------------------------------------------------------------
// Ray Generation Programs
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__closest() {
    const int idx = optixGetLaunchIndex().x;
    if (idx >= params.num_rays) return;
  
    const float3 origin = params.ray_origins[idx];
    const float3 direction = params.ray_directions[idx];
    const float t_max = params.ray_t_max ? params.ray_t_max[idx] : 1e20f;
  
    ClosestHitPayload payload;
    payload.t_hit = 1e20f;
  
    unsigned int u0, u1;
    packPointer(&payload, u0, u1);
  
    optixTrace(
        params.traversable,
        origin,
        direction,
        0.0f,           // tmin
        t_max,          // tmax
        0.0f,           // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        RAY_TYPE_CLOSEST,   // SBT offset
        NUM_RAY_TYPES,      // SBT stride
        RAY_TYPE_CLOSEST,   // miss SBT index
        u0, u1
    );
  
    params.t_hit_closest[idx] = payload.t_hit;
}

extern "C" __global__ void __raygen__all_hits() {
    const int idx = optixGetLaunchIndex().x;
    if (idx >= params.num_rays) return;
  
    const float3 origin = params.ray_origins[idx];
    const float3 direction = params.ray_directions[idx];
    const float t_max = params.ray_t_max ? params.ray_t_max[idx] : 1e20f;
  
    // We'll trace multiple times, advancing past each hit
    float t_min = 0.0f;
    int hit_count = 0;
    const int offset = params.hit_offsets[idx];
    const int max_hits = params.max_hits_per_ray;
  
    while (hit_count < max_hits) {
        ClosestHitPayload payload;
        payload.t_hit = 1e20f;
      
        unsigned int u0, u1;
        packPointer(&payload, u0, u1);
      
        optixTrace(
            params.traversable,
            origin,
            direction,
            t_min,
            t_max,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_CLOSEST,
            NUM_RAY_TYPES,
            RAY_TYPE_CLOSEST,
            u0, u1
        );
      
        if (payload.t_hit >= t_max || payload.t_hit >= 1e19f) {
            break;
        }
      
        params.hit_t_buffer[offset + hit_count] = payload.t_hit;
        hit_count++;
      
        // Advance slightly past this hit
        t_min = payload.t_hit + 1e-6f;
    }
  
    params.hit_counts[idx] = hit_count;
}

extern "C" __global__ void __raygen__occupancy() {
    const int idx = optixGetLaunchIndex().x;
    if (idx >= params.num_rays) return;
  
    // Cast ray in +Z direction and count intersections
    const float3 origin = params.ray_origins[idx];
    const float3 direction = make_float3(0.0f, 0.0f, 1.0f);
  
    int intersection_count = 0;
    float t_min = 0.0f;
    const float t_max = 1e20f;
  
    // Count all intersections by repeated tracing
    for (int i = 0; i < 256; i++) {  // Max iterations
        ClosestHitPayload payload;
        payload.t_hit = 1e20f;
      
        unsigned int u0, u1;
        packPointer(&payload, u0, u1);
      
        optixTrace(
            params.traversable,
            origin,
            direction,
            t_min,
            t_max,
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_NONE,
            RAY_TYPE_CLOSEST,
            NUM_RAY_TYPES,
            RAY_TYPE_CLOSEST,
            u0, u1
        );
      
        if (payload.t_hit >= 1e19f) {
            break;
        }
      
        intersection_count++;
        t_min = payload.t_hit + 1e-6f;
    }
  
    // Odd number of intersections = inside (for watertight mesh)
    params.occupancy[idx] = (intersection_count & 1) ? 1 : 0;
}

//------------------------------------------------------------------------------
// Miss Programs
//------------------------------------------------------------------------------

extern "C" __global__ void __miss__closest() {
    ClosestHitPayload* payload = getPayload<ClosestHitPayload>();
    payload->t_hit = 1e20f;  // No hit
}

extern "C" __global__ void __miss__all_hits() {
    // Nothing to do - payload already initialized
}

extern "C" __global__ void __miss__occupancy() {
    // Nothing to do
}

//------------------------------------------------------------------------------
// Closest Hit Programs
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__closest() {
    ClosestHitPayload* payload = getPayload<ClosestHitPayload>();
    payload->t_hit = optixGetRayTmax();
}

extern "C" __global__ void __closesthit__all_hits() {
    ClosestHitPayload* payload = getPayload<ClosestHitPayload>();
    payload->t_hit = optixGetRayTmax();
}

extern "C" __global__ void __closesthit__occupancy() {
    // Just need to record that we hit something - t value is used in raygen
    ClosestHitPayload* payload = getPayload<ClosestHitPayload>();
    payload->t_hit = optixGetRayTmax();
}

//------------------------------------------------------------------------------
// Any Hit Programs (optional, for future optimization)
//------------------------------------------------------------------------------

extern "C" __global__ void __anyhit__closest() {
    // Accept the hit
}

extern "C" __global__ void __anyhit__all_hits() {
    // Accept all hits
}

extern "C" __global__ void __anyhit__occupancy() {
    // Accept all hits
}
```

---

## 4. `src/optix_raycaster.h`

```cpp
#pragma once

#include <vector>
#include <memory>
#include <tuple>
#include <string>

#include <cuda_runtime.h>
#include <optix.h>

namespace optix_raycaster {

class RaycastScene {
public:
    RaycastScene(const float* vertices, int num_vertices,
                 const uint32_t* faces, int num_faces,
                 int device_id = 0);
    ~RaycastScene();
  
    // Disable copy
    RaycastScene(const RaycastScene&) = delete;
    RaycastScene& operator=(const RaycastScene&) = delete;
  
    // Move operations
    RaycastScene(RaycastScene&&) noexcept;
    RaycastScene& operator=(RaycastScene&&) noexcept;
  
    /**
     * Cast rays and return closest hit t values.
     * @param origins    Ray origins (num_rays * 3 floats, row-major)
     * @param directions Ray directions (num_rays * 3 floats, row-major)
     * @param num_rays   Number of rays
     * @param t_max      Optional max t per ray (nullptr = infinity)
     * @return Vector of t values (INFINITY if no hit)
     */
    std::vector<float> cast_rays(
        const float* origins,
        const float* directions,
        int num_rays,
        const float* t_max = nullptr
    );
  
    /**
     * List all intersections for each ray.
     * @param origins    Ray origins (num_rays * 3 floats)
     * @param directions Ray directions (num_rays * 3 floats)
     * @param t_max      Max t per ray (num_rays floats)
     * @param num_rays   Number of rays
     * @param max_hits_per_ray Maximum hits to record per ray
     * @return Tuple of (hit_counts[num_rays], ray_splits[num_rays+1], all_t_values[total_hits])
     */
    std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>
    list_intersections(
        const float* origins,
        const float* directions,
        const float* t_max,
        int num_rays,
        int max_hits_per_ray = 64
    );
  
    /**
     * Compute occupancy (inside/outside) for query points.
     * Uses ray casting in +Z direction and odd/even rule.
     * @param points     Query points (num_points * 3 floats)
     * @param num_points Number of points
     * @return Vector of occupancy values (1 = inside, 0 = outside)
     */
    std::vector<int8_t> compute_occupancy(
        const float* points,
        int num_points
    );
  
    int get_device_id() const { return device_id_; }
  
private:
    void initialize_optix();
    void create_module();
    void create_programs();
    void create_pipeline();
    void create_sbt();
    void build_gas(const float* vertices, int num_vertices,
                   const uint32_t* faces, int num_faces);
  
    // CUDA device
    int device_id_;
    CUcontext cuda_context_;
    CUstream stream_;
  
    // OptiX handles
    OptixDeviceContext optix_context_;
    OptixModule module_;
    OptixPipeline pipeline_;
  
    // Program groups
    OptixProgramGroup raygen_prog_closest_;
    OptixProgramGroup raygen_prog_all_hits_;
    OptixProgramGroup raygen_prog_occupancy_;
    OptixProgramGroup miss_prog_closest_;
    OptixProgramGroup miss_prog_all_hits_;
    OptixProgramGroup miss_prog_occupancy_;
    OptixProgramGroup hitgroup_prog_closest_;
    OptixProgramGroup hitgroup_prog_all_hits_;
    OptixProgramGroup hitgroup_prog_occupancy_;
  
    // Shader Binding Table
    CUdeviceptr sbt_raygen_record_;
    CUdeviceptr sbt_miss_records_;
    CUdeviceptr sbt_hitgroup_records_;
    OptixShaderBindingTable sbt_closest_;
    OptixShaderBindingTable sbt_all_hits_;
    OptixShaderBindingTable sbt_occupancy_;
  
    // Acceleration structure
    OptixTraversableHandle gas_handle_;
    CUdeviceptr gas_buffer_;
  
    // Geometry data on device
    CUdeviceptr d_vertices_;
    CUdeviceptr d_indices_;
    int num_vertices_;
    int num_faces_;
  
    // Launch params buffer
    CUdeviceptr d_launch_params_;
};

// PTX code (embedded at compile time)
extern const char* embedded_ptx_code;

} // namespace optix_raycaster
```

---

## 5. `src/optix_raycaster.cpp`

```cpp
#include "optix_raycaster.h"
#include "launch_params.h"

#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <numeric>

namespace optix_raycaster {

//------------------------------------------------------------------------------
// CUDA helper macros
//------------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t error = call;                                            \
        if (error != cudaSuccess) {                                          \
            throw std::runtime_error(                                        \
                std::string("CUDA error: ") + cudaGetErrorString(error) +    \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                    \
    } while (0)

#define CUDA_DRIVER_CHECK(call)                                               \
    do {                                                                      \
        CUresult result = call;                                              \
        if (result != CUDA_SUCCESS) {                                        \
            const char* errorStr;                                            \
            cuGetErrorString(result, &errorStr);                             \
            throw std::runtime_error(                                        \
                std::string("CUDA driver error: ") + errorStr +              \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                    \
    } while (0)

#define OPTIX_CHECK(call)                                                     \
    do {                                                                      \
        OptixResult result = call;                                           \
        if (result != OPTIX_SUCCESS) {                                       \
            throw std::runtime_error(                                        \
                std::string("OptiX error: ") + optixGetErrorString(result) + \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                    \
    } while (0)

//------------------------------------------------------------------------------
// OptiX log callback
//------------------------------------------------------------------------------

static void optix_log_callback(unsigned int level, const char* tag, const char* message, void*) {
    // Suppress logs in production; uncomment for debugging:
    // fprintf(stderr, "[OptiX][%d][%s]: %s\n", level, tag, message);
}

//------------------------------------------------------------------------------
// Constructor / Destructor
//------------------------------------------------------------------------------

RaycastScene::RaycastScene(const float* vertices, int num_vertices,
                           const uint32_t* faces, int num_faces,
                           int device_id)
    : device_id_(device_id)
    , cuda_context_(nullptr)
    , stream_(nullptr)
    , optix_context_(nullptr)
    , module_(nullptr)
    , pipeline_(nullptr)
    , raygen_prog_closest_(nullptr)
    , raygen_prog_all_hits_(nullptr)
    , raygen_prog_occupancy_(nullptr)
    , miss_prog_closest_(nullptr)
    , miss_prog_all_hits_(nullptr)
    , miss_prog_occupancy_(nullptr)
    , hitgroup_prog_closest_(nullptr)
    , hitgroup_prog_all_hits_(nullptr)
    , hitgroup_prog_occupancy_(nullptr)
    , sbt_raygen_record_(0)
    , sbt_miss_records_(0)
    , sbt_hitgroup_records_(0)
    , gas_handle_(0)
    , gas_buffer_(0)
    , d_vertices_(0)
    , d_indices_(0)
    , num_vertices_(num_vertices)
    , num_faces_(num_faces)
    , d_launch_params_(0)
{
    initialize_optix();
    create_module();
    create_programs();
    create_pipeline();
    build_gas(vertices, num_vertices, faces, num_faces);
    create_sbt();
  
    // Allocate launch params buffer
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_launch_params_), sizeof(LaunchParams)));
}

RaycastScene::~RaycastScene() {
    try {
        CUDA_CHECK(cudaSetDevice(device_id_));
      
        if (d_launch_params_) cudaFree(reinterpret_cast<void*>(d_launch_params_));
        if (sbt_raygen_record_) cudaFree(reinterpret_cast<void*>(sbt_raygen_record_));
        if (sbt_miss_records_) cudaFree(reinterpret_cast<void*>(sbt_miss_records_));
        if (sbt_hitgroup_records_) cudaFree(reinterpret_cast<void*>(sbt_hitgroup_records_));
        if (gas_buffer_) cudaFree(reinterpret_cast<void*>(gas_buffer_));
        if (d_vertices_) cudaFree(reinterpret_cast<void*>(d_vertices_));
        if (d_indices_) cudaFree(reinterpret_cast<void*>(d_indices_));
      
        if (hitgroup_prog_occupancy_) optixProgramGroupDestroy(hitgroup_prog_occupancy_);
        if (hitgroup_prog_all_hits_) optixProgramGroupDestroy(hitgroup_prog_all_hits_);
        if (hitgroup_prog_closest_) optixProgramGroupDestroy(hitgroup_prog_closest_);
        if (miss_prog_occupancy_) optixProgramGroupDestroy(miss_prog_occupancy_);
        if (miss_prog_all_hits_) optixProgramGroupDestroy(miss_prog_all_hits_);
        if (miss_prog_closest_) optixProgramGroupDestroy(miss_prog_closest_);
        if (raygen_prog_occupancy_) optixProgramGroupDestroy(raygen_prog_occupancy_);
        if (raygen_prog_all_hits_) optixProgramGroupDestroy(raygen_prog_all_hits_);
        if (raygen_prog_closest_) optixProgramGroupDestroy(raygen_prog_closest_);
        if (pipeline_) optixPipelineDestroy(pipeline_);
        if (module_) optixModuleDestroy(module_);
        if (optix_context_) optixDeviceContextDestroy(optix_context_);
        if (stream_) cudaStreamDestroy(stream_);
    } catch (...) {
        // Suppress exceptions in destructor
    }
}

RaycastScene::RaycastScene(RaycastScene&& other) noexcept {
    *this = std::move(other);
}

RaycastScene& RaycastScene::operator=(RaycastScene&& other) noexcept {
    if (this != &other) {
        device_id_ = other.device_id_;
        cuda_context_ = other.cuda_context_;
        stream_ = other.stream_;
        optix_context_ = other.optix_context_;
        module_ = other.module_;
        pipeline_ = other.pipeline_;
        raygen_prog_closest_ = other.raygen_prog_closest_;
        raygen_prog_all_hits_ = other.raygen_prog_all_hits_;
        raygen_prog_occupancy_ = other.raygen_prog_occupancy_;
        miss_prog_closest_ = other.miss_prog_closest_;
        miss_prog_all_hits_ = other.miss_prog_all_hits_;
        miss_prog_occupancy_ = other.miss_prog_occupancy_;
        hitgroup_prog_closest_ = other.hitgroup_prog_closest_;
        hitgroup_prog_all_hits_ = other.hitgroup_prog_all_hits_;
        hitgroup_prog_occupancy_ = other.hitgroup_prog_occupancy_;
        sbt_raygen_record_ = other.sbt_raygen_record_;
        sbt_miss_records_ = other.sbt_miss_records_;
        sbt_hitgroup_records_ = other.sbt_hitgroup_records_;
        sbt_closest_ = other.sbt_closest_;
        sbt_all_hits_ = other.sbt_all_hits_;
        sbt_occupancy_ = other.sbt_occupancy_;
        gas_handle_ = other.gas_handle_;
        gas_buffer_ = other.gas_buffer_;
        d_vertices_ = other.d_vertices_;
        d_indices_ = other.d_indices_;
        num_vertices_ = other.num_vertices_;
        num_faces_ = other.num_faces_;
        d_launch_params_ = other.d_launch_params_;
      
        // Clear other
        other.cuda_context_ = nullptr;
        other.stream_ = nullptr;
        other.optix_context_ = nullptr;
        other.module_ = nullptr;
        other.pipeline_ = nullptr;
        other.raygen_prog_closest_ = nullptr;
        other.raygen_prog_all_hits_ = nullptr;
        other.raygen_prog_occupancy_ = nullptr;
        other.miss_prog_closest_ = nullptr;
        other.miss_prog_all_hits_ = nullptr;
        other.miss_prog_occupancy_ = nullptr;
        other.hitgroup_prog_closest_ = nullptr;
        other.hitgroup_prog_all_hits_ = nullptr;
        other.hitgroup_prog_occupancy_ = nullptr;
        other.sbt_raygen_record_ = 0;
        other.sbt_miss_records_ = 0;
        other.sbt_hitgroup_records_ = 0;
        other.gas_handle_ = 0;
        other.gas_buffer_ = 0;
        other.d_vertices_ = 0;
        other.d_indices_ = 0;
        other.d_launch_params_ = 0;
    }
    return *this;
}

//------------------------------------------------------------------------------
// Initialization
//------------------------------------------------------------------------------

void RaycastScene::initialize_optix() {
    CUDA_CHECK(cudaSetDevice(device_id_));
    CUDA_CHECK(cudaStreamCreate(&stream_));
  
    CUDA_DRIVER_CHECK(cuCtxGetCurrent(&cuda_context_));
  
    OPTIX_CHECK(optixInit());
  
    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = optix_log_callback;
    options.logCallbackLevel = 4;
  
    OPTIX_CHECK(optixDeviceContextCreate(cuda_context_, &options, &optix_context_));
}

void RaycastScene::create_module() {
    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifdef NDEBUG
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
#else
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 2;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";
    pipeline_compile_options.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    const char* ptx = embedded_ptx_code;
    size_t ptx_size = strlen(ptx);
  
    char log[2048];
    size_t log_size = sizeof(log);
  
    OPTIX_CHECK(optixModuleCreate(
        optix_context_,
        &module_compile_options,
        &pipeline_compile_options,
        ptx,
        ptx_size,
        log,
        &log_size,
        &module_
    ));
}

void RaycastScene::create_programs() {
    char log[2048];
    size_t log_size;
  
    OptixProgramGroupOptions pg_options = {};
  
    // Ray generation programs
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module_;
        desc.raygen.entryFunctionName = "__raygen__closest";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &desc, 1, &pg_options, log, &log_size, &raygen_prog_closest_));
      
        desc.raygen.entryFunctionName = "__raygen__all_hits";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &desc, 1, &pg_options, log, &log_size, &raygen_prog_all_hits_));
      
        desc.raygen.entryFunctionName = "__raygen__occupancy";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &desc, 1, &pg_options, log, &log_size, &raygen_prog_occupancy_));
    }
  
    // Miss programs
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = module_;
        desc.miss.entryFunctionName = "__miss__closest";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &desc, 1, &pg_options, log, &log_size, &miss_prog_closest_));
      
        desc.miss.entryFunctionName = "__miss__all_hits";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &desc, 1, &pg_options, log, &log_size, &miss_prog_all_hits_));
      
        desc.miss.entryFunctionName = "__miss__occupancy";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &desc, 1, &pg_options, log, &log_size, &miss_prog_occupancy_));
    }
  
    // Hit group programs
    {
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = module_;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__closest";
        desc.hitgroup.moduleAH = module_;
        desc.hitgroup.entryFunctionNameAH = "__anyhit__closest";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &desc, 1, &pg_options, log, &log_size, &hitgroup_prog_closest_));
      
        desc.hitgroup.entryFunctionNameCH = "__closesthit__all_hits";
        desc.hitgroup.entryFunctionNameAH = "__anyhit__all_hits";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &desc, 1, &pg_options, log, &log_size, &hitgroup_prog_all_hits_));
      
        desc.hitgroup.entryFunctionNameCH = "__closesthit__occupancy";
        desc.hitgroup.entryFunctionNameAH = "__anyhit__occupancy";
        log_size = sizeof(log);
        OPTIX_CHECK(optixProgramGroupCreate(optix_context_, &desc, 1, &pg_options, log, &log_size, &hitgroup_prog_occupancy_));
    }
}

void RaycastScene::create_pipeline() {
    OptixProgramGroup program_groups[] = {
        raygen_prog_closest_,
        raygen_prog_all_hits_,
        raygen_prog_occupancy_,
        miss_prog_closest_,
        miss_prog_all_hits_,
        miss_prog_occupancy_,
        hitgroup_prog_closest_,
        hitgroup_prog_all_hits_,
        hitgroup_prog_occupancy_
    };
  
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
  
    char log[2048];
    size_t log_size = sizeof(log);
  
    OPTIX_CHECK(optixPipelineCreate(
        optix_context_,
        nullptr,  // Will use options from module creation
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &log_size,
        &pipeline_
    ));
  
    // Set stack sizes
    OptixStackSizes stack_sizes = {};
    for (auto& pg : program_groups) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(pg, &stack_sizes, pipeline_));
    }
  
    uint32_t direct_callable_stack_size_from_traversal;
    uint32_t direct_callable_stack_size_from_state;
    uint32_t continuation_stack_size;
    OPTIX_CHECK(optixUtilComputeStackSizes(
        &stack_sizes,
        1,  // maxTraceDepth
        0,  // maxCCDepth
        0,  // maxDCDepth
        &direct_callable_stack_size_from_traversal,
        &direct_callable_stack_size_from_state,
        &continuation_stack_size
    ));
  
    OPTIX_CHECK(optixPipelineSetStackSize(
        pipeline_,
        direct_callable_stack_size_from_traversal,
        direct_callable_stack_size_from_state,
        continuation_stack_size,
        1  // maxTraversableGraphDepth
    ));
}

void RaycastScene::build_gas(const float* vertices, int num_vertices,
                              const uint32_t* faces, int num_faces) {
    // Upload geometry to device
    size_t vertices_size = num_vertices * 3 * sizeof(float);
    size_t indices_size = num_faces * 3 * sizeof(uint32_t);
  
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices_), vertices_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_indices_), indices_size));
  
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices_), vertices, vertices_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_indices_), faces, indices_size, cudaMemcpyHostToDevice));
  
    // Build input
    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
  
    CUdeviceptr vertex_buffers[] = { d_vertices_ };
    build_input.triangleArray.vertexBuffers = vertex_buffers;
    build_input.triangleArray.numVertices = num_vertices;
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = 3 * sizeof(float);
  
    build_input.triangleArray.indexBuffer = d_indices_;
    build_input.triangleArray.numIndexTriplets = num_faces;
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = 3 * sizeof(uint32_t);
  
    uint32_t input_flags[] = { OPTIX_GEOMETRY_FLAG_NONE };
    build_input.triangleArray.flags = input_flags;
    build_input.triangleArray.numSbtRecords = 1;
  
    // Get memory requirements
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;
  
    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
        optix_context_,
        &accel_options,
        &build_input,
        1,
        &gas_buffer_sizes
    ));
  
    // Allocate temp and output buffers
    CUdeviceptr d_temp_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer), gas_buffer_sizes.tempSizeInBytes));
  
    CUdeviceptr d_output_buffer;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_output_buffer), gas_buffer_sizes.outputSizeInBytes));
  
    // Compaction size
    CUdeviceptr d_compacted_size;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_compacted_size), sizeof(size_t)));
  
    OptixAccelEmitDesc emit_desc = {};
    emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = d_compacted_size;
  
    // Build
    OPTIX_CHECK(optixAccelBuild(
        optix_context_,
        stream_,
        &accel_options,
        &build_input,
        1,
        d_temp_buffer,
        gas_buffer_sizes.tempSizeInBytes,
        d_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle_,
        &emit_desc,
        1
    ));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  
    // Compact
    size_t compacted_size;
    CUDA_CHECK(cudaMemcpy(&compacted_size, reinterpret_cast<void*>(d_compacted_size), sizeof(size_t), cudaMemcpyDeviceToHost));
  
    if (compacted_size < gas_buffer_sizes.outputSizeInBytes) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&gas_buffer_), compacted_size));
        OPTIX_CHECK(optixAccelCompact(optix_context_, stream_, gas_handle_, gas_buffer_, compacted_size, &gas_handle_));
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        cudaFree(reinterpret_cast<void*>(d_output_buffer));
    } else {
        gas_buffer_ = d_output_buffer;
    }
  
    cudaFree(reinterpret_cast<void*>(d_temp_buffer));
    cudaFree(reinterpret_cast<void*>(d_compacted_size));
}

void RaycastScene::create_sbt() {
    // Raygen records (3 programs)
    std::vector<RayGenRecord> raygen_records(3);
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_closest_, &raygen_records[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_all_hits_, &raygen_records[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_prog_occupancy_, &raygen_records[2]));
  
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt_raygen_record_), sizeof(RayGenRecord) * 3));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt_raygen_record_), raygen_records.data(),
                          sizeof(RayGenRecord) * 3, cudaMemcpyHostToDevice));
  
    // Miss records (3 programs)
    std::vector<MissRecord> miss_records(3);
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_closest_, &miss_records[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_all_hits_, &miss_records[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_prog_occupancy_, &miss_records[2]));
  
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt_miss_records_), sizeof(MissRecord) * 3));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt_miss_records_), miss_records.data(),
                          sizeof(MissRecord) * 3, cudaMemcpyHostToDevice));
  
    // Hit group records (3 ray types)
    std::vector<HitGroupRecord> hitgroup_records(3);
    HitGroupData hg_data;
    hg_data.vertices = reinterpret_cast<float3*>(d_vertices_);
    hg_data.indices = reinterpret_cast<uint3*>(d_indices_);
  
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_closest_, &hitgroup_records[0]));
    hitgroup_records[0].data = hg_data;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_all_hits_, &hitgroup_records[1]));
    hitgroup_records[1].data = hg_data;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_prog_occupancy_, &hitgroup_records[2]));
    hitgroup_records[2].data = hg_data;
  
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbt_hitgroup_records_), sizeof(HitGroupRecord) * 3));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbt_hitgroup_records_), hitgroup_records.data(),
                          sizeof(HitGroupRecord) * 3, cudaMemcpyHostToDevice));
  
    // Setup SBT for closest-hit mode
    sbt_closest_ = {};
    sbt_closest_.raygenRecord = sbt_raygen_record_;
    sbt_closest_.missRecordBase = sbt_miss_records_;
    sbt_closest_.missRecordStrideInBytes = sizeof(MissRecord);
    sbt_closest_.missRecordCount = 3;
    sbt_closest_.hitgroupRecordBase = sbt_hitgroup_records_;
    sbt_closest_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    sbt_closest_.hitgroupRecordCount = 3;
  
    // Setup SBT for all-hits mode (use second raygen record)
    sbt_all_hits_ = sbt_closest_;
    sbt_all_hits_.raygenRecord = sbt_raygen_record_ + sizeof(RayGenRecord);
  
    // Setup SBT for occupancy mode (use third raygen record)
    sbt_occupancy_ = sbt_closest_;
    sbt_occupancy_.raygenRecord = sbt_raygen_record_ + 2 * sizeof(RayGenRecord);
}

//------------------------------------------------------------------------------
// Ray casting methods
//------------------------------------------------------------------------------

std::vector<float> RaycastScene::cast_rays(
    const float* origins,
    const float* directions,
    int num_rays,
    const float* t_max)
{
    if (num_rays == 0) return {};
  
    CUDA_CHECK(cudaSetDevice(device_id_));
  
    // Allocate device memory
    CUdeviceptr d_origins, d_directions, d_t_max, d_t_hit;
    size_t rays_size = num_rays * 3 * sizeof(float);
  
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_origins), rays_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_directions), rays_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_t_hit), num_rays * sizeof(float)));
  
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_origins), origins, rays_size, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_directions), directions, rays_size, cudaMemcpyHostToDevice, stream_));
  
    if (t_max) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_t_max), num_rays * sizeof(float)));
        CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_t_max), t_max, num_rays * sizeof(float), cudaMemcpyHostToDevice, stream_));
    } else {
        d_t_max = 0;
    }
  
    // Setup launch params
    LaunchParams params = {};
    params.traversable = gas_handle_;
    params.ray_origins = reinterpret_cast<float3*>(d_origins);
    params.ray_directions = reinterpret_cast<float3*>(d_directions);
    params.ray_t_max = t_max ? reinterpret_cast<float*>(d_t_max) : nullptr;
    params.num_rays = num_rays;
    params.t_hit_closest = reinterpret_cast<float*>(d_t_hit);
    params.mode = MODE_CLOSEST;
  
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_launch_params_), &params, sizeof(params), cudaMemcpyHostToDevice, stream_));
  
    // Launch
    OPTIX_CHECK(optixLaunch(
        pipeline_,
        stream_,
        d_launch_params_,
        sizeof(LaunchParams),
        &sbt_closest_,
        num_rays,
        1,
        1
    ));
  
    // Copy results back
    std::vector<float> t_hit(num_rays);
    CUDA_CHECK(cudaMemcpyAsync(t_hit.data(), reinterpret_cast<void*>(d_t_hit), num_rays * sizeof(float), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  
    // Cleanup
    cudaFree(reinterpret_cast<void*>(d_origins));
    cudaFree(reinterpret_cast<void*>(d_directions));
    cudaFree(reinterpret_cast<void*>(d_t_hit));
    if (d_t_max) cudaFree(reinterpret_cast<void*>(d_t_max));
  
    return t_hit;
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<float>>
RaycastScene::list_intersections(
    const float* origins,
    const float* directions,
    const float* t_max,
    int num_rays,
    int max_hits_per_ray)
{
    if (num_rays == 0) {
        return {{}, {0}, {}};
    }
  
    CUDA_CHECK(cudaSetDevice(device_id_));
  
    // Allocate device memory
    CUdeviceptr d_origins, d_directions, d_t_max, d_hit_counts, d_hit_offsets, d_hit_t_buffer;
    size_t rays_size = num_rays * 3 * sizeof(float);
  
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_origins), rays_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_directions), rays_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_t_max), num_rays * sizeof(float)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_counts), num_rays * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_offsets), num_rays * sizeof(int)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hit_t_buffer), num_rays * max_hits_per_ray * sizeof(float)));
  
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_origins), origins, rays_size, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_directions), directions, rays_size, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_t_max), t_max, num_rays * sizeof(float), cudaMemcpyHostToDevice, stream_));
  
    // Initialize offsets (each ray gets max_hits_per_ray slots)
    std::vector<int> offsets(num_rays);
    for (int i = 0; i < num_rays; i++) {
        offsets[i] = i * max_hits_per_ray;
    }
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_hit_offsets), offsets.data(), num_rays * sizeof(int), cudaMemcpyHostToDevice, stream_));
  
    // Setup launch params
    LaunchParams params = {};
    params.traversable = gas_handle_;
    params.ray_origins = reinterpret_cast<float3*>(d_origins);
    params.ray_directions = reinterpret_cast<float3*>(d_directions);
    params.ray_t_max = reinterpret_cast<float*>(d_t_max);
    params.num_rays = num_rays;
    params.hit_counts = reinterpret_cast<int*>(d_hit_counts);
    params.hit_offsets = reinterpret_cast<int*>(d_hit_offsets);
    params.hit_t_buffer = reinterpret_cast<float*>(d_hit_t_buffer);
    params.max_hits_per_ray = max_hits_per_ray;
    params.mode = MODE_ALL_HITS;
  
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_launch_params_), &params, sizeof(params), cudaMemcpyHostToDevice, stream_));
  
    // Launch
    OPTIX_CHECK(optixLaunch(
        pipeline_,
        stream_,
        d_launch_params_,
        sizeof(LaunchParams),
        &sbt_all_hits_,
        num_rays,
        1,
        1
    ));
  
    // Copy results back
    std::vector<int> hit_counts(num_rays);
    CUDA_CHECK(cudaMemcpyAsync(hit_counts.data(), reinterpret_cast<void*>(d_hit_counts), num_rays * sizeof(int), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  
    // Compute ray_splits and total hits
    std::vector<int> ray_splits(num_rays + 1);
    ray_splits[0] = 0;
    for (int i = 0; i < num_rays; i++) {
        ray_splits[i + 1] = ray_splits[i] + hit_counts[i];
    }
    int total_hits = ray_splits[num_rays];
  
    // Copy hit t values (only the actual hits, not the full buffer)
    std::vector<float> all_t_values(total_hits);
    std::vector<float> full_buffer(num_rays * max_hits_per_ray);
    CUDA_CHECK(cudaMemcpy(full_buffer.data(), reinterpret_cast<void*>(d_hit_t_buffer), 
                          num_rays * max_hits_per_ray * sizeof(float), cudaMemcpyDeviceToHost));
  
    // Compact the buffer
    for (int i = 0; i < num_rays; i++) {
        int src_offset = i * max_hits_per_ray;
        int dst_offset = ray_splits[i];
        for (int j = 0; j < hit_counts[i]; j++) {
            all_t_values[dst_offset + j] = full_buffer[src_offset + j];
        }
    }
  
    // Cleanup
    cudaFree(reinterpret_cast<void*>(d_origins));
    cudaFree(reinterpret_cast<void*>(d_directions));
    cudaFree(reinterpret_cast<void*>(d_t_max));
    cudaFree(reinterpret_cast<void*>(d_hit_counts));
    cudaFree(reinterpret_cast<void*>(d_hit_offsets));
    cudaFree(reinterpret_cast<void*>(d_hit_t_buffer));
  
    return {hit_counts, ray_splits, all_t_values};
}

std::vector<int8_t> RaycastScene::compute_occupancy(const float* points, int num_points) {
    if (num_points == 0) return {};
  
    CUDA_CHECK(cudaSetDevice(device_id_));
  
    // Allocate device memory
    CUdeviceptr d_points, d_occupancy;
    size_t points_size = num_points * 3 * sizeof(float);
  
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_points), points_size));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_occupancy), num_points * sizeof(int8_t)));
  
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_points), points, points_size, cudaMemcpyHostToDevice, stream_));
  
    // Setup launch params
    LaunchParams params = {};
    params.traversable = gas_handle_;
    params.ray_origins = reinterpret_cast<float3*>(d_points);
    params.num_rays = num_points;
    params.occupancy = reinterpret_cast<int8_t*>(d_occupancy);
    params.mode = MODE_OCCUPANCY;
  
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_launch_params_), &params, sizeof(params), cudaMemcpyHostToDevice, stream_));
  
    // Launch
    OPTIX_CHECK(optixLaunch(
        pipeline_,
        stream_,
        d_launch_params_,
        sizeof(LaunchParams),
        &sbt_occupancy_,
        num_points,
        1,
        1
    ));
  
    // Copy results back
    std::vector<int8_t> occupancy(num_points);
    CUDA_CHECK(cudaMemcpyAsync(occupancy.data(), reinterpret_cast<void*>(d_occupancy), num_points * sizeof(int8_t), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  
    // Cleanup
    cudaFree(reinterpret_cast<void*>(d_points));
    cudaFree(reinterpret_cast<void*>(d_occupancy));
  
    return occupancy;
}

} // namespace optix_raycaster
```

---

## 6. `src/python_bindings.cpp`

```cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "optix_raycaster.h"

namespace py = pybind11;
using namespace optix_raycaster;

PYBIND11_MODULE(optix_raycaster_cpp, m) {
    m.doc() = "OptiX-based ray casting for mesh intersection queries";
  
    py::class_<RaycastScene>(m, "RaycastScene")
        .def(py::init([](py::array_t<float> vertices, py::array_t<uint32_t> faces, int device_id) {
            py::buffer_info v_info = vertices.request();
            py::buffer_info f_info = faces.request();
          
            if (v_info.ndim != 2 || v_info.shape[1] != 3) {
                throw std::runtime_error("vertices must be (N, 3) array");
            }
            if (f_info.ndim != 2 || f_info.shape[1] != 3) {
                throw std::runtime_error("faces must be (M, 3) array");
            }
          
            // Ensure contiguous
            auto v_cont = vertices.attr("astype")("float32").attr("copy")();
            auto f_cont = faces.attr("astype")("uint32").attr("copy")();
          
            py::buffer_info v_info2 = py::cast<py::array_t<float>>(v_cont).request();
            py::buffer_info f_info2 = py::cast<py::array_t<uint32_t>>(f_cont).request();
          
            return new RaycastScene(
                static_cast<float*>(v_info2.ptr),
                static_cast<int>(v_info2.shape[0]),
                static_cast<uint32_t*>(f_info2.ptr),
                static_cast<int>(f_info2.shape[0]),
                device_id
            );
        }), py::arg("vertices"), py::arg("faces"), py::arg("device_id") = 0)
      
        .def("cast_rays", [](RaycastScene& self,
                             py::array_t<float> origins,
                             py::array_t<float> directions,
                             std::optional<py::array_t<float>> t_max) {
            py::buffer_info o_info = origins.request();
            py::buffer_info d_info = directions.request();
          
            if (o_info.ndim != 2 || o_info.shape[1] != 3) {
                throw std::runtime_error("origins must be (N, 3) array");
            }
            if (d_info.ndim != 2 || d_info.shape[1] != 3) {
                throw std::runtime_error("directions must be (N, 3) array");
            }
            if (o_info.shape[0] != d_info.shape[0]) {
                throw std::runtime_error("origins and directions must have same length");
            }
          
            int num_rays = static_cast<int>(o_info.shape[0]);
          
            // Ensure contiguous float32
            auto o_cont = py::array_t<float>::ensure(origins.attr("astype")("float32"));
            auto d_cont = py::array_t<float>::ensure(directions.attr("astype")("float32"));
          
            const float* t_max_ptr = nullptr;
            py::array_t<float> t_max_cont;
            if (t_max.has_value()) {
                t_max_cont = py::array_t<float>::ensure(t_max.value().attr("astype")("float32"));
                t_max_ptr = static_cast<const float*>(t_max_cont.request().ptr);
            }
          
            auto result = self.cast_rays(
                static_cast<const float*>(o_cont.request().ptr),
                static_cast<const float*>(d_cont.request().ptr),
                num_rays,
                t_max_ptr
            );
          
            return py::array_t<float>(result.size(), result.data());
        }, py::arg("origins"), py::arg("directions"), py::arg("t_max") = py::none(),
           "Cast rays and return closest hit t values (inf if no hit)")
      
        .def("list_intersections", [](RaycastScene& self,
                                       py::array_t<float> origins,
                                       py::array_t<float> directions,
                                       py::array_t<float> t_max,
                                       int max_hits_per_ray) {
            py::buffer_info o_info = origins.request();
            py::buffer_info d_info = directions.request();
            py::buffer_info t_info = t_max.request();
          
            if (o_info.ndim != 2 || o_info.shape[1] != 3) {
                throw std::runtime_error("origins must be (N, 3) array");
            }
            if (d_info.ndim != 2 || d_info.shape[1] != 3) {
                throw std::runtime_error("directions must be (N, 3) array");
            }
          
            int num_rays = static_cast<int>(o_info.shape[0]);
          
            auto o_cont = py::array_t<float>::ensure(origins.attr("astype")("float32"));
            auto d_cont = py::array_t<float>::ensure(directions.attr("astype")("float32"));
            auto t_cont = py::array_t<float>::ensure(t_max.attr("astype")("float32"));
          
            auto [hit_counts, ray_splits, all_t] = self.list_intersections(
                static_cast<const float*>(o_cont.request().ptr),
                static_cast<const float*>(d_cont.request().ptr),
                static_cast<const float*>(t_cont.request().ptr),
                num_rays,
                max_hits_per_ray
            );
          
            return py::make_tuple(
                py::array_t<int>(hit_counts.size(), hit_counts.data()),
                py::array_t<int>(ray_splits.size(), ray_splits.data()),
                py::array_t<float>(all_t.size(), all_t.data())
            );
        }, py::arg("origins"), py::arg("directions"), py::arg("t_max"), py::arg("max_hits_per_ray") = 64,
           "List all intersections for each ray. Returns (hit_counts, ray_splits, t_values)")
      
        .def("compute_occupancy", [](RaycastScene& self, py::array_t<float> points) {
            py::buffer_info p_info = points.request();
          
            if (p_info.ndim != 2 || p_info.shape[1] != 3) {
                throw std::runtime_error("points must be (N, 3) array");
            }
          
            int num_points = static_cast<int>(p_info.shape[0]);
          
            auto p_cont = py::array_t<float>::ensure(points.attr("astype")("float32"));
          
            auto result = self.compute_occupancy(
                static_cast<const float*>(p_cont.request().ptr),
                num_points
            );
          
            return py::array_t<int8_t>(result.size(), result.data());
        }, py::arg("points"),
           "Compute occupancy (inside/outside) for query points. Returns 1 if inside, 0 if outside")
      
        .def_property_readonly("device_id", &RaycastScene::get_device_id);
}
```

---

## 7. `python/raycaster.py`

This is a drop-in replacement Python wrapper matching your original API:

```python
"""
OptiX-based raycaster - drop-in replacement for Open3D raycasting.
"""

import numpy as np
from tqdm import tqdm

# Import the compiled C++ module
try:
    import optix_raycaster_cpp
except ImportError as e:
    raise ImportError(
        "Could not import optix_raycaster_cpp. "
        "Please build the C++ extension first using: "
        "mkdir build && cd build && cmake .. && make"
    ) from e


def disk_samples_fibonacci(K: int, radius: float) -> np.ndarray:
    """Generate K Fibonacci-distributed samples on a disk of given radius."""
    ga = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(K, dtype=np.float32)
    r = radius * np.sqrt((i + 0.5) / K)
    theta = i * ga
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y], axis=1).astype(np.float32)


def pair_indices(idx_list):
    """Generate all pairs of indices from a list."""
    idx = np.asarray(idx_list)
    i, j = np.triu_indices(len(idx), k=1)
    pair_a = idx[i]
    pair_b = idx[j]
    return pair_a, pair_b


class RaycastingScene:
    """
    OptiX-accelerated raycasting scene.
  
    Drop-in replacement for Open3D's RaycastingScene with matching API.
    """
  
    def __init__(self, vertices: np.ndarray, faces: np.ndarray, device_id: int = 0):
        """
        Build a raycasting scene from mesh data.
      
        Args:
            vertices: (N, 3) float32 array of vertex positions
            faces: (M, 3) uint32 array of triangle indices
            device_id: CUDA device ID (default: 0)
        """
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.uint32)
        self._scene = optix_raycaster_cpp.RaycastScene(vertices, faces, device_id)
        self._device_id = device_id
  
    def cast_rays(self, rays: np.ndarray) -> dict:
        """
        Cast rays and return first hit information.
      
        Args:
            rays: (N, 6) array where each row is [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
      
        Returns:
            dict with 't_hit' key containing (N,) array of hit distances (inf if no hit)
        """
        rays = np.ascontiguousarray(rays, dtype=np.float32)
        origins = rays[:, :3]
        directions = rays[:, 3:]
      
        t_hit = self._scene.cast_rays(origins, directions)
      
        return {'t_hit': t_hit}
  
    def list_intersections(self, rays: np.ndarray) -> dict:
        """
        List all intersections for each ray.
      
        Args:
            rays: (N, 6) array where each row is [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
      
        Returns:
            dict with:
                'ray_splits': (N+1,) array of offsets into t_hit
                't_hit': flat array of all intersection t values
        """
        rays = np.ascontiguousarray(rays, dtype=np.float32)
        origins = rays[:, :3]
        directions = rays[:, 3:]
      
        # Use large t_max
        t_max = np.full(len(origins), 1e20, dtype=np.float32)
      
        hit_counts, ray_splits, t_hit = self._scene.list_intersections(
            origins, directions, t_max, max_hits_per_ray=64
        )
      
        return {
            'ray_splits': ray_splits.astype(np.int64),
            't_hit': t_hit
        }
  
    def compute_occupancy(self, points: np.ndarray, nsamples: int = 1) -> np.ndarray:
        """
        Compute occupancy (inside/outside) for query points.
      
        Args:
            points: (N, 3) array of query points
            nsamples: ignored (for API compatibility with Open3D)
      
        Returns:
            (N,) int8 array, 1 if inside mesh, 0 if outside
        """
        points = np.ascontiguousarray(points, dtype=np.float32)
        return self._scene.compute_occupancy(points)


def build_scene(vertices: np.ndarray, faces: np.ndarray, 
                nthreads: int = 0, device: str = "CUDA:0") -> RaycastingScene:
    """
    Build a raycasting scene from mesh vertices and faces.
  
    Args:
        vertices: (N, 3) vertex positions
        faces: (M, 3) triangle indices
        nthreads: ignored (for API compatibility)
        device: device string, e.g. "CUDA:0" or "CPU:0" (CPU falls back to CUDA:0)
  
    Returns:
        RaycastingScene object
    """
    # Parse device string
    if device.upper().startswith("CUDA:"):
        device_id = int(device.split(":")[1])
    else:
        device_id = 0  # Default to GPU 0
  
    return RaycastingScene(vertices, faces, device_id)


def segment_occlusion_and_thickness(
    scene: RaycastingScene,
    p0: np.ndarray,
    p1: np.ndarray,
    endpoint_eps: float = 1e-4,
    t_eps: float = 1e-6,
    dev: str = "CUDA:0",
    *,
    compute_occlusion: bool = True,
    occupancy_nsamples: int = 3,
):
    """
    Compute segment occlusion and thickness through mesh.
  
    For each segment from p0[i] to p1[i]:
    - occluded: True if any surface is hit before segment end
    - L_inside: total length of segment inside the mesh
    - L_total: segment length after endpoint trimming
  
    Args:
        scene: RaycastingScene object
        p0: (M, 3) segment start points
        p1: (M, 3) segment end points
        endpoint_eps: distance to shrink endpoints to avoid self-intersection
        t_eps: tolerance for hit detection
        dev: device string (for API compatibility)
        compute_occlusion: whether to compute occlusion
        occupancy_nsamples: ignored (for API compatibility)
  
    Returns:
        occluded: (M,) bool array
        L_inside: (M,) float32 array of inside lengths
        L_total: (M,) float32 array of total lengths
    """
    endpoint_eps = np.float32(endpoint_eps)
    t_eps = np.float32(t_eps)
  
    p0 = np.ascontiguousarray(p0, dtype=np.float32)
    p1 = np.ascontiguousarray(p1, dtype=np.float32)
  
    d = p1 - p0
    L = np.linalg.norm(d, axis=1).astype(np.float32)
    M = p0.shape[0]
  
    occluded = np.zeros(M, dtype=bool)
    L_inside = np.zeros(M, dtype=np.float32)
    L_total = np.zeros(M, dtype=np.float32)
  
    good = L > np.float32(1e-12)
    if not np.any(good):
        return occluded, L_inside, L_total
  
    # Unit direction
    u = np.zeros_like(d, dtype=np.float32)
    u[good] = d[good] / L[good, None]
  
    # Trim endpoints to avoid self-hit
    shrink = np.minimum(endpoint_eps, np.float32(0.49) * L).astype(np.float32)
    o = p0 + u * shrink[:, None]
    e = p1 - u * shrink[:, None]
  
    seg = e - o
    L2 = np.linalg.norm(seg, axis=1).astype(np.float32)
    L_total[good] = L2[good]
  
    good2 = L2 > np.float32(1e-12)
    if not np.any(good2):
        return occluded, L_inside, L_total
  
    idx = np.flatnonzero(good2)
    o2 = o[idx]
    seg2 = seg[idx]
    L2_2 = L2[idx]
  
    u2 = seg2 / L2_2[:, None]
    R = idx.size
  
    # Build ray array for OptiX
    rays = np.empty((R, 6), dtype=np.float32)
    rays[:, :3] = o2
    rays[:, 3:] = u2
  
    # 1) Occlusion via first hit
    if compute_occlusion:
        result = scene.cast_rays(rays)
        t_first = result['t_hit'].astype(np.float32)
        occluded2 = np.isfinite(t_first) & (t_first < (L2_2 - t_eps))
        occluded[idx] = occluded2
  
    # 2) Thickness via all intersections + inside/outside toggling
    # Get occupancy at start points
    inside0 = scene.compute_occupancy(rays[:, :3]).astype(np.int8)
  
    # Get all intersections
    inter = scene.list_intersections(rays)
    ray_splits = inter['ray_splits']
    t_all = inter['t_hit'].astype(np.float32)
  
    counts = np.diff(ray_splits)
    N = t_all.size
  
    if N == 0:
        L_inside[idx] = L2_2 * inside0.astype(np.float32)
        return occluded, L_inside, L_total
  
    # Map each hit to its ray id
    ray_id = np.repeat(np.arange(R, dtype=np.int64), counts)
  
    # Filter hits to (0, L2)
    in_seg = (t_all > 0.0) & (t_all < L2_2[ray_id])
    t = t_all[in_seg]
    rid = ray_id[in_seg]
  
    if t.size == 0:
        counts2 = np.zeros(R, dtype=np.int64)
        alt = np.zeros(R, dtype=np.float32)
    else:
        # Sort by (ray_id, t)
        order = np.lexsort((t, rid))
        t = t[order]
        rid = rid[order]
      
        counts2 = np.bincount(rid, minlength=R).astype(np.int64)
      
        splits2 = np.empty(R + 1, dtype=np.int64)
        splits2[0] = 0
        np.cumsum(counts2, out=splits2[1:])
      
        idx_in = np.arange(t.size, dtype=np.int64) - splits2[rid]
      
        # Alternating-sum sign pattern
        s = inside0[rid].astype(np.int64)
        parity = (idx_in + (1 - s)) & 1
        sign = (1 - 2 * parity).astype(np.float32)
      
        weighted = t.astype(np.float32) * sign
        alt = np.bincount(rid, weights=weighted, minlength=R).astype(np.float32)
  
    # Tail term
    tail_inside = (inside0.astype(np.int64) ^ (counts2 & 1)).astype(np.float32)
    L_inside2 = alt + L2_2 * tail_inside
  
    L_inside[idx] = L_inside2.astype(np.float32)
    return occluded, L_inside, L_total


def bundle_scores(
    scene: RaycastingScene,
    p0: np.ndarray,
    p1: np.ndarray,
    xy: np.ndarray,
    endpoint_eps: float = 1e-4,
    t_eps: float = 1e-6,
    dev: str = "CUDA:0"
):
    """
    Compute bundle occlusion scores for segment pairs with disk offsets.
  
    Args:
        scene: RaycastingScene object
        p0: (M, 3) segment start points
        p1: (M, 3) segment end points
        xy: (K, 2) disk sample offsets
        endpoint_eps: endpoint shrinkage
        t_eps: hit tolerance
        dev: device string (for API compatibility)
  
    Returns:
        blocked_fraction: (M,) fraction of disk samples blocked
        mean_L_inside: (M,) mean inside length for blocked rays
        min_L_inside: (M,) minimum inside length
    """
    p0 = np.asarray(p0, np.float32)
    p1 = np.asarray(p1, np.float32)
    xy = np.asarray(xy, np.float32)
    assert xy.ndim == 2 and xy.shape[1] == 2, "xy must be (K,2)"
  
    K = xy.shape[0]
  
    d = p1 - p0
    L = np.linalg.norm(d, axis=1)
    M = p0.shape[0]
    good = L > 1e-12
  
    u = np.zeros_like(d)
    u[good] = d[good] / L[good, None]
  
    # Per-segment perpendicular basis (v, w)
    a = np.tile(np.array([1, 0, 0], np.float32), (M, 1))
    mask = np.abs(u[:, 0]) > 0.9
    a[mask] = np.array([0, 1, 0], np.float32)
  
    v = np.cross(u, a)
    v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    w = np.cross(u, v)
  
    ox = xy[:, 0]
    oy = xy[:, 1]
  
    # (M, K, 3) offsets in 3D
    off = v[:, None, :] * ox[None, :, None] + w[:, None, :] * oy[None, :, None]
  
    p0k = (p0[:, None, :] + off).reshape(-1, 3)
    p1k = (p1[:, None, :] + off).reshape(-1, 3)
  
    _, L_inside_k, _ = segment_occlusion_and_thickness(
        scene, p0k, p1k, endpoint_eps=endpoint_eps, t_eps=t_eps, dev=dev
    )
    L_inside_k = L_inside_k.reshape(M, K)
  
    blocked = L_inside_k > 0.0
    blocked_fraction = blocked.mean(axis=1)
  
    blocked_count = np.maximum(blocked.sum(axis=1), 1)
    mean_L_inside = (L_inside_k * blocked).sum(axis=1) / blocked_count
  
    min_L_inside = L_inside_k.min(axis=1)
  
    blocked_fraction[~good] = 0.0
    mean_L_inside[~good] = 0.0
    min_L_inside[~good] = 0.0
  
    return (
        blocked_fraction.astype(np.float32),
        mean_L_inside.astype(np.float32),
        min_L_inside.astype(np.float32)
    )


def process_frames(
    vertices: np.ndarray,
    faces: np.ndarray,
    idx_list,
    disk_samples: int = 100,
    disk_radius: float = 0.1,
    device: str = "CUDA:0"
) -> np.ndarray:
    """
    Process multiple frames computing bundle scores.
  
    Args:
        vertices: (N, V, 3) array of vertex positions per frame
        faces: (F, 3) triangle indices (shared across frames)
        idx_list: list of vertex indices to compute pairs for
        disk_samples: number of Fibonacci disk samples
        disk_radius: radius of sampling disk
        device: device string, e.g. "CUDA:0"
  
    Returns:
        (N, num_pairs, 3) array of [blocked_fraction, mean_L_inside, min_L_inside]
    """
    disk = disk_samples_fibonacci(disk_samples, disk_radius)
    pair_a, pair_b = pair_indices(idx_list)
    n = len(vertices)
    num_pairs = len(pair_a)
    result = np.zeros((n, num_pairs, 3), dtype=np.float32)
  
    # Parse device
    if device.upper().startswith("CUDA:"):
        device_id = int(device.split(":")[1])
    else:
        device_id = 0
  
    for i, vts in tqdm(enumerate(vertices), total=n, desc="Processing frames"):
        scene = build_scene(vts, faces, device=device)
        scores = bundle_scores(scene, vts[pair_a], vts[pair_b], disk, dev=device)
        result[i] = np.array(scores).T
  
    return result
```

---

## 8. `setup.py`

```python
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def run(self):
        try:
            subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed")
      
        for ext in self.extensions:
            self.build_extension(ext)
  
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
      
        cmake_args = [
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}',
            f'-DPYTHON_EXECUTABLE={sys.executable}',
            f'-DCMAKE_BUILD_TYPE=Release',
        ]
      
        # Check for OptiX path
        optix_dir = os.environ.get('OptiX_INSTALL_DIR')
        if optix_dir:
            cmake_args.append(f'-DOptiX_INSTALL_DIR={optix_dir}')
      
        build_args = ['--config', 'Release', '-j']
      
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
      
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=build_temp)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_temp)

setup(
    name='optix_raycaster',
    version='1.0.0',
    author='Your Name',
    description='OptiX-accelerated ray casting for mesh intersection',
    long_description='',
    ext_modules=[CMakeExtension('optix_raycaster_cpp')],
    cmdclass=dict(build_ext=CMakeBuild),
    packages=[''],
    package_dir={'': 'python'},
    zip_safe=False,
    python_requires='>=3.8',
    install_requires=['numpy', 'tqdm'],
)
```

---

## 9. Build Instructions

### Prerequisites

1. **CUDA Toolkit** (11.0+)
2. **OptiX SDK 7.0+** - Download from [NVIDIA Developer](https://developer.nvidia.com/optix)
3. **pybind11** - `pip install pybind11` or install via conda
4. **CMake 3.18+**

### Build Steps

```bash
# Set OptiX path
export OptiX_INSTALL_DIR=/path/to/optix

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
make -j$(nproc)

# Install (optional)
pip install -e ..
```

Or use pip directly:

```bash
export OptiX_INSTALL_DIR=/path/to/optix
pip install -e .
```

---

## 10. Usage Example

```python
import numpy as np
from python.raycaster import process_frames, build_scene, bundle_scores, disk_samples_fibonacci

# Load your mesh data
vertices = np.random.randn(100, 1000, 3).astype(np.float32)  # 100 frames, 1000 vertices
faces = np.random.randint(0, 1000, (2000, 3)).astype(np.uint32)  # 2000 triangles
idx_list = [0, 1, 2, 3, 4]  # vertices to compute pairs for

# Process all frames
result = process_frames(
    vertices, 
    faces, 
    idx_list,
    disk_samples=100,
    disk_radius=0.1,
    device="CUDA:0"
)

print(f"Result shape: {result.shape}")  # (100, 10, 3) for 10 pairs
```

---

## Key Differences from Open3D Version

| Feature | Open3D | OptiX |
|---------|--------|-------|
| Backend | CPU or GPU (limited) | NVIDIA RTX/CUDA |
| BVH Build | Per-frame | Per-frame (could be optimized for deforming meshes) |
| Ray Batching | Built-in | Custom implementation |
| Occupancy | `compute_occupancy()` | Ray casting + odd/even rule |
| All Intersections | `list_intersections()` | Iterative closest-hit queries |

The OptiX version should be significantly faster on modern NVIDIA GPUs due to hardware-accelerated ray tracing.
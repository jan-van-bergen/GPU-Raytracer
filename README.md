# Advanced-Graphics-Pathtracer

## Features

### GPU implementation
The Pathtracer was implemented on the GPU using Cuda. A wavefront approach is used where separate kernels are invoked to the perform separate stages of Pathtracing.
The implementation supports basic triangle intersection, BVH traversal using a high-quality SBVH, three types of materials/BRDFS (diffuse, dielectrics, and glossy (Beckmann))

Rays are generated with coherence in mind

### Importance Sampling
Various forms of importance sampling were implemented.
The BRDF for Diffuse materials is importance sampled using a cosine weighted distribution. 
The BRDF for Glossy materials is importance sampled using the technique described in Walter et al. 2007.
Next Event estimtation is used by Diffuse and Glossy materials (for Dielectris it doesn't really make sense to do this). 
Multiple Importance Sampling is used by Diffuse and Glossy materials.

### Microfacet Materials
Glossy materials are implemented using the Beckmann microfacet model.

### BVH Serialization
Less serious feature, but because SBVH construction took quite a while for larger Scenes, BVH's are now constructed once and then stored to disk for later reuse.

## Papers
- Megakernels Considered Harmful: Wavefront Path Tracing on GPUs - Laine et al.
- Microfacet Models for Refraction through Rough Surfaces - Walter et al.

## Attribution
- http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
- https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/

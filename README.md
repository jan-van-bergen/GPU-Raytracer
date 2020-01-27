# Advanced-Graphics-Pathtracer

## Features

### GPU implementation
The Pathtracer was implemented on the GPU using Cuda. 
I have implemented it using two different techniques, Megakernel vs Wavefront, for comparison purposes.
Both implementations supports basic triangle intersection, texture mapping, triangle lights, scene traversal using an MBVH, and 3 types of materials/BRDFS (diffuse, dielectrics, and glossy (microfacet using Beckmann distribution)).
For the Wavefront Pathtracer the different types of materials are implemented in different kernels.

Rays are generated with coherence in mind when using the Wavefront approach. Instead of simply assigning each consecutive thread a consecutive pixel index in the frame buffer, every 32 threads (size of a warp) gets assigned an 8x4 block of pixels. This increases coherence for primary Rays, which slightly improves frame times.

### Importance Sampling
Various forms of importance sampling were implemented.
The BRDF for Diffuse materials is importance sampled using a cosine weighted distribution. 
The BRDF for Glossy materials is importance sampled using the technique described in Walter et al. 2007.
Next Event estimtation is used by Diffuse and Glossy materials (for Dielectris it doesn't really make sense to do this). 
Multiple Importance Sampling is used by Diffuse and Glossy materials.

### Microfacet Materials
Glossy materials are implemented using the Beckmann microfacet model.
Glossy materials also use NEE and MIS.
When tracing non-shadow rays (i.e. looking for indirect light) the BRDF is importance sampled using the formulas described in Walter et al. 2007.

### MBVH
A 4-way MBVH is constructed by collapsing the SBVH. The collapsing procedure was implemented as described in the slides. The MBVH results in a huge speedup on larger scenes. Sponza went from ~180 ms to ~100 ms per frame! But smaller scenes also benefit.

### BVH Serialization
A less serious feature, but because SBVH construction took quite a while for larger Scenes, BVH's are now constructed once and then stored to disk for later reuse.

## Papers
- Megakernels Considered Harmful: Wavefront Path Tracing on GPUs - Laine et al.
- Microfacet Models for Refraction through Rough Surfaces - Walter et al.

## Attribution
- http://www.reedbeta.com/blog/quick-and-easy-gpu-random-numbers-in-d3d11/
- https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/

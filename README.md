# CUDA Pathtracer

- CUDA Pathtracer that uses Wavefront rendering. 
- Supports Diffuse, Dielectric, and Glossy (Microfacets) materials.
- Multiple BVH types: SBVH, QBVH, and Compressed Wide BVH.

![Sponza](Screenshots/Sponza.png "Sponza")
![Glass](Screenshots/Glass.png "Dielectrics")

## Features

### GPU Implementation

The Pathtracer was implemented on the GPU using Cuda. The renderer uses a Wavefront approach, as described in [a paper by Laine et al. 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
This means different stages of a path are implemented in different kernels.

Rays are generated with coherence in mind. Instead of simply assigning each consecutive thread a consecutive pixel index in the frame buffer, every 32 threads (size of a warp) gets assigned an 8x4 block of pixels. This increases coherence for primary Rays, which slightly improves frame times.

### Importance Sampling

Various forms of importance sampling are implemented.
The BRDF for Diffuse materials is importance sampled using a cosine weighted distribution. 
The BRDF for Glossy materials is importance sampled using the technique described in Walter et al. 2007.
Next Event estimtation is used by Diffuse and Glossy materials. 
Multiple Importance Sampling is used by Diffuse and Glossy materials.

### Microfacet Materials

![Microfacet Model](Screenshots/Microfacets.png "Glossy materials using the Beckmann microfacet model")

Glossy materials are implemented using the Beckmann microfacet model.
Glossy materials also use NEE and MIS.
When tracing non-shadow rays (i.e. looking for indirect light) the BRDF is importance sampled using the formulas described in [Walter et al. 2007](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf).

### QBVH

Instead of a binary BVH, the Pathtracer uses a quaternary BVH (QBVH). This QBVH is constructed by iteratively collapsing the Nodes of a binary SBVH. The collapsing procedure was implemented as described in [Wald et al. 2008](https://graphics.stanford.edu/~boulos/papers/multi_rt08.pdf).

### Blue Noise Sampling

The Pathtracer uses a [low discrepancy sampler by Heitz et al.](https://eheitzresearch.wordpress.com/762-2/) that distributes Monte Carlo errors as a blue noise in screen space. This even works with rasterized primary rays.

### SVGF

A spatio-temporal filter is implemented, as described in the paper by Schied et al.

## Dependencies

The project uses SDL and GLEW. Their dll's for both x86 and x64 targets are included in the repositories, as well as all required headers.

The project uses CUDA 10.2 and requires that the ```CUDA_PATH``` system variable is set to the path where CUDA 10.2 is installed.

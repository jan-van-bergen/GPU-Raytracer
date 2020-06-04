# CUDA Pathtracer

![Sponza](Screenshots/Sponza.png "Sponza")

Efficient CUDA pathtracer using Wavefront rendering. 

## Features

- Wavefront rendering, [Laine et al. 2013](https://research.nvidia.com/sites/default/files/pubs/2013-07_Megakernels-Considered-Harmful/laine2013hpg_paper.pdf)
- Multiple BVH types
  - Standard SAH-based BVH
  - SBVH (Spatial BVH), see [Stich et al. 2009](https://www.nvidia.in/docs/IO/77714/sbvh.pdf)
  - QBVH (Quaternary BVH). The QBVH is constructed by iteratively collapsing the Nodes of the SBVH. The collapsing procedure was implemented as described in [Wald et al. 2008](https://graphics.stanford.edu/~boulos/papers/multi_rt08.pdf).
  - CWBVH (Compressed Wide BVH), see [Ylitie et al. 2017](https://research.nvidia.com/sites/default/files/publications/ylitie2017hpg-paper.pdf). The CWBVH outperforms both the QBVH and SBVH.
  - All BVH types use Dynamic Ray Fetching, see [Aila et al. 2009](https://www.nvidia.com/docs/IO/76976/HPG2009-Trace-Efficiency.pdf)
- SVGF (Spatio-Temporal Variance Guided Filter), see [Schied et al](https://cg.ivd.kit.edu/publications/2017/svgf/svgf_preprint.pdf). Denoiser 
- Importance Sampling
  - Next Event Estimation (NEE)
  - Multiple Importance Sampling (MIS)
  - Cosine weighted direction sampling for diffuse bounces.
  - Microfacet sampling as described in [Walter et al. 2007](https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf)
- Blue Noise Sampling: The low discrepency sampler by [Heitz et al.](https://eheitzresearch.wordpress.com/762-2/) is used. This sampler distributes Monte Carlo errors as a blue noise in screen space.
- Multiple Material types
  - Diffuse
  - Dielectrics
  - Microfacets (Beckmann and GGX)

## Screenshots

![SVGF Denoising](Screenshots/SVGF.png "SVGF Denoising")

![Microfacet Model](Screenshots/Microfacets.png "Glossy materials using the Beckmann microfacet model")

## Dependencies

The project uses SDL and GLEW. Their dll's for both x86 and x64 targets are included in the repositories, as well as all required headers.

The project uses CUDA 10.2 and requires that the ```CUDA_PATH``` system variable is set to the path where CUDA 10.2 is installed.

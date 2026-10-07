# FlowNet2 attribution

Nine NVIDIA FlowNet2 network/notice files are included unchanged from
`NVIDIA/flownet2-pytorch` commit `bce69c881ad2b11bef1e6ce9fd0dfd45bd8a1756`.
`flownet2/provenance.json` records original Git blob SHA and SHA256 for each file.
The original license and embedded copyright notices remain in place.

`src/native_flow_ops.py` is an independent PyTorch port of the forward formulas in
NVIDIA's channelnorm, resample2d and correlation CUDA implementations. It restricts
support to FlowNet2's operator configurations, uses chunked correlation and native
autograd/manual correlation backward, and is not a bitwise reproduction of the
legacy CUDA kernels. No upstream network source is patched; operator modules are
registered under the package's expected imports at runtime.

Original work: E.Ilg et al., FlowNet2.0: Evolution of Optical Flow Estimation with Deep
Networks, CVPR2017. PyTorch implementation: Fitsum Reda, Robert Pottorff, Jon Barker
and Bryan Catanzaro, NVIDIA. Source references and checkpoint URL are in README.md.

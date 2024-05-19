Fork of [Fast Total Variation Proximal](https://github.com/svaiter/ftvp)
========================================================================

Reasons to fork 
---------------

It was not working for my application out of the box, so I am trying to fix memory management and possibly even the computation of proximal operator for nonsquare arrays.

**Changelog**

Relative to 4b993f commit:

* FIX dealing with `dev_xioswp` variable. It does not have to be allocated by cudaMalloc since it is just a pointer to swap buffers.
* FIX cpu memory management of `init_memory` function. CPU arrays were prealocated, copied to GPU buffers and not freed. This was slow and might cause crash of the program as it would eventually lead to the memory overflow.
* IMPROVEMENT instead of memset and copying to GPU buffer now the cudaMemset is used to zero CUDA buffers
* IMPROVEMENT out of source compilation in the main github directory
    * `mkdir build`
    * `cmake ..`
    * `make`
    * `make install`
    * by default it adds libraries to ${HOME}/lib
    * `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${HOME}/lib`
* FIX There was a alignment problem between C arrays and numpy arrays as numpy arrays are indexed u[ycorrd, xcoord] but the sizes of axes sx and sy were swapped

TODO
* in ftvp.cuh there is minimization formula `u = argmin_x lambda/2 ||x - u||_2^2 + TV(x)` but it seems it actualy is `u = argmin_x ||x - u||_2^2 + lambda/2 TV(x)` as larger lambda causes stronger regularization
* that would be consistent with standard definition of proximal operator but need to look into the source code
* FIX numpy and Python bindings for 3D images 

`ftvp` is a CUDA library dedicated to the computation of the proximal operator
of the isotropic Total Variation in 2D and 3D on Nvidia GPU. This repository
includes examples of use in C, along with bindings for Python. It is licensed
under the New BSD licence, see LICENSE file.

* Issues: [GitHub Issues](https://github.com/svaiter/ftvp/issues)

Status
------

** EXPERIMENTAL CODE: may or may not work on your computer **

`ftvp` is currently under active development, and the API is susceptible to
change at any time. At the moment, the following features are available:

- 2D (exact/smoothed)-isotropic TV with different optimization schemes (Newton
  descent, Gradient descent, primal-dual scheme) in B/W or color.
- fast CPU implementation in a single file.
- Performs on float arrays only.
- Python 3 bindings available.

Basic use
---------
The library is at the moment splitted in two part `libftvp` for B/W images and
`libftvp-color` for color images.

For the sake of conciseness, we exemplify the use of the library for the B/W
version only. If `u` is float array of size `nxm`, the call to

    prox_tv(n, m, 1, u, lambda, 100, 16, 2, 1, 0.25, OE_SPLIT_NEWTON, 0);
    
will compute the TV regularization of `u` *in place* with regularization
parameter `lambda`, for 100 global iterations, with GPU blocks of size 16, with
2 inner iterations, testing the gap each iterations, stopping when the gap
factor is less than 0.25, where the inner iteration are given by a Newton scheme
and with standard over-relaxation parameter.

If one wants to perform the smoothed (Huber-like) TV regularization with the
same parameter, the following call should be used:

    prox_tv(n, m, 1, u, lambda, epsilon 100, 16, 2, 1, 0.25, OE_SPLIT_NEWTON, 0);

A more low-level call is possible in order to control the memory on GPU if
necessary, for instance using `prox_tv` as a inner iteration in a first-order
method for inverse problems. We refer to `init_memory` and
`prox_tv_eps_2d_noalloc` for more information.

On contrary, for the standard user, the number of parameter can be overwhelming.
In this case, we refer the user to the python 3 binding where a similar call is
done by executing

    prox_tv(u, lambda, epsilon=epsilon)
    
Please see the `examples` directory for more examples.

Build instructions
------------------
You will need [CUDA tools](https://developer.nvidia.com/cuda-toolkit) >= 5.5
installed to compile this library. On most systems you can build the library
using the following commands

    $ cd ftvp
    $ make
    
To install the library you can use the following command with the appropriate
permissions

    $ make install

`ftvp` has been successfully tested on a Jetson TK1 Embedded Development Kit
(NVIDIA Tegra K1) on Linux 3.10.24 with CUDA 6.0.1, and on a Amazon EC2
g2.2xlarge instance on Linux Ubuntu Server 14.04 LTS with CUDA 6.5. Please
report any issues with your configuration on the
[GitHub Issues](https://github.com/svaiter/ftvp/issues) page.

References
----------
If you use this library for academic work, please cite the following paper
and/or preprint

[1] Chambolle, A., & Pock, T. (2015). A remark on accelerated block coordinate
descent for computing the proximity operators of a sum of convex functions. SMAI
Journal of Computational Mathematics, 1, 29-54.

[2] Chambolle, A., Tan, P., & Vaiter S. (2016). Accelerated Alternating Descent
Methods for Dykstra-like problems. arXiV preprint.

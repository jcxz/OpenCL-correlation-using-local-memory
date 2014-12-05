OpenCL-correlation-using-local-memory
=====================================

This short demo implements a correlation-like algorithm in OpenCL, but the main
reason I wrote it, was to test out local memory optimizations in OpenCL for
algorithms that work with a kind of "halo" (i.e like convolution or correlation
that need an extra layer of additional pixels on each side). 
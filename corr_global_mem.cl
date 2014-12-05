#define IDX(x, y, size) ((x) + (size) * (y))

//#pragma OPENCL EXTENSION cl_amd_printf : enable


__kernel void corr(__global   const float *in,
                   __constant const float *mask,
                   __global         float *out,
                   const int in_row_pitch,
                   const int out_row_pitch)
{
  int i = get_global_id(0) + 1;
  int j = get_global_id(1) + 1;

  float sum = 0.0f;

  for (int jj = -1; jj <= 1; ++jj)
  {
    for (int ii = -1; ii <= 1; ++ii)
    {
      sum += in[IDX(i + ii, j + jj, in_row_pitch)] * mask[IDX(ii + 1, jj + 1, 3)];
    }
  }

  out[IDX(i - 1, j - 1, out_row_pitch)] = sum;
}

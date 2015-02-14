#define IDX(x, y, size) ((x) + (size) * (y))

//#pragma OPENCL EXTENSION cl_amd_printf : enable

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP |
                               CLK_FILTER_NEAREST;


#if 0
// Verzia ked mam fyzicky v pamati na GPU ulozeny aj okraj (halo)
__kernel void corr(__global   __read_only image2d_t in,
                   __constant const       float    *mask,
                   __global               float    *out,
                   const int out_row_pitch)
{
  int i = get_global_id(0);
  int j = get_global_id(1);
  int2 idx = (int2) (i + 1, j + 1);
  float sum = 0.0f;

  for (int jj = -1; jj <= 1; ++jj)
  {
    for (int ii = -1; ii <= 1; ++ii)
    {
      sum += read_imagef(in, sampler, idx + (int2) (ii, jj)).s0 * mask[IDX(ii + 1, jj + 1, 3)];
    }
  }

  out[IDX(i, j, out_row_pitch)] = sum;
}
#else
// v tejto verzii halo nie je ulozene fyzicky na GPU, ale vyuzivam clampovaci mod CLK_ADDRESS_CLAMP,
// ktory mi pre hodnoty mimo definovaneho pola vracia cislo 0
__kernel void corr(__read_only       image2d_t in,
                   __constant  const float    *mask,
                   __global          float    *out,
                   const int out_row_pitch)
{
  int2 idx = (int2) (get_global_id(0), get_global_id(1));
  float sum = 0.0f;

  for (int jj = -1; jj <= 1; ++jj)
  {
    for (int ii = -1; ii <= 1; ++ii)
    {
      sum += read_imagef(in, sampler, idx + (int2) (ii, jj)).s0 * mask[IDX(ii + 1, jj + 1, 3)];
    }
  }

  out[IDX(idx.x, idx.y, out_row_pitch)] = sum;
}
#endif

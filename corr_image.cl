#define IDX(x, y, size) ((x) + (size) * (y))

//#pragma OPENCL EXTENSION cl_amd_printf : enable

__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                               CLK_ADDRESS_CLAMP |
                               CLK_FILTER_NEAREST;

#ifndef TILE_W
#define TILE_W 32
#endif

#ifndef TILE_H
#define TILE_H 32
#endif

#ifndef WG_W
#define WG_W 32
#endif

#ifndef WG_H
#define WG_H 8
#endif

#define TILE_SIZE ((TILE_W) * (TILE_H))

#define WG_SIZE ((WG_W) * (WG_H))


__kernel void corr(__read_only       image2d_t in,
                   __constant  const float    *mask,
                   __global          float    *out,
                   const int out_row_pitch)
{
  int2 idx = (int2) (get_global_id(0), get_global_id(1));

  for (int k = 0; k < TILE_H; k += WG_H)
  {
    float sum = 0.0f;

    for (int jj = -1; jj <= 1; ++jj)
    {
      for (int ii = -1; ii <= 1; ++ii)
      {
        sum += read_imagef(in, sampler, idx + (int2) (ii, jj)).s0 * mask[IDX(ii + 1, jj + 1, 3)];
      }
    }

    out[IDX(idx.x, idx.y, out_row_pitch)] = sum;

    idx.y += WG_H;
  }
}

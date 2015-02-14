#define IDX(x, y, size) ((x) + (size) * (y))

//#pragma OPENCL EXTENSION cl_amd_printf : enable


/**********************************************
 * Every workgroup is processing data in tiles of width TILE_W and height TILE_H.
 * The tiles are organized so, that the width of the tiles is equal to the number
 * of work-items in a warp on given architecture. Processing of each tile is split
 * into several steps along the tile height.
 * Loading of tiles happens so that first the upper left rectangle of pixels is loaded
 * and the two columns of pixels on the left, two rows of pixel on the bottom and lastly
 * the 4 remaining pixels in the bottom right corner.
 */

//#define TILE_W 32 //64
//#define TILE_H 32 //64
#define TILE_SIZE ((TILE_W) * (TILE_H))  // 4096

//#define WG_W 32 //64
//#define WG_H 8  //4
#define WG_SIZE ((WG_W) * (WG_H))  // 256


__kernel void corr(__global   const float *in,
                   __constant const float *mask,
                   __global         float *out,
                   const int in_row_pitch,
                   const int out_row_pitch)
{
  __local float cache[TILE_W + 2][TILE_H + 2];

  int gi_0 = get_group_id(0) * TILE_W;
  int gj_0 = get_group_id(1) * TILE_H;

  int li = get_local_id(0);
  int lj = get_local_id(1);
  int lid = li + lj * WG_W;

  // nacitanie prostriedku z globalnej do lokalnej pamate
  for (int k = 0; k < TILE_H; k += WG_H)
  {
    cache[li][lj + k] = in[(gi_0 + li) + (gj_0 + lj + k) * in_row_pitch];
  }

  // nacitanie prveho dolneho riadku
  if (lid < WG_W)
  {
    cache[li][TILE_H] = in[(gi_0 + li) + (gj_0 + TILE_H) * in_row_pitch];
  }

  // nacitanie druheho dolneho riadku
  if ((lid >= WG_W) && (lid < (WG_W * 2)))
  {
    cache[li][TILE_H + 1] = in[(gi_0 + li) + (gj_0 + TILE_H + 1) * in_row_pitch];
  }

#if 0
  // nacitanie prveho praveho stlpca
  if ((lid >= (WG_W * 2)) && (lid < (WG_W * 3)))
  {
    cache[TILE_W][li] = in[(gi_0 + TILE_W) + (gj_0 + li) * in_row_pitch];
  }

  // nacitanie druheho praveho stlpca
  if ((lid >= (WG_W * 3)) && (lid < (WG_W * 4)))
  {
    cache[TILE_W + 1][li] = in[(gi_0 + TILE_W + 1) + (gj_0 + li) * in_row_pitch];
  }
#elif 0
  // nacitanie prveho praveho stlpca
  if ((lid >= (WG_W * 2)) && (lid < (WG_W * 3)))
  {
    cache[TILE_W + (li & 1)][(li >> 1)] = in[(gi_0 + TILE_W + (li & 1)) + (gj_0 + (li >> 1)) * in_row_pitch];
  }

  // nacitanie druheho praveho stlpca
  if ((lid >= (WG_W * 3)) && (lid < (WG_W * 4)))
  {
    cache[TILE_W + (li & 1)][(TILE_H / 2) + (li >> 1)] = in[(gi_0 + TILE_W + (li & 1)) + (gj_0 + (TILE_H / 2) + (li >> 1)) * in_row_pitch];
  }
#elif 1
  if ((lid >= (WG_W * 2)) && (lid < (WG_W * 3)))
  {
    cache[TILE_W]    [li] = in[(gi_0 + TILE_W    ) + (gj_0 + li) * in_row_pitch];
    cache[TILE_W + 1][li] = in[(gi_0 + TILE_W + 1) + (gj_0 + li) * in_row_pitch];
  }
#elif 0
  if ((lid >= (WG_W * 2)) && (lid < (WG_W * 3)))
  {
    cache[TILE_W + (li & 1)][(li >> 1)]                = in[(gi_0 + TILE_W + (li & 1)) + (gj_0 + (li >> 1))                * in_row_pitch];
    cache[TILE_W + (li & 1)][(TILE_H / 2) + (li >> 1)] = in[(gi_0 + TILE_W + (li & 1)) + (gj_0 + (TILE_H / 2) + (li >> 1)) * in_row_pitch];
  }
#endif

  // nacitanie rohov
  if (lid == 0)
  //if (lid == (WG_W * 4))
  {
    cache[TILE_W]    [TILE_H]     = in[(gi_0 + TILE_W)     + (gj_0 + TILE_H)     * in_row_pitch];
    cache[TILE_W + 1][TILE_H]     = in[(gi_0 + TILE_W + 1) + (gj_0 + TILE_H)     * in_row_pitch];
    cache[TILE_W]    [TILE_H + 1] = in[(gi_0 + TILE_W)     + (gj_0 + TILE_H + 1) * in_row_pitch];
    cache[TILE_W + 1][TILE_H + 1] = in[(gi_0 + TILE_W + 1) + (gj_0 + TILE_H + 1) * in_row_pitch];
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Vypocet korelacie
  for (int k = 0; k < TILE_H; k += WG_H)
  {
    float sum = 0.0f;

    for (int j = -1; j <= 1; ++j)
    {
      for (int i = -1; i <= 1; ++i)
      {
        sum += cache[li + 1 + i][lj + 1 + k + j] * mask[IDX(i + 1, j + 1, 3)];
      }
    }

    out[IDX(gi_0 + li, gj_0 + lj + k, out_row_pitch)] = sum;
  }
}

#define IDX(x, y, size) ((x) + (size) * (y))

//#pragma OPENCL EXTENSION cl_amd_printf : enable


/**********************************************
 * Every workgroup is processing data in tiles of width TILE_W and height TILE_H.
 * The tiles are organized so, that the width of the tiles is equal to the number
 * of work-items in a warp on given architecture. Processing of each tile is split
 * into several steps along the tile height.
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
  //__local float cache[TILE_W + 2][TILE_H + 2];
  __local float cache[TILE_H + 2][TILE_W + 2];

  int gi_0 = get_group_id(0) * TILE_W;
  int gj_0 = get_group_id(1) * TILE_H;

  int li = get_local_id(0);
  int lj = get_local_id(1);
  int lid = li + lj * WG_W;

  // nacitanie prostriedku z globalnej do lokalnej pamate
  for (int k = 0; k < TILE_H; k += WG_H)
  {
    //cache[li + 1][lj + 1 + k] = in[(gi_0 + li + PADDING) + (gj_0 + lj + 1 + k) * in_row_pitch];
    cache[lj + 1 + k][li + 1] = in[(gi_0 + li + PADDING) + (gj_0 + lj + 1 + k) * in_row_pitch];
  }

  // nacitanie horneho okraju
  if (lid < WG_W)
  {
    //cache[li + 1][0] = in[(gi_0 + li + PADDING) + gj_0 * in_row_pitch];
    cache[0][li + 1] = in[(gi_0 + li + PADDING) + gj_0 * in_row_pitch];
  }

  // nacitanie dolneho okraju
  if ((lid >= WG_W) && (lid < (WG_W * 2)))
  {
    //cache[li + 1][TILE_H + 1] = in[(gi_0 + li + PADDING) + (gj_0 + TILE_H + 1) * in_row_pitch];
    cache[TILE_H + 1][li + 1] = in[(gi_0 + li + PADDING) + (gj_0 + TILE_H + 1) * in_row_pitch];
  }

  // nacitanie laveho okraju
  if ((lid >= (WG_W * 2)) && (lid < (WG_W * 3)))
  {
    //cache[0][li + 1] = in[(gi_0 + PADDING - 1) + (gj_0 + li + 1) * in_row_pitch];
    cache[li + 1][0] = in[(gi_0 + PADDING - 1) + (gj_0 + li + 1) * in_row_pitch];
  }

  // nacitanie praveho okraju
  if ((lid >= (WG_W * 3)) && (lid < (WG_W * 4)))
  {
    //cache[TILE_W + 1][li + 1] = in[(gi_0 + PADDING + TILE_W) + (gj_0 + li + 1) * in_row_pitch];
    cache[li + 1][TILE_W + 1] = in[(gi_0 + PADDING + TILE_W) + (gj_0 + li + 1) * in_row_pitch];
  }

  // nacitanie rohov
  if (lid < WG_W)
  {
    cache[0][0] = in[(gi_0 + PADDING - 1) + gj_0 * in_row_pitch];  // lavy horny roh
  }

  if ((lid >= WG_W) && (lid < (WG_W * 2)))
  {
    //cache[0][TILE_H + 1] = in[(gi_0 + PADDING - 1) + (gj_0 + TILE_H + 1) * in_row_pitch];  // lavy dolny roh
    cache[TILE_H + 1][0] = in[(gi_0 + PADDING - 1) + (gj_0 + TILE_H + 1) * in_row_pitch];  // lavy dolny roh
  }

  if ((lid >= (WG_W * 2)) && (lid < (WG_W * 3)))
  {
    //cache[TILE_W + 1][0] = in[(gi_0 + PADDING + TILE_W) + gj_0 * in_row_pitch];  // pravy horny roh
    cache[0][TILE_W + 1] = in[(gi_0 + PADDING + TILE_W) + gj_0 * in_row_pitch];  // pravy horny roh
  }

  if ((lid >= (WG_W * 3)) && (lid < (WG_W * 4)))
  {
    //cache[TILE_W + 1][TILE_H + 1] = in[(gi_0 + PADDING + TILE_W) + (gj_0 + TILE_H + 1) * in_row_pitch];  // pravy dolny roh
    cache[TILE_H + 1][TILE_W + 1] = in[(gi_0 + PADDING + TILE_W) + (gj_0 + TILE_H + 1) * in_row_pitch];  // pravy dolny roh
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  // Vypocet korelacie
#if 1
  for (int k = 0; k < TILE_H; k += WG_H)
  {
    float sum = 0.0f;

    for (int j = -1; j <= 1; ++j)
    {
      for (int i = -1; i <= 1; ++i)
      {
        //sum += cache[li + 1 + i][lj + 1 + k + j] * mask[IDX(i + 1, j + 1, 3)];
        sum += cache[lj + 1 + k + j][li + 1 + i] * mask[IDX(i + 1, j + 1, 3)];
      }
    }

    out[IDX(gi_0 + li, gj_0 + lj + k, out_row_pitch)] = sum;
  }
#else
  for (int k = 0; k < TILE_H; k += WG_H)
  {
    //out[IDX(gi_0 + li, gj_0 + lj + k, out_row_pitch)] = cache[li + 0][lj + 0 + k];
    //out[IDX(gi_0 + li, gj_0 + lj + k, out_row_pitch)] = cache[li + 1][lj + 1 + k];
    out[IDX(gi_0 + li, gj_0 + lj + k, out_row_pitch)] = cache[li + 2][lj + 2 + k];
  }
#endif
}

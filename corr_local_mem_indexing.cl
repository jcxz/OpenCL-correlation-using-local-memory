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
 *
 * The important change here is that the local memory indexing is done so that it does not
 * cause shared memory bank conflicts:
 * Indexing memory like this shared_mem[local_id_x][local_id_y] in C does actually index
 * step through the memory by columns.
 *
 * Napriklad definicia v C/C++:
 *   int arr[x][y];
 *
 * definuje pole arr typu integer o x RIADKOCH a y STLPCOCH, teda pole bude definovane takto:
 *
 *         |   Col 0   |   Col 1   |   Col 2   |   Col 3   | ...
 *  -------+-----------+-----------+-----------+-----------+------
 *   Row 0 | arr[0][0] | arr[0][1] | arr[0][2] | arr[0][3] | ...
 *   Row 1 | arr[1][0] | arr[1][1] | arr[1][2] | arr[1][3] | ...
 *   Row 2 | arr[2][0] | arr[2][1] | arr[2][2] | arr[2][3] | ...
 *     .   |     .     |     .     |     .     |     .     | ...
 *     .   |     .     |     .     |     .     |     .     | ...
 *     .   |     .     |     .     |     .     |     .     | ...
 *         |           |           |           |           |
 *
 * a ked ja do prvej zatvorky dam x-ovu suradnicu, tak vlastne iterujem po riadkoch
 */

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

#ifndef WARP_SHIFT
#define WARP_SHIFT 5
#endif

#define WARP_MODULO ((WARP_SIZE) - 1)

//#define TILE_W 32 //64
//#define TILE_H 32 //64
#define TILE_SIZE ((TILE_W) * (TILE_H))  // 4096

//#define WG_W 32 //64
//#define WG_H 8  //4
#define WG_SIZE ((WG_W) * (WG_H))  // 256

#define TILE_STRIDE (TILE_W + 2)


#define WARP_ID(lid) ((lid) >> (WARP_SHIFT))
#define IS_WARP0(lid) ((WARP_ID(lid)) == 0)
#define IS_WARP1(lid) ((WARP_ID(lid)) == 1)
#define IS_WARP2(lid) ((WARP_ID(lid)) == 2)
#define IS_WARP3(lid) ((WARP_ID(lid)) == 3)
#define IS_WARP4(lid) ((WARP_ID(lid)) == 4)
#define IS_WARP5(lid) ((WARP_ID(lid)) == 5)
#define IS_WARP6(lid) ((WARP_ID(lid)) == 6)
#define IS_WARP7(lid) ((WARP_ID(lid)) == 7)



__kernel void corr(__global   const float *in,
                   __constant const float *mask,
                   __global         float *out,
                   const int in_row_pitch,
                   const int out_row_pitch)
{
  __local float cache[(TILE_W + 2) * (TILE_H + 2)];

  int gi_0 = get_group_id(0) * TILE_W;
  int gj_0 = get_group_id(1) * TILE_H;

  int li = get_local_id(0);
  int lj = get_local_id(1);
  int lid = li + lj * WG_W;

  // nacitanie prostriedku z globalnej do lokalnej pamate
  for (int k = 0; k < TILE_H; k += WG_H)
  {
    cache[li + (lj + k) * TILE_STRIDE] = in[(gi_0 + li) + (gj_0 + lj + k) * in_row_pitch];
  }

  // nacitanie prveho dolneho riadku
  if (IS_WARP0(lid))
  {
    cache[li + TILE_H * TILE_STRIDE] = in[(gi_0 + li) + (gj_0 + TILE_H) * in_row_pitch];
  }

  // nacitanie druheho dolneho riadku
  if (IS_WARP1(lid))
  {
    cache[li + (TILE_H + 1) * TILE_STRIDE] = in[(gi_0 + li) + (gj_0 + TILE_H + 1) * in_row_pitch];
  }

#if 1
  if (IS_WARP2(lid))
  {
    cache[TILE_W +     li * TILE_STRIDE] = in[(gi_0 + TILE_W    ) + (gj_0 + li) * in_row_pitch];
    cache[TILE_W + 1 + li * TILE_STRIDE] = in[(gi_0 + TILE_W + 1) + (gj_0 + li) * in_row_pitch];
  }
#elif 0
  if (IS_WARP2(lid))
  {
    cache[TILE_W + (li & 1) + (li >> 1) * TILE_STRIDE]                  = in[(gi_0 + TILE_W + (li & 1)) + (gj_0 + (li >> 1))                * in_row_pitch];
    cache[TILE_W + (li & 1) + ((TILE_H / 2) + (li >> 1)) * TILE_STRIDE] = in[(gi_0 + TILE_W + (li & 1)) + (gj_0 + (TILE_H / 2) + (li >> 1)) * in_row_pitch];
  }
#endif

  // nacitanie rohov
  if (lid == 0)
  //if (IS_WARP3(lid))
  {
    cache[TILE_W +     TILE_H * TILE_STRIDE]       = in[(gi_0 + TILE_W)     + (gj_0 + TILE_H)     * in_row_pitch];
    cache[TILE_W + 1 + TILE_H * TILE_STRIDE]       = in[(gi_0 + TILE_W + 1) + (gj_0 + TILE_H)     * in_row_pitch];
    cache[TILE_W     + (TILE_H + 1) * TILE_STRIDE] = in[(gi_0 + TILE_W)     + (gj_0 + TILE_H + 1) * in_row_pitch];
    cache[TILE_W + 1 + (TILE_H + 1) * TILE_STRIDE] = in[(gi_0 + TILE_W + 1) + (gj_0 + TILE_H + 1) * in_row_pitch];
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
        sum += cache[li + 1 + i + (lj + 1 + k + j) * TILE_STRIDE] * mask[IDX(i + 1, j + 1, 3)];
      }
    }

    out[IDX(gi_0 + li, gj_0 + lj + k, out_row_pitch)] = sum;
  }
}

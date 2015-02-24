#define IDX(x, y, size) ((x) + (size) * (y))

//#pragma OPENCL EXTENSION cl_amd_printf : enable


/**********************************************
 * Every workgroup is processing data in tiles of width TILE_W and height TILE_H.
 * The tiles are organized so, that the width of the tiles is equal to the number
 * of work-items in a warp on given architecture. Processing of each tile is split
 * into several steps along the tile height.
 */



__kernel void corr(__global   const float *in,
                   __constant const float *mask,
                   __global         float *out,
                   const int in_row_pitch,
                   const int out_row_pitch)
{
  __local float cache[IN_TILE_H][IN_TILE_W];

  // local ids of each thread in work group
  int li = get_local_id(0);
  int lj = get_local_id(1);

  // indices of the output tile
  //int gi_o = get_group_id(0) * OUT_TILE_W + li;
  //int gj_o = get_group_id(1) * OUT_TILE_H + lj;

  // indices of the input tile
  //int gi_i = gi_o - 1;
  //int gj_i = gj_o - 1;   // -1 because of the correleation kernel is of size 3 and thus has the radius of 1

  //cache[lj][li] = in[IDX(gi_i, gj_i, in_row_pitch)];

  int gi = get_group_id(0) * OUT_TILE_W + li;
  int gj = get_group_id(1) * OUT_TILE_H + lj;
  cache[lj][li] = in[IDX(gi, gj, in_row_pitch)];

  barrier(CLK_LOCAL_MEM_FENCE);

  // Vypocet korelacie
  if ((li < OUT_TILE_W) && (lj < OUT_TILE_H))
  {
    float sum = 0.0f;

    for (int j = -1; j <= 1; ++j)
    {
      for (int i = -1; i <= 1; ++i)
      {
        sum += cache[lj + 1 + j][li + 1 + i] * mask[IDX(i + 1, j + 1, 3)];
      }
    }

    out[IDX(gi, gj, out_row_pitch)] = sum;
  }
}

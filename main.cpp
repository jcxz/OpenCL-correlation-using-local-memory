#include "input.h"

#include <QtOpenCL/qclcontext.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

#define IDX(x, y, size) ((x) + (size) * (y))

#define OCL_REPORT(msg) \
  do { \
    std::cerr << msg << std::endl; \
    return false; \
  } while (0)

//#define DEBUG




/**************************************** POMOCNE FUNCKIE ****************************************/

static void printArray2d(const float *p, int w, int h)
{
  for (int j = 0; j < h; ++j)
  {
    for (int i = 0; i < w; ++i)
    {
      std::cout << std::fixed << std::setprecision(2) << std::setw(12) << p[IDX(i, j, w)] << ", ";
    }
    std::cout << std::endl;
  }
}


static float cmpArray2d(const float *p1, const float *p2, const int n)
{
  float sum = 0.0f;

  for (int i = 0; i < n; ++i)
  {
    sum += fabs(p1[i] - p2[i]);
  }

  return (sum / float(n));
}


/**************************************** REFERENCNA C++ IMPLEMENTACIA ****************************************/

static bool corrReference(const float *in, const float *mask, float *out, const int w, const int h)
{
  const int in_row_pitch = w + 2;

  for (int j = 1; j <= h; ++j)
  {
    for (int i = 1; i <= w; ++i)
    {
      float sum = 0.0f;

      for (int jj = -1; jj <= 1; ++jj)
      {
        for (int ii = -1; ii <= 1; ++ii)
        {
          sum += in[IDX(i + ii, j + jj, in_row_pitch)] * mask[IDX(ii + 1, jj + 1, 3)];
        }
      }

      out[IDX(i - 1, j - 1, w)] = sum;
    }
  }

  return true;
}


/**************************************** OPENCL IMPLEMENTACIA ****************************************/

static bool corrOCLGlobalMem(const float *in, const float *mask, float *out, const int w, const int h, const char *program_name, bool /* dummy */)
{
  std::cout << "*** OpenCL kernel that uses only global memory ***" << std::endl;

  // Vytvorenie kontextu
  QCLContext ctx;
  if (!ctx.create(QCLDevice::GPU)) OCL_REPORT("Failed to create OpenCL context");
  //if (!ctx.create(QCLDevice::CPU)) REPORT("Failed to create OpenCL context");

  // Vytvorenie fronty prikazov
  QCLCommandQueue queue(ctx.createCommandQueue(CL_QUEUE_PROFILING_ENABLE));
  if (queue.isNull()) OCL_REPORT("Failed to enable profiling on command queue");
  ctx.setCommandQueue(queue);

  // Alokacia pamate
  QCLBuffer buf_in = ctx.createBufferCopy(in, sizeof(float) * (w + 2) * (h + 2), QCLBuffer::ReadWrite);
  if (buf_in.isNull()) OCL_REPORT("Failed to create input buffer");

  QCLBuffer buf_mask = ctx.createBufferCopy(mask, sizeof(float) * 3 * 3, QCLBuffer::ReadWrite);
  if (buf_mask.isNull()) OCL_REPORT("Failed to create mask buffer");

  QCLBuffer buf_out = ctx.createBufferDevice(sizeof(float) * w * h, QCLBuffer::ReadWrite);
  if (buf_out.isNull()) OCL_REPORT("Failed to create output buffer");

  // Skompilovanie programu a vytvorenie kernelu
  //QCLProgram program = ctx.buildProgramFromSourceFile(":/corr_global_mem.cl");
  QCLProgram program = ctx.buildProgramFromSourceFile(QString(":/%1.cl").arg(program_name));
  if (program.isNull()) OCL_REPORT("Failed to compile program");

  QCLKernel kernel = program.createKernel("corr");
  if (kernel.isNull()) OCL_REPORT("Failed to create kernel");

  // Nastavenie parametrov kernelu
  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_mask);
  kernel.setArg(2, buf_out);
  kernel.setArg(3, w + 2);
  kernel.setArg(4, w);

  kernel.setGlobalWorkSize(w, h);

  // Spustenie kernelu
  QCLEvent ev(kernel.run());
  ev.waitForFinished();

  std::cout << "Execution time of kernel: " << ((ev.finishTime() - ev.runTime()) * 1e-6) << " ms" << std::endl;

  // Nacitanie vysledku
  if (!buf_out.read(out, sizeof(float) * w * h)) OCL_REPORT("Failed to read output");

  return true;
}


static bool corrOCLLocalMem(const float *in, const float *mask, float *out, const int w, const int h, const char *program_name, bool use_v2 = false)
{
  //std::cout << "*** OpenCL kernel that utilizes local memory " << ((use_v2) ? "second version ***" : "***") << std::endl;
  std::cout << "*** " << program_name << ((use_v2) ? " second version ***" : " ***") << std::endl;

  // Vytvorenie kontextu
  QCLContext ctx;
  if (!ctx.create(QCLDevice::GPU)) OCL_REPORT("Failed to create OpenCL context");
  //if (!ctx.create(QCLDevice::CPU)) OCL_REPORT("Failed to create OpenCL context");

  // Vytvorenie fronty prikazov
  QCLCommandQueue queue(ctx.createCommandQueue(CL_QUEUE_PROFILING_ENABLE));
  if (queue.isNull()) OCL_REPORT("Failed to enable profiling on command queue");
  ctx.setCommandQueue(queue);

  // Vypocet optimalnej local a global work_size
  int warp_size = 32; //64;
      // nastavenie workgroup-y (cize local work size)
  int block_width  = warp_size;                                                             // sirka work-groupy = local width/local_size(0)
  int block_height = ctx.defaultDevice().maximumWorkItemsPerGroup() / warp_size;            // vyska work-groupy = local height/local_size(1)
      // nastavenie tilu (bloku po ktorom sa budu spracovavat data)
  int tile_width = warp_size;
  int tile_height = use_v2 ? block_height : warp_size;
      // nastavenie gridu (pocet tilov na vysku a sirku)
  int grid_width  = ((w % tile_width) == 0) ? (w / tile_width) : (w / tile_width) + 1;      // pocet tilov na sirku
  int grid_height = ((h % tile_height) == 0) ? (h / tile_height) : (h / tile_height) + 1;   // pocet tilov na vysku

  // Alokacia pamate
  int in_w  = grid_width  * tile_width + 2;
  int in_h  = grid_height * tile_height + 2;
  int out_w = grid_width  * tile_width;
  int out_h = grid_height * tile_height;

  std::cerr << "grid_width=" << grid_width << ", grid_height=" << grid_height
            << ", block_width=" << block_width << ", block_height=" << block_height
            << ", tile_width=" << tile_width << ", tile_height=" << tile_height
            << ", in_w=" << in_w << ", in_h=" << in_h
            << ", out_w=" << out_w << ", out_h=" << out_h
            << std::endl;

  QCLBuffer buf_in = ctx.createBufferDevice(sizeof(float) * in_w * in_h, QCLBuffer::ReadWrite);
  if (buf_in.isNull()) OCL_REPORT("Failed to create input buffer");

  if (!buf_in.writeRect(QRect(0, 0, (w + 2) * sizeof(float), (h + 2)),
                        in,
                        in_w * sizeof(float),
                        (w + 2) * sizeof(float)))
  {
    OCL_REPORT("Failed to write data input buffer");
  }

  QCLBuffer buf_mask = ctx.createBufferCopy(mask, sizeof(float) * 3 * 3, QCLBuffer::ReadWrite);
  if (buf_mask.isNull()) OCL_REPORT("Failed to create mask buffer");

  QCLBuffer buf_out = ctx.createBufferDevice(sizeof(float) * out_w * out_h, QCLBuffer::ReadWrite);
  if (buf_out.isNull()) OCL_REPORT("Failed to create output buffer");

  // Skompilovanie programu a vytvorenie kernelu
  QString opts("-DTILE_W=%1 -DTILE_H=%2 -DWG_W=%3 -DWG_H=%4");
  //QString opts("-DSTR=\\\"test\\\"");

  //QCLProgram program = ctx.buildProgramFromSourceFile(use_v2 ? ":/corr_local_mem_v2.cl" : ":/corr_local_mem.cl",
  QCLProgram program = ctx.buildProgramFromSourceFile(QString(":/%1%2").arg(program_name).arg(use_v2 ? "_v2.cl" : ".cl"),
                                                      opts.arg(warp_size).arg(warp_size)
                                                          .arg(warp_size).arg(block_height));
  if (program.isNull()) OCL_REPORT("Failed to compile program");

  QCLKernel kernel = program.createKernel("corr");
  if (kernel.isNull()) OCL_REPORT("Failed to create kernel");

  // Nastavenie parametrov kernelu
  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_mask);
  kernel.setArg(2, buf_out);
  kernel.setArg(3, in_w);
  kernel.setArg(4, out_w);

  // Nastavenie work size-ov
  kernel.setLocalWorkSize(block_width, block_height);
  kernel.setGlobalWorkSize(grid_width * block_width, grid_height * block_height);

  // Spustenie kernelu
  QCLEvent ev(kernel.run());
  ev.waitForFinished();

  std::cout << "Execution time of kernel: " << ((ev.finishTime() - ev.runTime()) * 1e-6) << " ms" << std::endl;

  // Nacitanie vysledku
  if (!buf_out.readRect(QRect(0, 0, w * sizeof(float), h),
                        out,
                        sizeof(float) * out_w,
                        sizeof(float) * w))
  {
    OCL_REPORT("Failed to read output");
  }

  return true;
}


static bool corrOCLLocalMemInner(const float *in, const float *mask, float *out, const int w, const int h, const char *program_name, bool use_v2 = false)
{
  std::cout << "*** " << program_name << ((use_v2) ? " second version ***" : " ***") << std::endl;

  // Vytvorenie kontextu
  QCLContext ctx;
  if (!ctx.create(QCLDevice::GPU)) OCL_REPORT("Failed to create OpenCL context");
  //if (!ctx.create(QCLDevice::CPU)) OCL_REPORT("Failed to create OpenCL context");

  // Vytvorenie fronty prikazov
  QCLCommandQueue queue(ctx.createCommandQueue(CL_QUEUE_PROFILING_ENABLE));
  if (queue.isNull()) OCL_REPORT("Failed to enable profiling on command queue");
  ctx.setCommandQueue(queue);

  // Vypocet optimalnej local a global work_size
  int warp_size = 32; //64;
      // nastavenie workgroup-y (cize local work size)
  int block_width  = warp_size;                                                             // sirka work-groupy = local width/local_size(0)
  int block_height = ctx.defaultDevice().maximumWorkItemsPerGroup() / warp_size;            // vyska work-groupy = local height/local_size(1)
      // nastavenie velkosti vystupneho tilu
  int tile_width = block_width - 2; // -2 pretoze mam korelacnu masku o velkosti 3 a polomere 1
  int tile_height = block_height - 2;
      // nastavenie gridu (pocet tilov na vysku a sirku)
  int grid_width  = (w - 1) / tile_width + 1;    // pocet tilov na sirku
  int grid_height = (h - 1) / tile_height + 1;   // pocet tilov na vysku

  // Alokacia pamate
  int in_w  = grid_width  * tile_width + 2;
  int in_h  = grid_height * tile_height + 2;
  int out_w = grid_width  * tile_width;
  int out_h = grid_height * tile_height;

  std::cerr << "grid_width=" << grid_width << ", grid_height=" << grid_height
            << ", block_width=" << block_width << ", block_height=" << block_height
            << ", tile_width=" << tile_width << ", tile_height=" << tile_height
            << ", in_w=" << in_w << ", in_h=" << in_h
            << ", out_w=" << out_w << ", out_h=" << out_h
            << std::endl;

  QCLBuffer buf_in = ctx.createBufferDevice(sizeof(float) * in_w * in_h, QCLBuffer::ReadWrite);
  if (buf_in.isNull()) OCL_REPORT("Failed to create input buffer");

  if (!buf_in.writeRect(QRect(0, 0, (w + 2) * sizeof(float), (h + 2)),
                        in,
                        in_w * sizeof(float),
                        (w + 2) * sizeof(float)))
  {
    OCL_REPORT("Failed to write data input buffer");
  }

  QCLBuffer buf_mask = ctx.createBufferCopy(mask, sizeof(float) * 3 * 3, QCLBuffer::ReadWrite);
  if (buf_mask.isNull()) OCL_REPORT("Failed to create mask buffer");

  QCLBuffer buf_out = ctx.createBufferDevice(sizeof(float) * out_w * out_h, QCLBuffer::ReadWrite);
  if (buf_out.isNull()) OCL_REPORT("Failed to create output buffer");

  // Skompilovanie programu a vytvorenie kernelu
  QString opts("-DIN_TILE_W=%1 -DIN_TILE_H=%2 -DOUT_TILE_W=%3 -DOUT_TILE_H=%4");
  QCLProgram program = ctx.buildProgramFromSourceFile(QString(":/%1%2").arg(program_name).arg(use_v2 ? "_v2.cl" : ".cl"),
                                                      opts.arg(block_width).arg(block_height)
                                                          .arg(tile_width).arg(tile_height));
  if (program.isNull()) OCL_REPORT("Failed to compile program");

  QCLKernel kernel = program.createKernel("corr");
  if (kernel.isNull()) OCL_REPORT("Failed to create kernel");

  // Nastavenie parametrov kernelu
  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_mask);
  kernel.setArg(2, buf_out);
  kernel.setArg(3, in_w);
  kernel.setArg(4, out_w);

  // Nastavenie work size-ov
  kernel.setLocalWorkSize(block_width, block_height);
  kernel.setGlobalWorkSize(grid_width * block_width, grid_height * block_height);

  // Spustenie kernelu
  QCLEvent ev(kernel.run());
  ev.waitForFinished();

  std::cout << "Execution time of kernel: " << ((ev.finishTime() - ev.runTime()) * 1e-6) << " ms" << std::endl;

  // Nacitanie vysledku
  if (!buf_out.readRect(QRect(0, 0, w * sizeof(float), h),
                        out,
                        sizeof(float) * out_w,
                        sizeof(float) * w))
  {
    OCL_REPORT("Failed to read output");
  }

  return true;
}


static bool corrOCLLocalMemPadding(const float *in, const float *mask, float *out, const int w, const int h, const char *program_name, bool use_v2 = false)
{
  //std::cout << "*** OpenCL kernel that utilizes local memory and aligns global data " << ((use_v2) ? "second version ***" : "***") << std::endl;
  std::cout << "*** " << program_name << ((use_v2) ? " second version ***" : " ***") << std::endl;

  // Vytvorenie kontextu
  QCLContext ctx;
  if (!ctx.create(QCLDevice::GPU)) OCL_REPORT("Failed to create OpenCL context");
  //if (!ctx.create(QCLDevice::CPU)) OCL_REPORT("Failed to create OpenCL context");

  // Vytvorenie fronty prikazov
  QCLCommandQueue queue(ctx.createCommandQueue(CL_QUEUE_PROFILING_ENABLE));
  if (queue.isNull()) OCL_REPORT("Failed to enable profiling on command queue");
  ctx.setCommandQueue(queue);

  // Vypocet optimalnej local a global work_size
  int warp_size = 32; //64;
      // nastavenie workgroup-y (cize local work size)
  int block_width  = warp_size;                                                             // sirka work-groupy = local width/local_size(0)
  int block_height = ctx.defaultDevice().maximumWorkItemsPerGroup() / warp_size;            // vyska work-groupy = local height/local_size(1)
      // nastavenie tilu (bloku po ktorom sa budu spracovavat data)
  int tile_width = warp_size;
  int tile_height = use_v2 ? block_height : warp_size;
      // nastavenie gridu (pocet tilov na vysku a sirku)
  int grid_width  = ((w % tile_width) == 0) ? (w / tile_width) : (w / tile_width) + 1;      // pocet tilov na sirku
  int grid_height = ((h % tile_height) == 0) ? (h / tile_height) : (h / tile_height) + 1;   // pocet tilov na vysku

  // Alokacia pamate
  int in_w  = grid_width  * tile_width;
  int in_h  = grid_height * tile_height;
  int out_w = grid_width  * tile_width;
  int out_h = grid_height * tile_height;

  // uprava in_w, in_h, out_w a out_h podla velkosti halo a nastavenia paddingu
  int alignment = 32; //128; //64; //32;                        // 32 float numbers = 128 bytes
  int padding_in = (in_w + 1) % alignment;
  if (padding_in == 0) padding_in = 0; else padding_in = alignment - padding_in;
  int padding_out = out_w % alignment;
  if (padding_out == 0) padding_out = 0; else padding_out = alignment - padding_out;

  in_w = alignment + in_w + 1 + padding_in;  // kazdy riadok vstupnych dat ma padding na zaciatku aj na konci (kvoli halo)
  in_h = in_h + 2;                           // potrebujem navyse jeden riadok nad a pod
  out_w = out_w + padding_out;
  out_h = out_h;

  std::cerr << "grid_width=" << grid_width << ", grid_height=" << grid_height
            << ", block_width=" << block_width << ", block_height=" << block_height
            << ", tile_width=" << tile_width << ", tile_height=" << tile_height
            << ", in_w=" << in_w << ", in_h=" << in_h
            << ", out_w=" << out_w << ", out_h=" << out_h
            << ", alignment=" << alignment << ", padding_in=" << padding_in << ", padding_out=" << padding_out
            << std::endl;

  QCLBuffer buf_in = ctx.createBufferDevice(sizeof(float) * in_w * in_h, QCLBuffer::ReadWrite);
  if (buf_in.isNull()) OCL_REPORT("Failed to create input buffer");

  if (!buf_in.writeRect(QRect((alignment - 1) * sizeof(float), 0, (w + 2) * sizeof(float), (h + 2)),
                        in,
                        in_w * sizeof(float),
                        (w + 2) * sizeof(float)))
  {
    OCL_REPORT("Failed to write data input buffer");
  }

  QCLBuffer buf_mask = ctx.createBufferCopy(mask, sizeof(float) * 3 * 3, QCLBuffer::ReadWrite);
  if (buf_mask.isNull()) OCL_REPORT("Failed to create mask buffer");

  QCLBuffer buf_out = ctx.createBufferDevice(sizeof(float) * out_w * out_h, QCLBuffer::ReadWrite);
  if (buf_out.isNull()) OCL_REPORT("Failed to create output buffer");

  // Skompilovanie programu a vytvorenie kernelu
  QString opts("-DTILE_W=%1 -DTILE_H=%2 -DWG_W=%3 -DWG_H=%4 -DPADDING=%5");
  //QCLProgram program = ctx.buildProgramFromSourceFile(use_v2 ? ":/corr_local_mem_v2_padding.cl" : ":/corr_local_mem_padding.cl",
  QCLProgram program = ctx.buildProgramFromSourceFile(QString(":/%1%2").arg(program_name).arg(use_v2 ? "_v2.cl" : ".cl"),
                                                      opts.arg(warp_size).arg(warp_size)
                                                          .arg(warp_size).arg(block_height)
                                                          .arg(alignment));
  if (program.isNull()) OCL_REPORT("Failed to compile program");

  QCLKernel kernel = program.createKernel("corr");
  if (kernel.isNull()) OCL_REPORT("Failed to create kernel");

  // Nastavenie parametrov kernelu
  kernel.setArg(0, buf_in);
  kernel.setArg(1, buf_mask);
  kernel.setArg(2, buf_out);
  kernel.setArg(3, in_w);
  kernel.setArg(4, out_w);

  // Nastavenie work size-ov
  kernel.setLocalWorkSize(block_width, block_height);
  kernel.setGlobalWorkSize(grid_width * block_width, grid_height * block_height);

  // Spustenie kernelu
  QCLEvent ev(kernel.run());
  ev.waitForFinished();

  std::cout << "Execution time of kernel: " << ((ev.finishTime() - ev.runTime()) * 1e-6) << " ms" << std::endl;

  // Nacitanie vysledku
  if (!buf_out.readRect(QRect(0, 0, w * sizeof(float), h),
                        out,
                        sizeof(float) * out_w,
                        sizeof(float) * w))
  {
    OCL_REPORT("Failed to read output");
  }

  return true;
}


static bool corrOCLImage(const float *in, const float *mask, float *out, const int w, const int h, const char *program_name, bool use_v2 = false)
{
  //std::cout << "*** OpenCL kernel that uses textures ***" << std::endl;
  std::cout << "*** " << program_name << ((use_v2) ? " second version ***" : " ***") << std::endl;

  // Vytvorenie kontextu
  QCLContext ctx;
  if (!ctx.create(QCLDevice::GPU)) OCL_REPORT("Failed to create OpenCL context");
  //if (!ctx.create(QCLDevice::CPU)) OCL_REPORT("Failed to create OpenCL context");

  // Vytvorenie fronty prikazov
  QCLCommandQueue queue(ctx.createCommandQueue(CL_QUEUE_PROFILING_ENABLE));
  if (queue.isNull()) OCL_REPORT("Failed to enable profiling on command queue");
  ctx.setCommandQueue(queue);

  // Vypocet optimalnej local a global work_size
  int warp_size = 32; //64;
      // nastavenie workgroup-y (cize local work size)
  int block_width  = warp_size;                                                             // sirka work-groupy = local width/local_size(0)
  int block_height = ctx.defaultDevice().maximumWorkItemsPerGroup() / warp_size;            // vyska work-groupy = local height/local_size(1)
      // nastavenie tilu (bloku po ktorom sa budu spracovavat data)
  int tile_width = warp_size;
  int tile_height = use_v2 ? block_height : warp_size;
      // nastavenie gridu (pocet blokov na vysku a sirku)
  int grid_width  = (w + tile_width  - 1) / tile_width;    // pocet blokov na sirku
  int grid_height = (h + tile_height - 1) / tile_height;   // pocet blokov na vysku

  // Alokacia pamate
  int out_w = grid_width  * tile_width;
  int out_h = grid_height * tile_height;

  std::cerr << "grid_width=" << grid_width << ", grid_height=" << grid_height
            << ", block_width=" << block_width << ", block_height=" << block_height
            << ", tile_width=" << tile_width << ", tile_height=" << tile_height
            << ", in_w=" << w << ", in_h=" << h
            << ", out_w=" << out_w << ", out_h=" << out_h
            << std::endl;

  // Alokacia pamate
  QCLImageFormat fmt(QCLImageFormat::Order_R, QCLImageFormat::Type_Float);
  QCLImage2D img_in = ctx.createImage2DDevice(fmt, QSize(w, h), QCLBuffer::ReadOnly);
  if (img_in.isNull()) OCL_REPORT("Failed to create input GPU image");

  float *ptr = (float *) img_in.map(QRect(0, 0, w, h), QCLImage2D::WriteOnly);
  if (ptr == nullptr) OCL_REPORT("Failed to map input GPU image");

  for (int j = 0; j < h; ++j)
  {
    for (int i = 0; i < w; ++i)
    {
      ptr[i + j * w] = in[(i + 1) + (j + 1) * (w + 2)];
    }
  }

  img_in.unmap(ptr);

  QCLBuffer buf_mask = ctx.createBufferCopy(mask, sizeof(float) * 3 * 3, QCLBuffer::ReadWrite);
  if (buf_mask.isNull()) OCL_REPORT("Failed to create mask buffer");

  QCLBuffer buf_out = ctx.createBufferDevice(sizeof(float) * out_w * out_h, QCLBuffer::ReadWrite);
  if (buf_out.isNull()) OCL_REPORT("Failed to create output buffer");

  // Skompilovanie programu a vytvorenie kernelu
  QString opts("-DTILE_W=%1 -DTILE_H=%2 -DWG_W=%3 -DWG_H=%4");

  QCLProgram program = ctx.buildProgramFromSourceFile(QString(":/%1%2").arg(program_name).arg(use_v2 ? "_v2.cl" : ".cl"),
                                                      opts.arg(tile_width).arg(tile_height)
                                                          .arg(block_width).arg(block_height));
  if (program.isNull()) OCL_REPORT("Failed to compile program");

  QCLKernel kernel = program.createKernel("corr");
  if (kernel.isNull()) OCL_REPORT("Failed to create kernel");

  // Nastavenie parametrov kernelu
  kernel.setArg(0, img_in);
  kernel.setArg(1, buf_mask);
  kernel.setArg(2, buf_out);
  kernel.setArg(3, out_w);

  //kernel.setGlobalWorkSize(w, h);
  kernel.setLocalWorkSize(block_width, block_height);
  kernel.setGlobalWorkSize(grid_width * block_width, grid_height * block_height);

  // Spustenie kernelu
  QCLEvent ev(kernel.run());
  ev.waitForFinished();

  std::cout << "Execution time of kernel: " << ((ev.finishTime() - ev.runTime()) * 1e-6) << " ms" << std::endl;

  // Nacitanie vysledku
  if (!buf_out.readRect(QRect(0, 0, w * sizeof(float), h),
                        out,
                        sizeof(float) * out_w,
                        sizeof(float) * w))
  {
    OCL_REPORT("Failed to read output");
  }

  return true;
}


/**************************************** SPUSTANIE TESTOV ****************************************/

//typedef bool (* TCorrFunc)(const float *in, const float *mask, float *out, const int w, const int h, bool use_v2);
typedef bool (* TCorrFunc)(const float *in, const float *mask, float *out, const int w, const int h, const char *program_name, bool use_v2);

//static bool testFunc(TCorrFunc f, const float *ref, const float *in, const float *mask, float *out, const int w, const int h, bool use_v2)
static bool testFunc(TCorrFunc f, const float *ref, const float *in, const float *mask, float *out, const int w, const int h, const char *program_name, bool use_v2)
{
  //if (!f(in, mask, out, w, h, use_v2)) return false;
  if (!f(in, mask, out, w, h, program_name, use_v2)) return false;

#ifdef DEBUG
  std::cout << "C++:" << std::endl;    printArray2d(ref, w, h); std::cout << std::endl;
  std::cout << "OpenCL:" << std::endl; printArray2d(out, w, h); std::cout << std::endl;
#endif

  std::cout << "Average difference between elements of arrays: " << cmpArray2d(ref, out, w * h) << std::endl;

  return true;
}


static bool runTestDebug(void)
{
  // Vygenerovanie testovacich dat
  const int mask_w = 3;
  const float mask[mask_w * mask_w] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  };

  int w = 40, h = 40;
  //int w = 10, h = 10;
  const float *in;
  float *out_cpp, *out_ocl;

  input::genSequential(in, out_cpp, out_ocl, w, h, mask_w / 2);
  std::cout << "Input:" << std::endl;    printArray2d(in, w + 2, h + 2); std::cout << std::endl;
  if (!corrReference(in, mask, out_cpp, w, h)) return false;
  //if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, w, h, true)) return false;
  //if (!testFunc(corrOCLLocalMemPadding, out_cpp, in, mask, out_ocl, w, h, false)) return false;
  //if (!testFunc(corrOCLLocalMemPadding, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_padding", true)) return false;
  //if (!testFunc(corrOCLImage, out_cpp, in, mask, out_ocl, w, h, "corr_image", false)) return false;
  if (!testFunc(corrOCLLocalMemInner, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_inner_tile", false)) return false;

  delete [] in;
  delete [] out_cpp;
  delete [] out_ocl;

  return true;
}

static bool runTest1(void)
{
  // Vygenerovanie testovacich dat
  const int mask_w = 3;
  const float mask[mask_w * mask_w] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  };

  int w = 10000, h = 10000;
  const float *in;
  float *out_cpp, *out_ocl;

#ifdef DEBUG
  input::genDebug(in, out_cpp, out_ocl, w, h);
#else
  input::genSequential(in, out_cpp, out_ocl, w, h, mask_w / 2);
#endif

  std::cout << "Test size: w=" << w << ", h=" << h << std::endl;

  // vypocet referencnej implementacie
  if (!corrReference(in, mask, out_cpp, w, h)) return false;

  // OpenCL implementacia
  if (!testFunc(corrOCLGlobalMem, out_cpp, in, mask, out_ocl, w, h, "corr_global_mem", false)) return false;
  if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem", false)) return false;
  if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem", true)) return false;
  if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_corners", false)) return false;
  if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_right_border", false)) return false;
  if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_right_border_2", false)) return false;
  if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_rows_joint", false)) return false;
  //if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_float4", false)) return false;
  if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_indexing", false)) return false;
  if (!testFunc(corrOCLLocalMemPadding, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_padding", false)) return false;
  if (!testFunc(corrOCLLocalMemPadding, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_padding", true)) return false;
  if (!testFunc(corrOCLImage, out_cpp, in, mask, out_ocl, w, h, "corr_image", false)) return false;
  if (!testFunc(corrOCLImage, out_cpp, in, mask, out_ocl, w, h, "corr_image", true)) return false;
  if (!testFunc(corrOCLLocalMemInner, out_cpp, in, mask, out_ocl, w, h, "corr_local_mem_inner_tile", false)) return false;

  delete [] in;
  delete [] out_cpp;
  delete [] out_ocl;

  return true;
}


static bool runTest2(void)
{
  // Vygenerovanie testovacich dat
  const int mask_w = 3;
  const float mask[mask_w * mask_w] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  };

  const int n = 4;
  int tests_w[n] = { 1000, 4000, 8000 , 8190 };
  int tests_h[n] = { 1000, 2000, 10000, 8190 };

  for (int i = 0; i < n; ++i)
  {
    const float *in;
    float *out_cpp, *out_ocl;

    input::genRandom(in, out_cpp, out_ocl, tests_w[i], tests_h[i], mask_w / 2);

    std::cout << "==========================================================================" << std::endl;
    std::cout << "Test size: w=" << tests_w[i] << ", h=" << tests_h[i] << std::endl;

    // vypocet referencnej implementacie
    auto start = std::chrono::steady_clock::now();
    if (!corrReference(in, mask, out_cpp, tests_w[i], tests_h[i])) return false;
    auto end = std::chrono::steady_clock::now();
    std::cout << "Reference implementation total CPU time: " << std::chrono::duration <double, std::milli>(end - start).count() << " ms" << std::endl;

    // OpenCL implementacia
    if (!testFunc(corrOCLGlobalMem, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_global_mem", false)) return false;
    if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem", false)) return false;
    if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem", true)) return false;
    if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem_corners", false)) return false;
    if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem_right_border", false)) return false;
    if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem_right_border_2", false)) return false;
    if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem_rows_joint", false)) return false;
    //if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem_float4", false)) return false;
    if (!testFunc(corrOCLLocalMem, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem_indexing", false)) return false;
    if (!testFunc(corrOCLLocalMemPadding, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem_padding", false)) return false;
    if (!testFunc(corrOCLLocalMemPadding, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem_padding", true)) return false;
    if (!testFunc(corrOCLImage, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_image", false)) return false;
    if (!testFunc(corrOCLImage, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_image", true)) return false;
    if (!testFunc(corrOCLLocalMemInner, out_cpp, in, mask, out_ocl, tests_w[i], tests_h[i], "corr_local_mem_inner_tile", false)) return false;

    delete [] in;
    delete [] out_cpp;
    delete [] out_ocl;
  }

  return true;
}

/**************************************** MAIN ****************************************/

int main(void)
{
  //if (!runTest1()) return 1;
  if (!runTest2()) return 1;
  //if (!runTestDebug()) return 1;

  return 0;
}

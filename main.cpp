#include <QtOpenCL/qclcontext.h>
#include <iostream>
#include <iomanip>
#include <cmath>

#define IDX(x, y, size) ((x) + (size) * (y))

#define OCL_REPORT(msg) \
  do { \
    std::cerr << msg << std::endl; \
    return false; \
  } while (0)

//#define SMALL




/**************************************** POMOCNE FUNCKIE ****************************************/

#ifdef SMALL
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
#endif


static float cmpArray2d(const float *p1, const float *p2, const int n)
{
  float sum = 0.0f;

  for (int i = 0; i < n; ++i)
  {
    sum += fabs(p1[i] - p2[i]);
  }

  return (sum / float(n));
}


/**************************************** FUNCKIE NA GENROVANIE TESTOVACICH DAT ****************************************/

#ifdef SMALL
static void genInputSmall(const float * & in, float * & out_cpp, float * & out_ocl, int & w, int & h)
{
  const int w_ = 10;
  const int h_ = 10;
  const int w_size_ = w_ + 2;   // sirka riadku
  const int h_size_ = h_ + 2;   // sirka stlpca

  static const float in_[w_size_ * h_size_] = {
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0, 0,
    0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 0,
    0, 11, 12, 13, 14, 15, 16, 17, 18, 19,  20, 0,
    0, 21, 22, 23, 24, 25, 26, 27, 28, 29,  30, 0,
    0, 31, 32, 33, 34, 35, 36, 37, 38, 39,  40, 0,
    0, 41, 42, 43, 44, 45, 46, 47, 48, 49,  50, 0,
    0, 51, 52, 53, 54, 55, 56, 57, 58, 59,  60, 0,
    0, 61, 62, 63, 64, 65, 66, 67, 68, 69,  70, 0,
    0, 71, 72, 73, 74, 75, 76, 77, 78, 79,  80, 0,
    0, 81, 82, 83, 84, 85, 86, 87, 88, 89,  90, 0,
    0, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0, 0
  };

  static float out_cpp_[w_ * h_];
  static float out_ocl_[w_ * h_];

  in = in_;
  out_cpp = out_cpp_;
  out_ocl = out_ocl_;
  w = w_;
  h = h_;
}
#endif


static void genInputBig(const float * & in, float * & out_cpp, float * & out_ocl, int & w, int & h, int border_size)
{
  const int w_ = 10000;
  const int h_ = 10000;
  const int w_size_ = w_ + border_size * 2;   // sirka riadku
  const int h_size_ = h_ + border_size * 2;   // sirka stlpca

  float *in_ = new float[w_size_ * h_size_];
  float *out_cpp_ = new float[w_ * h_];
  float *out_ocl_ = new float[w_ * h_];

  for (int j = 0; j < h_size_; ++j)
  {
    for (int i = 0; i < w_size_; ++i)
    {
      int idx = i + j * w_size_;
      if ((i < border_size) || (i > (w_size_ - 1 - border_size)) || (j < border_size) || (j > (h_size_ - 1 - border_size)))
      {
        in_[idx] = 0;
      }
      else
      {
        in_[idx] = (i - 1) + (j - 1) * w_;
      }
    }
  }

  in = in_;
  out_cpp = out_cpp_;
  out_ocl = out_ocl_;
  w = w_;
  h = h_;
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

static bool corrOCLGlobalMem(const float *in, const float *mask, float *out, const int w, const int h)
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
  QCLProgram program = ctx.buildProgramFromSourceFile(":/corr_global_mem.cl");
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


static bool corrOCLLocalMem(const float *in, const float *mask, float *out, const int w, const int h)
{
  std::cout << "*** OpenCL kernel that utilizes local memory ***" << std::endl;

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
  int block_width  = warp_size;
  int block_height = ctx.defaultDevice().maximumWorkItemsPerGroup() / warp_size;
  int grid_width  = ((w % warp_size) == 0) ? (w / warp_size) : (w / warp_size) + 1;
  int grid_height = ((h % warp_size) == 0) ? (h / warp_size) : (h / warp_size) + 1;

  // Alokacia pamate
  int in_w  = grid_width  * warp_size + 2;
  int in_h  = grid_height * warp_size + 2;
  int out_w = grid_width  * warp_size;
  int out_h = grid_height * warp_size;

  std::cerr << "grid_width=" << grid_width << ", grid_height=" << grid_height
            << ", block_width=" << block_width << ", block_height=" << block_height
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

  QCLProgram program = ctx.buildProgramFromSourceFile(":/corr_local_mem.cl",
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


/**************************************** MAIN ****************************************/

int main(void)
{
  // Vygenerovanie testovacich dat
  const int mask_w = 3;
  const float mask[mask_w * mask_w] = {
    1, 1, 1,
    1, 1, 1,
    1, 1, 1
  };

  int w, h;
  const float *in;
  float *out_cpp, *out_ocl;

#ifdef SMALL
  genInputSmall(in, out_cpp, out_ocl, w, h);
#else
  genInputBig(in, out_cpp, out_ocl, w, h, mask_w / 2);
#endif

  std::cerr << "Test size: w=" << w << ", h=" << h << std::endl;

  // vypocet referencnej implementacie
  if (!corrReference(in, mask, out_cpp, w, h)) return 1;

  // OpenCL implementacia
  if (!corrOCLGlobalMem(in, mask, out_ocl, w, h)) return 1;

#ifdef SMALL
  std::cout << "C++:" << std::endl;    printArray2d(out_cpp, w, h); std::cout << std::endl;
  std::cout << "OpenCL:" << std::endl; printArray2d(out_ocl, w, h); std::cout << std::endl;
#endif

  std::cout << "Average difference between elements of arrays: " << cmpArray2d(out_cpp, out_ocl, w * h) << std::endl;

  if (!corrOCLLocalMem(in, mask, out_ocl, w, h)) return 1;

#ifdef SMALL
  std::cout << "C++:" << std::endl;    printArray2d(out_cpp, w, h); std::cout << std::endl;
  std::cout << "OpenCL:" << std::endl; printArray2d(out_ocl, w, h); std::cout << std::endl;
#endif

  std::cout << "Average difference between elements of arrays: " << cmpArray2d(out_cpp, out_ocl, w * h) << std::endl;

  return 0;
}

#include "input.h"

#include <limits>
#include <cstdlib>
#include <ctime>
#include <cstring>



namespace {

float random(float min, float max) { return (float(rand()) / float(RAND_MAX)) * (max - min) + min; }
float random(float max) { return random(std::numeric_limits<float>::min(), max); }
float random(void) { return random(std::numeric_limits<float>::min(), std::numeric_limits<float>::max()); }

}


namespace input {

void genDebug(const float * & in, float * & out_cpp, float * & out_ocl, int & w, int & h)
{
  const int w_ = 10;
  const int h_ = 10;
  const int w_size_ = w_ + 2;   // sirka riadku
  const int h_size_ = h_ + 2;   // sirka stlpca

  static const float in_data[w_size_ * h_size_] = {
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

  float *in_ = new float[w_size_ * h_size_];
  float *out_cpp_ = new float[w_ * h_];
  float *out_ocl_ = new float[w_ * h_];

  std::memcpy(in_, in_data, sizeof(float) * w_size_ * h_size_);

  in = in_;
  out_cpp = out_cpp_;
  out_ocl = out_ocl_;
  w = w_;
  h = h_;
}


void genSequential(const float * & in, float * & out_cpp, float * & out_ocl, int w, int h, int border_size)
{
  const int w_size_ = w + border_size * 2;   // sirka riadku
  const int h_size_ = h + border_size * 2;   // sirka stlpca

  float *in_ = new float[w_size_ * h_size_];
  float *out_cpp_ = new float[w * h];
  float *out_ocl_ = new float[w * h];

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
        in_[idx] = (i - 1) + (j - 1) * w;
      }
    }
  }

  in = in_;
  out_cpp = out_cpp_;
  out_ocl = out_ocl_;
}


void genRandom(const float * & in, float * & out_cpp, float * & out_ocl, int w, int h, int border_size)
{
  int n = (w + 2 * border_size) * (h + 2 * border_size);
  float *in_ = new float[n];
  float *out_cpp_ = new float[n];
  float *out_ocl_ = new float[n];

  srand(time(nullptr));

  for (int i = 0; i < n; ++i)
  {
    in_[i] = random(0.0f, 100.0f);
  }

  in = in_;
  out_cpp = out_cpp_;
  out_ocl = out_ocl_;
}

}

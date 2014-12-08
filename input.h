#ifndef INPUT_H
#define INPUT_H

namespace input {

void genDebug(const float * & in, float * & out_cpp, float * & out_ocl, int & w, int & h);
void genSequential(const float * & in, float * & out_cpp, float * & out_ocl, int w, int h, int border_size);
void genRandom(const float * & in, float * & out_cpp, float * & out_ocl, int w, int h, int border_size);

} // End of input namespace

#endif // INPUT_H

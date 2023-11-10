#ifndef COL2IM_H
#define COL2IM_H

void col2im_cpu(float* data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float* data_im);

#ifdef GPU
void col2im_gpu(float *data_col,
        int channels, int height, int width,
        int ksize, int stride, int pad, float *data_im);
#endif

#ifdef FTP
void col2im_cpu_ftp(float* data_col,
         int channels,  int height,  int width, int height_out, int width_out,
         int ksize,  int stride, int pad, float* data_im);
#endif
#endif

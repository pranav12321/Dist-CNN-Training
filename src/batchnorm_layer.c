#include "convolutional_layer.h"
#include "batchnorm_layer.h"
#include "blas.h"
#include "fused_device.h"
#include <stdio.h>

layer make_batchnorm_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "Batch Normalization Layer: %d x %d x %d image\n", w,h,c);
    layer l = {0};
    l.type = BATCHNORM;
    l.batch = batch;
    l.h = l.out_h = h;
    l.w = l.out_w = w;
    l.c = l.out_c = c;
    l.output = calloc(h * w * c * batch, sizeof(float));
    l.delta  = calloc(h * w * c * batch, sizeof(float));
    l.inputs = w*h*c;
    l.outputs = l.inputs;

    l.scales = calloc(c, sizeof(float));
    l.scale_updates = calloc(c, sizeof(float));
    l.biases = calloc(c, sizeof(float));
    l.bias_updates = calloc(c, sizeof(float));
    int i;
    for(i = 0; i < c; ++i){
        l.scales[i] = 1;
    }

    l.mean = calloc(c, sizeof(float));
    l.variance = calloc(c, sizeof(float));

    l.rolling_mean = calloc(c, sizeof(float));
    l.rolling_variance = calloc(c, sizeof(float));

    l.forward = forward_batchnorm_layer;
    l.backward = backward_batchnorm_layer;
#ifdef GPU
    l.forward_gpu = forward_batchnorm_layer_gpu;
    l.backward_gpu = backward_batchnorm_layer_gpu;

    l.output_gpu =  cuda_make_array(l.output, h * w * c * batch);
    l.delta_gpu =   cuda_make_array(l.delta, h * w * c * batch);

    l.biases_gpu = cuda_make_array(l.biases, c);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, c);

    l.scales_gpu = cuda_make_array(l.scales, c);
    l.scale_updates_gpu = cuda_make_array(l.scale_updates, c);

    l.mean_gpu = cuda_make_array(l.mean, c);
    l.variance_gpu = cuda_make_array(l.variance, c);

    l.rolling_mean_gpu = cuda_make_array(l.mean, c);
    l.rolling_variance_gpu = cuda_make_array(l.variance, c);

    l.mean_delta_gpu = cuda_make_array(l.mean, c);
    l.variance_delta_gpu = cuda_make_array(l.variance, c);

    l.x_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    l.x_norm_gpu = cuda_make_array(l.output, l.batch*l.outputs);
    #ifdef CUDNN
    cudnnCreateTensorDescriptor(&l.normTensorDesc);
    cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w); 
    cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1); 

    #endif
#endif
    return l;
}

void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(0 - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}

void mean_delta_cpu_dist_ftp(ftp_config* ftp_params, float *delta, float *variance, int batch, int filters, int height_full, int width_full,
                             int height_core, int width_core, int left_offset, int top_offset, float *mean_delta)
{
    int i,j,h,w;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (h = 0; h < height_core; ++h) {
                for (w = 0; w < width_core; ++w) {
                    int index = j*filters*height_full*width_full + (h+top_offset)*width_full + w + left_offset;
                    mean_delta[i] += delta[index];
                }
            }
        }
        //mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }

    if(ftp_params->is_main_gateway){
        float* mean_delta_temp = calloc(filters, sizeof(float));
        for(i = 1; i < ftp_params->NUM_TILES_X * ftp_params->NUM_TILES_Y; i++){
            receive_data(mean_delta_temp, filters, i);
            for(j = 0; j < filters; j++)
                mean_delta[j] += mean_delta_temp[j];
        }
        for(i = 1; i < ftp_params->NUM_TILES_X * ftp_params->NUM_TILES_Y; i++)
            send_data(mean_delta, filters, i);
        free(mean_delta_temp);
    }
    else{
            send_data(mean_delta, filters, 0);
            receive_data(mean_delta, filters, 0);
    }
    for(i = 0; i < filters; ++i)
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
}

void variance_delta_cpu_dist_ftp(ftp_config* ftp_params, float *x, float *delta, float *mean, float *variance, int batch, int filters,
                                 int height_full, int width_full, int height_core, int width_core, int left_offset, int top_offset, float *variance_delta)
{
    int i,j,h,w;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (h = 0; h < height_core; ++h) {
                for (w = 0; w < width_core; ++w) {
                    int index = j*filters*height_full*width_full + (h+top_offset)*width_full + w + left_offset;
                    variance_delta[i] += delta[index]*(0 - mean[i]);;
                }
            }
        }
        //variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }

    if(ftp_params->is_main_gateway){
        float* variance_delta_temp = calloc(filters, sizeof(float));
        for(i = 1; i < ftp_params->NUM_TILES_X * ftp_params->NUM_TILES_Y; i++){
            receive_data(variance_delta_temp, filters, i);
            for(j = 0; j < filters; j++)
                variance_delta[j] += variance_delta_temp[j];
        }
        for(i = 1; i < ftp_params->NUM_TILES_X * ftp_params->NUM_TILES_Y; i++)
            send_data(variance_delta, filters, i);
        free(variance_delta_temp);
    }
    else{
            send_data(variance_delta, filters, 0);
            receive_data(variance_delta, filters, 0);
    }
    for(i = 0; i < filters; ++i)
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));

}

void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    printf("spatial %d term = %.4f\n", variance_delta[0] * 2. * (0.0 - mean[0]) / (spatial * batch) + mean_delta[0]/(spatial*batch));
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f]/(spatial * batch);// * (0.0 - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void normalize_delta_cpu_dist_ftp(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta,
                                  int batch, int filters, int spatial_core, int spatial_full, float *delta)
{
    printf("full %d core %d term %.4f\n", spatial_full, spatial_core, variance_delta[0] * 2. * (0.0 - mean[0]) / (spatial_core * batch) + mean_delta[0]/(spatial_core*batch));
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial_full; ++k){
                int index = j*filters*spatial_full + f*spatial_full + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f]/(spatial_core * batch);// * 2; //* (0.0 - mean[f]) / (spatial_core * batch) + mean_delta[f]/(spatial_core*batch);
            }
        }
    }
}

void resize_batchnorm_layer(layer *layer, int w, int h)
{
    fprintf(stderr, "Not implemented\n");
}

void forward_batchnorm_layer(layer l, network net)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
    if(net.train){
        mean_cpu(l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu(l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);

        scal_cpu(l.out_c, .99, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);

        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);   
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void forward_batchnorm_layer_dist_ftp(ftp_config* ftp_params, layer l, network net)
{
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    copy_cpu(l.outputs*l.batch, l.output, 1, l.x, 1);
    
    //int left_edges = (net->index == net->n - 1) ?  0 : l.left_boundry_edges_featuremap;
    if(net.train){
        //no boundary version
        mean_cpu_dist_ftp(ftp_params, l.output, l.batch, l.out_c, l.out_h*l.out_w, l.mean);
        variance_cpu_dist_ftp(ftp_params, l.output, l.mean, l.batch, l.out_c, l.out_h*l.out_w, l.variance);
     
        scal_cpu(l.out_c, .99, l.rolling_mean, 1);
        axpy_cpu(l.out_c, .01, l.mean, 1, l.rolling_mean, 1);
        scal_cpu(l.out_c, .99, l.rolling_variance, 1);
        axpy_cpu(l.out_c, .01, l.variance, 1, l.rolling_variance, 1);
        //boundary version
        normalize_cpu(l.output, l.mean, l.variance, l.batch, l.out_c, l.out_h*l.out_w);
        copy_cpu(l.outputs*l.batch, l.output, 1, l.x_norm, 1);
    } else {
        normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.out_c, l.out_h*l.out_w);
    }
    scale_bias(l.output, l.scales, l.batch, l.out_c, l.out_h*l.out_w);
    add_bias(l.output, l.biases, l.batch, l.out_c, l.out_h*l.out_w);
}

void backward_batchnorm_layer(layer l, network net)
{
    if(!net.train){
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates);

    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.out_h*l.out_w);

    mean_delta_cpu(l.delta, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta);
    variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta);
    normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c, l.out_w*l.out_h, l.delta);
    int i;
    for(i=0; i<l.out_c; i++)
    printf("i = %d %.4f %.4f %.4f %.4f\n", i, l.variance[i], l.mean[i], l.variance_delta[i], l.mean_delta[i]);
    printf("\n\n");
    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

void backward_batchnorm_layer_dist_ftp(ftp_config* ftp_params, layer l, network net)
{
    if(!net.train){
        l.mean = l.rolling_mean;
        l.variance = l.rolling_variance;
    }
    
    int left_offset = l.left_boundry_edges_delta;
    int right_offset = l.right_boundry_edges_delta;
    int bottom_offset = l.bottom_boundry_edges_delta;
    int top_offset = l.top_boundry_edges_delta;
    //no boundary version delta
    backward_bias(l.bias_updates, l.delta, l.batch, l.out_c, l.delta_in_h_without_boundry*l.delta_in_w_without_boundry);
    backward_scale_cpu(l.x_norm, l.delta, l.batch, l.out_c, l.delta_in_h_without_boundry*l.delta_in_w_without_boundry, l.scale_updates);
    //boundary version 
    scale_bias(l.delta, l.scales, l.batch, l.out_c, l.delta_in_h_with_boundry*l.delta_in_w_with_boundry);
    //no boundary version
    mean_delta_cpu_dist_ftp(ftp_params, l.delta, l.variance, l.batch, l.out_c, l.delta_in_h_with_boundry, l.delta_in_w_with_boundry,
                            l.delta_in_h_without_boundry, l.delta_in_w_without_boundry, left_offset, top_offset, l.mean_delta);

    variance_delta_cpu_dist_ftp(ftp_params, l.x, l.delta, l.mean, l.variance, l.batch, l.out_c, l.delta_in_h_with_boundry, l.delta_in_w_with_boundry,
                                l.delta_in_h_without_boundry, l.delta_in_w_without_boundry, left_offset, top_offset, l.variance_delta);
    //boundary version
    int total_tiles = ftp_params->NUM_TILES_X * ftp_params->NUM_TILES_Y;
    int spatial_core = l.delta_in_h_without_boundry*l.delta_in_w_without_boundry*total_tiles;
    int spatial_full = l.delta_in_h_with_boundry*l.delta_in_w_with_boundry*total_tiles;
    normalize_delta_cpu_dist_ftp(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.out_c,
                                 spatial_core, l.delta_in_h_with_boundry*l.delta_in_w_with_boundry, l.delta);
    int i;
    for(i=0; i<l.out_c; i++)
    printf("i = %d %.4f %.4f %.4f %.4f\n", i, l.variance[i], l.mean[i], l.variance_delta[i], l.mean_delta[i]);
    printf("\n\n");

    if(l.type == BATCHNORM) copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_batchnorm_layer(layer l)
{
    cuda_pull_array(l.scales_gpu, l.scales, l.c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}
void push_batchnorm_layer(layer l)
{
    cuda_push_array(l.scales_gpu, l.scales, l.c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.c);
}

void forward_batchnorm_layer_gpu(layer l, network net)
{
    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
    if (net.train) {
#ifdef CUDNN
        float one = 1;
        float zero = 0;
        cudnnBatchNormalizationForwardTraining(cudnn_handle(),
                CUDNN_BATCHNORM_SPATIAL,
                &one,
                &zero,
                l.dstTensorDesc,
                l.x_gpu,
                l.dstTensorDesc,
                l.output_gpu,
                l.normTensorDesc,
                l.scales_gpu,
                l.biases_gpu,
                .01,
                l.rolling_mean_gpu,
                l.rolling_variance_gpu,
                .00001,
                l.mean_gpu,
                l.variance_gpu);
#else
        fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.mean_gpu);
        fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h*l.out_w, l.variance_gpu);

        scal_gpu(l.out_c, .99, l.rolling_mean_gpu, 1);
        axpy_gpu(l.out_c, .01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
        scal_gpu(l.out_c, .99, l.rolling_variance_gpu, 1);
        axpy_gpu(l.out_c, .01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);

        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_gpu, 1);
        normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        copy_gpu(l.outputs*l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);

        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
#endif
    } else {
        normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    }

}

void backward_batchnorm_layer_gpu(layer l, network net)
{
    if(!net.train){
        l.mean_gpu = l.rolling_mean_gpu;
        l.variance_gpu = l.rolling_variance_gpu;
    }
#ifdef CUDNN
    float one = 1;
    float zero = 0;
    cudnnBatchNormalizationBackward(cudnn_handle(),
            CUDNN_BATCHNORM_SPATIAL,
            &one,
            &zero,
            &one,
            &one,
            l.dstTensorDesc,
            l.x_gpu,
            l.dstTensorDesc,
            l.delta_gpu,
            l.dstTensorDesc,
            l.x_norm_gpu,
            l.normTensorDesc,
            l.scales_gpu,
            l.scale_updates_gpu,
            l.bias_updates_gpu,
            .00001,
            l.mean_gpu,
            l.variance_gpu);
    copy_gpu(l.outputs*l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
#else
    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h);
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.scale_updates_gpu);

    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h*l.out_w);

    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w*l.out_h, l.delta_gpu);
#endif
    if(l.type == BATCHNORM) copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif

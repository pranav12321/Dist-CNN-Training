#include "darknet.h"
#include "convolutional_layer.h"


void* execute_dev(void* ptr);
void execute_dev_v2(void* ptr, int current_layer_idx);
void* execute_dev_gateway(void* ptr);

typedef struct device_ftp_args{
    int NUM_DEVICES;
    int device_id;
    int im_width;
    int im_height;
    int filter_size;
    float* image;
    float* delta;
    float* boundry_top;
    float* boundry_bottom;
    float* boundry_left;
    float* boundry_right;
    float* boundry_top_right;
    float* boundry_top_left;
    float* boundry_bottom_right;
    float* boundry_bottom_left;
    int num_layers;
    float* output;
    float*** SHARED_WEIGHT_UPDATES;  
} device_ftp_args;

typedef struct device_ftp_args_v2{
    int NUM_TILES_X;
    int NUM_TILES_Y;

    int DEVICE_ID_X;
    int DEVICE_ID_Y;

    int INPUT_WIDTH;
    int INPUT_HEIGHT;

    network* net;
    int filter_size;
    float* image_input;
    float* delta;
    int num_layers;
    float* output;
    network*** SHARED_NETWORKS;
    float*** SHARED_INPUT_IMAGES;
    float*** SHARED_EXP_DELTA;
    float*** SHARED_WEIGHT_UPDATES; 
    float** SHARED_DELTA_ALIGNMENT_BOTTOM; 
    float** SHARED_DELTA_ALIGNMENT_RIGHT;
    float** SHARED_DELTA_ALIGNMENT_BOTTOM_RIGHT;

} device_ftp_args_v2;
#ifndef FUSED_CONVOLUTION
#define FUSED_CONVOLUTION
#include "darknet.h"
#include "convolutional_layer.h"
#include "fused_device.h"
#include "ftp.h"

void send_forward_group_boundry_data_device(network* net, float* INPUT_IMAGE,
    int NUM_TILES_X, int NUM_TILES_Y,
    int current_layer_idx, 
    int device_id_x, int device_id_y);

void assemble_forward_group_data_device(network* net, 
                                float* INPUT_IMAGE,
                                int NUM_TILES_X, int NUM_TILES_Y,
                                 group_profile_forward group,
                                 int DEVICE_ID_X, int DEVICE_ID_Y
                                 );

void send_backward_group_boundry_data_device(network* net, float* OUTPUT_DELTA,
    int NUM_TILES_X, int NUM_TILES_Y,
    int current_layer_idx, int num_layers,
    int device_id_x, int device_id_y);

void assemble_backward_group_data_device(network* net, 
                                float* OUTPUT_DELTA,
                                int NUM_TILES_X, int NUM_TILES_Y,
                                 group_profile_backward group,
                                 int DEVICE_ID_X, int DEVICE_ID_Y,
                                 int num_layers
                                 );

void zero_out_edges_featuremap_device(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int DEVICE_ID_Y, int DEVICE_ID_X);
void zero_out_edges_delta_device(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int DEVICE_ID_Y, int DEVICE_ID_X);
#endif
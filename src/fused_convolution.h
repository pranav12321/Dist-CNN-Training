#ifndef FUSED_CONVOLUTION
#define FUSED_CONVOLUTION
#include "darknet.h"
#include "convolutional_layer.h"
#include "fused.h"

void execute_forward();

void execute_backward();

void assemble_forward_group_data(network*** SHARED_NETWORKS, 
                                float***SHARED_INPUT_IMAGES,
                                int NUM_TILES_X, int NUM_TILES_Y,
								 group_profile_forward group,
                                 int DEVICE_ID_X, int DEVICE_ID_Y
								 );

void assemble_backward_group_data(network*** SHARED_NETWORKS, 
                                float***SHARED_EXP_DELTAS,
                                int NUM_TILES_X, int NUM_TILES_Y,
                                 group_profile_backward group,
                                 int DEVICE_ID_X, int DEVICE_ID_Y,
                                 int num_layers
                                 );

void zero_out_edges_featuremap(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int DEVICE_ID_Y, int DEVICE_ID_X);
void zero_out_edges_delta(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int DEVICE_ID_Y, int DEVICE_ID_X);
#endif
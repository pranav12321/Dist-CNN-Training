#ifndef FUSED_CONVOLUTION_DEVICE
#define FUSED_CONVOLUTION_DEVICE
#include "darknet.h"
#include "convolutional_layer.h"
#include "fused_device.h"

void assemble_tile(network* net, int batch, int depth,
                   float* target, float* core_tile_data,
                   int core_tile_height, int core_tile_width,
                   int left_boundry_edges, int right_boundry_edges, int top_boundry_edges, int bottom_boundry_edges,
                   int device_id_x, int device_id_y, int NUM_TILES_X, int NUM_TILES_Y);

void clear_edges_featuremap_device(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int DEVICE_ID_Y, int DEVICE_ID_X);
void clear_edges_delta_device(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int DEVICE_ID_Y, int DEVICE_ID_X);

void clear_spurious_edges_featuremap(network* net, int layer_idx);
void clear_spurious_edges_delta(network* net, int layer_idx);

void receive_sum_broadcast_weight_updates(network* net, int NUM_TILES_Y, int NUM_TILES_X);
void sync_weight_updates(network* net, int NUM_TILES_Y, int NUM_TILES_X);
void receive_sum_transmit_device_weight_updates(network* net, int NUM_TILES_Y, int NUM_TILES_X);
void devices_send_partial_weight_updates(network* net, int NUM_TILES_Y, int NUM_TILES_X);

void pos_correct_maxpool_indices(int* data, int stride,
                                int width_pool_in_forward,
                                int width_pool_out_backward, int height_pool_out_backward, int depth_pool_out_backward, int batch);


void copy_slice(float* dst, float* src, int batch, int depth,
                int height_src, int width_src, int height_dst, int width_dst,
                int src_start_x, int src_start_y, int dst_start_x, int dst_start_y,
                int copy_height_src, int copy_width_src, int copy_height_dst, int copy_width_dst,
                float* workspace);

void assemble_tile(network* net, int batch, int depth,
                   float* target, float* core_tile_data,
                   int core_tile_height, int core_tile_width,
                   int left_boundry_edges, int right_boundry_edges, int top_boundry_edges, int bottom_boundry_edges,
                   int device_id_x, int device_id_y, int NUM_TILES_X, int NUM_TILES_Y);

void assemble_pool_indices(network* net, int l, int device_id_x, int device_id_y, int NUM_TILES_X, int NUM_TILES_Y);
void remove_extra_boundary_data(network* net, int l);


#endif
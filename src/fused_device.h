#ifndef FUSED_DEVICE
#define FUSED_DEVICE
#include "darknet.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"

void compute_tile_boundries(network* net,
                            int DEVICE_ID_X, int DEVICE_ID_Y,
                            network *** SHARED_NETWORKS,
                            float* COMBINED_INPUT_IMAGES, float* COMBINED_EXP_DELTAS,
                          int start_y_forward, int start_x_forward,
                          int end_y_forward, int end_x_forward,
                          int start_y_backward, int start_x_backward,
                          int end_y_backward, int end_x_backward);

typedef struct group_profile_forward{
    int layer_start_idx;
    int layer_end_idx;
    int start_x_forward;
    int start_y_forward;
    int end_x_forward;
    int end_y_forward;  
} group_profile_forward;

typedef struct group_profile_backward{
    int layer_start_idx;
    int layer_end_idx;
    int start_x_backward;
    int start_y_backward;
    int end_x_backward;
    int end_y_backward;
} group_profile_backward;

typedef struct train_groups_profile{
    int num_forward_groups;
    int num_backward_groups;
    group_profile_forward* fp;
    group_profile_backward* bp;
} train_groups_profile;

#define MAX_LAYERS 28

typedef struct dim{
    int x_dim;
    int y_dim;
    int depth;
} dim;

typedef struct coordinate_bounds{
    int start_x_coordinate;
    int start_y_coordinate;
    int end_x_coordinate;
    int end_y_coordinate;
    int start_depth_coordinate;
    int end_depth_coordinate;
} coordinate_bounds;

typedef struct edges{
    int left_boundry_edges;
    int right_boundry_edges;
    int top_boundry_edges;
    int bottom_boundry_edges;
} edges;

typedef struct network_config{
    int num_layers;
    int INPUT_WIDTH;
    int INPUT_HEIGHT;
    int stride_vector[MAX_LAYERS];
    int filter_size_vector[MAX_LAYERS];
    int filter_stack_vector[MAX_LAYERS];
    int back_pad[MAX_LAYERS];

    LAYER_TYPE layer_type_vector[MAX_LAYERS];

    dim featuremap_dim_without_boundry_vector[MAX_LAYERS];
    dim delta_dim_without_boundry_vector[MAX_LAYERS];
    dim featuremap_dim_with_boundry_vector[MAX_LAYERS];
    dim delta_dim_with_boundry_vector[MAX_LAYERS];
    coordinate_bounds featuremap_bounds_vector[MAX_LAYERS];
    coordinate_bounds delta_bounds_vector[MAX_LAYERS];
    edges featuremap_edges_vector[MAX_LAYERS];
    edges delta_edges_vector[MAX_LAYERS];

    coordinate_bounds spurious_blocks[MAX_LAYERS+1];
} network_config;

typedef struct ftp_config{
    int NUM_TILES_X;
    int NUM_TILES_Y;
    int NUM_GROUPS_FORWARD;
    int NUM_GROUPS_BACKWARD;
    int DEVICE_ID_X;
    int DEVICE_ID_Y;
    int sync_group_vector_forward[MAX_LAYERS];
    int sync_group_vector_backward[MAX_LAYERS];
} ftp_config;


#endif
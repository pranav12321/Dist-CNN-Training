#ifndef FUSED
#define FUSED
#include "darknet.h"
#include "convolutional_layer.h"

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

#endif
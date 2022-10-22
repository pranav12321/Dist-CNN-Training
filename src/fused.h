#ifndef FUSED
#define FUSED
#include "darknet.h"
void compute_input_layers();

void compute_tile_boundries(network* net,
                            int DEVICE_ID_X, int DEVICE_ID_Y,
                            network *** SHARED_NETWORKS,
                            float* COMBINED_INPUT_IMAGES, float* COMBINED_EXP_DELTAS,
                          int start_y_forward, int start_x_forward,
                          int end_y_forward, int end_x_forward,
                          int start_y_backward, int start_x_backward,
                          int end_y_backward, int end_x_backward);

#endif
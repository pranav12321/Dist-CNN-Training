#include "darknet.h"
#include "fused_device.h"

extern int stride_vector[8];
extern int filter_stack_vector[8];
extern int filter_size_vector[8];


void partition_forward_device(network* net,
                        train_groups_profile* profile,
                       group_profile_forward* group, 
                        int start_x_forward, int start_y_forward,
                        int end_x_forward, int end_y_forward){

    int num_layers = net->n;

    int left_boundry_edges = 0;
    int top_boundry_edges = 0;

    int right_boundry_edges = 0;
    int bottom_boundry_edges = 0;

    int start_x_coordinate = start_x_forward;
    int start_y_coordinate = start_y_forward;
    int end_x_coordinate = end_x_forward;
    int end_y_coordinate = end_y_forward;


    for (int j = 0; j < net->n; ++j)
    {
        net->layers[j].stride = stride_vector[j];
    }

    for (int j = 0; j < net->n; ++j)
    {
        net->layers[j].size = filter_size_vector[j];
    }


    for (int i = group->layer_end_idx; i >= group->layer_start_idx; i--)
    {
        int unit_boundry = ((filter_size_vector[i] & 0x1) == 1) ? ((filter_size_vector[i] - 1)/2) : (filter_size_vector[i]/2);
        int boundry_frames = unit_boundry;

        int stride = net->layers[i].stride;
        int filter_size = net->layers[i].size;

        int next_layer_left_edges;
        int next_layer_right_edges;

        if(i == (group->layer_end_idx)){
            next_layer_left_edges = 0;
            next_layer_right_edges = 0;
        }
        else{
            next_layer_left_edges = net->layers[i+1].left_boundry_edges_featuremap;
            next_layer_right_edges = net->layers[i+1].right_boundry_edges_featuremap;
        }

        left_boundry_edges = unit_boundry + (next_layer_left_edges*stride);
        top_boundry_edges = left_boundry_edges;

        right_boundry_edges = unit_boundry + (next_layer_right_edges*stride);;
        bottom_boundry_edges = right_boundry_edges;


        start_x_coordinate = (start_x_coordinate*stride) - unit_boundry;
        start_y_coordinate = (start_y_coordinate*stride) - unit_boundry;

        end_x_coordinate = (end_x_coordinate*stride) + unit_boundry + stride - 1;
        end_y_coordinate = (end_y_coordinate*stride) + unit_boundry + stride - 1;

        int featuremap_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1; 
        int featuremap_in_h_without_boundry = featuremap_in_h_with_boundry - (top_boundry_edges + bottom_boundry_edges);

        int featuremap_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1;
        int featuremap_in_w_without_boundry = featuremap_in_w_with_boundry - (left_boundry_edges + right_boundry_edges);

        int num_channels = ((i == 0) ? 3 : filter_stack_vector[i-1]);
        net->layers[i] = make_convolutional_layer(1, featuremap_in_h_with_boundry, featuremap_in_w_with_boundry, num_channels, filter_stack_vector[i], 1, filter_size_vector[i], stride, 0, RELU, 0, 0, 0, 0);

        for (int i_f = 0; i_f < filter_size*filter_size*net->layers[i].c*net->layers[i].n; ++i_f)
        {
                net->layers[i].weights[i_f] = 0.01;
        }

        net->layers[i].left_boundry_edges_featuremap = left_boundry_edges;
        net->layers[i].top_boundry_edges_featuremap = top_boundry_edges;
        net->layers[i].right_boundry_edges_featuremap = right_boundry_edges;
        net->layers[i].bottom_boundry_edges_featuremap = bottom_boundry_edges;

        net->layers[i].featuremap_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1; 
        net->layers[i].featuremap_in_h_without_boundry = net->layers[i].featuremap_in_h_with_boundry - (net->layers[i].top_boundry_edges_featuremap + net->layers[i].bottom_boundry_edges_featuremap);
        net->layers[i].original_featuremap_in_h = net->layers[i].featuremap_in_h_without_boundry;

        net->layers[i].featuremap_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1;
        net->layers[i].featuremap_in_w_without_boundry = net->layers[i].featuremap_in_w_with_boundry - (net->layers[i].left_boundry_edges_featuremap + net->layers[i].right_boundry_edges_featuremap);
        net->layers[i].original_featuremap_in_w = net->layers[i].featuremap_in_w_without_boundry;

        printf("Layer %d\n\n", i);
        printf("FEATUREMAP H with boundry/without boundry = %d %d\n", net->layers[i].featuremap_in_h_with_boundry, net->layers[i].featuremap_in_h_without_boundry);
        printf("FEATUREMAP W with boundry/without boundry = %d %d\n", net->layers[i].featuremap_in_w_with_boundry, net->layers[i].featuremap_in_w_without_boundry);

        printf("Top boundry edges = %d\n", top_boundry_edges);
        printf("Left boundry edges = %d\n", left_boundry_edges);
        printf("Right boundry edges = %d\n", right_boundry_edges);
        printf("Bottom boundry edges = %d\n\n", bottom_boundry_edges);
        printf("Start x coordinate = %d\n", start_x_coordinate);
        printf("Start y coordinte = %d\n", start_y_coordinate);
        printf("End x coordinate = %d\n", end_x_coordinate);
        printf("End y coordinate = %d\n\n", end_y_coordinate);
        
    }
}

void partition_backward_device(network* net, 
                        group_profile_backward* profile,
                        int start_x_backward, int start_y_backward,
                        int end_x_backward, int end_y_backward){

    int left_boundry_edges = 0;
    int top_boundry_edges = 0;

    int right_boundry_edges = 0;
    int bottom_boundry_edges = 0;
    
    int start_x_coordinate = start_x_backward;
    int start_y_coordinate = start_y_backward;
    int end_x_coordinate = end_x_backward;
    int end_y_coordinate = end_y_backward;

    int num_layers = net->n;

    int start_idx = 0;
    if(profile->layer_start_idx == 0)
        start_idx = 0;
    else
        start_idx = (profile->layer_start_idx - 1);
    for (int i = start_idx; i <= (profile->layer_end_idx); ++i)
    {
        int filter_size = net->layers[i].size;
        int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
        int stride = net->layers[i].stride;

        if(i == start_idx){
            left_boundry_edges = 0;
            top_boundry_edges = 0;
            right_boundry_edges = 0;
            bottom_boundry_edges = 0;
        }

        else{

            left_boundry_edges = (unit_boundry + net->layers[i-1].left_boundry_edges_delta) /(net->layers[i].stride);
            top_boundry_edges = (unit_boundry + net->layers[i-1].top_boundry_edges_delta) /(net->layers[i].stride);

            right_boundry_edges = (unit_boundry + net->layers[i-1].right_boundry_edges_delta + net->layers[i].stride - 1) /(net->layers[i].stride);
            bottom_boundry_edges = (unit_boundry + net->layers[i-1].bottom_boundry_edges_delta + net->layers[i].stride - 1) /(net->layers[i].stride);
        }

        if(i>start_idx){

            int prev_stride = net->layers[i-1].stride;

            if((start_x_coordinate - unit_boundry) > 0){
                start_x_coordinate = (start_x_coordinate - unit_boundry + (stride - 1))/stride;
            }
            else{
                start_x_coordinate = (start_x_coordinate - unit_boundry)/stride;         
            }

            if((end_x_coordinate + unit_boundry) < 0){
                end_x_coordinate = (end_x_coordinate + unit_boundry - (stride - 1))/stride;
            }
            else{
                end_x_coordinate = (end_x_coordinate + unit_boundry)/stride;          
            }


            if((start_y_coordinate - unit_boundry) > 0){
                start_y_coordinate = (start_y_coordinate - unit_boundry + (stride - 1))/stride;
            }
            else{
                start_y_coordinate = (start_y_coordinate - unit_boundry)/stride;          
            }

            if((end_y_coordinate + unit_boundry) < 0){
                end_y_coordinate = (end_y_coordinate + unit_boundry - (stride - 1))/stride;
            }
            else{
                end_y_coordinate = (end_y_coordinate + unit_boundry)/stride;          
            }
        }

        net->layers[i].left_boundry_edges_delta = left_boundry_edges;
        net->layers[i].top_boundry_edges_delta = top_boundry_edges;
        net->layers[i].right_boundry_edges_delta = right_boundry_edges;
        net->layers[i].bottom_boundry_edges_delta = bottom_boundry_edges;

        net->layers[i].delta_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1; 
        net->layers[i].delta_in_h_without_boundry = net->layers[i].delta_in_h_with_boundry - (net->layers[i].top_boundry_edges_delta + net->layers[i].bottom_boundry_edges_delta);

        net->layers[i].delta_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1;
        net->layers[i].delta_in_w_without_boundry = net->layers[i].delta_in_w_with_boundry - (net->layers[i].left_boundry_edges_delta + net->layers[i].right_boundry_edges_delta);

        printf("Layer %d \n\n", i);
        printf("DELTA H with boundry/without boundry = %d %d\n", net->layers[i].delta_in_h_with_boundry, net->layers[i].delta_in_h_without_boundry);
        printf("DELTA W with boundry/without boundry = %d %d\n", net->layers[i].delta_in_w_with_boundry, net->layers[i].delta_in_w_without_boundry);
        printf("Top boundry edges = %d\n", top_boundry_edges);
        printf("Left boundry edges = %d\n", left_boundry_edges);
        printf("Right boundry edges = %d\n", right_boundry_edges);
        printf("Bottom boundry edges = %d\n\n", bottom_boundry_edges);
        printf("Start x coordinate = %d\n", start_x_coordinate);
        printf("Start y coordinte = %d\n", start_y_coordinate);
        printf("End x coordinate = %d\n", end_x_coordinate);
        printf("End y coordinate = %d\n\n", end_y_coordinate);
    }

}
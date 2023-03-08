#include "ftp.h"
#include "darknet.h"
#include "fused.h"


void partition_forward(network* net,
                        train_groups_profile* profile,
                       group_profile_forward* group, 
                        int DEVICE_ID_X, int DEVICE_ID_Y,
                        network*** SHARED_NETWORKS,
                        //float* COMBINED_INPUT_IMAGES,
                        int filter_size,
                        int start_x_forward, int start_y_forward,
                        int end_x_forward, int end_y_forward){

    int num_layers = net->n;

    int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);

    int left_boundry_edges = 0;
    int top_boundry_edges = 0;

    int right_boundry_edges = 0;
    int bottom_boundry_edges = 0;

    int boundry_frames = unit_boundry;

    int start_x_coordinate = start_x_forward;
    int start_y_coordinate = start_y_forward;
    int end_x_coordinate = end_x_forward;
    int end_y_coordinate = end_y_forward;


    for (int i = group->layer_end_idx; i >= group->layer_start_idx; i--)
    {

        int stride = net->layers[i].stride;

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

        net->layers[i] = make_convolutional_layer(1, featuremap_in_h_with_boundry, featuremap_in_w_with_boundry, 1, 1, 1, filter_size, stride, 0, RELU, 0, 0, 0, 0);

        for (int i_f = 0; i_f < filter_size; ++i_f)
        {
            for (int j_f = 0; j_f < filter_size; ++j_f)
            {
                net->layers[i].weights[i_f*filter_size + j_f] = 0.1;
            }
        }

        net->layers[0].stride = 1;
        net->layers[1].stride = 1;
        net->layers[2].stride = 2;
        net->layers[3].stride = 1;
        net->layers[4].stride = 1;
        net->layers[5].stride = 2;
        net->layers[6].stride = 1;
        net->layers[7].stride = 1;

        net->layers[profile->fp[0].layer_start_idx].original_featuremap_in_h = 12;
        net->layers[profile->fp[0].layer_start_idx].original_featuremap_in_w = 12;

        net->layers[profile->fp[1].layer_start_idx].original_featuremap_in_h = 6;
        net->layers[profile->fp[1].layer_start_idx].original_featuremap_in_w = 6;

        net->layers[profile->fp[2].layer_start_idx].original_featuremap_in_h = 3;
        net->layers[profile->fp[2].layer_start_idx].original_featuremap_in_w = 3;

        net->layers[i].left_boundry_edges_featuremap = left_boundry_edges;
        net->layers[i].top_boundry_edges_featuremap = top_boundry_edges;
        net->layers[i].right_boundry_edges_featuremap = right_boundry_edges;
        net->layers[i].bottom_boundry_edges_featuremap = bottom_boundry_edges;

        net->layers[i].featuremap_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1; 
        net->layers[i].featuremap_in_h_without_boundry = net->layers[i].featuremap_in_h_with_boundry - (net->layers[i].top_boundry_edges_featuremap + net->layers[i].bottom_boundry_edges_featuremap);

        net->layers[i].featuremap_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1;
        net->layers[i].featuremap_in_w_without_boundry = net->layers[i].featuremap_in_w_with_boundry - (net->layers[i].left_boundry_edges_featuremap + net->layers[i].right_boundry_edges_featuremap);


        //TODO: Put this into separate variable
        if(i==group->layer_start_idx){
            right_boundry_edges = (net->layers[i].featuremap_in_w_with_boundry - net->layers[i].original_featuremap_in_w - left_boundry_edges);
            bottom_boundry_edges = (net->layers[i].featuremap_in_h_with_boundry - net->layers[i].original_featuremap_in_h - top_boundry_edges);
            net->layers[i].right_boundry_edges_featuremap = right_boundry_edges;
            net->layers[i].bottom_boundry_edges_featuremap = bottom_boundry_edges;
        }


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

    // for (int i = start_y_coordinate; i <= end_y_coordinate; ++i)
    // {
    //     for (int j = start_x_coordinate; j <= end_x_coordinate; ++j)
    //     {
    //         int pos = (i - start_y_coordinate)*(end_x_coordinate - start_x_coordinate + 1) + j - start_x_coordinate;
    //         if (i < 0 || i >= 12 || j < 0 || j >= 12)
    //         {
    //             net->input[pos] = 0.0;
    //         }
    //         else{
    //             net->input[pos] = COMBINED_INPUT_IMAGES[i*12  + j];                
    //         }
    //     }
    // }

}

void partition_backward(network* net, 
                        int DEVICE_ID_X, int DEVICE_ID_Y,
                        network *** SHARED_NETWORKS,
                        float*** SHARED_EXP_DELTAS,
                        int filter_size,
                        group_profile_backward* profile,
                        int start_x_backward, int start_y_backward,
                        int end_x_backward, int end_y_backward){

    int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
    int delta_boundry_edges_vertical = net->layers[0].out_h + unit_boundry;
    int delta_boundry_edges_horizontal = net->layers[0].out_w + unit_boundry;

    int left_boundry_edges = 0;
    int top_boundry_edges = 0;

    int right_boundry_edges = 0;
    int bottom_boundry_edges = 0;

    int boundry_frames = unit_boundry;

    int start_x_coordinate = start_x_backward;
    int start_y_coordinate = start_y_backward;
    int end_x_coordinate = end_x_backward;
    int end_y_coordinate = end_y_backward;

    int num_layers = net->n;

    // net->layers[0].left_boundry_edges_delta = 0;
    // net->layers[0].top_boundry_edges_delta = 0;
    // net->layers[0].right_boundry_edges_delta = 0;
    // net->layers[0].bottom_boundry_edges_delta = 0;

    // // // net->layers[0].delta_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1;
    // // // net->layers[0].delta_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1;

    // net->layers[0].delta_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1; 
    // net->layers[0].delta_in_h_without_boundry = net->layers[0].delta_in_h_with_boundry;

    // net->layers[0].delta_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1;
    // net->layers[0].delta_in_w_without_boundry = net->layers[0].delta_in_w_with_boundry;

    // printf("DELTA H with boundry/without boundry = %d %d\n", net->layers[0].delta_in_h_with_boundry, net->layers[0].delta_in_h_without_boundry);
    // printf("DELTA W with boundry/without boundry = %d %d\n", net->layers[0].delta_in_w_with_boundry, net->layers[0].delta_in_w_without_boundry);

    int start_idx = 0;
    if(profile->layer_start_idx == 0)
        start_idx = 0;
    else
        start_idx = (profile->layer_start_idx - 1);
    for (int i = start_idx; i <= (profile->layer_end_idx); ++i)
    {

        int stride = net->layers[i].stride;
        int filter_size = net->layers[i].size;
        unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);

        if(net->layers[i].type == MAXPOOL){
            unit_boundry = 0;
        }

        if(i == start_idx){
            left_boundry_edges = 0;
            top_boundry_edges = 0;
            right_boundry_edges = 0;
            bottom_boundry_edges = 0;
        }

        else{

            if(net->layers[i].type == CONVOLUTIONAL){
                left_boundry_edges = (unit_boundry + net->layers[i-1].left_boundry_edges_delta) /(net->layers[i].stride);
                top_boundry_edges = (unit_boundry + net->layers[i-1].top_boundry_edges_delta) /(net->layers[i].stride);

                right_boundry_edges = (unit_boundry + net->layers[i-1].right_boundry_edges_delta + net->layers[i].stride - 1) /(net->layers[i].stride);
                bottom_boundry_edges = (unit_boundry + net->layers[i-1].bottom_boundry_edges_delta + net->layers[i].stride - 1) /(net->layers[i].stride);
            }
            else if(net->layers[i].type == MAXPOOL){
                left_boundry_edges = net->layers[i-1].left_boundry_edges_delta;
                top_boundry_edges = net->layers[i-1].top_boundry_edges_delta;

                right_boundry_edges = net->layers[i-1].right_boundry_edges_delta;
                bottom_boundry_edges = net->layers[i-1].bottom_boundry_edges_delta;
            }

        }
        // net->layers[i].left_boundry_edges_delta = left_boundry_edges;
        // net->layers[i].top_boundry_edges_delta = top_boundry_edges;
        // net->layers[i].right_boundry_edges_delta = right_boundry_edges;
        // net->layers[i].bottom_boundry_edges_delta = bottom_boundry_edges;

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

    // printf("Top boundry edges = %d\n", top_boundry_edges);
    // printf("Left boundry edges = %d\n", left_boundry_edges);
    // printf("Right boundry edges = %d\n", right_boundry_edges);
    // printf("Bottom boundry edges = %d\n\n", bottom_boundry_edges);


    // printf("Start x coordinate = %d\n", start_x_coordinate);
    // printf("Start y coordinte = %d\n", start_y_coordinate);
    // printf("End x coordinate = %d\n", end_x_coordinate);
    // printf("End y coordinate = %d\n\n", end_y_coordinate);

    // for (int i = start_y_coordinate; i <= end_y_coordinate; ++i)
    // {
    //     for (int j = start_x_coordinate; j <= end_x_coordinate; ++j)
    //     {
    //         int pos = (i - start_y_coordinate)*(end_x_coordinate - start_x_coordinate + 1) + j - start_x_coordinate;
    //         if (i < 0 || i >= 6 || j < 0 || j >= 6)
    //         {
    //             net->layers[3].delta_with_boundry[pos]
    //                                         = 0.0;
    //         }
    //         else{
    //             net->layers[3].delta_with_boundry[pos]
    //                                         = COMBINED_EXP_DELTAS[i*6  + j];                
    //         }
    //     }
    // }
}

void compute_tile_boundries(network* net,
                            int DEVICE_ID_X, int DEVICE_ID_Y,
                            network *** SHARED_NETWORKS,
                            float* COMBINED_INPUT_IMAGES, float* COMBINED_EXP_DELTAS,
                          int start_y_forward, int start_x_forward,
                          int end_y_forward, int end_x_forward,
                          int start_y_backward, int start_x_backward,
                          int end_y_backward, int end_x_backward){
	int num_layers_fuse = 3;
	int num_layers = 4;
	int filter_size = 3;//net->layers[0].size;

    // partition_forward(net, DEVICE_ID_X, DEVICE_ID_Y,
    //                   SHARED_NETWORKS,
    //                   filter_size,
    //                   start_x_forward, start_y_forward,
    //                   end_x_forward, end_y_forward);
    // partition_backward(net, DEVICE_ID_X, DEVICE_ID_Y,
    //                    SHARED_NETWORKS,
    //                    COMBINED_EXP_DELTAS,
    //                    filter_size,
    //                    start_x_backward, start_y_backward,
    //                    end_x_backward, end_y_backward);
}


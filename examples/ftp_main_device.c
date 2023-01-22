#include "fused_device.h"
#include "fused_convolution_device.h"

#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"
#include "maxpool_layer.h"

#include "transport.h"


//12 12 12 6 6 6 3 3 
//1  1  2  1 1 2 1 1


int main_device(){

    int NUM_TILES_X = 2;
    int NUM_TILES_Y = 2;
    int INPUT_WIDTH = 12;
    int INPUT_HEIGHT = 12;
    int INPUT_CHANNELS = 2;


    train_groups_profile profile;

    #ifdef SERVER
        init_server();
    #else
        init_transport();
    #endif

    profile.num_forward_groups = 1;
    profile.num_backward_groups = 1;

    profile.fp = calloc(profile.num_forward_groups, sizeof(group_profile_forward));
    profile.bp = calloc(profile.num_backward_groups, sizeof(group_profile_backward));

    profile.fp[0].layer_start_idx = 0;
    profile.fp[0].layer_end_idx = 4;
    profile.fp[0].start_x_forward = 0;
    profile.fp[0].start_y_forward = 0;
    profile.fp[0].end_x_forward = 5;
    profile.fp[0].end_y_forward = 5;

    // profile.fp[1].layer_start_idx = 4;
    // profile.fp[1].layer_end_idx = 5;
    // profile.fp[1].start_x_forward = 0;
    // profile.fp[1].start_y_forward = 0;
    // profile.fp[1].end_x_forward = 2;
    // profile.fp[1].end_y_forward = 2;

    // profile.fp[2].layer_start_idx = 6;
    // profile.fp[2].layer_end_idx = 7;
    // profile.fp[2].start_x_forward = 0;
    // profile.fp[2].start_y_forward = 0;
    // profile.fp[2].end_x_forward = 2;
    // profile.fp[2].end_y_forward = 2;

    profile.bp[0].layer_start_idx = 0;
    profile.bp[0].layer_end_idx = 4;
    profile.bp[0].start_x_backward = 0;
    profile.bp[0].start_y_backward = 0;
    profile.bp[0].end_x_backward = 5;
    profile.bp[0].end_y_backward = 5;

    // profile.bp[1].layer_start_idx = 4;
    // profile.bp[1].layer_end_idx = 5;
    // profile.bp[1].start_x_backward = 0;
    // profile.bp[1].start_y_backward = 0;
    // profile.bp[1].end_x_backward = 5;
    // profile.bp[1].end_y_backward = 5;

    // profile.bp[2].layer_start_idx = 6;
    // profile.bp[2].layer_end_idx = 7;
    // profile.bp[2].start_x_backward = 0;
    // profile.bp[2].start_y_backward = 0;
    // profile.bp[2].end_x_backward = 2;
    // profile.bp[2].end_y_backward = 2;



    int filter_size = 3;
    int num_layers = 5;
    int unit_boundry = 1;


    network* net = calloc(1, sizeof(network));

    net->n = 5;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    // for (int l = 0; l < net->n; ++l)
    // {
    //     net->layers[l] = 
    //     //make_convolutional_layer(1, 22, 22, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0);
    // }


    net->layers[0].stride = 1;
    net->layers[1].stride = 1;
    net->layers[2].stride = 1;
    net->layers[3].stride = 1;
    net->layers[4].stride = 1;
    // net->layers[5].stride = 2;
    // net->layers[6].stride = 1;
    // net->layers[7].stride = 1;

    float* INPUT_IMAGE = calloc(INPUT_CHANNELS*(INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), sizeof(float));
    fill_cpu(INPUT_CHANNELS*(INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), 1, INPUT_IMAGE, 1);

    net->workspace = calloc(50000000, sizeof(float));
    net->inputs = 10*INPUT_CHANNELS*(INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y);
    //net->input = calloc(484, sizeof(float));


    for (int g = 0; g < profile.num_forward_groups; ++g)
    {

        partition_forward_device(net, 
                          &profile,
                          &(profile.fp[g]),
                            3,
                            profile.fp[g].start_x_forward, profile.fp[g].start_y_forward,
                            profile.fp[g].end_x_forward, profile.fp[g].end_y_forward);

        net->input = calloc(net->inputs, sizeof(float));

        //get boundry data here
        assemble_forward_group_data_device(net, 
                                        INPUT_IMAGE,
                                        NUM_TILES_X, NUM_TILES_Y,
                                         profile.fp[g],
                                         DEVICE_ID_X, DEVICE_ID_Y
                                         );


        for (int l = profile.fp[g].layer_start_idx; l <= profile.fp[g].layer_end_idx; ++l)
        {

            zero_out_edges_featuremap_device(net, l, NUM_TILES_Y, NUM_TILES_X, DEVICE_ID_Y, DEVICE_ID_X);
            net->index = l;
            
            forward_convolutional_layer(net->layers[l], *net);



            for (int i_s = 0; i_s < net->layers[l].out_h; ++i_s)
            {
                for (int j_s = 0; j_s < net->layers[l].out_w; ++j_s)
                {
                    printf("%.2f ", net->layers[l].output[(i_s*net->layers[l].out_w) + j_s]);
                }
                printf("\n");
            }

            printf("\n");


            for (int i_s = 0; i_s < net->layers[l].out_h; ++i_s)
            {
                for (int j_s = 0; j_s < net->layers[l].out_w; ++j_s)
                {
                    printf("%.2f ", net->layers[l].output[36 + (i_s*net->layers[l].out_w) + j_s]);
                }
                printf("\n");
            }

            printf("\n");



            net->input = net->layers[l].output;
        }
        int last_layer = profile.fp[g].layer_end_idx;
        for (int i_s = 0; i_s < net->layers[last_layer].out_h; ++i_s)
        {
            for (int j_s = 0; j_s < net->layers[last_layer].out_w; ++j_s)
            {
                net->layers[last_layer].output_without_boundry[(i_s*net->layers[last_layer].out_w) + j_s] = net->layers[last_layer].output[(i_s*net->layers[last_layer].out_w) + j_s];
            }
        }
    }

    printf("Inference complete\n");
    //while(1);

    float* OUTPUT_DELTA = calloc(net->layers[net->n - 1].outputs, sizeof(float));
    fill_cpu(net->layers[net->n - 1].outputs, 1, OUTPUT_DELTA, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 1, net->layers[net->n - 1].delta_with_boundry, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 1, net->layers[net->n - 1].delta, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 1, net->layers[net->n - 1].delta_without_boundry, 1);

            // for (int m = 0; m < net->layers[7].out_h; ++m)
            // {
            //     for (int n = 0; n < net->layers[7].out_w; ++n)
            //     {
            //         printf("%.2f ", net->layers[7].delta_with_boundry[m*net->layers[7].out_w + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");
            // while(1);


    for (int g = (profile.num_backward_groups-1); g >= 0; --g)
    {

        partition_backward_device(net, 
                            filter_size,
                            &profile.bp[g],
                            profile.bp[g].start_x_backward, profile.bp[g].start_y_backward,
                            profile.bp[g].end_x_backward, profile.bp[g].end_y_backward);

        assemble_backward_group_data_device(net, 
                                        OUTPUT_DELTA,
                                        NUM_TILES_X, NUM_TILES_Y,
                                         profile.bp[g],
                                         DEVICE_ID_X, DEVICE_ID_Y,
                                         net->n
                                         );

        
        int start_idx = (g==0) ? 1 : profile.bp[g].layer_start_idx;
        for (int l = profile.bp[g].layer_end_idx; l >= start_idx; --l)
        {



            int featuremap_without_boundry_width = net->layers[l].featuremap_in_w_without_boundry + (2*unit_boundry);
            int featuremap_without_boundry_height = net->layers[l].featuremap_in_h_without_boundry + (2*unit_boundry);
            

            printf("output without boundry\n");
            for (int m = 0; m < featuremap_without_boundry_height; ++m)
            {
                for (int n = 0; n < featuremap_without_boundry_width; ++n)
                {
                    int left_edges = net->layers[l].left_boundry_edges_featuremap;
                    int right_edges = net->layers[l].right_boundry_edges_featuremap;
                    int top_edges = net->layers[l].top_boundry_edges_featuremap;
                    int bottom_edges = net->layers[l].bottom_boundry_edges_featuremap;

                    net->layers[l-1].output_without_boundry[m*featuremap_without_boundry_width + n] = 
                        net->layers[l-1].output[(m+top_edges - unit_boundry)*(net->layers[l].featuremap_in_w_with_boundry) + n + left_edges - unit_boundry];

                    //printf("%.2f ", net->layers[l-1].output_without_boundry[m*featuremap_without_boundry_width + n]); 
                }

               // printf("\n");

            }

            printf("\ndelta no boundry\n");

            for (int m = 0; m < net->layers[l].delta_in_h_without_boundry; ++m)
            {
                for (int n = 0; n < net->layers[l].delta_in_w_without_boundry; ++n)
                {
                    int left_edges = net->layers[l].left_boundry_edges_delta;
                    int right_edges = net->layers[l].right_boundry_edges_delta;
                    int top_edges = net->layers[l].top_boundry_edges_delta;
                    int bottom_edges = net->layers[l].bottom_boundry_edges_delta;

                    net->layers[l].delta_without_boundry[m*net->layers[l].delta_in_w_without_boundry + n] = 
                                                                net->layers[l].delta_with_boundry[(m+top_edges)*(net->layers[l].delta_in_w_with_boundry) + n + left_edges];
                    //printf("%.2f ", net->layers[l].delta_without_boundry[m*net->layers[l].delta_in_w_without_boundry + n]); 
                }

               // printf("\n");
            }

       //     printf("\n");





            net->input = net->layers[l-1].output_without_boundry;
            net->delta = net->layers[l-1].delta_with_boundry;


            int stride = net->layers[l].stride;

            net->layers[l].out_w = net->layers[l].delta_in_w_without_boundry;
            net->layers[l].out_h = net->layers[l].delta_in_h_without_boundry;
            net->layers[l].h = featuremap_without_boundry_height;
            net->layers[l].w = featuremap_without_boundry_width;

            net->layers[l].pad = 0;

            net->index = l;
            
           // while(1);


            backward_convolutional_layer_dist_filters(net->layers[l], *net);

            net->layers[l].out_w = net->layers[l].delta_in_w_with_boundry;
            net->layers[l].out_h = net->layers[l].delta_in_h_with_boundry;

            net->layers[l].pad = filter_size - 1;

            zero_out_edges_delta_device(net, l, NUM_TILES_Y, NUM_TILES_X, DEVICE_ID_Y, DEVICE_ID_X);

            int dilated_delta_dim_x = net->layers[l].out_w*stride;
            int dilated_delta_dim_y = net->layers[l].out_h*stride;
            net->layers[l].w = net->layers[l-1].delta_in_w_with_boundry;
            net->layers[l].h = net->layers[l-1].delta_in_h_with_boundry;


            backward_convolutional_layer_dist_delta(net->layers[l], *net);

            
            // printf("Delta with boundry layer %d\n", l-1);


            // for (int m = 0; m < net->layers[l-1].delta_in_h_with_boundry; ++m)
            // {
            //     for (int n = 0; n < net->layers[l-1].delta_in_w_with_boundry; ++n)
            //     {
            //         printf("%.2f ", net->layers[l-1].delta_with_boundry[m*net->layers[l-1].delta_in_w_with_boundry + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");


            printf("Delta without boundry layer %d\n", l-1);


            for (int m = 0; m < 6; ++m)
            {
                for (int n = 0; n < 6; ++n)
                {
                    printf("%.2f ", net->layers[l].delta_with_boundry[m*6 + n]);
                }
                printf("\n");
                
            }
            printf("\n");


            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[l].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[l].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            //while(1);




        }

       // while(1);



        int start_layer = start_idx;//profile.bp[g].layer_start_idx - 1;

        if(start_layer > 0){
            for (int m = 0; m < net->layers[start_layer-1].delta_in_h_without_boundry; ++m)
            {
                for (int n = 0; n < net->layers[start_layer-1].delta_in_w_without_boundry; ++n)
                {
                    int left_edges = net->layers[start_layer-1].left_boundry_edges_delta;
                    int right_edges = net->layers[start_layer-1].right_boundry_edges_delta;
                    int top_edges = net->layers[start_layer-1].top_boundry_edges_delta;
                    int bottom_edges = net->layers[start_layer-1].bottom_boundry_edges_delta;

                    net->layers[start_layer-1].delta_without_boundry[m*net->layers[start_layer-1].delta_in_w_without_boundry + n] = 
                                                                net->layers[start_layer-1].delta_with_boundry[(m)*(net->layers[start_layer-1].delta_in_w_without_boundry) + n];
                }
            }
        }

    }

            #ifdef SERVER
                receive_sum_broadcast_weight_updates(net, NUM_TILES_Y, NUM_TILES_X);
            #endif
            #ifdef CLIENT
                sync_weight_updates(net, NUM_TILES_Y, NUM_TILES_X);
            #endif

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[0].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[1].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[2].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[3].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[4].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            // for (int m = 0; m < 3; ++m)
            // {
            //     for (int n = 0; n < 3; ++n)
            //     {
            //         printf("%.2f ", net->layers[5].weight_updates[m*3 + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

            // for (int m = 0; m < 3; ++m)
            // {
            //     for (int n = 0; n < 3; ++n)
            //     {
            //         printf("%.2f ", net->layers[6].weight_updates[m*3 + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

            // for (int m = 0; m < 3; ++m)
            // {
            //     for (int n = 0; n < 3; ++n)
            //     {
            //         printf("%.2f ", net->layers[7].weight_updates[m*3 + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

}

int main(){
    main_device();
}

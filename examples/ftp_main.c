#include "ftp.h"
#include "fused.h"
#include "fused_convolution.h"

#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"
#include "maxpool_layer.h"

int main_maxpool(){

    network *net = calloc(1, sizeof(network));
    net->n = 1;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    //(int batch, int h, int w, int c, int size, int stride, int padding)
    net->layers[0] = make_maxpool_layer(1, 10, 10, 1, 3, 1, 0); 
    net->layers[0].batch_normalize = 0;
    net->input = calloc(100, sizeof(float));
    net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));


    float* image_merged = calloc(10*10, sizeof(float));
    fill_cpu(100, 1, image_merged, 1);

    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            net->input[i*10 + j] = image_merged[i*10 + j];
        }
    }

    // for (int i = 0; i < 10; ++i)
    // {
    //     net->input[i*10 + 9] = -1;
    //     net->input[90 + i] = -1;
    // }


    forward_maxpool_layer(net->layers[0], *net);

    fill_cpu(net->layers[0].outputs, 1, net->layers[0].delta, 1);

    net->delta = calloc(100, sizeof(float));
    backward_maxpool_layer(net->layers[0], *net);


    int out_size = net->layers[0].out_h * net->layers[0].out_w * net->layers[0].out_c * net->layers[0].batch;

    for (int i = 0; i < net->layers[0].out_h; ++i)
    {
        for (int j = 0; j < net->layers[0].out_w; ++j)
        {
            printf("%.4f ", net->layers[0].output[(i*net->layers[0].out_w) + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            printf("%.4f ", net->delta[(i*10) + j]);
        }
        printf("\n");
    }
}





//12 12 12 6 6 6 3 3 
//1  1  2  1 1 2 1 1

int main_distributed(){

    int NUM_TILES_X = 2;
    int NUM_TILES_Y = 2;
    int INPUT_WIDTH = 24;
    int INPUT_HEIGHT = 24;


    network*** SHARED_NETWORKS = calloc(NUM_TILES_Y, sizeof(network**));

    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        SHARED_NETWORKS[i] = calloc(NUM_TILES_X, sizeof(network*));
    }

    float*** SHARED_INPUT_IMAGES = calloc(NUM_TILES_Y, sizeof(float**));

    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        SHARED_INPUT_IMAGES[i] = calloc(NUM_TILES_X, sizeof(float*));
    }

    float*** SHARED_EXP_DELTAS = calloc(NUM_TILES_Y, sizeof(float**));

    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        SHARED_EXP_DELTAS[i] = calloc(NUM_TILES_X, sizeof(float*));
    }


    train_groups_profile profile;

    profile.num_forward_groups = 3;
    profile.num_backward_groups = 3;

    profile.fp = calloc(profile.num_forward_groups, sizeof(group_profile_forward));
    profile.bp = calloc(profile.num_backward_groups, sizeof(group_profile_backward));

    profile.fp[0].layer_start_idx = 0;
    profile.fp[0].layer_end_idx = 3;
    profile.fp[0].start_x_forward = 0;
    profile.fp[0].start_y_forward = 0;
    profile.fp[0].end_x_forward = 5;
    profile.fp[0].end_y_forward = 5;

    profile.fp[1].layer_start_idx = 4;
    profile.fp[1].layer_end_idx = 5;
    profile.fp[1].start_x_forward = 0;
    profile.fp[1].start_y_forward = 0;
    profile.fp[1].end_x_forward = 2;
    profile.fp[1].end_y_forward = 2;

    profile.fp[2].layer_start_idx = 6;
    profile.fp[2].layer_end_idx = 7;
    profile.fp[2].start_x_forward = 0;
    profile.fp[2].start_y_forward = 0;
    profile.fp[2].end_x_forward = 2;
    profile.fp[2].end_y_forward = 2;

    profile.bp[0].layer_start_idx = 0;
    profile.bp[0].layer_end_idx = 3;
    profile.bp[0].start_x_backward = 0;
    profile.bp[0].start_y_backward = 0;
    profile.bp[0].end_x_backward = 11;
    profile.bp[0].end_y_backward = 11;

    profile.bp[1].layer_start_idx = 4;
    profile.bp[1].layer_end_idx = 5;
    profile.bp[1].start_x_backward = 0;
    profile.bp[1].start_y_backward = 0;
    profile.bp[1].end_x_backward = 5;
    profile.bp[1].end_y_backward = 5;

    profile.bp[2].layer_start_idx = 6;
    profile.bp[2].layer_end_idx = 7;
    profile.bp[2].start_x_backward = 0;
    profile.bp[2].start_y_backward = 0;
    profile.bp[2].end_x_backward = 2;
    profile.bp[2].end_y_backward = 2;



    int filter_size = 3;
    int num_layers = 4;
    int unit_boundry = 1;


    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        for (int j = 0; j < NUM_TILES_X; ++j)
        {


                SHARED_NETWORKS[i][j] = calloc(1, sizeof(network));
                network* net = SHARED_NETWORKS[i][j];

                net->n = 8;
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
                net->layers[2].stride = 2;
                net->layers[3].stride = 1;
                net->layers[4].stride = 1;
                net->layers[5].stride = 2;
                net->layers[6].stride = 1;
                net->layers[7].stride = 1;

            SHARED_INPUT_IMAGES[i][j] = calloc((INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), sizeof(float));
            fill_cpu((INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), 1, SHARED_INPUT_IMAGES[i][j], 1);

            net->workspace = calloc(5000, sizeof(float));
            net->inputs = 484;
            //net->input = calloc(484, sizeof(float));
        }
    }

    for (int g = 0; g < profile.num_forward_groups; ++g)
    {
        for (int i = 0; i < NUM_TILES_Y; ++i)
        {
            for (int j = 0; j < NUM_TILES_X; ++j)
            {


                network* net = SHARED_NETWORKS[i][j];

                partition_forward(net, 
                                  &profile,
                                  &(profile.fp[g]),
                                    0, 0,
                                    NULL,
                                    3,
                                    profile.fp[g].start_x_forward, profile.fp[g].start_y_forward,
                                    profile.fp[g].end_x_forward, profile.fp[g].end_y_forward);

                net->input = calloc(484, sizeof(float));
                //get boundry data here
                assemble_forward_group_data(SHARED_NETWORKS, 
                                                SHARED_INPUT_IMAGES,
                                                NUM_TILES_X, NUM_TILES_Y,
                                                 profile.fp[g],
                                                 j, i
                                                 );
            }
        }

        for (int i = 0; i < NUM_TILES_Y; ++i)
        {
            for (int j = 0; j < NUM_TILES_X; ++j)
            {
                network* net = SHARED_NETWORKS[i][j];

                for (int l = profile.fp[g].layer_start_idx; l <= profile.fp[g].layer_end_idx; ++l)
                {
                    // if(l > 0){
                    //     for (int m = 0; m < net->layers[l].top_boundry_edges_featuremap; ++m)
                    //     {
                    //         for (int n = 0; n < net->layers[l].featuremap_in_w_with_boundry; ++n)
                    //         {
                    //             net->layers[l-1].output[m*net->layers[l].featuremap_in_w_with_boundry + n] = 0.0;
                    //         }
                    //     }

                    //     for (int m = 0; m < net->layers[l].featuremap_in_h_with_boundry; ++m)
                    //     {
                    //         for (int n = 0; n < net->layers[l].left_boundry_edges_featuremap; ++n)
                    //         {
                    //             net->layers[l-1].output[(m*net->layers[l].featuremap_in_w_with_boundry) + n] = 0.0;
                    //         }
                    //     }
                    // }
                    zero_out_edges_featuremap(net, l, NUM_TILES_Y, NUM_TILES_X, i, j);
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
        }
    }


    //while(1);

    // float* COMBINED_INPUT_IMAGES = calloc(144, sizeof(float));
    // fill_cpu(144, 1, COMBINED_INPUT_IMAGES, 1);
    // float* COMBINED_EXP_DELTAS = calloc(36, sizeof(float));
    // fill_cpu(36, 1, COMBINED_EXP_DELTAS, 1);

    // partition_forward(net, 
    //                         0, 0,
    //                         NULL,
    //                         COMBINED_INPUT_IMAGES,
    //                         3,
    //                         0, 0,
    //                         2, 2);


    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        for (int j = 0; j < NUM_TILES_X; ++j)
        {

            network* net = SHARED_NETWORKS[i][j];
            SHARED_EXP_DELTAS[i][j] = calloc(net->layers[net->n - 1].outputs, sizeof(float));
            fill_cpu(net->layers[net->n - 1].outputs, 1, SHARED_EXP_DELTAS[i][j], 1);
            fill_cpu(net->layers[net->n - 1].outputs, 1, net->layers[net->n - 1].delta_with_boundry, 1);
            fill_cpu(net->layers[net->n - 1].outputs, 1, net->layers[net->n - 1].delta, 1);
            fill_cpu(net->layers[net->n - 1].outputs, 1, net->layers[net->n - 1].delta_without_boundry, 1);
        }
    }


    for (int g = (profile.num_backward_groups-1); g >= 0; --g)
    {
        for (int i = 0; i < NUM_TILES_Y; ++i)
        {
            for (int j = 0; j < NUM_TILES_X; ++j)
            {

                network* net = SHARED_NETWORKS[i][j];
                partition_backward(net, 
                                    j, i,
                                    SHARED_NETWORKS,
                                    SHARED_EXP_DELTAS,
                                    filter_size,
                                    &profile.bp[g],
                                    profile.bp[g].start_x_backward, profile.bp[g].start_y_backward,
                                    profile.bp[g].end_x_backward, profile.bp[g].end_y_backward);

                assemble_backward_group_data(SHARED_NETWORKS, 
                                                SHARED_EXP_DELTAS,
                                                NUM_TILES_X, NUM_TILES_Y,
                                                 profile.bp[g],
                                                 j, i,
                                                 net->n
                                                 );

            }
        }
        
        for (int i = 0; i < NUM_TILES_Y; ++i)
        {
            for (int j = 0; j < NUM_TILES_X; ++j)
            {
                network* net = SHARED_NETWORKS[i][j];

                int start_idx = (g==0) ? 1 : profile.bp[g].layer_start_idx;
                for (int l = profile.bp[g].layer_end_idx; l >= start_idx; --l)
                {



                    int featuremap_without_boundry_width = net->layers[l].featuremap_in_w_without_boundry + (2*unit_boundry);
                    int featuremap_without_boundry_height = net->layers[l].featuremap_in_h_without_boundry + (2*unit_boundry);
                    for (int m = 0; m < featuremap_without_boundry_height; ++m)
                    {
                        for (int n = 0; n < featuremap_without_boundry_width; ++n)
                        {
                            int left_edges = net->layers[l].left_boundry_edges_featuremap;
                            int right_edges = net->layers[l].right_boundry_edges_featuremap;
                            int top_edges = net->layers[l].top_boundry_edges_featuremap;
                            int bottom_edges = net->layers[l].bottom_boundry_edges_featuremap;
                            // printf("%d %d %d\n", net->layers[l].featuremap_start_coordinate_x, featuremap_without_boundry_width, featuremap_without_boundry_height);

                            net->layers[l-1].output_without_boundry[m*featuremap_without_boundry_width + n] = 
                                net->layers[l-1].output[(m+top_edges - unit_boundry)*(net->layers[l].featuremap_in_w_with_boundry) + n + left_edges - unit_boundry];
                        }

                    }

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
                        }
                    }





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

                    zero_out_edges_delta(net, l, NUM_TILES_Y, NUM_TILES_X, i, j);

                    int dilated_delta_dim_x = net->layers[l].out_w*stride;
                    int dilated_delta_dim_y = net->layers[l].out_h*stride;
                    net->layers[l].w = net->layers[l-1].delta_in_w_with_boundry;
                    net->layers[l].h = net->layers[l-1].delta_in_h_with_boundry;

                    backward_convolutional_layer_dist_delta(net->layers[l], *net);

                    


                    for (int m = 0; m < net->layers[l].delta_in_h_with_boundry; ++m)
                    {
                        for (int n = 0; n < net->layers[l].delta_in_w_with_boundry; ++n)
                        {
                            printf("%.2f ", net->layers[l].delta_with_boundry[m*net->layers[l].delta_in_w_with_boundry + n]);
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
        }
    }
    //while(1);

        // net->input = net->layers[current_layer_idx - 1].output;
        // net->delta = net->layers[current_layer_idx - 1].delta_with_boundry;

    // int total_edges_left = net->layers[0].left_boundry_edges_featuremap;
    // int total_edges_right = net->layers[0].right_boundry_edges_featuremap;
    // int total_edges_bottom = net->layers[0].bottom_boundry_edges_featuremap;
    // int total_edges_top = net->layers[0].top_boundry_edges_featuremap;


    // for (int l = num_layers - 1; l >= 1; --l)
    // {
    //             int featuremap_without_boundry_width = net->layers[l].featuremap_in_w_without_boundry + (2*unit_boundry);
    //             int featuremap_without_boundry_height = net->layers[l].featuremap_in_h_without_boundry + (2*unit_boundry);
    //             //printf("%d %d\n", featuremap_without_boundry_width, featuremap_without_boundry_height);

    //             for (int m = 0; m < featuremap_without_boundry_height; ++m)
    //             {
    //                 for (int n = 0; n < featuremap_without_boundry_width; ++n)
    //                 {
    //                     int left_edges = net->layers[l].left_boundry_edges_featuremap;
    //                     int right_edges = net->layers[l].right_boundry_edges_featuremap;
    //                     int top_edges = net->layers[l].top_boundry_edges_featuremap;
    //                     int bottom_edges = net->layers[l].bottom_boundry_edges_featuremap;
    //                     // printf("%d %d %d\n", net->layers[l].featuremap_start_coordinate_x, featuremap_without_boundry_width, featuremap_without_boundry_height);

    //                     net->layers[l-1].output_without_boundry[m*featuremap_without_boundry_width + n] = 
    //                         net->layers[l-1].output[(m+top_edges - unit_boundry)*(net->layers[l].featuremap_in_w_with_boundry) + n + left_edges - unit_boundry];
    //                 }

    //             }

    //                 for (int m = 0; m < featuremap_without_boundry_height; ++m)
    //                 {
    //                     for (int n = 0; n < featuremap_without_boundry_width; ++n)
    //                     {
    //                         printf("%.2f ", net->layers[l-1].output_without_boundry[m*featuremap_without_boundry_width + n]);
    //                     }
    //                     printf("\n");
                        
    //                 }
    //                 printf("\n");
    //                 //while(1);


    //             for (int m = 0; m < net->layers[l].delta_in_h_without_boundry; ++m)
    //             {
    //                 for (int n = 0; n < net->layers[l].delta_in_w_without_boundry; ++n)
    //                 {
    //                     int left_edges = net->layers[l].left_boundry_edges_delta;
    //                     int right_edges = net->layers[l].right_boundry_edges_delta;
    //                     int top_edges = net->layers[l].top_boundry_edges_delta;
    //                     int bottom_edges = net->layers[l].bottom_boundry_edges_delta;

    //                     net->layers[l].delta_without_boundry[m*net->layers[l].delta_in_w_without_boundry + n] = 
    //                                                                 net->layers[l].delta_with_boundry[(m+top_edges)*(net->layers[l].delta_in_w_with_boundry) + n + left_edges];
    //                 }
    //             }




    //             net->input = net->layers[l-1].output_without_boundry;
    //             net->delta = net->layers[l-1].delta_with_boundry;


    //             int stride = net->layers[l].stride;

    //             net->layers[l].out_w = net->layers[l].delta_in_w_without_boundry;
    //             net->layers[l].out_h = net->layers[l].delta_in_h_without_boundry;
    //             net->layers[l].h = featuremap_without_boundry_height;
    //             net->layers[l].w = featuremap_without_boundry_width;

    //             net->layers[l].pad = 0;

    //             net->index = l;
                
    //            // while(1);

    //             backward_convolutional_layer_dist_filters(net->layers[l], *net);
                

    //             net->layers[l].out_w = net->layers[l].delta_in_w_with_boundry;
    //             net->layers[l].out_h = net->layers[l].delta_in_h_with_boundry;

    //             net->layers[l].pad = filter_size - 1;

    //             for (int m = 0; m < net->layers[l].top_boundry_edges_delta; ++m)
    //             {
    //                 for (int n = 0; n < net->layers[l].delta_in_w_with_boundry; ++n)
    //                 {
    //                     net->layers[l].delta_with_boundry[m*net->layers[l].delta_in_w_with_boundry + n] = 0.0;
    //                 }
    //             }

    //             for (int m = 0; m < net->layers[l].delta_in_h_with_boundry; ++m)
    //             {
    //                 for (int n = 0; n < net->layers[l].left_boundry_edges_delta; ++n)
    //                 {
    //                     net->layers[l].delta_with_boundry[(m*net->layers[l].delta_in_w_with_boundry) + n] = 0.0;
    //                 }
    //             }

    //             int dilated_delta_dim_x = net->layers[l].out_w*stride;
    //             int dilated_delta_dim_y = net->layers[l].out_h*stride;
    //             net->layers[l].w = net->layers[l-1].delta_in_w_with_boundry;
    //             net->layers[l].h = net->layers[l-1].delta_in_h_with_boundry;

    //             backward_convolutional_layer_dist_delta(net->layers[l], *net);

    // }
            network* net = SHARED_NETWORKS[0][0];

            for (int m = 0; m < net->layers[0].delta_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[0].delta_in_w_with_boundry; ++n)
                {
                    printf("%.2f ", net->layers[0].delta_with_boundry[m*net->layers[0].delta_in_w_with_boundry + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            net = SHARED_NETWORKS[0][1];

            for (int m = 0; m < net->layers[0].delta_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[0].delta_in_w_with_boundry; ++n)
                {
                    printf("%.2f ", net->layers[0].delta_with_boundry[m*net->layers[0].delta_in_w_with_boundry + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            net = SHARED_NETWORKS[1][0];

            for (int m = 0; m < net->layers[0].delta_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[0].delta_in_w_with_boundry; ++n)
                {
                    printf("%.2f ", net->layers[0].delta_with_boundry[m*net->layers[0].delta_in_w_with_boundry + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            net = SHARED_NETWORKS[1][1];

            for (int m = 0; m < net->layers[0].delta_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[0].delta_in_w_with_boundry; ++n)
                {
                    printf("%.2f ", net->layers[0].delta_with_boundry[m*net->layers[0].delta_in_w_with_boundry + n]);
                }
                printf("\n");
                
            }
            printf("\n");



            net = SHARED_NETWORKS[0][0];

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    net->layers[1].weight_updates[m*3 + n] += SHARED_NETWORKS[0][1]->layers[1].weight_updates[m*3 + n] + SHARED_NETWORKS[1][0]->layers[1].weight_updates[m*3 + n] + SHARED_NETWORKS[1][1]->layers[1].weight_updates[m*3 + n];
                    net->layers[2].weight_updates[m*3 + n] += SHARED_NETWORKS[0][1]->layers[2].weight_updates[m*3 + n] + SHARED_NETWORKS[1][0]->layers[2].weight_updates[m*3 + n] + SHARED_NETWORKS[1][1]->layers[2].weight_updates[m*3 + n];
                    net->layers[3].weight_updates[m*3 + n] += SHARED_NETWORKS[0][1]->layers[3].weight_updates[m*3 + n] + SHARED_NETWORKS[1][0]->layers[3].weight_updates[m*3 + n] + SHARED_NETWORKS[1][1]->layers[3].weight_updates[m*3 + n];
                    net->layers[4].weight_updates[m*3 + n] += SHARED_NETWORKS[0][1]->layers[4].weight_updates[m*3 + n] + SHARED_NETWORKS[1][0]->layers[4].weight_updates[m*3 + n] + SHARED_NETWORKS[1][1]->layers[4].weight_updates[m*3 + n];
                    net->layers[5].weight_updates[m*3 + n] += SHARED_NETWORKS[0][1]->layers[5].weight_updates[m*3 + n] + SHARED_NETWORKS[1][0]->layers[5].weight_updates[m*3 + n] + SHARED_NETWORKS[1][1]->layers[5].weight_updates[m*3 + n];
                    net->layers[6].weight_updates[m*3 + n] += SHARED_NETWORKS[0][1]->layers[6].weight_updates[m*3 + n] + SHARED_NETWORKS[1][0]->layers[6].weight_updates[m*3 + n] + SHARED_NETWORKS[1][1]->layers[6].weight_updates[m*3 + n];
                    net->layers[7].weight_updates[m*3 + n] += SHARED_NETWORKS[0][1]->layers[7].weight_updates[m*3 + n] + SHARED_NETWORKS[1][0]->layers[7].weight_updates[m*3 + n] + SHARED_NETWORKS[1][1]->layers[7].weight_updates[m*3 + n];
                }
                printf("\n");
                
            }

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

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[5].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[6].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[7].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");
            // net = SHARED_NETWORKS[0][1];

            // for (int m = 0; m < 3; ++m)
            // {
            //     for (int n = 0; n < 3; ++n)
            //     {
            //         printf("%.2f ", net->layers[1].weight_updates[m*3 + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");
            // net = SHARED_NETWORKS[1][0];

            // for (int m = 0; m < 3; ++m)
            // {
            //     for (int n = 0; n < 3; ++n)
            //     {
            //         printf("%.2f ", net->layers[1].weight_updates[m*3 + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

            // net = SHARED_NETWORKS[1][1];

            // for (int m = 0; m < 3; ++m)
            // {
            //     for (int n = 0; n < 3; ++n)
            //     {
            //         printf("%.2f ", net->layers[1].weight_updates[m*3 + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

            // for (int m = 0; m < net->layers[2].delta_in_h_with_boundry; ++m)
            // {
            //     for (int n = 0; n < net->layers[2].delta_in_w_with_boundry; ++n)
            //     {
            //         printf("%.2f ", net->layers[2].delta_with_boundry[m*net->layers[2].delta_in_w_with_boundry + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

}

int main_reference(){
    network* net = calloc(1, sizeof(network));//SHARED_NETWORKS[i][j];

    net->n = 8;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    int filter_size = 3;
    int num_layers = 8;
    int unit_boundry = 1;

    net->layers[0] = make_convolutional_layer(1, 24, 24, 3, 2, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, 24, 24, 2, 2, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, 24, 24, 2, 2, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[3] = make_convolutional_layer(1, 24, 24, 2, 2, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[4] = make_convolutional_layer(1, 24, 24, 2, 2, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[5] = make_convolutional_layer(1, 24, 24, 2, 2, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[6] = make_convolutional_layer(1, 24, 24, 2, 2, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[7] = make_convolutional_layer(1, 24, 24, 2, 2, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);


    for (int l = 0; l < net->n; ++l)
    {
        if(net->layers[l].type == CONVOLUTIONAL){

            int filter_size = net->layers[l].size;
            int num_filters = net->layers[l].n;
            int num_channels = net->layers[l].c;

            for (int i = 0; i < (filter_size*filter_size*num_filters*num_channels); ++i)
            {

                net->layers[l].weights[i] = 0.1;
            }        


        }
    }

    net->workspace = calloc(net->layers[0].workspace_size*5, sizeof(float));
    net->inputs = 625*3;
    net->input = calloc(625*3, sizeof(float));

    fill_cpu(576*3, 1, net->input, 1);
    fill_cpu(576*3, 1, net->layers[net->n - 1].delta, 1);

    for (int l = 0; l < num_layers; ++l)
    {
        net->index = l;
        forward_convolutional_layer(net->layers[l], *net);
        // for (size_t i = 0; i < net->layers[l].out_h; i++)
        // {
        //     for (size_t j = 0; j < net->layers[l].out_w; j++)
        //     {
        //         printf("%.2f ", net->layers[l].output[i*net->layers[l].out_w + j]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");

        net->input = net->layers[l].output;
    }

    for (int l = num_layers-1; l >=1; --l)
    {
        net->index = l;
        net->input = net->layers[l-1].output;
        net->delta = net->layers[l-1].delta;
        backward_convolutional_layer(net->layers[l], *net);

        printf("map at layer %d\n", l);

        for (size_t i = 0; i < net->layers[l-1].out_h; i++)
        {
            for (size_t j = 0; j < net->layers[l-1].out_w; j++)
            {
                printf("%.5f ", net->layers[l-1].output[i*net->layers[l-1].out_w + j]);
            }
            printf("\n");
            
        }
        printf("\n");

        printf("Delta at layer %d\n", l);

        for (size_t i = 0; i < 24; i++)
        {
            for (size_t j = 0; j < 24; j++)
            {
                printf("%.5f ", net->layers[l].delta[i*24 + j]);
            }
            printf("\n");
            
        }
        printf("\n");


            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[l].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[l].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[l].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[l].weight_updates[27 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");
    }

        for (size_t i = 0; i < 24; i++)
        {
            for (size_t j = 0; j < 24; j++)
            {
                printf("%.2f ", net->layers[0].delta[i*24 + j]);
            }
            printf("\n");
            
        }
        printf("\n");


            // for (size_t i = 0; i < 24; i++)
            // {
            //     for (size_t j = 0; j < 24; j++)
            //     {
            //         printf("%.2f ", net->layers[1].delta[i*24 + j]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");


            // for (size_t i = 0; i < 12; i++)
            // {
            //     for (size_t j = 0; j < 12; j++)
            //     {
            //         printf("%.2f ", net->layers[2].delta[i*12 + j]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");


            // for (size_t i = 0; i < 12; i++)
            // {
            //     for (size_t j = 0; j < 12; j++)
            //     {
            //         printf("%.2f ", net->layers[3].delta[i*12 + j]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

            // for (size_t i = 0; i < 12; i++)
            // {
            //     for (size_t j = 0; j < 12; j++)
            //     {
            //         printf("%.2f ", net->layers[4].delta[i*12 + j]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

            // for (size_t i = 0; i < 6; i++)
            // {
            //     for (size_t j = 0; j < 6; j++)
            //     {
            //         printf("%.2f ", net->layers[5].delta[i*6 + j]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

            // for (size_t i = 0; i < 6; i++)
            // {
            //     for (size_t j = 0; j < 6; j++)
            //     {
            //         printf("%.2f ", net->layers[6].delta[i*6 + j]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

            // for (size_t i = 0; i < 6; i++)
            // {
            //     for (size_t j = 0; j < 6; j++)
            //     {
            //         printf("%.2f ", net->layers[7].delta[i*6 + j]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");


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
                    printf("%.2f ", net->layers[0].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[0].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[0].weight_updates[27 + m*3 + n]);
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
                    printf("%.2f ", net->layers[1].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[1].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[1].weight_updates[27 + m*3 + n]);
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
                    printf("%.2f ", net->layers[2].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[2].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[2].weight_updates[27 + m*3 + n]);
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
                    printf("%.2f ", net->layers[3].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[3].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[3].weight_updates[27 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");


            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[4].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[4].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[4].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[4].weight_updates[27 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");


}

#define LAYER_SIZE 608
#define FILTER_SIZE 32

int main_yolo(){
    //make_convolutional_layer(int batch, int h,
    // int w, int c, int n, int groups, int size, int stride, int padding,
    // ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)

    network* net = calloc(1, sizeof(network));//SHARED_NETWORKS[i][j];

    net->n = 11;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    int filter_size = 3;
    int num_layers = 11;
    int unit_boundry = 1;

    //yolo v2
    net->layers[0] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, 3, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    //(int batch, int h, int w, int c, int size, int stride, int padding)
    //net->layers[1] = make_maxpool_layer(1, 604, 604, 32, 2, 2, 0); 
    net->layers[1] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, 32, FILTER_SIZE, 1, 3, 2, 1, LEAKY, 0, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, LAYER_SIZE/2, LAYER_SIZE/2, 32, 64, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    //net->layers[3] = make_maxpool_layer(1, 302, 302, 64, 2, 2, 0);
    net->layers[3] = make_convolutional_layer(1, LAYER_SIZE/2, LAYER_SIZE/2, 64, 64, 1, 3, 2, 1, LEAKY, 0, 0, 0, 0);
    net->layers[4] = make_convolutional_layer(1, LAYER_SIZE/4, LAYER_SIZE/4, 64, 128, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);


    // net->layers[1] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    // net->layers[2] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    // net->layers[3] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    // net->layers[4] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    net->layers[5] = make_convolutional_layer(1, LAYER_SIZE/4, LAYER_SIZE/4, 128, 64, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);
    net->layers[6] = make_convolutional_layer(1, LAYER_SIZE/4, LAYER_SIZE/4, 64, 128, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);


    //net->layers[7] = make_maxpool_layer(1, 152, 152, 128, 2, 2, 0);
    net->layers[7] = make_convolutional_layer(1, LAYER_SIZE/4, LAYER_SIZE/4, 128, 128, 1, 3, 2, 1, LEAKY, 0, 0, 0, 0);


    net->layers[8] = make_convolutional_layer(1, LAYER_SIZE/8, LAYER_SIZE/8, 128, 256, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[9] = make_convolutional_layer(1, LAYER_SIZE/8, LAYER_SIZE/8, 256, 128, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    net->layers[10] = make_convolutional_layer(1, LAYER_SIZE/8, LAYER_SIZE/8, 128, 256, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[11] = make_maxpool_layer(1, 76/2, 76/2, 256, 2, 2, 0);


    // net->layers[12] = make_convolutional_layer(1, 38/2, 38/2, 256, 512, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[13] = make_convolutional_layer(1, 38/2, 38/2, 512, 256, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    // net->layers[14] = make_convolutional_layer(1, 38/2, 38/2, 256, 512, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[15] = make_convolutional_layer(1, 38/2, 38/2, 512, 256, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    // net->layers[16] = make_convolutional_layer(1, 38/2, 38/2, 256, 512, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[17] = make_maxpool_layer(1, 38/2, 38/2, 512, 2, 2, 0);

    // net->layers[18] = make_convolutional_layer(1, 19/2, 19/2, 512, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[19] = make_convolutional_layer(1, 19/2, 19/2, 1024, 512, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    // net->layers[20] = make_convolutional_layer(1, 19/2, 19/2, 512, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[21] = make_convolutional_layer(1, 19/2, 19/2, 1024, 512, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    // net->layers[22] = make_convolutional_layer(1, 19/2, 19/2, 512, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[23] = make_convolutional_layer(1, 19/2, 19/2, 1024, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[24] = make_convolutional_layer(1, 19/2, 19/2, 1024, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    // net->layers[2] = make_convolutional_layer(1, 24, 24, 1, 1, 1, filter_size, 2, 1, RELU, 0, 0, 0, 0);
    // net->layers[3] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    // net->layers[4] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    // net->layers[5] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 2, 1, RELU, 0, 0, 0, 0);
    // net->layers[6] = make_convolutional_layer(1, 6, 6, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    // net->layers[7] = make_convolutional_layer(1, 6, 6, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0); 


    for (int l = 0; l < net->n; ++l)
    {
        if(net->layers[l].type == CONVOLUTIONAL){

            int filter_size = net->layers[l].size;
            int num_filters = net->layers[l].n;
            int num_channels = net->layers[l].c;

            for (int i = 0; i < (filter_size*filter_size*num_filters*num_channels); ++i)
            {

                net->layers[l].weights[i] = 0.01;
            }        


        }
    }



    net->workspace = calloc(net->layers[0].workspace_size*5, sizeof(float));
    net->inputs = 608*608*3;
    int outputs = 19*19*1024;
    net->input = calloc(net->inputs, sizeof(float));

    fill_cpu(net->inputs, 0.1, net->input, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);

    for (int l = 0; l < net->n; ++l)
    {
        net->index = l;
        printf("Filter stacks = %d\n", net->layers[l].n);
        net->layers[l].forward(net->layers[l], *net);
        net->input = net->layers[l].output;


        // for (size_t i = 0; i < net->layers[l].out_h; i++)
        // {
        //     for (size_t j = 0; j < net->layers[l].out_w; j++)
        //     {
        //         printf("%.2f ", net->layers[l].output[i*net->layers[l].out_w + j]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");

        // for (size_t i = 0; i < 12; i++)
        // {
        //     for (size_t j = 0; j < 12; j++)
        //     {
        //         printf("%.2f ", net->layers[l].output[144 + i*12 + j]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");

    }


    for (int l = (net->n - 1); l >=1; --l)
    {
        net->index = l;
        net->input = net->layers[l-1].output;
        net->delta = net->layers[l-1].delta;
        printf("Filter stacks = %d\n", net->layers[l].n);
        net->layers[l].backward(net->layers[l], *net);

        printf("Delta layer %d\n", l);

        // for (int m = 0; m < net->layers[l].out_h; ++m)
        // {
        //     for (int n = 0; n < net->layers[l].out_w; ++n)
        //     {
        //         printf("%.2f ", net->layers[l].delta[m*net->layers[l].out_w + n]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");


    //     for (int m = 0; m < 12; ++m)
    //     {
    //         for (int n = 0; n < 12; ++n)
    //         {
    //             printf("%.2f ", net->layers[l].delta[144 + m*12 + n]);
    //         }
    //         printf("\n");
            
    //     }
    //     printf("\n");
    }



    //     printf("Delta layer 0\n");

    //     for (int m = 0; m < 12; ++m)
    //     {
    //         for (int n = 0; n < 12; ++n)
    //         {
    //             printf("%.2f ", net->layers[0].delta[m*12 + n]);
    //         }
    //         printf("\n");
            
    //     }
    //     printf("\n");

        // for (int m = 0; m < 12; ++m)
        // {
        //     for (int n = 0; n < 12; ++n)
        //     {
        //         printf("%.2f ", net->layers[0].delta[144 + m*12 + n]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");



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
                    printf("%.2f ", net->layers[0].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[0].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[0].weight_updates[27 + m*3 + n]);
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
                    printf("%.2f ", net->layers[1].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[1].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[1].weight_updates[27 + m*3 + n]);
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
                    printf("%.2f ", net->layers[2].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[2].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[2].weight_updates[27 + m*3 + n]);
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
                    printf("%.2f ", net->layers[3].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[3].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[3].weight_updates[27 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");


            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[4].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[4].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[4].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[4].weight_updates[27 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");
}


int main(int argc, char* argv[]){
    //main_distributed();
    //main_reference();
    //main_maxpool();
    for (int i = 1; i < argc; i++)
    {
        printf("argv[%u] = %s\n", i, argv[i]);
        if (argv[i][0] == '-')
        {
        }
    }
    //main_yolo();
}
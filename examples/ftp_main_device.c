#include "fused_device.h"
#include "fused_convolution_device.h"

#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"
#include "maxpool_layer.h"

#include "transport.h"
#include "fused_device.h"


extern network_config network_params_original;
extern network_config network_params_tile;
extern ftp_config ftp_params;

int main_device(int argc, char* argv[]){

    // NUM_TILES_X = atoi(argv[1]);
    // NUM_TILES_Y = atoi(argv[2]);
    // DEVICE_ID_X = atoi(argv[19]);
    // DEVICE_ID_Y = atoi(argv[20]);

    // printf("%d %d %d %d\n", NUM_TILES_X, NUM_TILES_Y, DEVICE_ID_X, DEVICE_ID_Y);


    // train_groups_profile profile;

    // #ifdef SERVER
    //     init_server();
    // #else
    //     init_transport(argv);
    // #endif

    config_init(argc, argv);
    forward_pass();
    backward_pass();

    network* net;
    init_network(&net);


    printf("%d\n", net->n);
    float* OUTPUT_DELTA = calloc(net->layers[net->n - 1].outputs, sizeof(float));
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, OUTPUT_DELTA, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);




    for (int g = 0; g < ftp_params.NUM_GROUPS_FORWARD; ++g)
    {

        int group_start_idx = ftp_params.sync_group_vector_forward[g];
        net->inputs = net->layers[group_start_idx].featuremap_in_h_with_boundry*
                        net->layers[group_start_idx].featuremap_in_w_with_boundry*
                        net->layers[group_start_idx].c;

        net->input = calloc(net->inputs, sizeof(float));

        //get boundry data here
        assemble_forward_group_data_device(net, 
                                        INPUT_IMAGE,
                                        NUM_TILES_X, NUM_TILES_Y,
                                         group_start_idx,
                                         DEVICE_ID_X, DEVICE_ID_Y
                                         );

        printf("Received input boundary. Starting inference\n");

        for (int l = profile.fp[g].layer_start_idx; l <= profile.fp[g].layer_end_idx; ++l)
        {

            zero_out_edges_featuremap_device(net, l, NUM_TILES_Y, NUM_TILES_X, DEVICE_ID_Y, DEVICE_ID_X);
            net->index = l;

            if(net->layers[l].type == CONVOLUTIONAL){
                forward_convolutional_layer(net->layers[l], *net);
            }

            else if(net->layers[l].type == MAXPOOL){
                forward_maxpool_layer(net->layers[l], *net);
            }

            if(l == profile.fp[g].layer_start_idx){
                free(net->input);
            }

            net->input = net->layers[l].output;
        }
    }

    printf("Inference complete\n");


    // for (int g = (profile.num_backward_groups-1); g >= 0; --g)
    // {

    //     assemble_backward_group_data_device(net, 
    //                                     OUTPUT_DELTA,
    //                                     NUM_TILES_X, NUM_TILES_Y,
    //                                      profile.bp[g],
    //                                      DEVICE_ID_X, DEVICE_ID_Y,
    //                                      net->n
    //                                      );


    //     printf("Received delta boundary. Starting backprop\n");

    //     int start_idx = (g==0) ? 1 : profile.bp[g].layer_start_idx;
    //     for (int l = profile.bp[g].layer_end_idx; l >= start_idx; --l)
    //     {
    //         printf("propagating at Layer %d\n", l);

    //         int unit_boundry = (net->layers[l].size / 2);

    //         int featuremap_without_boundry_width = net->layers[l].featuremap_in_w_without_boundry + (2*unit_boundry);
    //         int featuremap_without_boundry_height = net->layers[l].featuremap_in_h_without_boundry + (2*unit_boundry);



    //         for (int c = 0; c < net->layers[l].c; ++c)
    //         {
    //             for (int m = 0; m < featuremap_without_boundry_height; ++m)
    //             {
    //                 for (int n = 0; n < featuremap_without_boundry_width; ++n)
    //                 {
    //                     int left_edges = net->layers[l].left_boundry_edges_featuremap;
    //                     int right_edges = net->layers[l].right_boundry_edges_featuremap;
    //                     int top_edges = net->layers[l].top_boundry_edges_featuremap;
    //                     int bottom_edges = net->layers[l].bottom_boundry_edges_featuremap;

    //                     int x_dim_nob = net->layers[l].featuremap_in_w_with_boundry;
    //                     int y_dim_nob = net->layers[l].featuremap_in_h_with_boundry;

    //                     net->workspace[(c*featuremap_without_boundry_width*featuremap_without_boundry_height) + m*featuremap_without_boundry_width + n] = 
    //                         net->layers[l-1].output[(c*x_dim_nob*y_dim_nob) + (m+top_edges - unit_boundry)*(x_dim_nob) + n + left_edges - unit_boundry];

    //                 }
    //             }
    //         }



    //         memcpy(net->layers[l-1].output, net->workspace, net->layers[l].c*featuremap_without_boundry_height*featuremap_without_boundry_width*sizeof(float));


    //         net->input = net->layers[l-1].output;
    //         net->delta = net->layers[l-1].delta;

    //         net->layers[l].w = net->layers[l-1].delta_in_w_with_boundry;
    //         net->layers[l].h = net->layers[l-1].delta_in_h_with_boundry;
    //         net->layers[l].out_w = net->layers[l].delta_in_w_with_boundry;
    //         net->layers[l].out_h = net->layers[l].delta_in_h_with_boundry;

    //         printf("dims: %d %d %d %d\n", net->layers[l].w, net->layers[l].h, net->layers[l].out_w, net->layers[l].out_h);

    //         zero_out_edges_delta_device(net, l, NUM_TILES_Y, NUM_TILES_X, DEVICE_ID_Y, DEVICE_ID_X);

    //         if(net->layers[l].type == CONVOLUTIONAL)
    //             backward_convolutional_layer_dist_delta(net->layers[l], *net);
    //         else if(net->layers[l].type == MAXPOOL)
    //             backward_maxpool_layer(net->layers[l], *net);



    //         if(net->layers[l].type == CONVOLUTIONAL){

    //             int x_dim = net->layers[l].delta_in_w_without_boundry;
    //             int y_dim = net->layers[l].delta_in_h_without_boundry;
    //             int depth = net->layers[l].n;

    //             for (int c = 0; c < net->layers[l].n; ++c)
    //             {
    //                 int x_dim_nob = net->layers[l].delta_in_w_with_boundry;
    //                 int y_dim_nob = net->layers[l].delta_in_h_with_boundry;

    //                 for (int m = 0; m < net->layers[l].delta_in_h_without_boundry; ++m)
    //                 {
    //                     for (int n = 0; n < net->layers[l].delta_in_w_without_boundry; ++n)
    //                     {
    //                         int left_edges = net->layers[l].left_boundry_edges_delta;
    //                         int right_edges = net->layers[l].right_boundry_edges_delta;
    //                         int top_edges = net->layers[l].top_boundry_edges_delta;
    //                         int bottom_edges = net->layers[l].bottom_boundry_edges_delta;

    //                         net->workspace[(c*x_dim*y_dim) + m*net->layers[l].delta_in_w_without_boundry + n] = 
    //                             net->layers[l].delta[(c*x_dim_nob*y_dim_nob) + (m+top_edges)*(net->layers[l].delta_in_w_with_boundry) + n + left_edges];
    //                     }
    //                 }
    //             }

    //             memcpy(net->layers[l].delta, net->workspace, x_dim*y_dim*depth*sizeof(float));

    //             net->layers[l].out_w = net->layers[l].delta_in_w_without_boundry;
    //             net->layers[l].out_h = net->layers[l].delta_in_h_without_boundry;
    //             net->layers[l].h = featuremap_without_boundry_height;
    //             net->layers[l].w = featuremap_without_boundry_width;

    //             net->layers[l].pad = 0;

    //             net->index = l;

    //             printf("filter layer %d \n", l);

    //             backward_convolutional_layer_dist_filters(net->layers[l], *net);
    //         }

    //     }

    // }

    // free(OUTPUT_DELTA);

    // printf("Backprop complete\n");


    //         if(DEVICE_ID_X == 0 && DEVICE_ID_Y == 0)
    //             receive_sum_broadcast_weight_updates(net, NUM_TILES_Y, NUM_TILES_X);
    //         else{
    //         //#ifdef CLIENT
    //             sync_weight_updates(net, NUM_TILES_Y, NUM_TILES_X);
    //         }



    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[0].weight_updates[m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[0].weight_updates[9 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[0].weight_updates[18 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[0].weight_updates[27 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");



    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[2].weight_updates[m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[2].weight_updates[9 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[2].weight_updates[18 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[2].weight_updates[27 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");


    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[4].weight_updates[m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[4].weight_updates[9 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[4].weight_updates[18 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[4].weight_updates[27 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");



    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[8].weight_updates[m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[8].weight_updates[9 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[8].weight_updates[18 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.2f ", net->layers[8].weight_updates[27 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");


    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.4f ", net->layers[10].weight_updates[m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.4f ", net->layers[10].weight_updates[9 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.4f ", net->layers[10].weight_updates[18 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");

    //         for (int m = 0; m < 3; ++m)
    //         {
    //             for (int n = 0; n < 3; ++n)
    //             {
    //                 printf("%.4f ", net->layers[10].weight_updates[27 + m*3 + n]);
    //             }
    //             printf("\n");
                
    //         }
    //         printf("\n");


    //         printf("Done\n");
}

int main(int argc, char* argv[]){
    main_device(argc, argv);
}

#include "fused_device.h"
#include "fused_convolution_device.h"

#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"
#include "maxpool_layer.h"

#include "transport.h"

int NUM_TILES_X;
int NUM_TILES_Y;
int DEVICE_ID_X;
int DEVICE_ID_Y;


int stride_vector[16] = {1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1};
int filter_stack_vector[16] = {32, 32, 64, 64, 128, 64, 128, 128, 256, 128, 256, 256, 512, 256, 512, 256};
LAYER_TYPE layer_vector[16] = {CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL,
                               CONVOLUTIONAL, CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, CONVOLUTIONAL,
                               CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, CONVOLUTIONAL, CONVOLUTIONAL, CONVOLUTIONAL};
int filter_size_vector[16] = {3, 2, 3, 2, 3, 1, 1, 2, 3, 1, 3, 3, 3, 1, 3, 1};

// int stride_vector[11] = {1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1};
// int filter_stack_vector[11] = {32, 32, 64, 64, 128, 64, 128, 128, 256, 128, 256};
// LAYER_TYPE layer_vector[11] = {CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL,
//                                CONVOLUTIONAL, CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, CONVOLUTIONAL, CONVOLUTIONAL};
// int filter_size_vector[11] = {3, 3, 3, 3, 3, 1, 1, 3, 3, 1, 3};

int sync_group_vector[11] = {1, 1, 1, 1, 1, 1, 1 ,1 ,1 ,1 ,1};

// int stride_vector[8] = {1, 1, 1, 1, 1, 1, 1, 1};
// int filter_stack_vector[8] = {2, 2, 2, 2, 2, 2, 2, 2};
// int filter_size_vector[8] = {3, 3, 3, 3, 3, 3, 3, 3};

int main_device(int argc, char* argv[]){

    NUM_TILES_X = atoi(argv[1]);
    NUM_TILES_Y = atoi(argv[2]);
    DEVICE_ID_X = atoi(argv[19]);
    DEVICE_ID_Y = atoi(argv[20]);

    printf("%d %d %d %d\n", NUM_TILES_X, NUM_TILES_Y, DEVICE_ID_X, DEVICE_ID_Y);

    int INPUT_WIDTH = 608;
    int INPUT_HEIGHT = 608;
    int INPUT_CHANNELS = 3;


    train_groups_profile profile;

    #ifdef SERVER
        init_server();
    #else
        init_transport(argv);
    #endif

    profile.num_forward_groups = 1;
    profile.num_backward_groups = 6;

    profile.fp = calloc(profile.num_forward_groups, sizeof(group_profile_forward));
    profile.bp = calloc(profile.num_backward_groups, sizeof(group_profile_backward));

    // for (int i = 0; i < profile.num_forward_groups; ++i)
    // {
    //     /* code */
    // }

    profile.fp[0].layer_start_idx = 0;
    profile.fp[0].layer_end_idx = 10;
    profile.fp[0].start_x_forward = 0;
    profile.fp[0].start_y_forward = 0;
    profile.fp[0].end_x_forward = (76/NUM_TILES_X) - 1;
    profile.fp[0].end_y_forward = (76/NUM_TILES_Y) - 1;

    // profile.fp[1].layer_start_idx = 1;
    // profile.fp[1].layer_end_idx = 1;
    // profile.fp[1].start_x_forward = 0;
    // profile.fp[1].start_y_forward = 0;
    // profile.fp[1].end_x_forward = 151;
    // profile.fp[1].end_y_forward = 151;

    // profile.fp[2].layer_start_idx = 2;
    // profile.fp[2].layer_end_idx = 2;
    // profile.fp[2].start_x_forward = 0;
    // profile.fp[2].start_y_forward = 0;
    // profile.fp[2].end_x_forward = 151;
    // profile.fp[2].end_y_forward = 151;

    // profile.fp[3].layer_start_idx = 3;
    // profile.fp[3].layer_end_idx = 3;
    // profile.fp[3].start_x_forward = 0;
    // profile.fp[3].start_y_forward = 0;
    // profile.fp[3].end_x_forward = 75;
    // profile.fp[3].end_y_forward = 75;

    // profile.fp[4].layer_start_idx = 4;
    // profile.fp[4].layer_end_idx = 4;
    // profile.fp[4].start_x_forward = 0;
    // profile.fp[4].start_y_forward = 0;
    // profile.fp[4].end_x_forward = 75;
    // profile.fp[4].end_y_forward = 75;

    // profile.fp[5].layer_start_idx = 5;
    // profile.fp[5].layer_end_idx = 5;
    // profile.fp[5].start_x_forward = 0;
    // profile.fp[5].start_y_forward = 0;
    // profile.fp[5].end_x_forward = 75;
    // profile.fp[5].end_y_forward = 75;

    // profile.fp[6].layer_start_idx = 6;
    // profile.fp[6].layer_end_idx = 6;
    // profile.fp[6].start_x_forward = 0;
    // profile.fp[6].start_y_forward = 0;
    // profile.fp[6].end_x_forward = 75;
    // profile.fp[6].end_y_forward = 75;

    // profile.fp[7].layer_start_idx = 7;
    // profile.fp[7].layer_end_idx = 7;
    // profile.fp[7].start_x_forward = 0;
    // profile.fp[7].start_y_forward = 0;
    // profile.fp[7].end_x_forward = 37;
    // profile.fp[7].end_y_forward = 37;

    // profile.fp[8].layer_start_idx = 8;
    // profile.fp[8].layer_end_idx = 8;
    // profile.fp[8].start_x_forward = 0;
    // profile.fp[8].start_y_forward = 0;
    // profile.fp[8].end_x_forward = 37;
    // profile.fp[8].end_y_forward = 37;

    // profile.fp[9].layer_start_idx = 9;
    // profile.fp[9].layer_end_idx = 9;
    // profile.fp[9].start_x_forward = 0;
    // profile.fp[9].start_y_forward = 0;
    // profile.fp[9].end_x_forward = 37;
    // profile.fp[9].end_y_forward = 37;

    // profile.fp[10].layer_start_idx = 10;
    // profile.fp[10].layer_end_idx = 10;
    // profile.fp[10].start_x_forward = 0;
    // profile.fp[10].start_y_forward = 0;
    // profile.fp[10].end_x_forward = 37;
    // profile.fp[10].end_y_forward = 37;

    // profile.bp[0].layer_start_idx = 0;
    // profile.bp[0].layer_end_idx = 0;
    // profile.bp[0].start_x_backward = 0;
    // profile.bp[0].start_y_backward = 0;
    // profile.bp[0].end_x_backward = 303;//151;
    // profile.bp[0].end_y_backward = 303;

    profile.bp[0].layer_start_idx = 1;
    profile.bp[0].layer_end_idx = 1;
    profile.bp[0].start_x_backward = 0;
    profile.bp[0].start_y_backward = 0;
    profile.bp[0].end_x_backward = 303;
    profile.bp[0].end_y_backward = 303;

    profile.bp[1].layer_start_idx = 2;
    profile.bp[1].layer_end_idx = 2;
    profile.bp[1].start_x_backward = 0;
    profile.bp[1].start_y_backward = 0;
    profile.bp[1].end_x_backward = 151;
    profile.bp[1].end_y_backward = 151;

    profile.bp[2].layer_start_idx = 3;
    profile.bp[2].layer_end_idx = 3;
    profile.bp[2].start_x_backward = 0;
    profile.bp[2].start_y_backward = 0;
    profile.bp[2].end_x_backward = 151;
    profile.bp[2].end_y_backward = 151;

    profile.bp[3].layer_start_idx = 4;
    profile.bp[3].layer_end_idx = 6;
    profile.bp[3].start_x_backward = 0;
    profile.bp[3].start_y_backward = 0;
    profile.bp[3].end_x_backward = 75;
    profile.bp[3].end_y_backward = 75;

    profile.bp[4].layer_start_idx = 7;
    profile.bp[4].layer_end_idx = 7;
    profile.bp[4].start_x_backward = 0;
    profile.bp[4].start_y_backward = 0;
    profile.bp[4].end_x_backward = 75;
    profile.bp[4].end_y_backward = 75;

    profile.bp[5].layer_start_idx = 8;
    profile.bp[5].layer_end_idx = 10;
    profile.bp[5].start_x_backward = 0;
    profile.bp[5].start_y_backward = 0;
    profile.bp[5].end_x_backward = 37;
    profile.bp[5].end_y_backward = 37;

    // profile.bp[6].layer_start_idx = 12;
    // profile.bp[6].layer_end_idx = 15;
    // profile.bp[6].start_x_backward = 0;
    // profile.bp[6].start_y_backward = 0;
    // profile.bp[6].end_x_backward = 2;
    // profile.bp[6].end_y_backward = 2;


    network* net = calloc(1, sizeof(network));

    net->n = 11;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    float* INPUT_IMAGE = calloc(INPUT_CHANNELS*(INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), sizeof(float));
    fill_cpu(INPUT_CHANNELS*(INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), 0.1, INPUT_IMAGE, 1);


    for (int g = 0; g < profile.num_forward_groups; ++g)
    {

        partition_forward_device(net, 
                          &profile,
                          &(profile.fp[g]),
                            profile.fp[g].start_x_forward, profile.fp[g].start_y_forward,
                            profile.fp[g].end_x_forward, profile.fp[g].end_y_forward);

        net->inputs = net->layers[profile.fp[g].layer_start_idx].featuremap_in_h_with_boundry*
                        net->layers[profile.fp[g].layer_start_idx].featuremap_in_w_with_boundry*
                        net->layers[profile.fp[g].layer_start_idx].c;

        net->input = calloc(net->inputs, sizeof(float));

        //TODO Find this actual value
        int max = 0;
        for (int i = profile.fp[g].layer_start_idx; i <= profile.fp[g].layer_end_idx; ++i)
        {
            if(net->layers[i].workspace_size > max){
                max = net->layers[i].workspace_size;
            }
        }
        printf("wsize = %d inputs = %d outputs = %d\n", max*sizeof(float), net->inputs, net->layers[0].outputs);
        net->workspace = calloc(max, sizeof(float));

        //get boundry data here
        assemble_forward_group_data_device(net, 
                                        INPUT_IMAGE,
                                        NUM_TILES_X, NUM_TILES_Y,
                                         profile.fp[g],
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

        // free(net->input);
        int last_layer = profile.fp[g].layer_end_idx;

       free(net->workspace);
    }

    //free(INPUT_IMAGE);

    printf("Inference complete\n");

    int max = 0;
    for (int i = 0; i < net->n; ++i)
    {
        if(net->layers[i].workspace_size > max){
            max = net->layers[i].workspace_size;
        }
    }
    printf("wsize = %d inputs = %d outputs = %d\n", max*sizeof(float), net->inputs, net->layers[0].outputs);
    net->workspace = calloc(max, sizeof(float));

    float* OUTPUT_DELTA = calloc(net->layers[net->n - 1].outputs, sizeof(float));
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, OUTPUT_DELTA, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);


    for (int g = (profile.num_backward_groups-1); g >= 0; --g)
    {

        partition_backward_device(net, 
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


        printf("Received delta boundary. Starting backprop\n");

        int start_idx = (g==0) ? 1 : profile.bp[g].layer_start_idx;
        for (int l = profile.bp[g].layer_end_idx; l >= start_idx; --l)
        {
            printf("propagating at Layer %d\n", l);

            int unit_boundry = (net->layers[l].size / 2);

            int featuremap_without_boundry_width = net->layers[l].featuremap_in_w_without_boundry + (2*unit_boundry);
            int featuremap_without_boundry_height = net->layers[l].featuremap_in_h_without_boundry + (2*unit_boundry);



            for (int c = 0; c < net->layers[l].c; ++c)
            {
                for (int m = 0; m < featuremap_without_boundry_height; ++m)
                {
                    for (int n = 0; n < featuremap_without_boundry_width; ++n)
                    {
                        int left_edges = net->layers[l].left_boundry_edges_featuremap;
                        int right_edges = net->layers[l].right_boundry_edges_featuremap;
                        int top_edges = net->layers[l].top_boundry_edges_featuremap;
                        int bottom_edges = net->layers[l].bottom_boundry_edges_featuremap;

                        int x_dim_nob = net->layers[l].featuremap_in_w_with_boundry;
                        int y_dim_nob = net->layers[l].featuremap_in_h_with_boundry;

                        net->workspace[(c*featuremap_without_boundry_width*featuremap_without_boundry_height) + m*featuremap_without_boundry_width + n] = 
                            net->layers[l-1].output[(c*x_dim_nob*y_dim_nob) + (m+top_edges - unit_boundry)*(x_dim_nob) + n + left_edges - unit_boundry];

                    }
                }
            }



            memcpy(net->layers[l-1].output, net->workspace, net->layers[l].c*featuremap_without_boundry_height*featuremap_without_boundry_width*sizeof(float));


            net->input = net->layers[l-1].output;
            net->delta = net->layers[l-1].delta;

            net->layers[l].w = net->layers[l-1].delta_in_w_with_boundry;
            net->layers[l].h = net->layers[l-1].delta_in_h_with_boundry;
            net->layers[l].out_w = net->layers[l].delta_in_w_with_boundry;
            net->layers[l].out_h = net->layers[l].delta_in_h_with_boundry;

            printf("dims: %d %d %d %d\n", net->layers[l].w, net->layers[l].h, net->layers[l].out_w, net->layers[l].out_h);

            net->layers[l].pad = (net->layers[l].size - 1 - (( ( (net->layers[l].size/2) & 0x1 ) == 1 ) ? (net->layers[l].stride - 1) : 0));
            //Temporary hack: Fix this ASAP
            // if(l == 3)
            //     net->layers[l].pad += 1;

            zero_out_edges_delta_device(net, l, NUM_TILES_Y, NUM_TILES_X, DEVICE_ID_Y, DEVICE_ID_X);

            if(net->layers[l].type == CONVOLUTIONAL)
                backward_convolutional_layer_dist_delta(net->layers[l], *net);
            else if(net->layers[l].type == MAXPOOL)
                backward_maxpool_layer(net->layers[l], *net);



            if(net->layers[l].type == CONVOLUTIONAL){

                int x_dim = net->layers[l].delta_in_w_without_boundry;
                int y_dim = net->layers[l].delta_in_h_without_boundry;
                int depth = net->layers[l].n;

                for (int c = 0; c < net->layers[l].n; ++c)
                {
                    int x_dim_nob = net->layers[l].delta_in_w_with_boundry;
                    int y_dim_nob = net->layers[l].delta_in_h_with_boundry;

                    for (int m = 0; m < net->layers[l].delta_in_h_without_boundry; ++m)
                    {
                        for (int n = 0; n < net->layers[l].delta_in_w_without_boundry; ++n)
                        {
                            int left_edges = net->layers[l].left_boundry_edges_delta;
                            int right_edges = net->layers[l].right_boundry_edges_delta;
                            int top_edges = net->layers[l].top_boundry_edges_delta;
                            int bottom_edges = net->layers[l].bottom_boundry_edges_delta;

                            net->workspace[(c*x_dim*y_dim) + m*net->layers[l].delta_in_w_without_boundry + n] = 
                                net->layers[l].delta[(c*x_dim_nob*y_dim_nob) + (m+top_edges)*(net->layers[l].delta_in_w_with_boundry) + n + left_edges];
                        }
                    }
                }

                memcpy(net->layers[l].delta, net->workspace, x_dim*y_dim*depth*sizeof(float));

                net->layers[l].out_w = net->layers[l].delta_in_w_without_boundry;
                net->layers[l].out_h = net->layers[l].delta_in_h_without_boundry;
                net->layers[l].h = featuremap_without_boundry_height;
                net->layers[l].w = featuremap_without_boundry_width;

                net->layers[l].pad = 0;

                net->index = l;

                printf("filter layer %d \n", l);



                backward_convolutional_layer_dist_filters(net->layers[l], *net);
            }

        }

    }

    free(OUTPUT_DELTA);

    printf("Backprop complete\n");


            if(DEVICE_ID_X == 0 && DEVICE_ID_Y == 0)
                receive_sum_broadcast_weight_updates(net, NUM_TILES_Y, NUM_TILES_X);
            else{
            //#ifdef CLIENT
                sync_weight_updates(net, NUM_TILES_Y, NUM_TILES_X);
            }



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
                    printf("%.2f ", net->layers[4].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[4].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[4].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[4].weight_updates[27 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");



            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[8].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[8].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[8].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[8].weight_updates[27 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");


            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[10].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[10].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[10].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.4f ", net->layers[10].weight_updates[27 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");


            printf("Done\n");
}

int main(int argc, char* argv[]){
    main_device(argc, argv);
}

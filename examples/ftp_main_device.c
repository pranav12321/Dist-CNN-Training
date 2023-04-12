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

extern int is_device_representative_tile;

void backprop_layer0(network* net, float* INPUT_BOUNDRY);

int main_device(int argc, char* argv[]){

    clock_t begin;
    clock_t end;

    clock_t total_begin;
    clock_t total_end;

    double total_time = 0.0;
    double total_computation_time = 0.0;
    double boundary_communication_time = 0.0;
    double inference_time = 0.0;
    double backprop_time = 0.0;
    double filter_partial_updates_time = 0.0;
    double total_communication_time = 0.0;

    total_begin = clock();

    begin = clock();

    config_init(argc, argv);
    forward_pass();
    backward_pass();

    init_transport(argv);

    network* net;
    init_network(&net);

    float* INPUT_IMAGE = calloc(net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, sizeof(float));
    fill_cpu(net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, 0.1, INPUT_IMAGE, 1);

    float* OUTPUT_DELTA = calloc(net->layers[net->n - 1].outputs, sizeof(float));
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, OUTPUT_DELTA, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);

    net->inputs = net->layers[0].featuremap_in_h_with_boundry*
                    net->layers[0].featuremap_in_w_with_boundry*
                    net->layers[0].c;

    float* INPUT_BOUNDRY = calloc(net->inputs, sizeof(float));

    end = clock();

    total_computation_time += (double)(end - begin) / CLOCKS_PER_SEC;

    for (int g = 0; g < ftp_params.NUM_GROUPS_FORWARD; ++g)
    {

        begin = clock();
        int group_start_idx = ftp_params.sync_group_vector_forward[g];
        int group_end_idx = (g == (ftp_params.NUM_GROUPS_FORWARD - 1)) ? (net->n - 1) : (ftp_params.sync_group_vector_forward[g+1] - 1);
        net->inputs = net->layers[group_start_idx].featuremap_in_h_with_boundry*
                        net->layers[group_start_idx].featuremap_in_w_with_boundry*
                        net->layers[group_start_idx].c;

        net->input = calloc(net->inputs, sizeof(float));

        //get boundry data here
        assemble_forward_group_data_device(net, 
                                        INPUT_IMAGE,
                                        ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y,
                                         group_start_idx,
                                         ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y
                                         );
        end = clock();

        total_communication_time += (double)(end - begin) / CLOCKS_PER_SEC;
        boundary_communication_time += (double)(end - begin) / CLOCKS_PER_SEC;

        begin = clock();

        if(g == 0){
            memcpy(INPUT_BOUNDRY, net->input, net->inputs*sizeof(float));
        }

        printf("Received input boundary. Starting inference\n");

        for (int l = group_start_idx; l <= group_end_idx; ++l)
        {


            zero_out_edges_featuremap_device(net, l, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X, ftp_params.DEVICE_ID_Y, ftp_params.DEVICE_ID_X);
            zero_out_spurious_edges_featuremap(net, l);
            net->index = l;

                // printf("%d\n", net->layers[l].featuremap_in_w_with_boundry);
                // for (int i = 0; i < net->layers[l].featuremap_in_h_with_boundry; ++i)
                // {
                //     for (int j = 0; j < net->layers[l].featuremap_in_w_with_boundry; ++j)
                //     {
                //         printf("%.4f ", net->input[(i*net->layers[l].featuremap_in_w_with_boundry) + j]);
                //     }
                //     printf("\n");
                // }
                // printf("\n");

                // printf("%d\n", net->layers[l].featuremap_in_w_with_boundry);
                // for (int i = 0; i < net->layers[l].featuremap_in_h_with_boundry; ++i)
                // {
                //     for (int j = 0; j < net->layers[l].featuremap_in_w_with_boundry; ++j)
                //     {
                //         printf("%.4f ", net->input[(i*net->layers[l].featuremap_in_w_with_boundry) + j]);
                //     }
                //     printf("\n");
                // }
                // printf("\n");


            if(net->layers[l].type == CONVOLUTIONAL){
                forward_convolutional_layer(net->layers[l], *net);
            }

            else if(net->layers[l].type == MAXPOOL){
                forward_maxpool_layer(net->layers[l], *net);
            }

            // if(l > 0){
            //     for (int i = 0; i < net->layers[l-1].out_h; ++i)
            //     {
            //         for (int j = 0; j < net->layers[l-1].out_w; ++j)
            //         {
            //             printf("%.4f ", net->layers[l-1].output[(i*net->layers[l-1].out_w) + j]);
            //         }
            //         printf("\n");
            //     }
            //     printf("\n");
            // }


            if(l == group_start_idx){
                free(net->input);
            }

            net->input = net->layers[l].output;
        }

        end = clock();
        total_computation_time += (double)(end - begin) / CLOCKS_PER_SEC;
        inference_time += (double)(end - begin) / CLOCKS_PER_SEC;
    }

    // for (int i = 0; i < net->layers[10].out_h; ++i)
    // {
    //     for (int j = 0; j < net->layers[10].out_w; ++j)
    //     {
    //         printf("%.4f ", net->layers[10].output[(i*net->layers[10].out_w) + j]);
    //     }
    //     printf("\n");
    // }

    printf("Inference complete\n");

    for (int g = (ftp_params.NUM_GROUPS_BACKWARD-1); g >= 0; --g)
    {

        begin = clock();

        int group_start_idx = (g == 0) ? (1) : (ftp_params.sync_group_vector_backward[g-1] + 1);
        int group_end_idx = ftp_params.sync_group_vector_backward[g];

        assemble_backward_group_data_device(net, 
                                        OUTPUT_DELTA,
                                        ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y,
                                         group_end_idx,
                                         ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y,
                                         net->n
                                         );
        end = clock();

        total_communication_time += (double)(end - begin) / CLOCKS_PER_SEC;
        boundary_communication_time += (double)(end - begin) / CLOCKS_PER_SEC;

        printf("Received delta boundary. Starting backprop\n");

        begin = clock();        

        for (int l = group_end_idx; l >= group_start_idx; --l)
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

            net->layers[l].w = (l == group_start_idx) ? net->layers[l-1].delta_in_w_without_boundry : net->layers[l-1].delta_in_w_with_boundry;
            net->layers[l].h = (l == group_start_idx) ? net->layers[l-1].delta_in_h_without_boundry : net->layers[l-1].delta_in_h_with_boundry;
            net->layers[l].out_w = net->layers[l].delta_in_w_with_boundry;
            net->layers[l].out_h = net->layers[l].delta_in_h_with_boundry;

            net->layers[l].pad = net->layers[l].size - 1;

            zero_out_edges_delta_device(net, l, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X, ftp_params.DEVICE_ID_Y, ftp_params.DEVICE_ID_X);
            zero_out_spurious_edges_delta(net, l);

            if(net->layers[l].type == CONVOLUTIONAL)
                backward_convolutional_layer_dist_delta(net->layers[l], *net);
            else if(net->layers[l].type == MAXPOOL)
                backward_maxpool_layer(net->layers[l], *net);

            // for (int m = 0; m < net->layers[l].out_h; ++m)
            // {
            //     for (int n = 0; n < net->layers[l].out_w; ++n)
            //     {
            //         printf("%.2f ", net->layers[l].delta[m*net->layers[l].out_w + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

            // for (int m = 0; m < net->layers[l].h; ++m)
            // {
            //     for (int n = 0; n < net->layers[l].w; ++n)
            //     {
            //         printf("%.2f ", net->layers[l-1].delta[m*net->layers[l].w + n]);
            //     }
            //     printf("\n");
                
            // }
            // printf("\n");

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
                // int spurious_coord_x = (network_params_tile.spurious_blocks[l].start_x_coordinate > -1) ? network_params_tile.spurious_blocks[l].start_x_coordinate : net->layers[l].featuremap_in_w_without_boundry;
                // int spurious_coord_y = (network_params_tile.spurious_blocks[l].start_y_coordinate > -1) ? network_params_tile.spurious_blocks[l].start_y_coordinate : net->layers[l].featuremap_in_h_without_boundry;
                // int spurious_edge_x = net->layers[l].featuremap_in_w_without_boundry - spurious_coord_x;
                // int spurious_edge_y = net->layers[l].featuremap_in_h_without_boundry - spurious_coord_y;

                // printf("sp %d\n", spurious_edge_x);
                // printf("sp %d\n", spurious_edge_y);
                net->layers[l].h = featuremap_without_boundry_height;
                net->layers[l].w = featuremap_without_boundry_width;

                net->layers[l].pad = 0;

                net->index = l;

                printf("filter layer %d \n", l);

                backward_convolutional_layer_dist_filters(net->layers[l], *net);

            }

        }

        end = clock();
        total_computation_time += (double)(end - begin) / CLOCKS_PER_SEC;
        backprop_time += (double)(end - begin) / CLOCKS_PER_SEC;

    }

    begin = clock();     

    backprop_layer0(net, INPUT_BOUNDRY);

    free(OUTPUT_DELTA);

    end = clock();
    total_computation_time += (double)(end - begin) / CLOCKS_PER_SEC;
    backprop_time += (double)(end - begin) / CLOCKS_PER_SEC;

    printf("Backprop complete\n");

    begin = clock();

    if(ftp_params.DEVICE_ID_X == 0 && ftp_params.DEVICE_ID_Y == 0){
        receive_sum_transmit_device_weight_updates(net, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X);
    }
    else if(is_device_representative_tile){
        devices_send_partial_weight_updates(net, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X);
    }

    end = clock();
    sleep(1);

    total_communication_time += (double)(end - begin) / CLOCKS_PER_SEC;
    filter_partial_updates_time += (double)(end - begin) / CLOCKS_PER_SEC;

    total_end = clock();
    total_time += (double)(total_end - total_begin) / CLOCKS_PER_SEC;


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
                    printf("%.2f ", net->layers[5].weight_updates[m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[5].weight_updates[9 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[5].weight_updates[18 + m*3 + n]);
                }
                printf("\n");
                
            }
            printf("\n");

            for (int m = 0; m < 3; ++m)
            {
                for (int n = 0; n < 3; ++n)
                {
                    printf("%.2f ", net->layers[5].weight_updates[27 + m*3 + n]);
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

            printf("Total Time = %.4f\n", total_time);
            printf("Total Communication Time = %.4f\n", total_communication_time);
            printf("Total Computation Time = %.4f\n", total_computation_time);
            printf("Inference Computation Time = %.4f\n", inference_time);
            printf("Backprop Computation Time = %.4f\n", backprop_time);
            printf("Boundary Communication time = %.4f\n", boundary_communication_time);
            printf("Filter Update Time = %.4f\n", filter_partial_updates_time);

           FILE *fptr;

           char* file_prefix = "weights_device_";
           char file_name[20];
           strcpy(file_name, file_prefix);

           file_name[strlen(file_prefix)] = (char)(ftp_params.DEVICE_ID_X + 48);
           file_name[strlen(file_prefix) + 1] = (char)(ftp_params.DEVICE_ID_Y + 48);
           file_name[strlen(file_prefix) + 2] = '\0';

           fptr = fopen(file_name,"w");

           if(fptr == NULL)
           {
              printf("Error!");   
              exit(1);             
           }
           int layer_cumulative_weights = 0;

            for (int l = 0; l < net->n; ++l){

                int num_filters = net->layers[l].n;
                int filter_size = net->layers[l].size;
                int channels = net->layers[l].c;

                for (int c = 0; c < channels; ++c)
                {
                    for (int f = 0; f < num_filters; ++f)
                    {                    
                        for (int m = 0; m < filter_size; ++m)
                        {
                            for (int n = 0; n < filter_size; ++n)
                            {
                             fprintf(fptr,"%.4f ", net->layers[l].weight_updates[(c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n]);
                            }
                            fprintf(fptr, "\n");
                        }
                        fprintf(fptr, "\n\n");
                    }
                    fprintf(fptr, "\n\n\n");
                }

                fprintf(fptr, "\n\n\n\n");

                layer_cumulative_weights += num_filters*channels*filter_size*filter_size;
            }

            fprintf(fptr, "\n\n\n\n\n\n\n\n");

           fclose(fptr);

            printf("Done\n");

}

void backprop_layer0(network* net, float* INPUT_BOUNDRY){
    int unit_boundry = (net->layers[0].size / 2);

    int featuremap_without_boundry_width = net->layers[0].featuremap_in_w_without_boundry + (2*unit_boundry);
    int featuremap_without_boundry_height = net->layers[0].featuremap_in_h_without_boundry + (2*unit_boundry);



    for (int c = 0; c < net->layers[0].c; ++c)
    {
        for (int m = 0; m < featuremap_without_boundry_height; ++m)
        {
            for (int n = 0; n < featuremap_without_boundry_width; ++n)
            {
                int left_edges = net->layers[0].left_boundry_edges_featuremap;
                int right_edges = net->layers[0].right_boundry_edges_featuremap;
                int top_edges = net->layers[0].top_boundry_edges_featuremap;
                int bottom_edges = net->layers[0].bottom_boundry_edges_featuremap;

                int x_dim_nob = net->layers[0].featuremap_in_w_with_boundry;
                int y_dim_nob = net->layers[0].featuremap_in_h_with_boundry;

                net->workspace[(c*featuremap_without_boundry_width*featuremap_without_boundry_height) + m*featuremap_without_boundry_width + n] = 
                    INPUT_BOUNDRY[(c*x_dim_nob*y_dim_nob) + (m+top_edges - unit_boundry)*(x_dim_nob) + n + left_edges - unit_boundry];

            }
        }
    }

    memcpy(INPUT_BOUNDRY, net->workspace, net->layers[0].c*featuremap_without_boundry_height*featuremap_without_boundry_width*sizeof(float));
   
    net->layers[0].out_w = net->layers[0].delta_in_w_with_boundry;
    net->layers[0].out_h = net->layers[0].delta_in_h_with_boundry;
    net->layers[0].h = featuremap_without_boundry_height;
    net->layers[0].w = featuremap_without_boundry_width;

    net->layers[0].pad = 0;

    net->index = 0;

    // for (int m = 0; m < net->layers[0].out_h; ++m)
    // {
    //     for (int n = 0; n < net->layers[0].out_w; ++n)
    //     {
    //         printf("%.2f ", net->layers[0].delta[m*net->layers[0].out_w + n]);
    //     }
    //     printf("\n");
        
    // }
    // printf("\n");

    net->input = INPUT_BOUNDRY;

    printf("filter layer %d \n", 0);

    backward_convolutional_layer_dist_filters(net->layers[0], *net);   
}

int main(int argc, char* argv[]){
    main_device(argc, argv);
}

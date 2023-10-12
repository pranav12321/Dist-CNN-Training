#include "fused_device.h"
#include "fused_convolution_device.h"

#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"
#include "maxpool_layer.h"

#include "transport.h"
#include "fused_device.h"

#include <sys/time.h>
#include <unistd.h>


extern network_config network_params_original;
extern network_config network_params_tile;
extern ftp_config ftp_params;

extern device_tile current_tile;
extern network_device current_device;
extern ftp_network ftp_cluster;

extern int cumulative;

void backprop_layer0(network* net, float* INPUT_BOUNDRY);

int main_device(int argc, char* argv[]){

    struct timeval total_time_before, total_time_after, total_time_result;
    struct timeval step_time_before, step_time_after, step_time_result;

    double total_time = 0.0;
    double total_computation_time = 0.0;
    double boundary_communication_time = 0.0;
    double input_output_comm_time = 0.0;
    double inference_time = 0.0;
    double backprop_time = 0.0;
    double filter_partial_updates_time = 0.0;
    double train_complete_wait_time = 0.0;
    double actual_filter_sync_time = 0.0;
    double total_communication_time = 0.0;

#ifdef GPU
    cudaError_t status;
#endif

    config_init(argc, argv);
    forward_pass();
    backward_pass();

    init_transport(argv);

    network* net;
    init_network(&net);

    float* INPUT_IMAGE = calloc(net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, sizeof(float));
    float* OUTPUT_DELTA = calloc(net->batch*net->layers[net->n - 1].outputs, sizeof(float));

#ifdef GPU
    float* INPUT_IMAGE_GPU = cuda_make_array(INPUT_IMAGE, net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c);
#endif

    if(current_tile.is_device_representative_tile){
        init_complete_sema_wait(current_device.num_tiles - 1);
    }
    else{
        init_complete_sema_post(1);
    }

    gettimeofday(&total_time_before, NULL);
    gettimeofday(&step_time_before, NULL);

    if(ftp_params.DEVICE_ID_X == 0 && ftp_params.DEVICE_ID_Y == 0){

        fill_cpu(net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, 0.1, INPUT_IMAGE, 1);
        fill_cpu(net->batch*net->layers[net->n - 1].outputs, 0.1, OUTPUT_DELTA, 1);
        fill_cpu(net->batch*net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);

        for (int i = 0; i < ftp_params.NUM_TILES_X; ++i)
        {
            for (int j = 0; j < ftp_params.NUM_TILES_Y; ++j)
            {
                if(!((i == 0) && (j == 0))){
                   send_boundry(INPUT_IMAGE, net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, i, j);
                   send_boundry(OUTPUT_DELTA, net->batch*net->layers[net->n - 1].outputs, i, j);
                }
            }
        }
    }
    else{
       cumulative += (net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c);
       //printf("CUMULATIVE = %d\n",  cumulative);
       receive_boundry(INPUT_IMAGE, net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, 0, 0);
       receive_boundry(OUTPUT_DELTA, net->batch*net->layers[net->n - 1].outputs, 0, 0);
    }

    gettimeofday(&step_time_after, NULL);

    memcpy(net->layers[net->n - 1].delta, OUTPUT_DELTA, net->batch*net->layers[net->n - 1].outputs*sizeof(float));

    timersub(&step_time_after, &step_time_before, &step_time_result);

    total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    input_output_comm_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

    net->inputs =   net->batch*
                    net->layers[0].featuremap_in_h_with_boundry*
                    net->layers[0].featuremap_in_w_with_boundry*
                    net->layers[0].c;

    float* NETWORK_INPUT_PTR;

    for (int g = 0; g < ftp_params.NUM_GROUPS_FORWARD; ++g)
    {

        gettimeofday(&step_time_before, NULL);

        int group_start_idx = ftp_params.sync_group_vector_forward[g];
        int group_end_idx = (g == (ftp_params.NUM_GROUPS_FORWARD - 1)) ? (net->n - 1) : (ftp_params.sync_group_vector_forward[g+1] - 1);
        
        net->inputs =   net->batch*
                        net->layers[group_start_idx].featuremap_in_h_with_boundry*
                        net->layers[group_start_idx].featuremap_in_w_with_boundry*
                        net->layers[group_start_idx].c;

        net->input = calloc(net->inputs, sizeof(float));
        float* group_input = (group_start_idx == 0) ?  INPUT_IMAGE : net->layers[group_start_idx - 1].output;

#ifdef GPU
        status = cudaMalloc((void **)&net->input_gpu, net->inputs*sizeof(float));
        check_error(status);
        float* group_input_gpu = (group_start_idx == 0) ?  INPUT_IMAGE_GPU : net->layers[group_start_idx - 1].output_gpu;
#endif

        //get boundry data here
#ifdef GPU
        assemble_tile_gpu(net, net->batch, net->layers[group_start_idx].c,
                        net->input_gpu, group_input_gpu,
                        net->layers[group_start_idx].featuremap_in_h_without_boundry, net->layers[group_start_idx].featuremap_in_w_without_boundry,
                        net->layers[group_start_idx].left_boundry_edges_featuremap, net->layers[group_start_idx].right_boundry_edges_featuremap,
                        net->layers[group_start_idx].top_boundry_edges_featuremap, net->layers[group_start_idx].bottom_boundry_edges_featuremap,
                       ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y);
        if(group_start_idx > 0){
            status = cudaMemcpy(net->layers[group_start_idx - 1].output_gpu, net->input_gpu, net->inputs*sizeof(float), cudaMemcpyDeviceToDevice);
            check_error(status);
        }
#else
        assemble_tile(net, net->batch, net->layers[group_start_idx].c,
                        net->input, group_input,
                        net->layers[group_start_idx].featuremap_in_h_without_boundry, net->layers[group_start_idx].featuremap_in_w_without_boundry,
                        net->layers[group_start_idx].left_boundry_edges_featuremap, net->layers[group_start_idx].right_boundry_edges_featuremap,
                        net->layers[group_start_idx].top_boundry_edges_featuremap, net->layers[group_start_idx].bottom_boundry_edges_featuremap,
                        ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y);
        if(group_start_idx > 0)
            memcpy(net->layers[group_start_idx - 1].output, net->input, net->inputs*sizeof(float));
#endif

        gettimeofday(&step_time_after, NULL);

        timersub(&step_time_after, &step_time_before, &step_time_result);

        total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        boundary_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        gettimeofday(&step_time_before, NULL);

        if(g == 0)
            NETWORK_INPUT_PTR = net->input;

        printf("Received input boundary. Starting inference\n");

        for (int l = group_start_idx; l <= group_end_idx; ++l)
        {

#ifdef GPU
            //clear_edges_featuremap_device_gpu(net, l, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X, ftp_params.DEVICE_ID_Y, ftp_params.DEVICE_ID_X);
            clear_spurious_edges_featuremap_gpu(net, l);
            net->index = l;
            net->layers[l].forward_gpu(net->layers[l], *net);

            if((l == group_start_idx) && (l > 0)){
                cuda_free(net->input_gpu);
            }

            net->input_gpu = net->layers[l].output_gpu;
#else
            clear_edges_featuremap_device(net, l, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X, ftp_params.DEVICE_ID_Y, ftp_params.DEVICE_ID_X);
            clear_spurious_edges_featuremap(net, l);

            net->index = l;
            net->layers[l].forward(net->layers[l], *net);

            if((l == group_start_idx) && (group_start_idx > 0)){
                free(net->input);
            }

            net->input = net->layers[l].output;
#endif
        }

        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);

        total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        inference_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    }

    printf("Inference complete\n");

    update_args a;
    a.batch = net->batch;
    a.learning_rate = 0.001;
    a.momentum = 0.9;
    a.decay = 0.0005;

    for (int g = (ftp_params.NUM_GROUPS_BACKWARD-1); g >= 0; --g)
    {

        gettimeofday(&step_time_before, NULL);

        int group_start_idx = (g == 0) ? (1) : (ftp_params.sync_group_vector_backward[g-1] + 1);
        int group_end_idx = ftp_params.sync_group_vector_backward[g];

        int depth = (net->layers[group_end_idx].type == CONVOLUTIONAL) ? (net->layers[group_end_idx].n) : (net->layers[group_end_idx].c);

#ifdef GPU
        assemble_tile_gpu(net, net->batch, depth,
                        net->layers[group_end_idx].delta_gpu, net->layers[group_end_idx].delta_gpu,
                        net->layers[group_end_idx].delta_in_h_without_boundry, net->layers[group_end_idx].delta_in_w_without_boundry,
                        net->layers[group_end_idx].left_boundry_edges_delta, net->layers[group_end_idx].right_boundry_edges_delta,
                        net->layers[group_end_idx].top_boundry_edges_delta, net->layers[group_end_idx].bottom_boundry_edges_delta,
                        ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y);
#else        
        assemble_tile(net, net->batch, depth,
                        net->layers[group_end_idx].delta, net->layers[group_end_idx].delta,
                        net->layers[group_end_idx].delta_in_h_without_boundry, net->layers[group_end_idx].delta_in_w_without_boundry,
                        net->layers[group_end_idx].left_boundry_edges_delta, net->layers[group_end_idx].right_boundry_edges_delta,
                        net->layers[group_end_idx].top_boundry_edges_delta, net->layers[group_end_idx].bottom_boundry_edges_delta,
                        ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y);
#endif

        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);

        total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        boundary_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        printf("Received delta boundary. Starting backprop\n");

        gettimeofday(&step_time_before, NULL);        

        for (int l = group_end_idx; l >= group_start_idx; --l)
        {
            gettimeofday(&step_time_before, NULL);

            if(net->layers[l].type == MAXPOOL){
                assemble_pool_indices(net, l, ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y);
            }

            gettimeofday(&step_time_after, NULL);
            timersub(&step_time_after, &step_time_before, &step_time_result);
            total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
            boundary_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

            gettimeofday(&step_time_before, NULL);  

            printf("propagating at Layer %d\n", l);

            int left_edges = net->layers[l].left_boundry_edges_featuremap;
            int right_edges = net->layers[l].right_boundry_edges_featuremap;
            int top_edges = net->layers[l].top_boundry_edges_featuremap;
            int bottom_edges = net->layers[l].bottom_boundry_edges_featuremap;

            int unit_boundry = (net->layers[l].size / 2);
            int featuremap_with_unit_boundry_width = net->layers[l].featuremap_in_w_without_boundry + (2*unit_boundry);
            int featuremap_with_unit_boundry_height = net->layers[l].featuremap_in_h_without_boundry + (2*unit_boundry);

            copy_slice(net->layers[l-1].output, net->layers[l-1].output, net->batch, net->layers[l].c,
                            net->layers[l].featuremap_in_h_with_boundry, net->layers[l].featuremap_in_w_with_boundry,
                            featuremap_with_unit_boundry_height, featuremap_with_unit_boundry_width,
                            net->layers[l].left_boundry_edges_featuremap - unit_boundry, net->layers[l].top_boundry_edges_featuremap - unit_boundry,
                            0, 0,
                            featuremap_with_unit_boundry_height, featuremap_with_unit_boundry_width, featuremap_with_unit_boundry_height, featuremap_with_unit_boundry_width,
                            net->workspace);
            
            net->input = net->layers[l-1].output;
            net->delta = net->layers[l-1].delta;

            net->layers[l].w = (l == group_start_idx) ? net->layers[l-1].delta_in_w_without_boundry : net->layers[l-1].delta_in_w_with_boundry;
            net->layers[l].h = (l == group_start_idx) ? net->layers[l-1].delta_in_h_without_boundry : net->layers[l-1].delta_in_h_with_boundry;
            net->layers[l].out_w = net->layers[l].delta_in_w_with_boundry;
            net->layers[l].out_h = net->layers[l].delta_in_h_with_boundry;

            net->layers[l].pad = net->layers[l].size - 1;

            clear_edges_delta_device(net, l, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X, ftp_params.DEVICE_ID_Y, ftp_params.DEVICE_ID_X);
            clear_spurious_edges_delta(net, l);

            if(net->layers[l].type == CONVOLUTIONAL)
                backward_convolutional_layer_dist_delta(net->layers[l], *net);
            else if(net->layers[l].type == MAXPOOL){
                backward_maxpool_layer(net->layers[l], *net);
                remove_extra_boundary_data(net, l);
            }

            if(net->layers[l].type == CONVOLUTIONAL){

                copy_slice(net->layers[l].delta, net->layers[l].delta, net->batch, net->layers[l].n,
                                net->layers[l].delta_in_h_with_boundry, net->layers[l].delta_in_w_with_boundry,
                                net->layers[l].delta_in_h_without_boundry, net->layers[l].delta_in_w_without_boundry,
                                net->layers[l].left_boundry_edges_delta, net->layers[l].top_boundry_edges_delta,
                                0, 0,
                                net->layers[l].delta_in_h_without_boundry, net->layers[l].delta_in_w_without_boundry,
                                net->layers[l].delta_in_h_without_boundry, net->layers[l].delta_in_w_without_boundry,
                                net->workspace);                    

                net->layers[l].out_w = net->layers[l].delta_in_w_without_boundry;
                net->layers[l].out_h = net->layers[l].delta_in_h_without_boundry;
                net->layers[l].h = featuremap_with_unit_boundry_height;
                net->layers[l].w = featuremap_with_unit_boundry_width;

                net->layers[l].pad = 0;

                net->index = l;

                printf("filter layer %d \n", l);

                backward_convolutional_layer_dist_filters(net->layers[l], *net);

            }

            gettimeofday(&step_time_after, NULL);
            timersub(&step_time_after, &step_time_before, &step_time_result);
            total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
            backprop_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        }

    }

    gettimeofday(&step_time_before, NULL);    

    backprop_layer0(net, NETWORK_INPUT_PTR);

    free(OUTPUT_DELTA);

    gettimeofday(&step_time_after, NULL);
    timersub(&step_time_after, &step_time_before, &step_time_result);

    total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    backprop_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

    printf("Backprop complete\n");

    gettimeofday(&step_time_before, NULL);

    if(ftp_params.DEVICE_ID_X == 0 && ftp_params.DEVICE_ID_Y == 0){

        gettimeofday(&step_time_before, NULL);
        train_cycle_complete_sema_wait(current_device.num_tiles - 1);
        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);
        train_complete_wait_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        gettimeofday(&step_time_before, NULL);
        receive_sum_transmit_device_weight_updates(net, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X);
        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);
        actual_filter_sync_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    }
    else if(current_tile.is_device_representative_tile){
        gettimeofday(&step_time_before, NULL);
        train_cycle_complete_sema_wait(current_device.num_tiles - 1);
        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);
        train_complete_wait_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        gettimeofday(&step_time_before, NULL);
        devices_send_partial_weight_updates(net, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X);
        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);
        actual_filter_sync_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    }
    else{
        gettimeofday(&step_time_before, NULL);
        train_cycle_complete_sema_post(1);
        filter_sync_complete_sema_wait(1);
        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);
        actual_filter_sync_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    }

    gettimeofday(&step_time_after, NULL);
    timersub(&step_time_after, &step_time_before, &step_time_result);

    total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    filter_partial_updates_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);


    gettimeofday(&step_time_before, NULL);
    for (int l = 0; l < net->n; l++)
    {
        if(net->layers[l].type == CONVOLUTIONAL){
            net->layers[l].learning_rate_scale = 1.0;
            update_convolutional_layer(net->layers[l], a);
        }
    }
    gettimeofday(&step_time_after, NULL);
    timersub(&step_time_after, &step_time_before, &step_time_result);
    total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    backprop_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

    gettimeofday(&total_time_after, NULL);
    timersub(&total_time_after, &total_time_before, &total_time_result);
    total_time += (double)(total_time_result.tv_sec + (total_time_result.tv_usec)/1000000.0);


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

            printf("Total Time = %.4f\n", total_time);
            printf("Total Communication Time = %.4f\n", total_communication_time);
            printf("Total Computation Time = %.4f\n", total_computation_time);
            printf("Inference Computation Time = %.4f\n", inference_time);
            printf("Backprop Computation Time = %.4f\n", backprop_time);
            printf("I/O comm time = %.4f\n", input_output_comm_time);
            printf("Boundary Communication time = %.4f\n", boundary_communication_time);
            printf("Filter Update Time = %.4f\n", filter_partial_updates_time);
            printf("\t Train complete wait time = %.4f\n", train_complete_wait_time);
            printf("\t Actual filter sync time = %.4f\n", actual_filter_sync_time);

           FILE *fptr;

           char* file_prefix = "weight_updates_device_";
           char file_name[30];
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

            for (int l = 0; l < net->n; ++l){
                
                if(net->layers[l].type == CONVOLUTIONAL){
                    int num_filters = net->layers[l].n;
                    int filter_size = net->layers[l].size;
                    int channels = net->layers[l].c;


                    for (int n = 0; n < (channels*filter_size*filter_size*num_filters); ++n)
                    {
                        fprintf(fptr,"%.4f\n", net->layers[l].weights[n]);
                    }
                    fprintf(fptr,"\n\n");
                }
            }

           fclose(fptr);

            printf("Done\n");

}

void backprop_layer0(network* net, float* INPUT_BOUNDRY){
    int unit_boundry = (net->layers[0].size / 2);

    int featuremap_without_boundry_width = net->layers[0].featuremap_in_w_without_boundry + (2*unit_boundry);
    int featuremap_without_boundry_height = net->layers[0].featuremap_in_h_without_boundry + (2*unit_boundry);

    int left_edges = net->layers[0].left_boundry_edges_featuremap;
    int right_edges = net->layers[0].right_boundry_edges_featuremap;
    int top_edges = net->layers[0].top_boundry_edges_featuremap;
    int bottom_edges = net->layers[0].bottom_boundry_edges_featuremap;

    int x_dim_nob = net->layers[0].featuremap_in_w_with_boundry;
    int y_dim_nob = net->layers[0].featuremap_in_h_with_boundry;

    int sample_size_unit_boundry = featuremap_without_boundry_width*featuremap_without_boundry_height*net->layers[0].c;
    int sample_size_with_boundry = x_dim_nob*y_dim_nob*net->layers[0].c;

    for(int b = 0; b < net->batch; b++)
    {
        for (int c = 0; c < net->layers[0].c; ++c)
        {
            for (int m = 0; m < featuremap_without_boundry_height; ++m)
            {
                for (int n = 0; n < featuremap_without_boundry_width; ++n)
                {
                    net->workspace[(b*sample_size_unit_boundry) + (c*featuremap_without_boundry_width*featuremap_without_boundry_height) + m*featuremap_without_boundry_width + n] = 
                        INPUT_BOUNDRY[(b*sample_size_with_boundry) + (c*x_dim_nob*y_dim_nob) + (m+top_edges - unit_boundry)*(x_dim_nob) + n + left_edges - unit_boundry];
                }
            }
        }
    }

    memcpy(INPUT_BOUNDRY, net->workspace, net->batch*net->layers[0].c*featuremap_without_boundry_height*featuremap_without_boundry_width*sizeof(float));
   
    net->layers[0].out_w = net->layers[0].delta_in_w_with_boundry;
    net->layers[0].out_h = net->layers[0].delta_in_h_with_boundry;
    net->layers[0].h = featuremap_without_boundry_height;
    net->layers[0].w = featuremap_without_boundry_width;

    net->layers[0].pad = 0;

    net->index = 0;

    net->input = INPUT_BOUNDRY;

    printf("filter layer %d \n", 0);

    backward_convolutional_layer_dist_filters(net->layers[0], *net);   
}


int main_inference(int argc, char* argv[]){

    struct timeval total_time_before, total_time_after, total_time_result;
    struct timeval step_time_before, step_time_after, step_time_result;

    double total_time = 0.0;
    double total_computation_time = 0.0;
    double boundary_communication_time = 0.0;
    double input_output_comm_time = 0.0;
    double inference_time = 0.0;
    double backprop_time = 0.0;
    double filter_partial_updates_time = 0.0;
    double total_communication_time = 0.0;

    config_init(argc, argv);
    forward_pass();
    backward_pass();

    init_transport(argv);

    network* net;
    init_network(&net);

    float* INPUT_IMAGE = calloc(net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, sizeof(float));
    float* OUTPUT_DELTA = calloc(net->batch*net->layers[net->n - 1].outputs, sizeof(float));

    if(current_tile.is_device_representative_tile){
        init_complete_sema_wait(current_device.num_tiles - 1);
    }
    else{
        init_complete_sema_post(1);
    }

    gettimeofday(&total_time_before, NULL);
    gettimeofday(&step_time_before, NULL);

    if(ftp_params.DEVICE_ID_X == 0 && ftp_params.DEVICE_ID_Y == 0){

        fill_cpu(net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, 0.1, INPUT_IMAGE, 1);
        fill_cpu(net->layers[net->n - 1].outputs, 0.1, OUTPUT_DELTA, 1);
        fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);

        for (int i = 0; i < ftp_params.NUM_TILES_X; ++i)
        {
            for (int j = 0; j < ftp_params.NUM_TILES_Y; ++j)
            {
                if(!((i == 0) && (j == 0))){
                   send_boundry(INPUT_IMAGE, net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, i, j);
                   send_boundry(OUTPUT_DELTA, net->batch*net->layers[net->n - 1].outputs, i, j);
                }
            }
        }
    }
    else{
       cumulative += (net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c);
       printf("CUMULATIVE = %d\n",  cumulative);
       usleep(net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c*50);
       receive_boundry(INPUT_IMAGE, net->batch*net->layers[0].featuremap_in_h_without_boundry*net->layers[0].featuremap_in_w_without_boundry*net->layers[0].c, 0, 0);
       receive_boundry(OUTPUT_DELTA, net->batch*net->layers[net->n - 1].outputs, 0, 0);
    }

    gettimeofday(&step_time_after, NULL);

    memcpy(net->layers[net->n - 1].delta, OUTPUT_DELTA, net->batch*net->layers[net->n - 1].outputs*sizeof(float));

    timersub(&step_time_after, &step_time_before, &step_time_result);

    total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    input_output_comm_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

    net->inputs =   net->batch*
                    net->layers[0].featuremap_in_h_with_boundry*
                    net->layers[0].featuremap_in_w_with_boundry*
                    net->layers[0].c;

    float* NETWORK_INPUT_PTR;

    for (int g = 0; g < ftp_params.NUM_GROUPS_FORWARD; ++g)
    {

        gettimeofday(&step_time_before, NULL);

        int group_start_idx = ftp_params.sync_group_vector_forward[g];
        int group_end_idx = (g == (ftp_params.NUM_GROUPS_FORWARD - 1)) ? (net->n - 1) : (ftp_params.sync_group_vector_forward[g+1] - 1);
        
        net->inputs =   net->batch*
                        net->layers[group_start_idx].featuremap_in_h_with_boundry*
                        net->layers[group_start_idx].featuremap_in_w_with_boundry*
                        net->layers[group_start_idx].c;

        net->input = calloc(net->inputs, sizeof(float));

        //get boundry data here
        // assemble_forward_group_data_device(net, 
        //                                 INPUT_IMAGE,
        //                                 ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y,
        //                                  group_start_idx,
        //                                  ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y
        //                                  );
        gettimeofday(&step_time_after, NULL);

        timersub(&step_time_after, &step_time_before, &step_time_result);

        total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        boundary_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        gettimeofday(&step_time_before, NULL);

        if(g == 0){
            NETWORK_INPUT_PTR = net->input;
        }

        printf("Received input boundary. Starting inference\n");

        for (int l = group_start_idx; l <= group_end_idx; ++l)
        {
            clear_edges_featuremap_device(net, l, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X, ftp_params.DEVICE_ID_Y, ftp_params.DEVICE_ID_X);
            clear_spurious_edges_featuremap(net, l);
            net->index = l;

            if(net->layers[l].type == CONVOLUTIONAL){
                forward_convolutional_layer(net->layers[l], *net);
            }

            else if(net->layers[l].type == MAXPOOL){
                forward_maxpool_layer(net->layers[l], *net);
            }

            if(l == group_start_idx){
                free(net->input);
            }

            net->input = net->layers[l].output;
        }

        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);

        total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        inference_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    }

    printf("Inference complete\n");

    gettimeofday(&total_time_after, NULL);
    timersub(&total_time_after, &total_time_before, &total_time_result);
    total_time += (double)(total_time_result.tv_sec + (total_time_result.tv_usec)/1000000.0);

    printf("Total Time = %.4f\n", total_time);
    printf("Total Communication Time = %.4f\n", total_communication_time);
    printf("Total Computation Time = %.4f\n", total_computation_time);
    printf("Inference Computation Time = %.4f\n", inference_time);
    printf("Backprop Computation Time = %.4f\n", backprop_time);
    printf("I/O comm time = %.4f\n", input_output_comm_time);
    printf("Boundary Communication time = %.4f\n", boundary_communication_time);
    printf("Filter Update Time = %.4f\n", filter_partial_updates_time);
}

int main_backprop(int argc, char* argv[]){

    struct timeval total_time_before, total_time_after, total_time_result;
    struct timeval step_time_before, step_time_after, step_time_result;

    double total_time = 0.0;
    double total_computation_time = 0.0;
    double boundary_communication_time = 0.0;
    double input_output_comm_time = 0.0;
    double inference_time = 0.0;
    double backprop_time = 0.0;
    double filter_partial_updates_time = 0.0;
    double total_communication_time = 0.0;

    config_init(argc, argv);
    forward_pass();
    backward_pass();

    init_transport(argv);

    network* net;
    init_network(&net);

    float* OUTPUT_DELTA = calloc(net->batch*net->layers[net->n - 1].outputs, sizeof(float));

    if(current_tile.is_device_representative_tile){
        init_complete_sema_wait(current_device.num_tiles - 1);
    }
    else{
        init_complete_sema_post(1);
    }

    gettimeofday(&total_time_before, NULL);
    gettimeofday(&step_time_before, NULL);

    if(ftp_params.DEVICE_ID_X == 0 && ftp_params.DEVICE_ID_Y == 0){

        fill_cpu(net->layers[net->n - 1].outputs, 0.1, OUTPUT_DELTA, 1);
        fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);

        for (int i = 0; i < ftp_params.NUM_TILES_X; ++i)
        {
            for (int j = 0; j < ftp_params.NUM_TILES_Y; ++j)
            {
                if(!((i == 0) && (j == 0))){
                   send_boundry(OUTPUT_DELTA, net->batch*net->layers[net->n - 1].outputs, i, j);
                }
            }
        }
    }
    else{
       cumulative += (net->batch*net->layers[net->n - 1].outputs);
       printf("CUMULATIVE = %d\n",  cumulative);
       receive_boundry(OUTPUT_DELTA, net->batch*net->layers[net->n - 1].outputs, 0, 0);
    }

    gettimeofday(&step_time_after, NULL);

    memcpy(net->layers[net->n - 1].delta, OUTPUT_DELTA, net->batch*net->layers[net->n - 1].outputs*sizeof(float));

    timersub(&step_time_after, &step_time_before, &step_time_result);

    total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    input_output_comm_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

    net->inputs =   net->batch*
                    net->layers[0].featuremap_in_h_with_boundry*
                    net->layers[0].featuremap_in_w_with_boundry*
                    net->layers[0].c;
    float* INPUT_BOUNDRY = calloc(net->inputs, sizeof(float));
    fill_cpu(net->inputs, 0.1, INPUT_BOUNDRY, 1);

    update_args a;
    a.batch = net->batch;
    a.learning_rate = 0.001;
    a.momentum = 0.9;
    a.decay = 0.0005;

    for (int g = (ftp_params.NUM_GROUPS_BACKWARD-1); g >= 0; --g)
    {

        gettimeofday(&step_time_before, NULL);

        int group_start_idx = (g == 0) ? (1) : (ftp_params.sync_group_vector_backward[g-1] + 1);
        int group_end_idx = ftp_params.sync_group_vector_backward[g];

        int depth = (net->layers[group_end_idx].type == CONVOLUTIONAL) ? (net->layers[group_end_idx].n) : (net->layers[group_end_idx].c);
        
        assemble_tile(net, net->batch, depth,
                        net->layers[group_end_idx].delta, net->layers[group_end_idx].delta,
                        net->layers[group_end_idx].delta_in_h_without_boundry, net->layers[group_end_idx].delta_in_w_without_boundry,
                        net->layers[group_end_idx].left_boundry_edges_delta, net->layers[group_end_idx].right_boundry_edges_delta,
                        net->layers[group_end_idx].top_boundry_edges_delta, net->layers[group_end_idx].bottom_boundry_edges_delta,
                        ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y);

        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);

        total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        boundary_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        printf("Received delta boundary. Starting backprop\n");

        gettimeofday(&step_time_before, NULL);        

        for (int l = group_end_idx; l >= group_start_idx; --l)
        {
            printf("propagating at Layer %d\n", l);

            int left_edges = net->layers[l].left_boundry_edges_featuremap;
            int right_edges = net->layers[l].right_boundry_edges_featuremap;
            int top_edges = net->layers[l].top_boundry_edges_featuremap;
            int bottom_edges = net->layers[l].bottom_boundry_edges_featuremap;

            int unit_boundry = (net->layers[l].size / 2);
            int featuremap_with_unit_boundry_width = net->layers[l].featuremap_in_w_without_boundry + (2*unit_boundry);
            int featuremap_with_unit_boundry_height = net->layers[l].featuremap_in_h_without_boundry + (2*unit_boundry);

            copy_slice(net->layers[l-1].output, net->layers[l-1].output, net->batch, net->layers[l].c,
                            net->layers[l].featuremap_in_h_with_boundry, net->layers[l].featuremap_in_w_with_boundry,
                            featuremap_with_unit_boundry_height, featuremap_with_unit_boundry_width,
                            net->layers[l].left_boundry_edges_featuremap - unit_boundry, net->layers[l].top_boundry_edges_featuremap - unit_boundry,
                            0, 0,
                            featuremap_with_unit_boundry_height, featuremap_with_unit_boundry_width, featuremap_with_unit_boundry_height, featuremap_with_unit_boundry_width,
                            net->workspace);
            if(net->layers[l].type == MAXPOOL){
                 assemble_pool_indices(net, l, ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y);
            }
            

            net->input = net->layers[l-1].output;
            net->delta = net->layers[l-1].delta;

            net->layers[l].w = (l == group_start_idx) ? net->layers[l-1].delta_in_w_without_boundry : net->layers[l-1].delta_in_w_with_boundry;
            net->layers[l].h = (l == group_start_idx) ? net->layers[l-1].delta_in_h_without_boundry : net->layers[l-1].delta_in_h_with_boundry;
            net->layers[l].out_w = net->layers[l].delta_in_w_with_boundry;
            net->layers[l].out_h = net->layers[l].delta_in_h_with_boundry;

            net->layers[l].pad = net->layers[l].size - 1;

            clear_edges_delta_device(net, l, ftp_params.NUM_TILES_Y, ftp_params.NUM_TILES_X, ftp_params.DEVICE_ID_Y, ftp_params.DEVICE_ID_X);
            clear_spurious_edges_delta(net, l);

            if(net->layers[l].type == CONVOLUTIONAL)
                backward_convolutional_layer_dist_delta(net->layers[l], *net);
            else if(net->layers[l].type == MAXPOOL){
                backward_maxpool_layer(net->layers[l], *net);
                remove_extra_boundary_data(net, l);
            }

            if(net->layers[l].type == CONVOLUTIONAL){

                copy_slice(net->layers[l].delta, net->layers[l].delta, net->batch, net->layers[l].n,
                                net->layers[l].delta_in_h_with_boundry, net->layers[l].delta_in_w_with_boundry,
                                net->layers[l].delta_in_h_without_boundry, net->layers[l].delta_in_w_without_boundry,
                                net->layers[l].left_boundry_edges_delta, net->layers[l].top_boundry_edges_delta,
                                0, 0,
                                net->layers[l].delta_in_h_without_boundry, net->layers[l].delta_in_w_without_boundry,
                                net->layers[l].delta_in_h_without_boundry, net->layers[l].delta_in_w_without_boundry,
                                net->workspace);                    

                net->layers[l].out_w = net->layers[l].delta_in_w_without_boundry;
                net->layers[l].out_h = net->layers[l].delta_in_h_without_boundry;
                net->layers[l].h = featuremap_with_unit_boundry_height;
                net->layers[l].w = featuremap_with_unit_boundry_width;

                net->layers[l].pad = 0;

                net->index = l;

                printf("filter layer %d \n", l);

                backward_convolutional_layer_dist_filters(net->layers[l], *net);

            }

        }

        gettimeofday(&step_time_after, NULL);
        timersub(&step_time_after, &step_time_before, &step_time_result);
        total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        backprop_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

    }

    gettimeofday(&step_time_before, NULL);    

    backprop_layer0(net, INPUT_BOUNDRY);

    free(OUTPUT_DELTA);

    gettimeofday(&step_time_after, NULL);
    timersub(&step_time_after, &step_time_before, &step_time_result);

    total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    backprop_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

    printf("Backprop complete\n");

    gettimeofday(&step_time_before, NULL);
    for (int l = 0; l < net->n; l++)
    {
        if(net->layers[l].type == CONVOLUTIONAL){
            net->layers[l].learning_rate_scale = 1.0;
            update_convolutional_layer(net->layers[l], a);
        }
    }
    gettimeofday(&step_time_after, NULL);
    timersub(&step_time_after, &step_time_before, &step_time_result);
    total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    backprop_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

    gettimeofday(&total_time_after, NULL);
    timersub(&total_time_after, &total_time_before, &total_time_result);
    total_time += (double)(total_time_result.tv_sec + (total_time_result.tv_usec)/1000000.0);


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

            printf("Total Time = %.4f\n", total_time);
            printf("Total Communication Time = %.4f\n", total_communication_time);
            printf("Total Computation Time = %.4f\n", total_computation_time);
            printf("Inference Computation Time = %.4f\n", inference_time);
            printf("Backprop Computation Time = %.4f\n", backprop_time);
            printf("I/O comm time = %.4f\n", input_output_comm_time);
            printf("Boundary Communication time = %.4f\n", boundary_communication_time);
            printf("Filter Update Time = %.4f\n", filter_partial_updates_time);

        //    FILE *fptr;

        //    char* file_prefix = "weight_updates_device_";
        //    char file_name[20];
        //    strcpy(file_name, file_prefix);

        //    file_name[strlen(file_prefix)] = (char)(ftp_params.DEVICE_ID_X + 48);
        //    file_name[strlen(file_prefix) + 1] = (char)(ftp_params.DEVICE_ID_Y + 48);
        //    file_name[strlen(file_prefix) + 2] = '\0';

        //    fptr = fopen(file_name,"w");

        //    if(fptr == NULL)
        //    {
        //       printf("Error!");   
        //       exit(1);             
        //    }
        //    int layer_cumulative_weights = 0;

        //     for (int l = 0; l < net->n; ++l){

        //         int num_filters = net->layers[l].n;
        //         int filter_size = net->layers[l].size;
        //         int channels = net->layers[l].c;

        //         for (int c = 0; c < channels; ++c)
        //         {
        //             for (int f = 0; f < num_filters; ++f)
        //             {                    
        //                 for (int m = 0; m < filter_size; ++m)
        //                 {
        //                     for (int n = 0; n < filter_size; ++n)
        //                     {
        //                      fprintf(fptr,"%.4f ", net->layers[l].weights[(c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n]);
        //                     }
        //                     fprintf(fptr, "\n");
        //                 }
        //                 fprintf(fptr, "\n\n");
        //             }
        //             fprintf(fptr, "\n\n\n");
        //         }

        //         fprintf(fptr, "\n\n\n\n");

        //         layer_cumulative_weights += num_filters*channels*filter_size*filter_size;
        //     }

        //     fprintf(fptr, "\n\n\n\n\n\n\n\n");

        //    fclose(fptr);

        //     printf("Done\n");    
}



int main(int argc, char* argv[]){
    main_device(argc, argv);
    printf("complete\n");
}
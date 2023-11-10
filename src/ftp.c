#include "ftp.h"
#include "tcp_utils.h"
#include "convolutional_layer.h"
#include "maxpool_layer.h"
#include "utils.h"

#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

void static evaluate_ftp_validity(ftp *ftp_params, network *net){

}

void static compute_ftp_input_dimensions(ftp *ftp_params, network *net){
    int last_fused_layer = ftp_params->fused_layers - 1;
    int last_layer_input_orig_dim_x = net->layers[last_fused_layer].w;
    int last_layer_input_orig_dim_y = net->layers[last_fused_layer].h;
    int last_layer_input_tile_dim_x = (net->layers[last_fused_layer].w + (ftp_params->num_tiles_x-1))/ftp_params->num_tiles_x;
    int last_layer_input_tile_dim_y = (net->layers[last_fused_layer].h + (ftp_params->num_tiles_y-1))/ftp_params->num_tiles_y;
    
    net->layers[last_fused_layer].featuremap_h_without_boundary = last_layer_input_tile_dim_y;
    net->layers[last_fused_layer].featuremap_w_without_boundary = last_layer_input_tile_dim_x;

    int layer_out_height = last_layer_input_tile_dim_y;
    int layer_out_width = last_layer_input_tile_dim_x;

    int i;
    for (i = (last_fused_layer - 1); i >= 0; i--)
    {
        net->layers[i].featuremap_h_without_boundary = layer_out_height*net->layers[i].stride;
        net->layers[i].featuremap_w_without_boundary = layer_out_width*net->layers[i].stride;
        layer_out_height = net->layers[i].featuremap_h_without_boundary;
        layer_out_width = net->layers[i].featuremap_w_without_boundary;
    }
    
    net->layers[last_fused_layer].delta_h_without_boundary = net->layers[last_fused_layer].featuremap_h_without_boundary/net->layers[last_fused_layer].stride;
    net->layers[last_fused_layer].delta_w_without_boundary = net->layers[last_fused_layer].featuremap_w_without_boundary/net->layers[last_fused_layer].stride;
    
    for (i = 0; i < last_fused_layer; i++)
    {
        net->layers[i].delta_h_without_boundary = net->layers[i+1].featuremap_h_without_boundary;
        net->layers[i].delta_w_without_boundary = net->layers[i+1].featuremap_w_without_boundary;
    }
    
    for(i = 0; i < (last_fused_layer + 1); i++){
        net->layers[i].original_w = net->layers[i].w;
        net->layers[i].original_h = net->layers[i].h;
        net->layers[i].original_out_w = net->layers[i].out_w;
        net->layers[i].original_out_h = net->layers[i].out_h;

        if(ftp_params->device_id_x == (ftp_params->num_tiles_x - 1)){
            net->layers[i].extra_input_width = (ftp_params->num_tiles_x*net->layers[i].featuremap_w_without_boundary) - net->layers[i].original_w;
            net->layers[i].extra_output_width = (ftp_params->num_tiles_x*net->layers[i].delta_w_without_boundary) - net->layers[i].original_out_w;
        }
        if(ftp_params->device_id_y == (ftp_params->num_tiles_y - 1)){
            net->layers[i].extra_input_height = (ftp_params->num_tiles_y*net->layers[i].featuremap_h_without_boundary) - net->layers[i].original_h;
            net->layers[i].extra_output_height = (ftp_params->num_tiles_y*net->layers[i].delta_h_without_boundary) - net->layers[i].original_out_h;
        }
    }   
}

void static compute_group_boundries(ftp *ftp_params, network *net){
    int last_fused_layer = ftp_params->fused_layers - 1;
    int i, j; 
    
    for (i = (ftp_params->num_groups_forward - 1); i >= 0; i--)
    {
        int group_start_idx = ftp_params->group_sync_forward[i];
        int group_end_idx = (i == (ftp_params->num_groups_forward - 1)) ? last_fused_layer : (ftp_params->group_sync_forward[i+1] - 1);
    
        int left_boundary_offset = 0;
        int right_boundary_offset = net->layers[group_end_idx].delta_w_without_boundary - 1;
        int top_boundary_offset = 0;
        int bottom_boundary_offset = net->layers[group_end_idx].delta_h_without_boundary - 1;;

        for (j = group_end_idx; j >= group_start_idx; j--)
        {
            if(net->layers[j].type == CONVOLUTIONAL){
                left_boundary_offset = (left_boundary_offset*net->layers[j].stride) - (int)(net->layers[j].size/2);
                top_boundary_offset = (top_boundary_offset*net->layers[j].stride) - (int)(net->layers[j].size/2);
                right_boundary_offset = (right_boundary_offset*net->layers[j].stride) + (net->layers[j].stride-1) +(int)(net->layers[j].size/2);
                bottom_boundary_offset = (bottom_boundary_offset*net->layers[j].stride) + (net->layers[j].stride-1) + (int)(net->layers[j].size/2);
            }
            else if(net->layers[j].type == MAXPOOL){
                left_boundary_offset = (left_boundary_offset*net->layers[j].stride);
                top_boundary_offset = (top_boundary_offset*net->layers[j].stride);
                right_boundary_offset = (right_boundary_offset*net->layers[j].stride) + (net->layers[j].stride-1);
                bottom_boundary_offset = (bottom_boundary_offset*net->layers[j].stride) + (net->layers[j].stride-1);
            }
            net->layers[j].featuremap_h_with_boundary = bottom_boundary_offset - top_boundary_offset + 1;
            net->layers[j].featuremap_w_with_boundary = right_boundary_offset - left_boundary_offset + 1;
            net->layers[j].left_boundary_edges_featuremap = -1*left_boundary_offset;
            net->layers[j].right_boundary_edges_featuremap = right_boundary_offset - (net->layers[j].featuremap_w_without_boundary - 1);
            net->layers[j].top_boundary_edges_featuremap = -1*top_boundary_offset;
            net->layers[j].bottom_boundary_edges_featuremap = bottom_boundary_offset - (net->layers[j].featuremap_h_without_boundary - 1);

        }  
    }
    
    for (i = 0; i < (ftp_params->num_groups_backward); i++)
    {
        int group_end_idx = ftp_params->group_sync_backward[i];
        int group_start_idx = (i == 0) ? 1 : (ftp_params->group_sync_backward[i-1] + 1);
    
        int left_boundary_offset = 0;
        int right_boundary_offset = net->layers[group_start_idx - 1].delta_w_without_boundary - 1;
        int top_boundary_offset = 0;
        int bottom_boundary_offset = net->layers[group_start_idx - 1].delta_h_without_boundary - 1;

        for (j = group_start_idx; j <= group_end_idx; j++)
        {
            if(net->layers[j].type == CONVOLUTIONAL){
                left_boundary_offset = (int)((left_boundary_offset - (int)(net->layers[j].size/2))/net->layers[j].stride);
                top_boundary_offset = (int)((top_boundary_offset - (int)(net->layers[j].size/2))/net->layers[j].stride);
                right_boundary_offset = (int)((right_boundary_offset + (int)(net->layers[j].size/2))/net->layers[j].stride);
                bottom_boundary_offset = (int)((bottom_boundary_offset + (int)(net->layers[j].size/2))/net->layers[j].stride);
            }
            else if(net->layers[j].type == MAXPOOL){
                left_boundary_offset = (int)((left_boundary_offset - (net->layers[j].stride - 1))/net->layers[j].stride);
                top_boundary_offset = (int)((top_boundary_offset - (net->layers[j].stride - 1))/net->layers[j].stride);
                right_boundary_offset = (int)(right_boundary_offset/net->layers[j].stride);
                bottom_boundary_offset = (int)(bottom_boundary_offset/net->layers[j].stride);
            }
            net->layers[j].delta_h_with_boundary = bottom_boundary_offset - top_boundary_offset + 1;
            net->layers[j].delta_w_with_boundary = right_boundary_offset - left_boundary_offset + 1;
            net->layers[j].left_boundary_edges_delta = -1*left_boundary_offset;
            net->layers[j].right_boundary_edges_delta = right_boundary_offset - (net->layers[j].delta_w_without_boundary - 1);
            net->layers[j].top_boundary_edges_delta = -1*top_boundary_offset;
            net->layers[j].bottom_boundary_edges_delta = bottom_boundary_offset - (net->layers[j].delta_h_without_boundary - 1);
        }
        net->layers[0].delta_h_with_boundary = net->layers[0].delta_h_without_boundary;  
        net->layers[0].delta_w_with_boundary = net->layers[0].delta_w_without_boundary;
    } 
}

void static resize_layers(ftp* ftp_params, network *net){
    int last_fused_layer = ftp_params->fused_layers - 1;
    int i;
    for (i = 0; i <= last_fused_layer; i++)
    {
        free(net->layers[i].output);
        free(net->layers[i].delta);
        net->layers[i].pad = 0;

        net->layers[i].w = net->layers[i].featuremap_w_with_boundary;
        net->layers[i].h = net->layers[i].featuremap_h_with_boundary;
        net->layers[i].inputs = net->layers[i].w * net->layers[i].h * net->layers[i].c;

        if(net->layers[i].type == CONVOLUTIONAL){
              net->layers[i].delta = calloc(net->layers[i].delta_h_with_boundary*net->layers[i].delta_w_with_boundary*net->layers[i].n*net->batch, sizeof(float));

            int out_w = convolutional_out_width(net->layers[i]);
            int out_h = convolutional_out_height(net->layers[i]);
            net->layers[i].out_h = out_h;
            net->layers[i].out_w = out_w;
            net->layers[i].out_c = net->layers[i].n;
            net->layers[i].outputs = net->layers[i].out_h * net->layers[i].out_w * net->layers[i].out_c;      
            int out_h_allocate = (i == last_fused_layer) ? net->layers[i].out_h : net->layers[i+1].featuremap_h_with_boundary;
            int out_w_allocate = (i == last_fused_layer) ? net->layers[i].out_w : net->layers[i+1].featuremap_w_with_boundary;
            net->layers[i].output = calloc(out_h_allocate*out_w_allocate*net->layers[i].n*net->batch, sizeof(float));
        }

        else if(net->layers[i].type == MAXPOOL){
            free(net->layers[i].indexes);
            net->layers[i].delta = calloc(net->layers[i].delta_h_with_boundary*net->layers[i].delta_w_with_boundary*net->layers[i].c*net->batch, sizeof(float));
            net->layers[i].out_w = (net->layers[i].w + net->layers[i].pad - net->layers[i].size)/net->layers[i].stride + 1;
            net->layers[i].out_h = (net->layers[i].h + net->layers[i].pad - net->layers[i].size)/net->layers[i].stride + 1;
            net->layers[i].outputs = net->layers[i].out_w * net->layers[i].out_h * net->layers[i].c;
            int output_size = net->layers[i].out_h * net->layers[i].out_w * net->layers[i].out_c * net->batch;
            net->layers[i].indexes = calloc(net->layers[i].delta_h_with_boundary*net->layers[i].delta_w_with_boundary*net->layers[i].c*net->batch, sizeof(int));
            int out_h_allocate = (i == last_fused_layer) ? net->layers[i].out_h : net->layers[i+1].featuremap_h_with_boundary;
            int out_w_allocate = (i == last_fused_layer) ? net->layers[i].out_w : net->layers[i+1].featuremap_w_with_boundary;
            net->layers[i].output = calloc(out_h_allocate*out_w_allocate*net->layers[i+1].c*net->batch, sizeof(float));
        }    
    }
   
    int max = 0;
    for (i = 0; i < net->n; ++i)
    {
        int size = net->batch*net->layers[i].out_h*net->layers[i].out_w*net->layers[i].c*net->layers[i].size*net->layers[i].size;
        if(size > max){
            max = size;
        }
    }
    net->workspace = calloc(max, sizeof(float));
 
    for(i=0; i< ftp_params->fused_layers; i++){
        printf("Layer %d\n\n", i);
        printf("orig-inp-h %d orig-inp-w %d\n", net->layers[i].original_h, net->layers[i].original_w);
        printf("orig-out-h %d orig-out-w %d\n", net->layers[i].original_out_h, net->layers[i].original_out_w);
        printf("fm-h-nb %d fm-w-nb %d fm-h-wb %d fm-w-wb %d\n", net->layers[i].featuremap_h_without_boundary, net->layers[i].featuremap_w_without_boundary,
                                                  net->layers[i].featuremap_h_with_boundary, net->layers[i].featuremap_w_with_boundary);

        printf("del-h-nb %d del-w-nb %d del-h-wb %d del-w-wb %d\n", net->layers[i].delta_h_without_boundary, net->layers[i].delta_w_without_boundary,
                                                  net->layers[i].delta_h_with_boundary, net->layers[i].delta_w_with_boundary);
        printf("fm-le %d fm-re %d fm-te %d fm-be %d\n", net->layers[i].left_boundary_edges_featuremap, net->layers[i].right_boundary_edges_featuremap,
                                                  net->layers[i].top_boundary_edges_featuremap, net->layers[i].bottom_boundary_edges_featuremap);

        printf("del-le %d del-re %d del-te %d del-be %d\n", net->layers[i].left_boundary_edges_delta, net->layers[i].right_boundary_edges_delta,
                                                  net->layers[i].top_boundary_edges_delta, net->layers[i].bottom_boundary_edges_delta);
        printf("extra-inp-h %d extra-inp-w %d\n", net->layers[i].extra_input_height, net->layers[i].extra_input_width);
        printf("extra-out-h %d extra-out-w %d\n\n", net->layers[i].extra_output_height, net->layers[i].extra_output_width);
        printf("out-h %d out-w %d\n", net->layers[i].out_h, net->layers[i].out_w);
        printf("h %d w %d\n\n", net->layers[i].h, net->layers[i].w);

    }
    printf("resized\n");    
}

void static configure_shared_memory(ftp *ftp_params, network *net){
    int i;
    printf("configuring sm %d\n", ftp_params->is_device_gateway);
    if(ftp_params->is_device_gateway){
        printf("came here main\n");
        create_process_semaphore("/sm-semas-created", &ftp_params->sm_semas_created);
        //create_process_semaphore("/sem-mutex-create", &ftp_params->data_shared);
        create_process_semaphore("/batch-backprop-completed", &ftp_params->batch_backprop_completed);
        create_process_semaphore("/weights-updated", &ftp_params->weights_updated);
    }
    else{
        printf("came here gate\n");
        while( ((ftp_params->sm_semas_created = sem_open ("/sm-semas-created", 0, 0, 0)) == SEM_FAILED) );
        while( ((ftp_params->batch_backprop_completed = sem_open ("/batch-backprop-completed", 0, 0, 0)) == SEM_FAILED) );
        while( ((ftp_params->weights_updated = sem_open ("/weights-updated", 0, 0, 0)) == SEM_FAILED) );
        process_sema_wait(1, ftp_params->sm_semas_created);
        printf("came here gate 2\n");
    }
    for(i = 0; i < ftp_params->fused_layers; ++i)
    {
        if(net->layers[i].type == CONVOLUTIONAL){
            float* device_weight_update_buffers;
            char shm_file[3];
            shm_file[0] = '/';
            shm_file[1] = '1' + i;
            shm_file[2] = '\0';
            if(ftp_params->is_device_gateway){
                create_shared_memory(shm_file, &device_weight_update_buffers, ftp_params->num_device_tiles, net->layers[i].nweights);
                net->layers[i].weight_updates = device_weight_update_buffers;
            }
            else{
                get_shared_memory(shm_file, &device_weight_update_buffers, ftp_params->num_device_tiles, net->layers[i].nweights);
                net->layers[i].weight_updates = device_weight_update_buffers + ftp_params->local_device_tile_idx*net->layers[i].nweights;
            }
        }
    }
    
    if(ftp_params->is_device_gateway)
        process_sema_post(ftp_params->num_device_tiles - 1, ftp_params->sm_semas_created);
}

void static copy_slice(float* dst, float* src, int batch, int depth,
                int height_src, int width_src, int height_dst, int width_dst,
                int src_start_x, int src_start_y, int dst_start_x, int dst_start_y,
                int copy_height_src, int copy_width_src, int copy_height_dst, int copy_width_dst,
                float* workspace){

    float* src_intermediate = src;
    int b, d, h, w;
    //if(dst == src){
        src_intermediate = workspace;

        for(b = 0; b < batch; b++){
            for(d = 0; d < depth; d++){
                for(h = 0; h < copy_height_src; h++){
                    for(w = 0; w < copy_width_src; w++){
                        workspace[b*depth*copy_height_src*copy_width_src + d*copy_height_src*copy_width_src + h*copy_width_src + w] =
                        src[b*depth*height_src*width_src + d*height_src*width_src + (h + src_start_y)*width_src + w + src_start_x];
                    }
                }
            }
        }
    //}

    for(b = 0; b < batch; b++){
        for(d = 0; d < depth; d++){
            for(h = 0; h < copy_height_dst; h++){
                for(w = 0; w < copy_width_dst; w++){
                    dst[b*depth*height_dst*width_dst + d*height_dst*width_dst + (h + dst_start_y)*width_dst + w + dst_start_x] =
                    workspace[b*depth*copy_height_dst*copy_width_dst + d*copy_height_dst*copy_width_dst + h*copy_width_dst + w];
                }
            }
        }
    }
}

void static clear_slice(float* target, int batch, int depth,
                int target_height, int target_width,
                int offset_y, int offset_x,
                int clear_height, int clear_width){
    int b, d, h, w;
    for(b = 0; b < batch; b++){
        for(d = 0; d < depth; d++){
            for(h = 0; h < clear_height; h++){
                for(w = 0; w < clear_width; w++){
                    target[b*depth*target_height*target_width + d*target_height*target_width + (h + offset_y)*target_width + w + offset_x] = 0.0;
                }
            }
        }
    }
}

typedef enum orientation{
    TOP,
    LEFT,
    BOTTOM,
    RIGHT,
    TOP_LEFT,
    BOTTOM_LEFT,
    BOTTOM_RIGHT,
    TOP_RIGHT,
} orientation;

void static get_group_boundary_data_device(
    int num_tiles_x, int num_tile_y,
    float** device_data, 
    int rows, int cols, int depth, int batch,
    orientation region,
    int device_src_id_x, int device_src_id_y, 
    int device_dst_id_x, int device_dst_id_y) {

    float* boundary_data = calloc(rows*cols*depth*batch, sizeof(float));
    *device_data = boundary_data;

    if((device_src_id_x >= num_tiles_x) && 
        (region == BOTTOM_LEFT || region == LEFT || region == TOP_LEFT)){
        fill_cpu(rows*cols*depth*batch, 0, *device_data, 1);
        return;
    }
    if((device_src_id_x < 0) && 
        (region == TOP_RIGHT || region == RIGHT || region == BOTTOM_RIGHT)){
        fill_cpu(rows*cols*depth*batch, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y >= num_tile_y) && 
        (region == TOP_LEFT || region == TOP || region == TOP_RIGHT)){
        fill_cpu(rows*cols*depth*batch, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y < 0) && 
        (region == BOTTOM_LEFT || region == BOTTOM || region == BOTTOM_RIGHT)){
        fill_cpu(rows*cols*depth*batch, 0, *device_data, 1);
        return;
    }
    printf("data transmission\n", region);
    receive_data(boundary_data, rows*cols*depth*batch, device_src_id_x + (device_src_id_y)*num_tiles_x);
}

void static assemble_tile(ftp* ftp_params, network* net, int batch, int depth,
                   float* target, float* core_tile_data,
                   int core_tile_height, int core_tile_width,
                   int left_boundary_edges, int right_boundary_edges, int top_boundary_edges, int bottom_boundary_edges){
     printf("te = %d be = %d le = %d re = %d\n", top_boundary_edges, bottom_boundary_edges, left_boundary_edges, right_boundary_edges);
    int num_tiles_x = ftp_params->num_tiles_x;
    int num_tiles_y = ftp_params->num_tiles_y;
    int device_id_x = ftp_params->device_id_x;
    int device_id_y = ftp_params->device_id_y;
    int src_node_id = ftp_params->device_id_x + (ftp_params->device_id_y * num_tiles_x);

    int full_height = core_tile_height + top_boundary_edges + bottom_boundary_edges;
    int full_width = core_tile_width + left_boundary_edges + right_boundary_edges;

    float* core_tile_temp = calloc((batch*depth*core_tile_height*core_tile_width), sizeof(float));
    memcpy(core_tile_temp, core_tile_data, batch*depth*core_tile_height*core_tile_width*sizeof(float));

    copy_slice(target, core_tile_data, batch, depth,
        core_tile_height, core_tile_width, full_height, full_width,
        0, 0, left_boundary_edges, top_boundary_edges,
        core_tile_height, core_tile_width, core_tile_height, core_tile_width,
        net->workspace);
    
    float* transmit_data;

    // //Top left
    if((top_boundary_edges > 0) && (left_boundary_edges > 0)){

        float* boundary_top_left;

        get_group_boundary_data_device(
            num_tiles_x, num_tiles_y,
            &boundary_top_left, 
            top_boundary_edges, left_boundary_edges, depth, batch,
            BOTTOM_RIGHT, 
            device_id_x-1, device_id_y-1,
            device_id_x, device_id_y);

        copy_slice(target, boundary_top_left, batch, depth,
            top_boundary_edges, left_boundary_edges, full_height, full_width,
            0, 0, 0, 0,
            top_boundary_edges, left_boundary_edges, top_boundary_edges, left_boundary_edges,
            net->workspace);

        free(boundary_top_left);

        //SEND TOP LEFT
        if((device_id_y > 0) && (device_id_x > 0)){
            int rows = bottom_boundary_edges;
            int cols = right_boundary_edges;
            transmit_data = calloc((batch*depth*rows*cols), sizeof(float));

            copy_slice(transmit_data, core_tile_temp, batch, depth,
                core_tile_height, core_tile_width, rows, cols,
                0, 0, 0, 0,
                rows, cols, rows, cols,
                net->workspace);
            send_data(transmit_data, batch*depth*rows*cols, device_id_x-1 + (device_id_y-1)*num_tiles_x);
            free(transmit_data);
        }
    }


    //Top
    if(top_boundary_edges > 0){
        //receive top edges
        float* boundary_top;

        get_group_boundary_data_device(
            num_tiles_x, num_tiles_y,
            &boundary_top, 
            top_boundary_edges, core_tile_width, depth, batch,
            BOTTOM, 
            device_id_x, device_id_y-1,
            device_id_x, device_id_y);

        copy_slice(target, boundary_top, batch, depth,
            top_boundary_edges, core_tile_width, full_height, full_width,
            0, 0, left_boundary_edges, 0,
            top_boundary_edges, core_tile_width, top_boundary_edges, core_tile_width,
            net->workspace);

        free(boundary_top);

        //SEND TOP
        if(device_id_y > 0){
            int rows = bottom_boundary_edges;
            int cols = core_tile_width;
            transmit_data = calloc((batch*depth*rows*cols), sizeof(float));

            copy_slice(transmit_data, core_tile_temp, batch, depth,
                core_tile_height, core_tile_width, rows, cols,
                0, 0, 0, 0,
                rows, cols, rows, cols,
                net->workspace);
            send_data(transmit_data, batch*depth*rows*cols, device_id_x + (device_id_y-1)*num_tiles_x);
            free(transmit_data);
        }
    } 

    //Top Right
    if(top_boundary_edges > 0 && right_boundary_edges > 0){
        //receive top-right edges
        float* boundary_top_right;

        get_group_boundary_data_device(
            num_tiles_x, num_tiles_y,
            &boundary_top_right, 
            top_boundary_edges, right_boundary_edges, depth, batch,
            BOTTOM_LEFT, 
            device_id_x+1, device_id_y-1,
            device_id_x, device_id_y);

        copy_slice(target, boundary_top_right, batch, depth,
            top_boundary_edges, right_boundary_edges, full_height, full_width,
            0, 0, left_boundary_edges + core_tile_width, 0,
            top_boundary_edges, right_boundary_edges, top_boundary_edges, right_boundary_edges,
            net->workspace);

        free(boundary_top_right);

        //SEND TOP-RIGHT
        if((device_id_y > 0) && (device_id_x < (num_tiles_x-1))){
            int rows = bottom_boundary_edges;
            int cols = left_boundary_edges;
            transmit_data = calloc((batch*depth*rows*cols), sizeof(float));

            copy_slice(transmit_data, core_tile_temp, batch, depth,
                core_tile_height, core_tile_width, rows, cols,
                core_tile_width - right_boundary_edges, 0, 0, 0,
                rows, cols, rows, cols,
                net->workspace);
            send_data(transmit_data, batch*depth*rows*cols, device_id_x+1 + (device_id_y-1)*num_tiles_x);
            free(transmit_data);
        }
    } 

    //LEFT
    if(left_boundary_edges > 0){
        //receive Left edges
        float* boundary_left;

        get_group_boundary_data_device(
            num_tiles_x, num_tiles_y,
            &boundary_left, 
            core_tile_height, left_boundary_edges, depth, batch,
            RIGHT, 
            device_id_x-1, device_id_y,
            device_id_x, device_id_y);

        copy_slice(target, boundary_left, batch, depth,
            core_tile_height, left_boundary_edges, full_height, full_width,
            0, 0, 0, top_boundary_edges,
            core_tile_height, left_boundary_edges, core_tile_height, left_boundary_edges,
            net->workspace);

        free(boundary_left);

        //SEND Left
        if(device_id_x > 0){
            int rows = core_tile_height;
            int cols = right_boundary_edges;
            transmit_data = calloc((batch*depth*rows*cols), sizeof(float));

            copy_slice(transmit_data, core_tile_temp, batch, depth,
                core_tile_height, core_tile_width, rows, cols,
                0, 0, 0, 0,
                rows, cols, rows, cols,
                net->workspace);
            send_data(transmit_data, batch*depth*rows*cols, device_id_x - 1 + (device_id_y*num_tiles_x));
            free(transmit_data);
        }
    } 

    //RIGHT
    if(right_boundary_edges > 0){
        //SEND Right
        if(device_id_x < (num_tiles_x-1)){
            int rows = core_tile_height;
            int cols = left_boundary_edges;
            transmit_data = calloc((batch*depth*rows*cols), sizeof(float));

            copy_slice(transmit_data, core_tile_temp, batch, depth,
                core_tile_height, core_tile_width, rows, cols,
                core_tile_width - right_boundary_edges, 0, 0, 0,
                rows, cols, rows, cols,
                net->workspace);
            send_data(transmit_data, batch*depth*rows*cols, device_id_x+1 + (device_id_y*num_tiles_x));
            free(transmit_data);
        }

        //receive Right edges
        float* boundary_right;

        get_group_boundary_data_device(
            num_tiles_x, num_tiles_y,
            &boundary_right, 
            core_tile_height, right_boundary_edges, depth, batch,
            LEFT, 
            device_id_x+1, device_id_y,
            device_id_x, device_id_y);

        copy_slice(target, boundary_right, batch, depth,
            core_tile_height, right_boundary_edges, full_height, full_width,
            0, 0, left_boundary_edges + core_tile_width, top_boundary_edges,
            core_tile_height, right_boundary_edges, core_tile_height, right_boundary_edges,
            net->workspace);

        free(boundary_right);
    }

    //BOTTOM LEFT
    if((bottom_boundary_edges > 0) && (left_boundary_edges > 0)){
        //SEND Bottom-Left
        if((device_id_y < (num_tiles_y-1)) && (device_id_x > 0)){
            int rows = top_boundary_edges;
            int cols = right_boundary_edges;
            transmit_data = calloc((batch*depth*rows*cols), sizeof(float));

            copy_slice(transmit_data, core_tile_temp, batch, depth,
                core_tile_height, core_tile_width, rows, cols,
                0, core_tile_height - top_boundary_edges, 0, 0,
                rows, cols, rows, cols,
                net->workspace);
            send_data(transmit_data, batch*depth*rows*cols, device_id_x-1 + (device_id_y+1)*num_tiles_x);
            free(transmit_data);
        }

        //receive Bottom-Left edges
        float* boundary_bottom_left;

        get_group_boundary_data_device(
            num_tiles_x, num_tiles_y,
            &boundary_bottom_left, 
            bottom_boundary_edges, left_boundary_edges, depth, batch,
            TOP_RIGHT, 
            device_id_x-1, device_id_y+1,
            device_id_x, device_id_y);

        copy_slice(target, boundary_bottom_left, batch, depth,
            bottom_boundary_edges, left_boundary_edges, full_height, full_width,
            0, 0, 0, top_boundary_edges + core_tile_height,
            bottom_boundary_edges, left_boundary_edges, bottom_boundary_edges, left_boundary_edges,
            net->workspace);

        free(boundary_bottom_left);
    }

    //BOTTOM
    if(bottom_boundary_edges > 0){
        //SEND Bottom
        if(device_id_y < (num_tiles_y-1)){
            int rows = top_boundary_edges;
            int cols = core_tile_width;
            transmit_data = calloc((batch*depth*rows*cols), sizeof(float));

            copy_slice(transmit_data, core_tile_temp, batch, depth,
                core_tile_height, core_tile_width, rows, cols,
                0, core_tile_height - top_boundary_edges, 0, 0,
                rows, cols, rows, cols,
                net->workspace);
            send_data(transmit_data, batch*depth*rows*cols, device_id_x + (device_id_y+1)*num_tiles_x);
            free(transmit_data);
        }

        //receive Bottom
        float* boundary_bottom;

        get_group_boundary_data_device(
            num_tiles_x, num_tiles_y,
            &boundary_bottom, 
            bottom_boundary_edges, core_tile_width, depth, batch,
            TOP, 
            device_id_x, device_id_y+1,
            device_id_x, device_id_y);

        copy_slice(target, boundary_bottom, batch, depth,
            bottom_boundary_edges, core_tile_width, full_height, full_width,
            0, 0, left_boundary_edges, top_boundary_edges + core_tile_height,
            bottom_boundary_edges, core_tile_width, bottom_boundary_edges, core_tile_width,
            net->workspace);

        free(boundary_bottom);
    }


    //BOTTOM RIGHT
    if(bottom_boundary_edges > 0 && right_boundary_edges > 0){
        //SEND Bottom-Right
        if((device_id_y < (num_tiles_y-1)) && (device_id_x < (num_tiles_x-1))){
            int rows = top_boundary_edges;
            int cols = left_boundary_edges;
            transmit_data = calloc((batch*depth*rows*cols), sizeof(float));

            copy_slice(transmit_data, core_tile_temp, batch, depth,
                core_tile_height, core_tile_width, rows, cols,
                core_tile_width - left_boundary_edges, core_tile_height - top_boundary_edges, 0, 0,
                rows, cols, rows, cols,
                net->workspace);
            send_data(transmit_data, batch*depth*rows*cols, device_id_x+1 + (device_id_y+1)*num_tiles_x);
            free(transmit_data);
        }

        //receive Bottom-Right
        float* boundary_bottom_right;

        get_group_boundary_data_device(
            num_tiles_x, num_tiles_y,
            &boundary_bottom_right, 
            bottom_boundary_edges, right_boundary_edges, depth, batch,
            TOP_LEFT, 
            device_id_x+1, device_id_y+1,
            device_id_x, device_id_y);

        copy_slice(target, boundary_bottom_right, batch, depth,
            bottom_boundary_edges, right_boundary_edges, full_height, full_width,
            0, 0, left_boundary_edges + core_tile_width, top_boundary_edges + core_tile_height,
            bottom_boundary_edges, right_boundary_edges, bottom_boundary_edges, right_boundary_edges,
            net->workspace);

        free(boundary_bottom_right);
    }

    free(core_tile_temp);

}

void static clear_edges_featuremap_device(ftp* ftp_params, network* net, int layer_idx){
    if(ftp_params->device_id_y == 0){
        clear_slice(net->layers[layer_idx-1].output, net->batch, net->layers[layer_idx].c,
                        net->layers[layer_idx].featuremap_h_with_boundary, net->layers[layer_idx].featuremap_w_with_boundary,
                        0, 0,
                        net->layers[layer_idx].top_boundary_edges_featuremap, net->layers[layer_idx].featuremap_w_with_boundary);
    }
    if(ftp_params->device_id_x == 0){
        clear_slice(net->layers[layer_idx-1].output, net->batch, net->layers[layer_idx].c,
                        net->layers[layer_idx].featuremap_h_with_boundary, net->layers[layer_idx].featuremap_w_with_boundary,
                        0, 0,
                        net->layers[layer_idx].featuremap_h_with_boundary, net->layers[layer_idx].left_boundary_edges_featuremap);
    }
    if(ftp_params->device_id_y == (ftp_params->num_tiles_y - 1)){
        clear_slice(net->layers[layer_idx-1].output, net->batch, net->layers[layer_idx].c,
                        net->layers[layer_idx].featuremap_h_with_boundary, net->layers[layer_idx].featuremap_w_with_boundary,
                        net->layers[layer_idx].featuremap_h_without_boundary + net->layers[layer_idx].top_boundary_edges_featuremap, 0,
                        net->layers[layer_idx].bottom_boundary_edges_featuremap, net->layers[layer_idx].featuremap_w_with_boundary);
    }
    if(ftp_params->device_id_x == (ftp_params->num_tiles_x - 1)){
        clear_slice(net->layers[layer_idx-1].output, net->batch, net->layers[layer_idx].c,
                        net->layers[layer_idx].featuremap_h_with_boundary, net->layers[layer_idx].featuremap_w_with_boundary,
                        0, net->layers[layer_idx].featuremap_w_without_boundary+net->layers[layer_idx].left_boundary_edges_featuremap,
                        net->layers[layer_idx].featuremap_h_with_boundary, net->layers[layer_idx].right_boundary_edges_featuremap);
    }
}

void static clear_spurious_edges_featuremap(ftp* ftp_params, network* net, int layer_idx){
    int depth = (net->layers[layer_idx].type == CONVOLUTIONAL) ? net->layers[layer_idx].n : net->layers[layer_idx].c;
    int extra_width = net->layers[layer_idx].extra_output_width;
    int extra_height = net->layers[layer_idx].extra_output_height;
    int output_height = (layer_idx == ftp_params->fused_layers - 1) ? net->layers[layer_idx].out_h : net->layers[layer_idx + 1].featuremap_h_with_boundary;
    int output_width = (layer_idx == ftp_params->fused_layers - 1) ? net->layers[layer_idx].out_w : net->layers[layer_idx + 1].featuremap_w_with_boundary;
    int right_edges = (layer_idx == ftp_params->fused_layers - 1) ? 0 : net->layers[layer_idx + 1].right_boundary_edges_featuremap;
    int bottom_edges = (layer_idx == ftp_params->fused_layers - 1) ? 0 : net->layers[layer_idx + 1].bottom_boundary_edges_featuremap;

    if(ftp_params->device_id_x == (ftp_params->num_tiles_x - 1)){
        clear_slice(net->layers[layer_idx].output, net->batch, depth,
                        output_height, output_width,
                        0, output_width - extra_width - right_edges,
                        output_height, net->layers[layer_idx].extra_output_width);        
    }
    if(ftp_params->device_id_y == (ftp_params->num_tiles_y - 1)){
        clear_slice(net->layers[layer_idx].output, net->batch, depth,
                        output_height, output_width,
                        0, output_width - extra_height - bottom_edges,
                        net->layers[layer_idx].extra_output_height, output_width);     
    }
}

void static clear_edges_delta_device(ftp* ftp_params, network* net, int layer_idx){
    int depth = (net->layers[layer_idx].type == CONVOLUTIONAL) ? net->layers[layer_idx].n : net->layers[layer_idx].c;
    if(ftp_params->device_id_y == 0){
        clear_slice(net->layers[layer_idx].delta, net->batch, depth,
                        net->layers[layer_idx].delta_h_with_boundary, net->layers[layer_idx].delta_w_with_boundary,
                        0, 0,
                        net->layers[layer_idx].top_boundary_edges_delta, net->layers[layer_idx].delta_w_with_boundary);
    }
    if(ftp_params->device_id_x == 0){
        clear_slice(net->layers[layer_idx].delta, net->batch, depth,
                        net->layers[layer_idx].delta_h_with_boundary, net->layers[layer_idx].delta_w_with_boundary,
                        0, 0,
                        net->layers[layer_idx].delta_h_with_boundary, net->layers[layer_idx].left_boundary_edges_delta);
    }
    if(ftp_params->device_id_y == (ftp_params->num_tiles_y - 1)){
        clear_slice(net->layers[layer_idx].delta, net->batch, depth,
                        net->layers[layer_idx].delta_h_with_boundary, net->layers[layer_idx].delta_w_with_boundary,
                        net->layers[layer_idx].delta_h_without_boundary + net->layers[layer_idx].top_boundary_edges_delta, 0,
                        net->layers[layer_idx].bottom_boundary_edges_delta, net->layers[layer_idx].delta_w_with_boundary);
    }
    if(ftp_params->device_id_x == (ftp_params->num_tiles_x - 1)){
        clear_slice(net->layers[layer_idx].delta, net->batch, depth,
                        net->layers[layer_idx].delta_h_with_boundary, net->layers[layer_idx].delta_w_with_boundary,
                        0, net->layers[layer_idx].delta_w_without_boundary+net->layers[layer_idx].left_boundary_edges_delta,
                        net->layers[layer_idx].delta_h_with_boundary, net->layers[layer_idx].right_boundary_edges_delta);
    }
}

void static clear_spurious_edges_delta(ftp* ftp_params, network* net, int layer_idx){
    int depth = (net->layers[layer_idx].type == CONVOLUTIONAL) ? net->layers[layer_idx].n : net->layers[layer_idx].c;
    int extra_width = net->layers[layer_idx].extra_output_width;
    int extra_height = net->layers[layer_idx].extra_output_height;
    int output_height = net->layers[layer_idx].delta_h_with_boundary;
    int output_width = net->layers[layer_idx].delta_w_with_boundary;
    int right_edges = net->layers[layer_idx].right_boundary_edges_delta;
    int bottom_edges = net->layers[layer_idx].bottom_boundary_edges_delta;

    if(ftp_params->device_id_x == (ftp_params->num_tiles_x - 1)){
        clear_slice(net->layers[layer_idx].delta, net->batch, depth,
                        output_height, output_width,
                        0, output_width - extra_width - right_edges,
                        output_height, net->layers[layer_idx].extra_output_width);        
    }
    if(ftp_params->device_id_y == (ftp_params->num_tiles_y - 1)){
        clear_slice(net->layers[layer_idx].delta, net->batch, depth,
                        output_height, output_width,
                        0, output_width - extra_height - bottom_edges,
                        net->layers[layer_idx].extra_output_height, output_width);     
    }
}

void init_ftp(ftp* ftp_params, network* net){
    int i;
    evaluate_ftp_validity(ftp_params, net);
    int node_id = (ftp_params->device_id_y*ftp_params->num_tiles_x) + ftp_params->device_id_x;
    printf("tiles-x %d tiles-y %d id-x %d id-y %d node-id %d\n", ftp_params->num_tiles_x, ftp_params->num_tiles_y, ftp_params->device_id_x, ftp_params->device_id_y, node_id);
    printf("IPs\n");
    for(i=0; i<(ftp_params->num_tiles_x*ftp_params->num_tiles_y); i++){
        printf("%s\n", ftp_params->IPs[i]);
    }
    printf("Forward groups\n");
    for(i=0; i<ftp_params->num_groups_forward; i++){
        printf("%d\n", ftp_params->group_sync_forward[i]);
    }
    printf("Backward groups\n");
    for(i=0; i<ftp_params->num_groups_backward; i++){
        printf("%d\n", ftp_params->group_sync_backward[i]);
    }
    tcp_connect(ftp_params->num_tiles_x*ftp_params->num_tiles_y, node_id, ftp_params->IPs);
    compute_ftp_input_dimensions(ftp_params, net);
    compute_group_boundries(ftp_params, net);
    resize_layers(ftp_params, net);
    configure_shared_memory(ftp_params, net);
}

void static read_input_chunk(ftp* ftp_params, network* net, float* tile_input, float* full_input, int dst_tile_x, int dst_tile_y){
    int in_height = net->layers[0].original_w;
    int in_width = net->layers[0].original_h;
    int depth = net->layers[0].c;

    int copy_height = net->layers[0].featuremap_h_without_boundary - net->layers[0].extra_input_height;
    int copy_width = net->layers[0].featuremap_w_without_boundary - net->layers[0].extra_input_width;
    printf("copy height %d width %d\n", copy_height, copy_width);
    copy_slice(tile_input, full_input, net->batch, depth,
                    in_height, in_width,
                    net->layers[0].featuremap_h_without_boundary, net->layers[0].featuremap_w_without_boundary,
                    dst_tile_x*net->layers[0].featuremap_w_without_boundary, dst_tile_y*net->layers[0].featuremap_h_without_boundary,
                    0, 0,
                    copy_height, copy_width, copy_height, copy_width,
                    net->workspace);   
}

int is_matching(float target, float actual){
    if(fabs(target) < 0.05 || fabs(actual) < 0.05)
        return 1;
    if(fabs(target - actual) < (0.3*fabs(target)))
        return 1;
    return 0;
}

void compare_ftp_to_ref(float* data_ref, float* data_ftp,
                        int ref_x_dim, int ref_y_dim, int ftp_x_dim, int ftp_y_dim, int ftp_offset_x, int ftp_offset_y,
                        int depth, int batch){
    size_t b, d, h, w;
    for (b = 0; b < batch; b++){
        for (d = 0; d < depth; d++){
            for (h = 0; h < ftp_y_dim; h++){
                for (w = 0; w < ftp_x_dim; w++){
                    int index_ref = b*depth*ref_x_dim*ref_y_dim + d*ref_x_dim*ref_y_dim + ((h+ftp_offset_y)*ref_x_dim) + w + ftp_offset_x;
                    int index_ftp = b*depth*ftp_x_dim*ftp_y_dim + d*ftp_x_dim*ftp_y_dim + (h*ftp_x_dim) + w;
                    if(!is_matching(data_ref[index_ref], data_ftp[index_ftp])){
                        printf("Mismatch found - Expected = %.8f Actual = %.8f h = %d w = %d refid = %d ftpid = %d\n", data_ref[index_ref], data_ftp[index_ftp], h, w, index_ref, index_ftp);
                        exit(1);
                    }
                }
            }
        }
    }
}


void static distribute_network_batch_input_data(ftp* ftp_params, network* net, float* full_input, float **input_tile_chunk){
   *input_tile_chunk = calloc(net->batch*net->layers[0].featuremap_h_without_boundary*net->layers[0].featuremap_w_without_boundary*net->layers[0].c, sizeof(float));
    int i, j;
    if(ftp_params->is_main_gateway){
       for (i = 0; i < ftp_params->num_tiles_x; ++i)
        {
            for (j = 0; j < ftp_params->num_tiles_y; ++j)
            {
                if(!((i == 0) && (j == 0))){
                   read_input_chunk(ftp_params, net, *input_tile_chunk, full_input, i, j);
                   send_data(*input_tile_chunk, net->batch*net->layers[0].featuremap_h_without_boundary*net->layers[0].featuremap_w_without_boundary*net->layers[0].c, i + j*ftp_params->num_tiles_x);
                }
            }
        }
        read_input_chunk(ftp_params, net, *input_tile_chunk, full_input, 0, 0);
    }
    else
       receive_data(*input_tile_chunk, net->batch*net->layers[0].featuremap_h_without_boundary*net->layers[0].featuremap_w_without_boundary*net->layers[0].c, 0);
}

void static read_delta_chunk(ftp* ftp_params, network* net, float* tile_delta, float* full_delta, int dst_tile_x, int dst_tile_y){
    int last_fused_layer = ftp_params->fused_layers - 1;
    int out_height = net->layers[last_fused_layer].original_out_w;
    int out_width = net->layers[last_fused_layer].original_out_h;
    int depth = (net->layers[last_fused_layer].type == CONVOLUTIONAL) ? net->layers[last_fused_layer].n : net->layers[last_fused_layer].c;

    int copy_height = net->layers[last_fused_layer].delta_h_without_boundary - net->layers[last_fused_layer].extra_output_height;
    int copy_width = net->layers[last_fused_layer].delta_w_without_boundary - net->layers[last_fused_layer].extra_output_width;

    copy_slice(tile_delta, full_delta, net->batch, depth,
                    out_height, out_width,
                    net->layers[last_fused_layer].delta_h_without_boundary, net->layers[last_fused_layer].delta_w_without_boundary,
                    dst_tile_x*net->layers[last_fused_layer].delta_w_without_boundary, dst_tile_y*net->layers[last_fused_layer].delta_h_without_boundary,
                    0, 0,
                    copy_height, copy_width, copy_height, copy_width,
                    net->workspace);   
}

void static distribute_network_batch_delta_data(ftp* ftp_params, network* net, float* full_delta, float** delta_tile_chunk){
    int last_fused_layer = ftp_params->fused_layers - 1;
    int depth = (net->layers[last_fused_layer].type == CONVOLUTIONAL) ? net->layers[last_fused_layer].n : net->layers[last_fused_layer].c;
    *delta_tile_chunk = calloc(net->batch*net->layers[last_fused_layer].delta_h_without_boundary*net->layers[last_fused_layer].delta_w_without_boundary*depth, sizeof(float));
    int i, j;
    if(ftp_params->is_main_gateway){
        for (i = 0; i < ftp_params->num_tiles_x; ++i)
        {
            for (j = 0; j < ftp_params->num_tiles_y; ++j)
            {
                if(!((i == 0) && (j == 0))){
                   read_delta_chunk(ftp_params, net, *delta_tile_chunk, full_delta, i, j);
                   send_data(*delta_tile_chunk, net->batch*net->layers[last_fused_layer].delta_h_without_boundary*net->layers[last_fused_layer].delta_w_without_boundary*depth, i + j*ftp_params->num_tiles_x);
                }
            }
        }
        read_delta_chunk(ftp_params, net, *delta_tile_chunk, full_delta, 0, 0);
    }
    else
       receive_data(*delta_tile_chunk, net->batch*net->layers[last_fused_layer].delta_h_without_boundary*net->layers[last_fused_layer].delta_w_without_boundary*depth, 0);
}

void static converge_fused_layers_output_data(ftp* ftp_params, network* net, float** aggregate_ftp_output){
    int last_fused_layer = ftp_params->fused_layers - 1;
    int last_fused_layer_out_w = net->layers[last_fused_layer].out_w;
    int last_fused_layer_out_h = net->layers[last_fused_layer].out_h;
    int last_fused_layer_out_depth = (net->layers[last_fused_layer].type == CONVOLUTIONAL) ? net->layers[last_fused_layer].n : net->layers[last_fused_layer].c;
    int last_fused_layer_orig_out_w = net->layers[last_fused_layer].original_out_w;
    int last_fused_layer_orig_out_h = net->layers[last_fused_layer].original_out_h;
    int copy_height = net->layers[last_fused_layer].out_h - net->layers[last_fused_layer].extra_output_height;
    int copy_width = net->layers[last_fused_layer].out_w - net->layers[last_fused_layer].extra_output_width;
    
    if(ftp_params->is_main_gateway){
        printf("came here converge\n");
        *aggregate_ftp_output = calloc(last_fused_layer_orig_out_w*last_fused_layer_orig_out_h*last_fused_layer_out_depth*net->batch, sizeof(float));
        float* temp_buffer = calloc(last_fused_layer_out_w*last_fused_layer_out_h*last_fused_layer_out_depth*net->batch, sizeof(float));
        printf("%p %d %d %d %d\n", temp_buffer, last_fused_layer_out_w, last_fused_layer_out_h, last_fused_layer_out_depth, net->batch);
        int i, j;
        for (i = 0; i < ftp_params->num_tiles_x; ++i)
        {
            for (j = 0; j < ftp_params->num_tiles_y; ++j)
            {
                if(!((i == 0) && (j == 0))){
                    receive_data(temp_buffer, last_fused_layer_out_w*last_fused_layer_out_h*last_fused_layer_out_depth*net->batch, i + j*ftp_params->num_tiles_x);
                    copy_slice(*aggregate_ftp_output, temp_buffer, net->batch, last_fused_layer_out_depth,
                        last_fused_layer_out_h, last_fused_layer_out_w,
                        last_fused_layer_orig_out_h, last_fused_layer_orig_out_w,
                        0, 0,
                        i*last_fused_layer_out_w, j*last_fused_layer_out_h,
                        copy_height, copy_width, copy_height, copy_width,
                        net->workspace);
                    printf("ch = %d cw = %d depth = %d batch = %d origh = %d origw = %d\n", copy_height, copy_width, last_fused_layer_out_depth, net->batch, last_fused_layer_orig_out_h, last_fused_layer_orig_out_w);  
                }
            }
        }
 
        copy_slice(*aggregate_ftp_output, net->layers[last_fused_layer].output, net->batch, last_fused_layer_out_depth,
                   last_fused_layer_out_h, last_fused_layer_out_w,
                   last_fused_layer_orig_out_h, last_fused_layer_orig_out_w,
                   0, 0,
                   0, 0,
                   copy_height, copy_width, copy_height, copy_width,
                   net->workspace);
        net->input = *aggregate_ftp_output;
        free(temp_buffer);
    }
    else
       send_data(net->layers[last_fused_layer].output, last_fused_layer_out_w*last_fused_layer_out_h*last_fused_layer_out_depth*net->batch, 0);    
}

void forward_network_distributed_ftp(ftp* ftp_params, network* net, network* net_ref){
    struct timeval step_time_before, step_time_after, step_time_result;
    double inference_time = 0.0;
    float* network_input_batch;
    float* full_input = (ftp_params->is_main_gateway) ? net->input : NULL;
    printf("%d %d\n", net_ref->h, net_ref->w);
    distribute_network_batch_input_data(ftp_params, net, full_input, &network_input_batch);
    free(net->input);

    int g, l;
    for (g = 0; g < ftp_params->num_groups_forward; ++g)
    {
        //gettimeofday(&step_time_before, NULL);
        int group_start_idx = ftp_params->group_sync_forward[g];
        int group_end_idx = (g == (ftp_params->num_groups_forward - 1)) ? (ftp_params->fused_layers - 1) : (ftp_params->group_sync_forward[g+1] - 1);
        net->inputs =   net->batch*
                        net->layers[group_start_idx].featuremap_h_with_boundary*
                        net->layers[group_start_idx].featuremap_w_with_boundary*
                        net->layers[group_start_idx].c;
        net->input = calloc(net->inputs, sizeof(float));
        float* group_input = (group_start_idx == 0) ?  network_input_batch : net->layers[group_start_idx - 1].output;

       assemble_tile(ftp_params, net, net->batch, net->layers[group_start_idx].c,
                        net->input, group_input,
                        net->layers[group_start_idx].featuremap_h_without_boundary, net->layers[group_start_idx].featuremap_w_without_boundary,
                        net->layers[group_start_idx].left_boundary_edges_featuremap, net->layers[group_start_idx].right_boundary_edges_featuremap,
                        net->layers[group_start_idx].top_boundary_edges_featuremap, net->layers[group_start_idx].bottom_boundary_edges_featuremap);
        if(group_start_idx > 0)
            memcpy(net->layers[group_start_idx - 1].output, net->input, net->inputs);
        if(g == 0){
            free(network_input_batch);
            network_input_batch = net->input;
        }
        //gettimeofday(&step_time_after, NULL);

        //timersub(&step_time_after, &step_time_before, &step_time_result);

        //total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        //boundary_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        //gettimeofday(&step_time_before, NULL);

        printf("Received input boundary. Starting inference\n");

        for (l = group_start_idx; l <= group_end_idx; ++l)
        {
            net->index = l;
            if(net->layers[l].type == CONVOLUTIONAL)
                forward_convolutional_layer_distributed_ftp(ftp_params, net->layers[l], *net);
            else
                forward_maxpool_layer(net->layers[l], *net);
            
            if(l > 0){
                clear_edges_featuremap_device(ftp_params, net, l);
                clear_spurious_edges_featuremap(ftp_params, net, l);
            }

            if((l == group_start_idx) && (group_start_idx > 0)){
                free(net->input);
            }
            net->input = net->layers[l].output;
           
            net_ref->layers[l].forward(net_ref->layers[l], *net_ref);
            net_ref->input = net_ref->layers[l].output;
            
     //       compare_ftp_to_ref(net_ref->layers[l].output, net->input,
     //                   net_ref->layers[l].out_w, net_ref->layers[l].out_h, net->layers[l].out_w, net->layers[l].out_h,
     //                   net->layers[l].out_w*ftp_params->device_id_x, net->layers[l].out_h*ftp_params->device_id_y,
     //                   net_ref->layers[l].n, net_ref->batch);

        }

       // gettimeofday(&step_time_after, NULL);
       // timersub(&step_time_after, &step_time_before, &step_time_result);

        //total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        //inference_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    }

    float* aggregate_output;
    converge_fused_layers_output_data(ftp_params, net, &aggregate_output);
    net->input = aggregate_output;

    l = ftp_params->fused_layers - 1;
 //   if(ftp_params->is_main_gateway)
 //       compare_ftp_to_ref(net_ref->layers[l].output, net->input,
 //                          net_ref->layers[l].out_w, net_ref->layers[l].out_h, net_ref->layers[l].out_w, net_ref->layers[l].out_h,
 //                          0, 0,
 //                          net_ref->layers[l].n, net_ref->batch);

    for(l = ftp_params->fused_layers; l < net->n; l++){
        if(ftp_params->is_main_gateway){
            printf("gateway inference layer %d outw %d outh %d refw %d refh %d\n", l, net->layers[l].out_w, net->layers[l].out_h, net_ref->layers[l].out_w, net_ref->layers[l].out_h);
            net->layers[l].forward(net->layers[l], *net);
            net->input = net->layers[l].output;
        }
   
        printf("ref inference layer %d\n", l);
        net_ref->layers[l].forward(net_ref->layers[l], *net_ref);
        net_ref->input = net_ref->layers[l].output;
     
 //       if(ftp_params->is_main_gateway)
 //           compare_ftp_to_ref(net_ref->layers[l].output, net->input,
 //               net_ref->layers[l].out_w, net_ref->layers[l].out_h, net->layers[l].out_w, net->layers[l].out_h,
 //               net->layers[l].out_w*ftp_params->device_id_x, net->layers[l].out_h*ftp_params->device_id_y,
 //               net_ref->layers[l].n, net_ref->batch);
    }
    ftp_params->last_fused_layer_output = aggregate_output;
    printf("Inference complete\n");
}

void static pos_correct_maxpool_indices(int* data, int stride,
                                int width_pool_in_forward,
                                int height_pool_out_backward, int width_pool_out_backward, int depth_pool_out_backward, int batch){
     int b, d, h, w;
     for(b = 0; b < batch; b++){
        for(d = 0; d < depth_pool_out_backward; d++){
            for(h = 0; h < height_pool_out_backward; h++){
                for(w = 0; w < width_pool_out_backward; w++){
                    int sample_size_out_backward = depth_pool_out_backward*height_pool_out_backward*width_pool_out_backward;
                    int channel_size_out_backward = height_pool_out_backward*width_pool_out_backward;
                    int element = data[(b*sample_size_out_backward) + (d*channel_size_out_backward) + (h*width_pool_out_backward) + w];
                    int offset_x = element % stride;
                    int offset_y = ((element - (element % width_pool_in_forward)) / width_pool_in_forward) % stride;
                    data[(b*sample_size_out_backward) + (d*channel_size_out_backward) + (h*width_pool_out_backward) + w] = 
                    stride*stride*((b*sample_size_out_backward) + 
                    (d*channel_size_out_backward) + (h*width_pool_out_backward)) + w*stride + 
                    offset_x + (stride*width_pool_out_backward*offset_y);
                    
                }
            }
        }
    }
}

void static assemble_pool_indices(ftp* ftp_params, network* net, int l){
    int* tile_core_pool_indices = calloc(net->batch*net->layers[l].delta_h_with_boundary*net->layers[l].delta_w_with_boundary*net->layers[l].c, sizeof(int));
    
    int left_extra_edges_pool_backward = net->layers[l].left_boundary_edges_delta - 
                                         (net->layers[l].left_boundary_edges_featuremap / net->layers[l].stride);
    int right_extra_edges_pool_backward = net->layers[l].right_boundary_edges_delta - 
                                         (net->layers[l].right_boundary_edges_featuremap / net->layers[l].stride);
    int bottom_extra_edges_pool_backward = net->layers[l].bottom_boundary_edges_delta - 
                                         (net->layers[l].bottom_boundary_edges_featuremap / net->layers[l].stride);
    int top_extra_edges_pool_backward = net->layers[l].top_boundary_edges_delta - 
                                         (net->layers[l].top_boundary_edges_featuremap / net->layers[l].stride);

    int left_extra_edges_pool_forward = (net->layers[l].left_boundary_edges_featuremap / net->layers[l].stride) -
                                        net->layers[l].left_boundary_edges_delta;
    int right_extra_edges_pool_forward = (net->layers[l].right_boundary_edges_featuremap / net->layers[l].stride) -
                                        net->layers[l].right_boundary_edges_delta;
    int bottom_extra_edges_pool_forward = (net->layers[l].bottom_boundary_edges_featuremap / net->layers[l].stride) -
                                        net->layers[l].bottom_boundary_edges_delta;
    int top_extra_edges_pool_forward = (net->layers[l].top_boundary_edges_featuremap / net->layers[l].stride) -
                                        net->layers[l].top_boundary_edges_delta;
    
    if(left_extra_edges_pool_backward < 0) left_extra_edges_pool_backward = 0;
    if(right_extra_edges_pool_backward < 0) right_extra_edges_pool_backward = 0;
    if(top_extra_edges_pool_backward < 0) top_extra_edges_pool_backward = 0;
    if(bottom_extra_edges_pool_backward < 0) bottom_extra_edges_pool_backward = 0;

    if(left_extra_edges_pool_forward < 0) left_extra_edges_pool_forward = 0;
    if(right_extra_edges_pool_forward < 0) right_extra_edges_pool_forward = 0;
    if(top_extra_edges_pool_forward < 0) top_extra_edges_pool_forward = 0;
    if(bottom_extra_edges_pool_forward < 0) bottom_extra_edges_pool_forward = 0;

    copy_slice((float*) tile_core_pool_indices, (float*) net->layers[l].indexes, net->batch, net->layers[l].c,
        net->layers[l].featuremap_h_with_boundary/net->layers[l].stride,
        net->layers[l].featuremap_w_with_boundary/net->layers[l].stride,
        net->layers[l].delta_h_with_boundary, net->layers[l].delta_w_with_boundary,
        left_extra_edges_pool_forward, top_extra_edges_pool_forward, left_extra_edges_pool_backward, top_extra_edges_pool_backward,
        net->layers[l].delta_h_with_boundary - top_extra_edges_pool_backward - bottom_extra_edges_pool_backward,
        net->layers[l].delta_w_with_boundary - left_extra_edges_pool_backward - right_extra_edges_pool_backward,
        net->layers[l].delta_h_with_boundary - top_extra_edges_pool_backward - bottom_extra_edges_pool_backward,
        net->layers[l].delta_w_with_boundary - left_extra_edges_pool_backward - right_extra_edges_pool_backward,
        net->workspace);

    if((left_extra_edges_pool_backward > 0) || (right_extra_edges_pool_backward > 0) || (top_extra_edges_pool_backward > 0) || (bottom_extra_edges_pool_backward > 0)){
        assemble_tile(ftp_params, net, net->batch, net->layers[l].c,
                        (float*)tile_core_pool_indices, (float*)net->layers[l].indexes,
                        net->layers[l].delta_h_with_boundary - top_extra_edges_pool_backward - bottom_extra_edges_pool_backward,
                        net->layers[l].delta_w_with_boundary - left_extra_edges_pool_backward - right_extra_edges_pool_backward,
                        left_extra_edges_pool_backward, right_extra_edges_pool_backward, top_extra_edges_pool_backward, bottom_extra_edges_pool_backward);
    }

    pos_correct_maxpool_indices(tile_core_pool_indices, net->layers[l].stride,
                                    net->layers[l].featuremap_w_with_boundary,
                                    net->layers[l].delta_h_with_boundary, net->layers[l].delta_w_with_boundary, net->layers[l].c, net->batch);
    
    memcpy(net->layers[l].indexes, tile_core_pool_indices, net->batch*net->layers[l].delta_h_with_boundary*net->layers[l].delta_w_with_boundary*net->layers[l].c*sizeof(float));
    free(tile_core_pool_indices); 

}

void static remove_extra_boundary_data(network* net, int l){
    int left_edges_extra = (net->layers[l].left_boundary_edges_delta * net->layers[l].stride) - net->layers[l-1].left_boundary_edges_delta;
    int top_edges_extra = (net->layers[l].top_boundary_edges_delta * net->layers[l].stride) - net->layers[l-1].top_boundary_edges_delta;
    int right_edges_extra = (net->layers[l].right_boundary_edges_delta * net->layers[l].stride) - net->layers[l-1].right_boundary_edges_delta;
    int bottom_edges_extra = (net->layers[l].bottom_boundary_edges_delta * net->layers[l].stride) - net->layers[l-1].bottom_boundary_edges_delta;

    if(left_edges_extra < 0) left_edges_extra = 0;
    if(right_edges_extra < 0) right_edges_extra = 0;    
    if(top_edges_extra < 0) top_edges_extra = 0;   
    if(bottom_edges_extra < 0) bottom_edges_extra = 0;
      
    copy_slice(net->layers[l-1].delta, net->layers[l-1].delta, net->batch, net->layers[l].c,
        net->layers[l-1].delta_h_with_boundary + top_edges_extra + bottom_edges_extra,
        net->layers[l-1].delta_w_with_boundary + left_edges_extra + right_edges_extra,
        net->layers[l-1].delta_h_with_boundary,
        net->layers[l-1].delta_w_with_boundary,
        left_edges_extra, top_edges_extra, 0, 0,
        net->layers[l-1].delta_h_with_boundary, net->layers[l-1].delta_w_with_boundary,
        net->layers[l-1].delta_h_with_boundary, net->layers[l-1].delta_w_with_boundary,
        net->workspace);    
}


void backward_network_distributed_ftp(ftp* ftp_params, network* net, network* net_ref){
    int g, l;
    int last_fused_layer = ftp_params->fused_layers - 1;
        
    for(l = net->n - 1; l > (last_fused_layer + 1); l--){
        net_ref->input = net_ref->layers[l-1].output;
        net_ref->delta = net_ref->layers[l-1].delta;
 
        net_ref->layers[l].backward(net_ref->layers[l], *net_ref);
            
        if(ftp_params->is_main_gateway){
            printf("main tile backward layer %d\n", l);
            net->input = net->layers[l-1].output;
            net->delta = net->layers[l-1].delta;
            
            net->layers[l].backward(net->layers[l], *net);
          
            compare_ftp_to_ref(net_ref->delta, net->delta,
                    net_ref->layers[l-1].out_w, net_ref->layers[l-1].out_h, net->layers[l-1].out_w, net->layers[l-1].out_h,
                    net->layers[l-1].out_w*ftp_params->device_id_x, net->layers[l-1].out_h*ftp_params->device_id_y,
                    net_ref->layers[l-1].n, net_ref->batch);
        }
    }
 
    if(ftp_params->is_main_gateway){
        net->input = ftp_params->last_fused_layer_output;
        //TODO allocate this and aggregate output in initialization in ftp_params
        net->delta = calloc(net->layers[last_fused_layer].original_out_h*net->layers[last_fused_layer].original_out_w*net->layers[last_fused_layer+1].c*net->batch , sizeof(float));
        net->layers[last_fused_layer + 1].backward(net->layers[last_fused_layer + 1], *net);
        ftp_params->last_fused_layer_delta = net->delta;
    }

    net_ref->input = net_ref->layers[last_fused_layer].output;
    net_ref->delta = net_ref->layers[last_fused_layer].delta;
    net_ref->layers[last_fused_layer + 1].backward(net_ref->layers[last_fused_layer + 1], *net_ref);
     printf("comparision start 1\n");
     if(ftp_params->is_main_gateway)
            compare_ftp_to_ref(net_ref->delta, net->delta,
                    net_ref->layers[last_fused_layer].out_w, net_ref->layers[last_fused_layer].out_h, net_ref->layers[last_fused_layer].out_w, net_ref->layers[last_fused_layer].out_h,
                    0, 0,
                    net_ref->layers[last_fused_layer].n, net_ref->batch);
    printf("compared 1\n");
    float* last_fused_layer_tile_delta;
    int depth = (net->layers[last_fused_layer].type == CONVOLUTIONAL) ? net->layers[last_fused_layer].n : net->layers[last_fused_layer].c;

    distribute_network_batch_delta_data(ftp_params, net, ftp_params->last_fused_layer_delta, &last_fused_layer_tile_delta);

    memcpy(net->layers[last_fused_layer].delta, last_fused_layer_tile_delta, 
    net->layers[last_fused_layer].delta_h_without_boundary*net->layers[last_fused_layer].delta_w_without_boundary*depth*net->batch*sizeof(float));
    free(last_fused_layer_tile_delta);
    //TODO: Likewise avoid this memcpy

    printf("comparision start %d %d %d %d\n", net->layers[last_fused_layer].out_w, net->layers[last_fused_layer].out_h, net_ref->layers[last_fused_layer].out_w, net_ref->layers[last_fused_layer].out_h);
    if(ftp_params->is_main_gateway)
           compare_ftp_to_ref(net_ref->layers[last_fused_layer].delta, net->layers[last_fused_layer].delta,
                    net_ref->layers[last_fused_layer].out_w, net_ref->layers[last_fused_layer].out_h, net->layers[last_fused_layer].out_w, net->layers[last_fused_layer].out_h,
                    0, 0,
                    net_ref->layers[last_fused_layer].n, net_ref->batch);
    printf("compared\n");
    int i;    
    for (g = (ftp_params->num_groups_backward - 1); g >= 0; --g)
    {
        //gettimeofday(&step_time_before, NULL);

        int group_start_idx = (g == 0) ? 1 : (ftp_params->group_sync_backward[g-1] + 1);
        int group_end_idx = ftp_params->group_sync_backward[g];

        int depth = (net->layers[group_end_idx].type == CONVOLUTIONAL) ? (net->layers[group_end_idx].n) : (net->layers[group_end_idx].c);

#ifdef GPU
        assemble_tile_gpu(net, net->batch, depth,
                        net->layers[group_end_idx].delta_gpu, net->layers[group_end_idx].delta_gpu,
                        net->layers[group_end_idx].delta_in_h_without_boundary, net->layers[group_end_idx].delta_in_w_without_boundary,
                        net->layers[group_end_idx].left_boundary_edges_delta, net->layers[group_end_idx].right_boundary_edges_delta,
                        net->layers[group_end_idx].top_boundary_edges_delta, net->layers[group_end_idx].bottom_boundary_edges_delta,
                        ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.NUM_TILES_X, ftp_params.NUM_TILES_Y);
#else      
        if(net->layers[group_end_idx].type == CONVOLUTIONAL){
            int l = group_end_idx;

            int out_h_orig = (l == last_fused_layer) ? net->layers[l].out_h : (net->layers[l+1].featuremap_h_without_boundary + 2*(net->layers[l+1].size/2));
            int out_w_orig = (l == last_fused_layer) ? net->layers[l].out_w : (net->layers[l+1].featuremap_w_without_boundary + 2*(net->layers[l+1].size/2));

            int out_h_nob = (l == last_fused_layer) ? net->layers[l].out_h : net->layers[l+1].featuremap_h_without_boundary;
            int out_w_nob = (l == last_fused_layer) ? net->layers[l].out_w : net->layers[l+1].featuremap_w_without_boundary;

            int left_offset = (l == last_fused_layer) ? 0 : net->layers[l+1].size/2;
            int top_offset = (l == last_fused_layer) ? 0 : net->layers[l+1].size/2;

            float* temp_output = calloc(out_h_nob*out_w_nob*net->batch*net->layers[l].n, sizeof(float));
            copy_slice(temp_output, net->layers[l].output, net->batch, net->layers[l].n,
                            out_h_orig, out_w_orig,
                            out_h_nob, out_w_nob,
                            left_offset, top_offset,
                            0, 0,
                            out_h_nob, out_w_nob, out_h_nob, out_w_nob,
                            net->workspace);
            gradient_array(temp_output, out_h_nob*out_w_nob*net->batch*net->layers[l].n, net->layers[l].activation, net->layers[l].delta);
            free(temp_output);
        }
  
        assemble_tile(ftp_params, net, net->batch, depth,
                        net->layers[group_end_idx].delta, net->layers[group_end_idx].delta,
                        net->layers[group_end_idx].delta_h_without_boundary, net->layers[group_end_idx].delta_w_without_boundary,
                        net->layers[group_end_idx].left_boundary_edges_delta, net->layers[group_end_idx].right_boundary_edges_delta,
                        net->layers[group_end_idx].top_boundary_edges_delta, net->layers[group_end_idx].bottom_boundary_edges_delta);
#endif

        //gettimeofday(&step_time_after, NULL);
        //timersub(&step_time_after, &step_time_before, &step_time_result);

        //total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
        //boundary_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        printf("Received delta boundary. Starting backprop\n");

        //gettimeofday(&step_time_before, NULL);        

        for (l = group_end_idx; l >= group_start_idx; --l)
        {
            //gettimeofday(&step_time_before, NULL);

            if(net->layers[l].type == MAXPOOL){
                assemble_pool_indices(ftp_params, net, l);
            }

            //gettimeofday(&step_time_after, NULL);
            //timersub(&step_time_after, &step_time_before, &step_time_result);
            //total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
            //boundary_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

            //gettimeofday(&step_time_before, NULL);  

            printf("propagating at Layer %d\n", l);

            int left_edges = net->layers[l].left_boundary_edges_featuremap;
            int right_edges = net->layers[l].right_boundary_edges_featuremap;
            int top_edges = net->layers[l].top_boundary_edges_featuremap;
            int bottom_edges = net->layers[l].bottom_boundary_edges_featuremap;

            net->input = net->layers[l-1].output;
            net->delta = net->layers[l-1].delta;

            net->layers[l].w = (l == group_start_idx) ? net->layers[l-1].delta_w_without_boundary : net->layers[l-1].delta_w_with_boundary;
            net->layers[l].h = (l == group_start_idx) ? net->layers[l-1].delta_h_without_boundary : net->layers[l-1].delta_h_with_boundary;
            net->layers[l].out_w = net->layers[l].delta_w_with_boundary;
            net->layers[l].out_h = net->layers[l].delta_h_with_boundary;

            net->layers[l].pad = net->layers[l].size - 1;

            clear_edges_delta_device(ftp_params, net, l);
            clear_spurious_edges_delta(ftp_params, net, l);

            if(net->layers[l].type == CONVOLUTIONAL)
                backward_convolutional_layer_delta_ftp(net->layers[l], *net);
            else if(net->layers[l].type == MAXPOOL){
                backward_maxpool_layer(net->layers[l], *net);
                remove_extra_boundary_data(net, l);
            }
            net_ref->layers[l].batch_normalize = 0;
            net_ref->input = net_ref->layers[l-1].output;
            net_ref->delta = net_ref->layers[l-1].delta;         
            net_ref->layers[l].backward(net_ref->layers[l], *net_ref);

            printf("ref %d %d act %d %d delact %d %d\n", net_ref->layers[l-1].out_w, net_ref->layers[l-1].out_h, net->layers[l].w, net->layers[l].h, net->layers[l].out_w, net->layers[l].out_h);
            compare_ftp_to_ref(net_ref->delta, net->delta,
                    net_ref->layers[l-1].out_w, net_ref->layers[l-1].out_h,
                    net->layers[l].w, net->layers[l].h,
                    net->layers[l].w*ftp_params->device_id_x, net->layers[l].h*ftp_params->device_id_y,
                    net_ref->layers[l].c, net_ref->batch);


            if(net->layers[l].type == CONVOLUTIONAL){
                int unit_boundary = net->layers[l].size / 2;
                int featuremap_with_unit_boundary_width = net->layers[l].featuremap_w_without_boundary + (2*unit_boundary);
                int featuremap_with_unit_boundary_height = net->layers[l].featuremap_h_without_boundary + (2*unit_boundary);
                copy_slice(net->layers[l-1].output, net->layers[l-1].output, net->batch, net->layers[l].c,
                           net->layers[l].featuremap_h_with_boundary, net->layers[l].featuremap_w_with_boundary,
                           featuremap_with_unit_boundary_height, featuremap_with_unit_boundary_width,
                           net->layers[l].left_boundary_edges_featuremap - unit_boundary, net->layers[l].top_boundary_edges_featuremap - unit_boundary,
                           0, 0,
                           featuremap_with_unit_boundary_height, featuremap_with_unit_boundary_width,
                           featuremap_with_unit_boundary_height, featuremap_with_unit_boundary_width,
                           net->workspace);

                copy_slice(net->layers[l].delta, net->layers[l].delta, net->batch, net->layers[l].n,
                           net->layers[l].delta_h_with_boundary, net->layers[l].delta_w_with_boundary,
                           net->layers[l].delta_h_without_boundary, net->layers[l].delta_w_without_boundary,
                           net->layers[l].left_boundary_edges_delta, net->layers[l].top_boundary_edges_delta,
                           0, 0,
                           net->layers[l].delta_h_without_boundary, net->layers[l].delta_w_without_boundary,
                           net->layers[l].delta_h_without_boundary, net->layers[l].delta_w_without_boundary,
                           net->workspace);                    
                 
                net->layers[l].out_w = net->layers[l].delta_w_without_boundary;
                net->layers[l].out_h = net->layers[l].delta_h_without_boundary;
                net->layers[l].h = featuremap_with_unit_boundary_height;
                net->layers[l].w = featuremap_with_unit_boundary_width;

                net->layers[l].pad = 0;

                net->index = l;

                printf("filter layer %d \n", l);

 //               backward_convolutional_layer_filters_ftp(net->layers[l], *net);

            }

            //gettimeofday(&step_time_after, NULL);
            //timersub(&step_time_after, &step_time_before, &step_time_result);
            //total_computation_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
            //backprop_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        }
    }
}

void receive_sum_transmit_device_weight_updates(ftp* ftp_params, network* net){
    int total_weights = 0;
    int i, l, n;
    for (l = 0; l < ftp_params->fused_layers; ++l){
        int num_filters = net->layers[l].n;
        int filter_size = net->layers[l].size;
        int channels = net->layers[l].c;
        total_weights += num_filters*channels*filter_size*filter_size;
    }    
    int num_tiles_in_device = ftp_params->num_device_tiles;
    int total_devices = ftp_params->num_unique_devices;

    for (i = 1; i < num_tiles_in_device; ++i)
    {
        for (l = 0; l < net->n; ++l){
            int num_filters = net->layers[l].n;
            int filter_size = net->layers[l].size;
            int channels = net->layers[l].c;

            float* tile_weight_updates_offset = (net->layers[l].weight_updates) + (i*num_filters*channels*filter_size*filter_size);
            for (n = 0; n < (num_filters*channels*filter_size*filter_size); ++n)
                net->layers[l].weight_updates[n] += tile_weight_updates_offset[n];
        }
    }

    float* data = calloc(total_weights, sizeof(float));
    for (i = 1; i < total_devices; ++i)
    {
        receive_data(data, total_weights, ftp_params->device_gateway_ids[i]);
        int layer_cumulative_weights = 0;
        for (l = 0; l < net->n; ++l){
            int num_filters = net->layers[l].n;
            int filter_size = net->layers[l].size;
            int channels = net->layers[l].c;
            for (n = 0; n < (num_filters*channels*filter_size*filter_size); ++n)
                net->layers[l].weight_updates[n] += data[layer_cumulative_weights + n];
            layer_cumulative_weights += (num_filters*channels*filter_size*filter_size);
        }
    }

    for (i = 1; i < total_devices; ++i)
    {
        int layer_cumulative_weights = 0;
        for (l = 0; l < net->n; ++l){
            int num_filters = net->layers[l].n;
            int filter_size = net->layers[l].size;
            int channels = net->layers[l].c;
            for (n = 0; n < (num_filters*channels*filter_size*filter_size); ++n)
                data[layer_cumulative_weights + n] = net->layers[l].weight_updates[n];
            layer_cumulative_weights += (num_filters*channels*filter_size*filter_size);
        }
        send_data(data, total_weights, ftp_params->device_gateway_ids[i]);
    }

    for (i = 1; i < num_tiles_in_device; ++i)
    {
        for (l = 0; l < net->n; ++l){
            int num_filters = net->layers[l].n;
            int filter_size = net->layers[l].size;
            int channels = net->layers[l].c;

            float* tile_weight_updates_offset = (net->layers[l].weight_updates) + (i*num_filters*channels*filter_size*filter_size);
            for (n = 0; n < (num_filters*channels*filter_size*filter_size); ++n)
                tile_weight_updates_offset[n] = net->layers[l].weight_updates[n];
        }
    }
   process_sema_post(ftp_params->num_device_tiles - 1, ftp_params->weights_updated);
}

void static devices_send_partial_weight_updates(ftp* ftp_params, network* net){
    int total_weights = 0;
    int num_tiles_in_device = ftp_params->num_device_tiles;
    int total_devices = ftp_params->num_unique_devices;
    int i, l, n;
    for (l = 0; l < ftp_params->fused_layers; ++l){
        int num_filters = net->layers[l].n;
        int filter_size = net->layers[l].size;
        int channels = net->layers[l].c;
        total_weights += num_filters*channels*filter_size*filter_size;
    }    

    for (i = 1; i < num_tiles_in_device; ++i)
    {
        for (l = 0; l < ftp_params->fused_layers; ++l){
            int num_filters = net->layers[l].n;
            int filter_size = net->layers[l].size;
            int channels = net->layers[l].c;

            float* tile_weight_updates_offset = (net->layers[l].weight_updates) + (i*num_filters*channels*filter_size*filter_size);

            for (n = 0; n < (num_filters*channels*filter_size*filter_size); ++n)
                net->layers[l].weight_updates[n] += tile_weight_updates_offset[n];
        }
    }

    float* data = malloc(total_weights * sizeof(float));

    int layer_cumulative_weights = 0;
    for (l = 0; l < ftp_params->fused_layers; ++l){
        int num_filters = net->layers[l].n;
        int filter_size = net->layers[l].size;
        int channels = net->layers[l].c;

        for (n = 0; n < (num_filters*channels*filter_size*filter_size); ++n)
            data[layer_cumulative_weights + n] = net->layers[l].weight_updates[n];
        layer_cumulative_weights += num_filters*channels*filter_size*filter_size;
    }

    send_data(data, total_weights, 0);
    receive_data(data, total_weights, 0);
    layer_cumulative_weights = 0;

    for (l = 0; l < ftp_params->fused_layers; ++l){
        int num_filters = net->layers[l].n;
        int filter_size = net->layers[l].size;
        int channels = net->layers[l].c;

        for (n = 0; n < (num_filters*channels*filter_size*filter_size); ++n)
            net->layers[l].weight_updates[n] = data[layer_cumulative_weights + n];
        layer_cumulative_weights += num_filters*channels*filter_size*filter_size;
    }
    free(data);

    for (i = 1; i < num_tiles_in_device; ++i)
    {
        for (l = 0; l < ftp_params->fused_layers; ++l){
            int num_filters = net->layers[l].n;
            int filter_size = net->layers[l].size;
            int channels = net->layers[l].c;

            float* tile_weight_updates_offset = (net->layers[l].weight_updates) + (i*num_filters*channels*filter_size*filter_size);

            for (n = 0; n < (num_filters*channels*filter_size*filter_size); ++n)
                tile_weight_updates_offset[n] = net->layers[l].weight_updates[n];
        }
    }
    process_sema_post(ftp_params->num_device_tiles - 1, ftp_params->weights_updated);
}

void update_weights_distributed_ftp(ftp* ftp_params, network* net){
    if(ftp_params->is_main_gateway){
        //gettimeofday(&step_time_before, NULL);
        //train_cycle_complete_sema_wait(current_device.num_tiles - 1);
        process_sema_wait(ftp_params->num_device_tiles - 1, ftp_params->batch_backprop_completed);
        //gettimeofday(&step_time_after, NULL);
        //timersub(&step_time_after, &step_time_before, &step_time_result);
        //train_complete_wait_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        //gettimeofday(&step_time_before, NULL);
        receive_sum_transmit_device_weight_updates(ftp_params, net);
        //gettimeofday(&step_time_after, NULL);
        //timersub(&step_time_after, &step_time_before, &step_time_result);
        //actual_filter_sync_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    }
    else if(ftp_params->is_device_gateway){
        //gettimeofday(&step_time_before, NULL);
        //train_cycle_complete_sema_wait(current_device.num_tiles - 1);
        process_sema_wait(ftp_params->num_device_tiles - 1, ftp_params->batch_backprop_completed);
        //gettimeofday(&step_time_after, NULL);
        //timersub(&step_time_after, &step_time_before, &step_time_result);
        //train_complete_wait_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);

        //gettimeofday(&step_time_before, NULL);
        devices_send_partial_weight_updates(ftp_params, net);
        //gettimeofday(&step_time_after, NULL);
        //timersub(&step_time_after, &step_time_before, &step_time_result);
        //actual_filter_sync_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    }
    else{
        //gettimeofday(&step_time_before, NULL);
        //train_cycle_complete_sema_post(1);
        process_sema_post(1, ftp_params->batch_backprop_completed);
        //filter_sync_complete_sema_wait(1);
        process_sema_wait(1, ftp_params->weights_updated);
        //gettimeofday(&step_time_after, NULL);
        //timersub(&step_time_after, &step_time_before, &step_time_result);
        //actual_filter_sync_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    }

    //gettimeofday(&step_time_after, NULL);
    //timersub(&step_time_after, &step_time_before, &step_time_result);

    //total_communication_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);
    //filter_partial_updates_time += (double)(step_time_result.tv_sec + (step_time_result.tv_usec)/1000000.0);


    //gettimeofday(&step_time_before, NULL);
    update_args a = {0};
    a.batch = net->batch*net->subdivisions;
    a.learning_rate = get_current_rate(net);
    a.momentum = net->momentum;
    a.decay = net->decay;
    a.adam = net->adam;
    a.B1 = net->B1;
    a.B2 = net->B2;
    a.eps = net->eps;
    ++*net->t;
    a.t = *net->t;

    int l;
    for (l = 0; l < net->n; l++)
    {
        if(net->layers[l].type == CONVOLUTIONAL){
            //net->layers[l].learning_rate_scale = 1.0;
            //update_convolutional_layer(net->layers[l], a);
        }
    }

}

float train_network_distributed_ftp(ftp* ftp_params, network *net, network* net_ref, data d)
{
    resize_layers(ftp_params, net);
    assert(d.X.rows % net->batch == 0);
    int batch = net->batch;
    int n = d.X.rows / batch;
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i){
        if(ftp_params->is_main_gateway)
            get_next_batch(d, batch, i*batch, net->input, net->truth);
        printf("%d %d\n", net_ref->h, net_ref->w);
        *net->seen += net->batch;
        net->train = 1;
        if(ftp_params->is_main_gateway){
            get_next_batch(d, batch, i*batch, net_ref->input, net_ref->truth);
            send_data(net_ref->input, 608*608*3*2, 1);
        }
        else
            receive_data(net_ref->input, 608*608*3*2, 0);
        forward_network_distributed_ftp(ftp_params, net, net_ref);
        backward_network_distributed_ftp(ftp_params, net, net_ref);
        float error = *net->cost;
        if(((*net->seen)/net->batch)%net->subdivisions == 0) update_weights_distributed_ftp(ftp_params, net);
        sum += error;
    }

    return (float)sum/(n*batch);
}

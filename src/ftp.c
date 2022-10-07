#include "ftp.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

extern sem_t filter_diverge;
extern sem_t filter_converge;

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

void get_boundry_data_forward(network*** SHARED_NETWORKS, float***SHARED_INPUT_IMAGES,
    int NUM_TILES_X, int NUM_TILES_Y,
    int current_layer_idx, 
    float** device_data, 
    int rows, int cols, orientation region,
    int device_src_id_x, int device_src_id_y, 
    int device_dst_id_x, int device_dst_id_y) {

    float* boundry_data = calloc(rows*cols, sizeof(float));
    *device_data = boundry_data;
    //TODO: ASSERT CHECK DIM OVER/UNDERFLOW

    int x_dim;
    int y_dim; 

    float* boundry_src_data;

    if((device_src_id_x >= NUM_TILES_X) && 
        (region == BOTTOM_LEFT || region == LEFT || region == TOP_LEFT)){
        fill_cpu(rows*cols, 0, *device_data, 1);
        return;
    }
    if((device_src_id_x < 0) && 
        (region == TOP_RIGHT || region == RIGHT || region == BOTTOM_RIGHT)){
        fill_cpu(rows*cols, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y >= NUM_TILES_Y) && 
        (region == TOP_LEFT || region == TOP || region == TOP_RIGHT)){
        fill_cpu(rows*cols, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y < 0) && 
        (region == BOTTOM_LEFT || region == BOTTOM || region == BOTTOM_RIGHT)){
        fill_cpu(rows*cols, 0, *device_data, 1);
        return;
    }

    if(current_layer_idx == 0){
        x_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[0].w;
        y_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[0].h;
        boundry_src_data = SHARED_INPUT_IMAGES[device_src_id_y][device_src_id_x];
    }
    else{
        x_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx - 1].out_w;
        y_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx - 1].out_h;
        boundry_src_data = SHARED_NETWORKS[device_src_id_y][device_src_id_x]->layers[current_layer_idx - 1].output;

    }

    switch(region) {
        case TOP_LEFT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j];
                }
            }
        break;

        case BOTTOM_LEFT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[(i+y_dim-rows)*x_dim + j];
                }
            }
        break;

        case TOP_RIGHT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j+x_dim-cols];
                }
            }
        break;

        case BOTTOM_RIGHT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[(i+y_dim-rows)*x_dim + j+x_dim-cols];
                }
            }
        break;

        case TOP:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j];
                }
            }
        break;

        case LEFT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j];
                }
            }
        break;

        case BOTTOM:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[(i+y_dim-rows)*x_dim + j];
                }
            }
        break;

        case RIGHT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j+x_dim-cols];
                }
            }
        break;
    }

}


void get_boundry_data_backward(network*** SHARED_NETWORKS, float*** SHARED_EXP_DELTAS,
    int NUM_TILES_X, int NUM_TILES_Y,
    int current_layer_idx, int num_layers,
    float** device_data, 
    int rows, int cols, orientation region,
    int device_src_id_x, int device_src_id_y, 
    int device_dst_id_x, int device_dst_id_y) {

    float* boundry_data = calloc(rows*cols, sizeof(float));
    *device_data = boundry_data;
    //TODO: ASSERT CHECK DIM OVER/UNDERFLOW

    int x_dim;
    int y_dim; 

    float* boundry_src_data;

    if((device_src_id_x >= NUM_TILES_X) && 
        (region == BOTTOM_LEFT || region == LEFT || region == TOP_LEFT)){
        fill_cpu(rows*cols, 0, *device_data, 1);
        return;
    }
    if((device_src_id_x < 0) && 
        (region == TOP_RIGHT || region == RIGHT || region == BOTTOM_RIGHT)){
        fill_cpu(rows*cols, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y >= NUM_TILES_Y) && 
        (region == TOP_LEFT || region == TOP || region == TOP_RIGHT)){
        fill_cpu(rows*cols, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y < 0) && 
        (region == BOTTOM_LEFT || region == BOTTOM || region == BOTTOM_RIGHT)){
        fill_cpu(rows*cols, 0, *device_data, 1);
        return;
    }

    if(current_layer_idx == (num_layers - 1)){
        x_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].out_w;
        y_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].out_h;
        boundry_src_data = SHARED_EXP_DELTAS[device_src_id_y][device_src_id_x];
    }
    else{
        x_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].out_w;
        y_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].out_h;
        boundry_src_data = SHARED_NETWORKS[device_src_id_y][device_src_id_x]->layers[current_layer_idx].delta;

    }

        // x_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].out_w;
        // y_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].out_h;
        // boundry_src_data = SHARED_NETWORKS[device_src_id_y][device_src_id_x]->layers[current_layer_idx].delta;


    switch(region) {
        case TOP_LEFT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j];
                }
            }
        break;

        case BOTTOM_LEFT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[(i+y_dim-rows)*x_dim + j];
                }
            }
        break;

        case TOP_RIGHT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j+x_dim-cols];
                }
            }
        break;

        case BOTTOM_RIGHT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[(i+y_dim-rows)*x_dim + j+x_dim-cols];
                }
            }
        break;

        case TOP:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j];
                }
            }
        break;

        case LEFT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j];
                }
            }
        break;

        case BOTTOM:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[(i+y_dim-rows)*x_dim + j];
                }
            }
        break;

        case RIGHT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[i*x_dim + j+x_dim-cols];
                }
            }
        break;
    }

}

static void setup_data_forward(network* net, device_ftp_args_v2* ftp_args, int current_layer_idx){

    int NUM_TILES_X = ftp_args->NUM_TILES_X;
    int NUM_TILES_Y = ftp_args->NUM_TILES_Y;

    network*** SHARED_NETWORKS = ftp_args->SHARED_NETWORKS; 
    float*** SHARED_INPUT_IMAGES = ftp_args->SHARED_INPUT_IMAGES;

    int DEVICE_ID_X = ftp_args->DEVICE_ID_X;
    int DEVICE_ID_Y = ftp_args->DEVICE_ID_Y;

    int INPUT_WIDTH = ftp_args->INPUT_WIDTH;
    int INPUT_HEIGHT = ftp_args->INPUT_HEIGHT;

    float* image_input = ftp_args->image_input;

    int filter_size = net->layers[current_layer_idx].size;
    int stride = net->layers[current_layer_idx].stride;

    int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
    
    int boundry_frames = unit_boundry; //unit_boundry*num_layers; //No fusing for now

    int tile_total_input_width = net->layers[current_layer_idx].w;
    int tile_total_input_height = net->layers[current_layer_idx].h;

    int tile_input_width = tile_total_input_width - (2*unit_boundry);
    int tile_input_height = tile_total_input_height - (2*unit_boundry);

    int tile_total_output_width = net->layers[current_layer_idx].out_w;
    int tile_total_output_height = net->layers[current_layer_idx].out_h;

    int conv_start_pos_x = (tile_total_output_width*DEVICE_ID_X)*stride;
    int conv_start_pos_y = (tile_total_output_height*DEVICE_ID_Y)*stride;

    int conv_end_pos_x = conv_start_pos_x + (tile_total_output_width-1)*stride + filter_size - 1;
    int conv_end_pos_y = conv_start_pos_y + (tile_total_output_height-1)*stride + filter_size - 1;

    int next_conv_start_pos_x = (tile_total_output_width*(DEVICE_ID_X+1))*stride;
    int next_conv_start_pos_y = (tile_total_output_height*(DEVICE_ID_Y+1))*stride;

    int prev_conv_end_pos_x = conv_start_pos_x - stride + filter_size - 1;
    int prev_conv_end_pos_y = conv_start_pos_y - stride + filter_size - 1;

    int tile_core_start_pos_x = DEVICE_ID_X*tile_input_width + boundry_frames;
    int tile_core_start_pos_y = DEVICE_ID_Y*tile_input_height + boundry_frames;

    int tile_core_end_pos_x = tile_core_start_pos_x + tile_input_width - 1;
    int tile_core_end_pos_y = tile_core_start_pos_y + tile_input_height - 1;

    int next_tile_core_start_pos_x = tile_core_start_pos_x + tile_input_width;
    int next_tile_core_start_pos_y = tile_core_start_pos_y + tile_input_height;

    int prev_tile_core_end_pos_x = tile_core_end_pos_x - tile_input_width;
    int prev_tile_core_end_pos_y = tile_core_end_pos_y - tile_input_height;

    int left_boundry_edges = (conv_start_pos_x < tile_core_start_pos_x) ? (tile_core_start_pos_x - conv_start_pos_x) : (conv_start_pos_x - tile_core_start_pos_x);
    int right_boundry_edges = (conv_end_pos_x > tile_core_end_pos_x) ? (conv_end_pos_x - tile_core_end_pos_x) : (tile_core_end_pos_x - conv_end_pos_x);

    int top_boundry_edges = (conv_start_pos_y < tile_core_start_pos_y) ? (tile_core_start_pos_y - conv_start_pos_y) : (conv_start_pos_y - tile_core_start_pos_y);
    int bottom_boundry_edges = (conv_end_pos_y > tile_core_end_pos_y) ? (conv_end_pos_y - tile_core_end_pos_y) : (tile_core_end_pos_y - conv_end_pos_y);

    int next_tile_left_boundry_edges = (next_conv_start_pos_x < next_tile_core_start_pos_x) ? (next_tile_core_start_pos_x - next_conv_start_pos_x) : (next_conv_start_pos_x - next_tile_core_start_pos_x);
    int prev_tile_right_boundry_edges = (prev_conv_end_pos_x > prev_tile_core_end_pos_x) ? (prev_conv_end_pos_x - prev_tile_core_end_pos_x) : (prev_tile_core_end_pos_x - prev_conv_end_pos_x);

    int next_tile_top_boundry_edges = (next_conv_start_pos_y < next_tile_core_start_pos_y) ? (next_tile_core_start_pos_y - next_conv_start_pos_y) : (next_conv_start_pos_y - next_tile_core_start_pos_y);
    int prev_tile_bottom_boundry_edges = (prev_conv_end_pos_y > prev_tile_core_end_pos_y) ? (prev_conv_end_pos_y - prev_tile_core_end_pos_y) : (prev_tile_core_end_pos_y - prev_conv_end_pos_y);


    if(current_layer_idx == 0){

        net->input = calloc(tile_total_input_width*tile_total_input_height, sizeof(float));
        fill_cpu(tile_total_input_width*tile_total_input_height, 1, net->input, 1);

        // for (int i = 0; i < tile_total_input_height; ++i)
        // {
        //     for (int j = 0; j < tile_total_input_width; ++j)
        //     {
        //         net->input[(i)*tile_total_input_width + (j)] = 1.0;//image_input[(i*tile_input_width) + j];
        //     }
        // }           
    }

    else{

        free(net->input);
        net->input = calloc(tile_total_input_width*tile_total_input_height, sizeof(float));
        fill_cpu(tile_total_input_width*tile_total_input_height, 0, net->input, 1);

        int core_img_read_start_offset_x = (left_boundry_edges >= 0) ? 0 : (-1*left_boundry_edges);
        int core_img_read_start_offset_y = (top_boundry_edges >= 0) ? 0 : (-1*top_boundry_edges);
        int core_img_write_start_offset_x = (left_boundry_edges >= 0) ? left_boundry_edges : 0;
        int core_img_write_start_offset_y = (top_boundry_edges >= 0) ? top_boundry_edges : 0;
        int num_core_img_elements_x = 0;
        int num_core_img_elements_y = 0;

        if(left_boundry_edges >= 0){
            if((tile_total_input_width - left_boundry_edges) >= tile_input_width){
                num_core_img_elements_x = tile_input_width;
            }
            else{
                num_core_img_elements_x = tile_total_input_width - left_boundry_edges;
            }
        }
        else{
            num_core_img_elements_x = tile_input_width + left_boundry_edges;
        }

        if(top_boundry_edges >= 0){
            if((tile_total_input_height - top_boundry_edges) >= tile_input_height){
                num_core_img_elements_y = tile_input_height;
            }
            else{
                num_core_img_elements_y = tile_total_input_height - top_boundry_edges;
            }
        }
        else{
            num_core_img_elements_y = tile_input_height + top_boundry_edges;
        }

        //Core tile image
        for (int i = 0; i < num_core_img_elements_y; ++i)
        {
            for (int j = 0; j < num_core_img_elements_x; ++j)
            {
                net->input[(i+core_img_write_start_offset_y)*tile_total_input_width + (j+core_img_write_start_offset_x)] = net->layers[current_layer_idx-1].output[((i+core_img_read_start_offset_y)*tile_input_width) + (j+core_img_read_start_offset_x)];
            }
        }

        //Top
        if(top_boundry_edges > 0){
            //receive top edges
            float* boundry_top;
            get_boundry_data_forward(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_top, 
                top_boundry_edges, tile_input_width, BOTTOM, 
                DEVICE_ID_X, DEVICE_ID_Y-1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges) : 0;
            for (int i = 0; i < top_boundry_edges; ++i)
            {
                for (int j = 0; j < tile_input_width; ++j)
                {

                    net->input[(i*tile_total_input_width) + (j+left_write_offset)] = boundry_top[(i*tile_input_width) + j];
                }
            }
            free(boundry_top);
        } 

        //Left
        if(left_boundry_edges > 0){
            //receive left edges

            float* boundry_left;
            get_boundry_data_forward(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_left, 
                tile_input_height, left_boundry_edges, RIGHT, 
                DEVICE_ID_X-1, DEVICE_ID_Y,
                DEVICE_ID_X, DEVICE_ID_Y);

            int top_write_offset = (top_boundry_edges >= 0) ? top_boundry_edges : 0;
            for (int i = 0; i < tile_input_height; ++i)
            {
                for (int j = 0; j < left_boundry_edges; ++j)
                {
                    net->input[(i+top_write_offset)*tile_total_input_width + j] = boundry_left[(i*left_boundry_edges) + j];
                }
            }
            free(boundry_left);
        } 
        // //Bottom 
        if(bottom_boundry_edges > 0){
            //receive bottom edges
            float* boundry_bottom;
            get_boundry_data_forward(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_bottom, 
                bottom_boundry_edges, tile_input_width, TOP, 
                DEVICE_ID_X, DEVICE_ID_Y+1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int bottom_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges + tile_input_height) : tile_input_height;
            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges) : 0;
            for (int i = 0; i < bottom_boundry_edges; ++i)
            {
                for (int j = 0; j < tile_input_width; ++j)
                {
                    net->input[(i+bottom_write_offset)*tile_total_input_width + (j+left_write_offset)] = boundry_bottom[(i*tile_input_width) + j];
                }
            }
            free(boundry_bottom);
        }   
        // //Right
        if(right_boundry_edges > 0){
            //receive right edges
            float* boundry_right;
            get_boundry_data_forward(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_right, 
                tile_input_height, right_boundry_edges, LEFT, 
                DEVICE_ID_X+1, DEVICE_ID_Y,
                DEVICE_ID_X, DEVICE_ID_Y);

            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges + tile_input_width) : tile_input_width;
            int top_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges) : 0;
            for (int i = 0; i < tile_input_height; ++i)
            {
                for (int j = 0; j < right_boundry_edges; ++j)
                {
                    net->input[(i+top_write_offset)*tile_total_input_width + (j+left_write_offset)] = boundry_right[(i*right_boundry_edges) + j];
                }
            }
            free(boundry_right);
        } 
        // //Top left
        if((top_boundry_edges > 0) && (left_boundry_edges > 0)){

            float* boundry_top_left;
            get_boundry_data_forward(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_top_left, 
                top_boundry_edges, left_boundry_edges, BOTTOM_RIGHT, 
                DEVICE_ID_X-1, DEVICE_ID_Y-1,
                DEVICE_ID_X, DEVICE_ID_Y);

            for (int i = 0; i < top_boundry_edges; ++i)
            {
                for (int j = 0; j < left_boundry_edges; ++j)
                {
                    net->input[(i*tile_total_input_width) + j] = boundry_top_left[(i*left_boundry_edges) + j];
                }
            }
            free(boundry_top_left);
        }

        // //Top right
        if((top_boundry_edges > 0) && (right_boundry_edges > 0)){
            float* boundry_top_right;
            get_boundry_data_forward(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_top_right, 
                top_boundry_edges, right_boundry_edges, BOTTOM_LEFT, 
                DEVICE_ID_X+1, DEVICE_ID_Y-1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges + tile_input_width) : tile_input_width;
            for (int i = 0; i < top_boundry_edges; ++i)
            {
                for (int j = 0; j < right_boundry_edges; ++j)
                {
                    net->input[(i*tile_total_input_width) + (j+left_write_offset)] = boundry_top_right[(i*right_boundry_edges) + j];
                }
            }
            free(boundry_top_right);
        }

        // //Bottom left
        if((bottom_boundry_edges > 0) && (left_boundry_edges > 0)){
            float* boundry_bottom_left;
            get_boundry_data_forward(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_bottom_left, 
                bottom_boundry_edges, left_boundry_edges, TOP_RIGHT, 
                DEVICE_ID_X-1, DEVICE_ID_Y+1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int top_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges + tile_input_height) : tile_input_height;
            for (int i = 0; i < bottom_boundry_edges; ++i)
            {
                for (int j = 0; j < left_boundry_edges; ++j)
                {
                    net->input[(i+top_write_offset)*(tile_total_input_width) + j] = boundry_bottom_left[(i*left_boundry_edges) + j];
                }
            }
            free(boundry_bottom_left);
        }

        // //Bottom right
        if((bottom_boundry_edges > 0) && (right_boundry_edges > 0)){
            float* boundry_bottom_right;
            get_boundry_data_forward(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_bottom_right, 
                bottom_boundry_edges, right_boundry_edges, TOP_LEFT, 
                DEVICE_ID_X+1, DEVICE_ID_Y+1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int top_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges + tile_input_height) : tile_input_height;
            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges + tile_input_width) : tile_input_width;
            for (int i = 0; i < bottom_boundry_edges; ++i)
            {
                for (int j = 0; j < right_boundry_edges; ++j)
                {
                    net->input[(i+top_write_offset)*tile_total_input_width + (j+left_write_offset)] = boundry_bottom_right[(i*right_boundry_edges) + j];
                }
            }
            free(boundry_bottom_right);
        }

    }
}






static void setup_data_backward(network* net, device_ftp_args_v2* ftp_args, int current_layer_idx){

    int NUM_TILES_X = ftp_args->NUM_TILES_X;
    int NUM_TILES_Y = ftp_args->NUM_TILES_Y;

    network*** SHARED_NETWORKS = ftp_args->SHARED_NETWORKS; 
    float*** SHARED_EXP_DELTA = ftp_args->SHARED_EXP_DELTA;

    int num_layers = ftp_args->num_layers;

    int DEVICE_ID_X = ftp_args->DEVICE_ID_X;
    int DEVICE_ID_Y = ftp_args->DEVICE_ID_Y;

    int OUPUT_WIDTH = net->layers[net->n - 1].out_w;
    int OUPUT_HEIGHT = net->layers[net->n - 1].out_h;

    float* image_input = ftp_args->image_input;

    int filter_size = net->layers[current_layer_idx].size;
    int stride = net->layers[current_layer_idx].stride;

    int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
    
    int boundry_frames = unit_boundry; //unit_boundry*num_layers; //No fusing for now

    //dimensions of required delta propagating backward
    int tile_delta_out_width = net->layers[current_layer_idx-1].out_w;
    int tile_delta_out_height = net->layers[current_layer_idx-1].out_h;

    int tile_total_input_width = net->layers[current_layer_idx].w;
    int tile_total_input_height = net->layers[current_layer_idx].h;

    int tile_input_width = tile_total_input_width - (2*unit_boundry);
    int tile_input_height = tile_total_input_height - (2*unit_boundry);

    //dimensions of delta coming in from next layer
    int tile_delta_in_width = net->layers[current_layer_idx].out_w;
    int tile_delta_in_height = net->layers[current_layer_idx].out_h;

    int dilated_tile_delta_in_width = tile_delta_in_width + (tile_delta_in_width)*stride;
    int dilated_tile_delta_in_height = tile_delta_in_height + (tile_delta_in_height)*stride;

    //computes how much actual data to get from neighboring edges. can be optimized in strided case
    //to avoid 0s communication
    int left_boundry_edges = (stride > 1 && boundry_frames > 0) ? (boundry_frames/(stride)) : boundry_frames;
    int top_boundry_edges = left_boundry_edges;

    int right_boundry_edges = (stride > 1 && boundry_frames > 0) ? ((boundry_frames+1)/(stride)) : boundry_frames;
    int bottom_boundry_edges = right_boundry_edges;

    int tile_total_delta_in_width = tile_delta_in_width + left_boundry_edges + right_boundry_edges;
    int tile_total_delta_in_height = tile_delta_in_height + bottom_boundry_edges + top_boundry_edges;

    net->input = net->layers[current_layer_idx - 1].output;
    net->delta = net->layers[current_layer_idx - 1].delta;

    //Core tile delta
    for (int i = 0; i < (tile_delta_in_height); ++i)
    {
        for (int j = 0; j < (tile_delta_in_width); ++j)
        {
            net->layers[current_layer_idx].delta_with_boundry[(i+top_boundry_edges)*tile_total_delta_in_width + j+left_boundry_edges] = net->layers[current_layer_idx].delta[(i)*tile_delta_in_width + (j)];
        }
    }

    //Boundry Delta
    //Top
    if(top_boundry_edges > 0){
        //receive top edges
        float* boundry_top;
        get_boundry_data_backward(SHARED_NETWORKS, SHARED_EXP_DELTA,
            NUM_TILES_X, NUM_TILES_Y,
            current_layer_idx, num_layers,
            &boundry_top, 
            top_boundry_edges, tile_delta_in_width, BOTTOM, 
            DEVICE_ID_X, DEVICE_ID_Y-1,
            DEVICE_ID_X, DEVICE_ID_Y);

        int left_write_offset = left_boundry_edges;
        for (int i = 0; i < top_boundry_edges; ++i)
        {
            for (int j = 0; j < tile_delta_in_width; ++j)
            {
                net->layers[current_layer_idx].delta_with_boundry[(i*tile_total_delta_in_width) + (j+left_boundry_edges)] = boundry_top[(i*tile_delta_in_width) + j];
            }
        }

        free(boundry_top);
    } 

    //Left
    if(left_boundry_edges > 0){
        //receive left edges

        float* boundry_left;
        get_boundry_data_backward(SHARED_NETWORKS, SHARED_EXP_DELTA,
            NUM_TILES_X, NUM_TILES_Y,
            current_layer_idx, num_layers,
            &boundry_left, 
            tile_delta_in_height, left_boundry_edges, RIGHT, 
            DEVICE_ID_X-1, DEVICE_ID_Y,
            DEVICE_ID_X, DEVICE_ID_Y);

        int top_write_offset = top_boundry_edges;
        for (int i = 0; i < tile_delta_in_height; ++i)
        {
            for (int j = 0; j < left_boundry_edges; ++j)
            {
                net->layers[current_layer_idx].delta_with_boundry[(i+top_write_offset)*tile_total_delta_in_width + j] = boundry_left[(i*left_boundry_edges) + j];
            }
        }
        free(boundry_left);
    } 
        // //Bottom 
    if(bottom_boundry_edges > 0){
        //receive bottom edges
        float* boundry_bottom;
        get_boundry_data_backward(SHARED_NETWORKS, SHARED_EXP_DELTA,
            NUM_TILES_X, NUM_TILES_Y,
            current_layer_idx, num_layers,
            &boundry_bottom, 
            bottom_boundry_edges, tile_delta_in_width, TOP, 
            DEVICE_ID_X, DEVICE_ID_Y+1,
            DEVICE_ID_X, DEVICE_ID_Y);

        int bottom_write_offset = top_boundry_edges + tile_delta_in_height;
        int left_write_offset = left_boundry_edges;

        for (int i = 0; i < bottom_boundry_edges; ++i)
        {
            for (int j = 0; j < tile_delta_in_width; ++j)
            {
                net->layers[current_layer_idx].delta_with_boundry[(i+bottom_write_offset)*tile_total_delta_in_width + (j+left_write_offset)] = boundry_bottom[(i*tile_delta_in_width) + j];
            }
        }
        free(boundry_bottom);
    }   
        // //Right
    if(right_boundry_edges > 0){
        //receive right edges
        float* boundry_right;
        get_boundry_data_backward(SHARED_NETWORKS, SHARED_EXP_DELTA,
            NUM_TILES_X, NUM_TILES_Y,
            current_layer_idx, num_layers,
            &boundry_right, 
            tile_delta_in_height, right_boundry_edges, LEFT, 
            DEVICE_ID_X+1, DEVICE_ID_Y,
            DEVICE_ID_X, DEVICE_ID_Y);

        int left_write_offset = left_boundry_edges + tile_delta_in_width;
        int top_write_offset = top_boundry_edges;
        for (int i = 0; i < tile_delta_in_height; ++i)
        {
            for (int j = 0; j < right_boundry_edges; ++j)
            {
                net->layers[current_layer_idx].delta_with_boundry[(i+top_write_offset)*tile_total_delta_in_width + (j+left_write_offset)] = boundry_right[(i*right_boundry_edges) + j];
            }
        }
        free(boundry_right);
    } 
    // //Top left
    if((top_boundry_edges > 0) && (left_boundry_edges > 0)){

        float* boundry_top_left;
        get_boundry_data_backward(SHARED_NETWORKS, SHARED_EXP_DELTA,
            NUM_TILES_X, NUM_TILES_Y,
            current_layer_idx, num_layers,
            &boundry_top_left, 
            top_boundry_edges, left_boundry_edges, BOTTOM_RIGHT, 
            DEVICE_ID_X-1, DEVICE_ID_Y-1,
            DEVICE_ID_X, DEVICE_ID_Y);

        for (int i = 0; i < top_boundry_edges; ++i)
        {
            for (int j = 0; j < left_boundry_edges; ++j)
            {
                net->layers[current_layer_idx].delta_with_boundry[(i*tile_total_delta_in_width) + j] = boundry_top_left[(i*left_boundry_edges) + j];
            }
        }
        free(boundry_top_left);
    }

    // //Top right
    if((top_boundry_edges > 0) && (right_boundry_edges > 0)){
        float* boundry_top_right;
        get_boundry_data_backward(SHARED_NETWORKS, SHARED_EXP_DELTA,
            NUM_TILES_X, NUM_TILES_Y,
            current_layer_idx, num_layers,
            &boundry_top_right, 
            top_boundry_edges, right_boundry_edges, BOTTOM_LEFT, 
            DEVICE_ID_X+1, DEVICE_ID_Y-1,
            DEVICE_ID_X, DEVICE_ID_Y);

        int left_write_offset = (left_boundry_edges + tile_delta_in_width);
        for (int i = 0; i < top_boundry_edges; ++i)
        {
            for (int j = 0; j < right_boundry_edges; ++j)
            {
                net->layers[current_layer_idx].delta_with_boundry[(i*tile_total_delta_in_width) + (j+left_write_offset)] = boundry_top_right[(i*right_boundry_edges) + j];
            }
        }
        free(boundry_top_right);
    }

    // //Bottom left
    if((bottom_boundry_edges > 0) && (left_boundry_edges > 0)){
        float* boundry_bottom_left;
        get_boundry_data_backward(SHARED_NETWORKS, SHARED_EXP_DELTA,
            NUM_TILES_X, NUM_TILES_Y,
            current_layer_idx, num_layers,
            &boundry_bottom_left, 
            bottom_boundry_edges, left_boundry_edges, TOP_RIGHT, 
            DEVICE_ID_X-1, DEVICE_ID_Y+1,
            DEVICE_ID_X, DEVICE_ID_Y);

        int top_write_offset = top_boundry_edges + tile_delta_in_height;
        for (int i = 0; i < bottom_boundry_edges; ++i)
        {
            for (int j = 0; j < left_boundry_edges; ++j)
            {
                net->layers[current_layer_idx].delta_with_boundry[(i+top_write_offset)*(tile_total_delta_in_width) + j] = boundry_bottom_left[(i*left_boundry_edges) + j];
            }
        }
        free(boundry_bottom_left);
    }

    // //Bottom right
    if((bottom_boundry_edges > 0) && (right_boundry_edges > 0)){
        float* boundry_bottom_right;
        get_boundry_data_backward(SHARED_NETWORKS, SHARED_EXP_DELTA,
            NUM_TILES_X, NUM_TILES_Y,
            current_layer_idx, num_layers,
            &boundry_bottom_right, 
            bottom_boundry_edges, right_boundry_edges, TOP_LEFT, 
            DEVICE_ID_X+1, DEVICE_ID_Y+1,
            DEVICE_ID_X, DEVICE_ID_Y);

        int top_write_offset = top_boundry_edges + tile_delta_in_height;
        int left_write_offset = left_boundry_edges + tile_delta_in_width;
        for (int i = 0; i < bottom_boundry_edges; ++i)
        {
            for (int j = 0; j < right_boundry_edges; ++j)
            {
                net->layers[current_layer_idx].delta_with_boundry[(i+top_write_offset)*tile_total_delta_in_width + (j+left_write_offset)] = boundry_bottom_right[(i*right_boundry_edges) + j];
            }
        }
        free(boundry_bottom_right);
    }


    int save_h = net->layers[current_layer_idx].h;
    int save_w = net->layers[current_layer_idx].w;

    int save_out_h = net->layers[current_layer_idx].out_h;
    int save_out_w = net->layers[current_layer_idx].out_w;

    int save_pad = net->layers[current_layer_idx].pad;

    //TODO Take care of this in col2im itself and avoid this confusing implementation
    int extra_zero_padded_edges = boundry_frames - (left_boundry_edges*stride); //accounts for the top left extra edge.

    net->layers[current_layer_idx].out_h = tile_delta_in_height + top_boundry_edges + bottom_boundry_edges;
    net->layers[current_layer_idx].out_w = tile_delta_in_width + left_boundry_edges + right_boundry_edges;

    int dilated_outh = net->layers[current_layer_idx].out_h + (stride-1)*(net->layers[current_layer_idx].out_h);
    int dilated_outw = net->layers[current_layer_idx].out_w + (stride-1)*(net->layers[current_layer_idx].out_w);
    net->layers[current_layer_idx].h = dilated_outh - filter_size + 1;
    net->layers[current_layer_idx].w = dilated_outw - filter_size + 1;

    //TODO: Make this generic across 2 D. now assumes both w and h are equal
    int base_pad = (dilated_outh - net->layers[current_layer_idx].h);

    net->layers[current_layer_idx].pad = base_pad-extra_zero_padded_edges;

    printf("%d %d %d %d\n", top_boundry_edges, left_boundry_edges, bottom_boundry_edges, right_boundry_edges);
    // printf("DIN_H %d DIN_W %d TOTDIN_H %d TOTDIN_W%d\n", tile_delta_in_height, tile_delta_in_width, tile_total_delta_in_height, tile_total_delta_in_height);
    // printf("DOUT_H %d DOUT_W %d\n", tile_delta_out_height, tile_delta_out_width);
    printf("%d %d %d\n", dilated_outh, dilated_outw, extra_zero_padded_edges);
    printf("%d %d %d %d\n", net->layers[current_layer_idx].out_h, net->layers[current_layer_idx].out_w, net->layers[current_layer_idx].h, net->layers[current_layer_idx].w);

    if(current_layer_idx == 3 && (DEVICE_ID_X == 1) && (DEVICE_ID_Y == 0)){
        for (int i = 0; i < net->layers[current_layer_idx].out_h; ++i)
        {
            for (int j = 0; j < net->layers[current_layer_idx].out_w; ++j)
            {
                printf("%.2f ", net->layers[current_layer_idx].delta_with_boundry[i*(net->layers[current_layer_idx].out_w) + j]);
            }
            printf("\n");
        }
        //while(1);
    }

    backward_convolutional_layer_dist_v2(net->layers[current_layer_idx], *net);

    // if(current_layer_idx == 3 && (DEVICE_ID_X == 1) && (DEVICE_ID_Y == 0)){
    //     for (int i = 0; i < net->layers[current_layer_idx].h; ++i)
    //     {
    //         for (int j = 0; j < net->layers[current_layer_idx].w; ++j)
    //         {
    //             printf("%.2f ", net->layers[current_layer_idx-1].delta[i*(net->layers[current_layer_idx].w) + j]);
    //         }
    //         printf("\n");
    //     }
    //     while(1);
    // }

    int extra_cols = net->layers[current_layer_idx].w - tile_input_width;
    int extra_rows = net->layers[current_layer_idx].h - tile_input_height;

    printf("Extras: %d %d\n", extra_rows, extra_cols);

    int left_realignment_edges_receive = (DEVICE_ID_X == 0) ? 0 : (DEVICE_ID_X*extra_cols);
    int top_realignment_edges_receive = (DEVICE_ID_Y == 0) ? 0 : (DEVICE_ID_Y*extra_rows);

    int bottom_realignment_edges_transmit = (DEVICE_ID_Y == NUM_TILES_Y - 1) ? 0 : ((DEVICE_ID_Y + 1)*extra_rows);
    int right_realignment_edges_transmit = (DEVICE_ID_X == NUM_TILES_X - 1) ? 0 : ((DEVICE_ID_X + 1)*extra_cols);   



    //For this single thread test purpose since the tiles are executes top left to left/bottom.
    //the tiles from which some tile is getting boundry data will already be done with back pass
    //In real implementation with actual communication, need some wait synchronization
    // float * delta_ftp_top_receive;
    // float * delta_ftp_left_receive;
    // float * delta_ftp_top_left_receive;
    // float * delta_ftp_bottom_transmit;
    // float * delta_ftp_bottom_right_transmit;
    // float * delta_ftp_right_transmit;

    if(right_realignment_edges_transmit > 0){
        //receive top edges

        int left_write_offset = net->layers[current_layer_idx].w - extra_cols;
        for (int i = 0; i < (net->layers[current_layer_idx].h - extra_rows); i++)
        {
            for (int j = left_write_offset; j < left_write_offset+extra_cols; ++j)
            {
                net->layers[current_layer_idx-1].delta_ftp_right_transmit[i*extra_cols + (j - left_write_offset)] =
                     net->layers[current_layer_idx-1].delta[i*net->layers[current_layer_idx].w + j];
            }
        }
    }
    if(bottom_realignment_edges_transmit > 0){
        //receive top edges

        int top_write_offset = net->layers[current_layer_idx].h - extra_rows;
        for (int i = top_write_offset; i < top_write_offset+extra_rows; ++i)
        {
            for (int j = 0; j < (net->layers[current_layer_idx].w - extra_cols); j++)
            {
                net->layers[current_layer_idx-1].delta_ftp_bottom_transmit[(i-top_write_offset)*(net->layers[current_layer_idx].w - extra_cols) + j] =
                     net->layers[current_layer_idx-1].delta[i*net->layers[current_layer_idx].w + j];
            }
        }
    }


    if(bottom_realignment_edges_transmit > 0 && right_realignment_edges_transmit > 0){
        //receive top edges
        int left_write_offset = net->layers[current_layer_idx].w - extra_cols;
        int top_write_offset = net->layers[current_layer_idx].h - extra_rows;
        for (int i = top_write_offset; i < top_write_offset+extra_rows; ++i)
        {
            for (int j = left_write_offset; j < left_write_offset+extra_cols; ++j)
            {
                net->layers[current_layer_idx-1].delta_ftp_bottom_right_transmit[(i-top_write_offset)*(extra_cols) + j - left_write_offset] =
                     net->layers[current_layer_idx-1].delta[i*net->layers[current_layer_idx].w + j];
            }
        }
    } 

    net->layers[current_layer_idx].pad = save_pad;

    net->layers[current_layer_idx].h = save_h;
    net->layers[current_layer_idx].w = save_w;

    net->layers[current_layer_idx].out_h = save_out_h;
    net->layers[current_layer_idx].out_w = save_out_w;


    float * temp = calloc(net->layers[current_layer_idx-1].outputs*10, sizeof(float));
    //Left
    if(left_realignment_edges_receive > 0){
        //receive left edges
        for(int i=0; i < net->layers[current_layer_idx].h; i++)
            for (int j = 0; j < left_realignment_edges_receive; ++j)
            {
                temp[((i+top_realignment_edges_receive)*net->layers[current_layer_idx].w) + j] = 
                    ftp_args->SHARED_NETWORKS[DEVICE_ID_Y][DEVICE_ID_X-1]->layers[current_layer_idx-1].
                        delta_ftp_right_transmit[i*left_realignment_edges_receive + j];
            }
    }

    if(top_realignment_edges_receive > 0){
        //receive left edges
        for(int i = 0; i < top_realignment_edges_receive; ++i)
            for (int j=0; j < net->layers[current_layer_idx].w; j++)
            {
                temp[(i*net->layers[current_layer_idx].w) + j + left_realignment_edges_receive] = 
                    ftp_args->SHARED_NETWORKS[DEVICE_ID_Y-1][DEVICE_ID_X]->layers[current_layer_idx-1].
                        delta_ftp_bottom_transmit[i*(net->layers[current_layer_idx].w) + j];
            }
    }

    // //Top left
    if((top_realignment_edges_receive > 0) && (left_realignment_edges_receive > 0)){

        for (int i = 0; i < top_realignment_edges_receive; ++i)
        {
            for (int j = 0; j < left_realignment_edges_receive; ++j)
            {
                temp[(i*left_realignment_edges_receive) + j] = 
                    ftp_args->SHARED_NETWORKS[DEVICE_ID_Y-1][DEVICE_ID_X-1]->layers[current_layer_idx-1].
                        delta_ftp_bottom_right_transmit[i*(left_realignment_edges_receive) + j];
            }
        }
    }
    int elongated_h = dilated_outh - filter_size + 1;
    int elongated_w = dilated_outw - filter_size + 1;

    for (int i = 0; i < (net->layers[current_layer_idx].h - top_realignment_edges_receive); ++i)
    {
        for (int j = 0; j < (net->layers[current_layer_idx].w - left_realignment_edges_receive); ++j)
        {
                temp[((i+top_realignment_edges_receive)*net->layers[current_layer_idx].w) + j + left_realignment_edges_receive] = 
                    net->layers[current_layer_idx-1].
                        delta[i*(elongated_h) + j];
        }
    }

    // for (int i = 0; i < (net->layers[current_layer_idx].h); ++i)
    // {
    //     for (int j = 0; j < (net->layers[current_layer_idx].w); ++j)
    //     {
    //             printf("%.2f ", temp[i*net->layers[current_layer_idx].w + j]);
    //     }
    //     printf("\n");
    // }
    // while(1);

    //TODO: realign data
    // free(net->layers[current_layer_idx-1].delta);
    // net->layers[current_layer_idx-1].delta = temp;
    
}






void execute_dev_v2_forward(void* ptr, int current_layer_idx){

    printf("Thread started exec_device\n\n");
    //while(1);

    device_ftp_args_v2* ftp_args = (device_ftp_args_v2*) ptr;

    int DEVICE_ID_Y = ftp_args->DEVICE_ID_Y;
    int DEVICE_ID_X = ftp_args->DEVICE_ID_X;    

    network *net = ftp_args->SHARED_NETWORKS[DEVICE_ID_Y][DEVICE_ID_X];
    net->index = current_layer_idx;
    
    setup_data_forward(net, ftp_args, current_layer_idx);

}

void execute_dev_v2_backward(void* ptr, int current_layer_idx){

    printf("Thread started exec_device\n\n");
    //while(1);

    device_ftp_args_v2* ftp_args = (device_ftp_args_v2*) ptr;

    int DEVICE_ID_Y = ftp_args->DEVICE_ID_Y;
    int DEVICE_ID_X = ftp_args->DEVICE_ID_X;    

    network *net = ftp_args->SHARED_NETWORKS[DEVICE_ID_Y][DEVICE_ID_X];
    net->index = current_layer_idx;

    if(current_layer_idx == (ftp_args->num_layers - 1)){
        int ydim = net->layers[current_layer_idx].out_h;
        int xdim = net->layers[current_layer_idx].out_w;
        for (int i = 0; i < ydim; ++i)
        {
            for (int j = 0; j < xdim; ++j)
            {
                net->layers[current_layer_idx].delta[i*xdim + j] = ftp_args->SHARED_EXP_DELTA[DEVICE_ID_Y][DEVICE_ID_X][i*xdim + j];
            }
        }
    }
    
    setup_data_backward(net, ftp_args, current_layer_idx);
}
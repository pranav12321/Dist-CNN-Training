#include "fused_convolution.h"
#include "ftp.h"
#include "fused.h"

void get_forward_group_boundry_data(network*** SHARED_NETWORKS, float***SHARED_INPUT_IMAGES,
    int NUM_TILES_X, int NUM_TILES_Y,
    int current_layer_idx, 
    float** device_data, 
    int rows, int cols, orientation region,
    int device_src_id_x, int device_src_id_y, 
    int device_dst_id_x, int device_dst_id_y){

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

    x_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].original_featuremap_in_w;
    y_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].original_featuremap_in_h;

    int top_boundry_edges = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].top_boundry_edges_featuremap;
    int bottom_boundry_edges = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].bottom_boundry_edges_featuremap;
    int right_boundry_edges = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].right_boundry_edges_featuremap;
    int left_boundry_edges = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].left_boundry_edges_featuremap;

    if(current_layer_idx == 0){
        boundry_src_data = SHARED_INPUT_IMAGES[device_src_id_y][device_src_id_x];
    }
    else{
        boundry_src_data = SHARED_NETWORKS[device_src_id_y][device_src_id_x]->layers[current_layer_idx-1].output_without_boundry;
    }


    // if(device_dst_id_x == 1 && device_dst_id_y == 0 && current_layer_idx == 4 && region == RIGHT){

    //     printf("%d %d\n\n", device_src_id_x, device_src_id_y);
    //     for (int i = 0; i < 6; ++i)
    //     {
    //         for (int j = 0; j < 6; ++j)
    //         {
    //             printf("%.2f ", SHARED_NETWORKS[0][0]->layers[3].output[(i)*6 + j]);
    //         }
    //         printf("\n");
    //     }

    //     for (int i = 0; i < rows; ++i)
    //     {
    //         for (int j = 0; j < cols; ++j)
    //         {
    //             printf("%.2f ", boundry_src_data[(i)*x_dim + j]);
    //         }
    //         printf("\n");
    //     }
    // while(1);
    // }

    printf("\n");      

    switch(region) {
        case TOP_LEFT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[(i)*x_dim + j];
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
                    boundry_data[i*cols + j] = boundry_src_data[(i)*x_dim + j+x_dim-cols];
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
            }        // if(group.layer_start_idx == 0){
        //     net->input = SHARED_INPUT_IMAGES[DEVICE_ID_Y][DEVICE_ID_X];
        // }
        // else{
        //     net->input = net->layers[net->layers[group.layer_start_idx - 1]].output;
        // }
        break;

        case TOP:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[(i)*x_dim + j];
                }
            }
        break;

        case LEFT:
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    boundry_data[i*cols + j] = boundry_src_data[(i)*x_dim + j];
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
                    boundry_data[i*cols + j] = boundry_src_data[(i)*x_dim + j+x_dim-cols];
                }
            }
        break;
    }

}


void get_backward_group_boundry_data(network*** SHARED_NETWORKS, float*** SHARED_EXP_DELTAS,
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

    x_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].delta_in_w_without_boundry;
    y_dim = SHARED_NETWORKS[device_dst_id_y][device_dst_id_x]->layers[current_layer_idx].delta_in_h_without_boundry;

    if(current_layer_idx == (num_layers - 1)){
        boundry_src_data = SHARED_EXP_DELTAS[device_src_id_y][device_src_id_x];
    }
    else{
        boundry_src_data = SHARED_NETWORKS[device_src_id_y][device_src_id_x]->layers[current_layer_idx].delta_without_boundry;
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


void assemble_forward_group_data(network*** SHARED_NETWORKS, 
                                float***SHARED_INPUT_IMAGES,
                                int NUM_TILES_X, int NUM_TILES_Y,
								 group_profile_forward group,
                                 int DEVICE_ID_X, int DEVICE_ID_Y
								 ){



        network* net = SHARED_NETWORKS[DEVICE_ID_Y][DEVICE_ID_X];

        layer start_layer = net->layers[group.layer_start_idx];

        int top_boundry_edges = start_layer.top_boundry_edges_featuremap;
        int bottom_boundry_edges = start_layer.bottom_boundry_edges_featuremap;
        int right_boundry_edges = start_layer.right_boundry_edges_featuremap;
        int left_boundry_edges = start_layer.left_boundry_edges_featuremap;

        int tile_input_height_original = start_layer.original_featuremap_in_h;
        int tile_input_width_original = start_layer.original_featuremap_in_w;
        //int tile_input_height = start_layer.featuremap_in_h_without_boundry;
        //int tile_input_width = start_layer.featuremap_in_w_without_boundry;
        int tile_total_input_height = start_layer.featuremap_in_h_with_boundry;
        int tile_total_input_width = start_layer.featuremap_in_w_with_boundry;

        int current_layer_idx = group.layer_start_idx;



        int core_img_read_start_offset_x = (left_boundry_edges >= 0) ? 0 : (-1*left_boundry_edges);
        int core_img_read_start_offset_y = (top_boundry_edges >= 0) ? 0 : (-1*top_boundry_edges);
        int core_img_write_start_offset_x = (left_boundry_edges >= 0) ? left_boundry_edges : 0;
        int core_img_write_start_offset_y = (top_boundry_edges >= 0) ? top_boundry_edges : 0;
        int num_core_img_elements_x = 0;
        int num_core_img_elements_y = 0;


        if(left_boundry_edges >= 0){
            if((tile_total_input_width - left_boundry_edges) >= tile_input_width_original){
                num_core_img_elements_x = tile_input_width_original;
            }
            else{
                num_core_img_elements_x = tile_total_input_width - left_boundry_edges;
            }
        }
        else{
            num_core_img_elements_x = tile_input_width_original + left_boundry_edges;
        }


        if(top_boundry_edges >= 0){
            if((tile_total_input_height - top_boundry_edges) >= tile_input_height_original){
                num_core_img_elements_y = tile_input_height_original;
            }
            else{
                num_core_img_elements_y = tile_total_input_height - top_boundry_edges;
            }
        }
        else{
            num_core_img_elements_y = tile_input_height_original + top_boundry_edges;
        }

        float* group_initial_featuremap;

        if(group.layer_start_idx == 0){
            group_initial_featuremap = SHARED_INPUT_IMAGES[DEVICE_ID_Y][DEVICE_ID_X];
        }
        else{
            group_initial_featuremap = net->layers[group.layer_start_idx - 1].output;
        }


        //Core tile image
        for (int i = 0; i < num_core_img_elements_y; ++i)
        {
            for (int j = 0; j < num_core_img_elements_x; ++j)
            {
                net->input[(i+core_img_write_start_offset_y)*tile_total_input_width + (j+core_img_write_start_offset_x)] = group_initial_featuremap[((i+core_img_read_start_offset_y)*tile_input_width_original) + (j+core_img_read_start_offset_x)];
            }
        }


        //Top
        if(top_boundry_edges > 0){
            //receive top edges
            float* boundry_top;
            get_forward_group_boundry_data(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_top, 
                top_boundry_edges, tile_input_width_original, BOTTOM, 
                DEVICE_ID_X, DEVICE_ID_Y-1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges) : 0;
            for (int i = 0; i < top_boundry_edges; ++i)
            {
                for (int j = 0; j < tile_input_width_original; ++j)
                {

                    net->input[(i*tile_total_input_width) + (j+left_write_offset)] = boundry_top[(i*tile_input_width_original) + j];
                }
            }
            free(boundry_top);
        } 



        //Left
        if(left_boundry_edges > 0){
            //receive left edges

            float* boundry_left;
            get_forward_group_boundry_data(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_left, 
                tile_input_height_original, left_boundry_edges, RIGHT, 
                DEVICE_ID_X-1, DEVICE_ID_Y,
                DEVICE_ID_X, DEVICE_ID_Y);

            int top_write_offset = (top_boundry_edges >= 0) ? top_boundry_edges : 0;
            for (int i = 0; i < tile_input_height_original; ++i)
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
            get_forward_group_boundry_data(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_bottom, 
                bottom_boundry_edges, tile_input_width_original, TOP, 
                DEVICE_ID_X, DEVICE_ID_Y+1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int bottom_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges + tile_input_height_original) : tile_input_height_original;
            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges) : 0;
            for (int i = 0; i < bottom_boundry_edges; ++i)
            {
                for (int j = 0; j < tile_input_width_original; ++j)
                {
                    net->input[(i+bottom_write_offset)*tile_total_input_width + (j+left_write_offset)] = boundry_bottom[(i*tile_input_width_original) + j];
                }
            }
            free(boundry_bottom);
        }   
        // //Right
        if(right_boundry_edges > 0){
            //receive right edges
            float* boundry_right;
            get_forward_group_boundry_data(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_right, 
                tile_input_height_original, right_boundry_edges, LEFT, 
                DEVICE_ID_X+1, DEVICE_ID_Y,
                DEVICE_ID_X, DEVICE_ID_Y);

            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges + tile_input_width_original) : tile_input_width_original;
            int top_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges) : 0;

            for (int i = 0; i < tile_input_height_original; ++i)
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
            get_forward_group_boundry_data(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
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
            get_forward_group_boundry_data(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_top_right, 
                top_boundry_edges, right_boundry_edges, BOTTOM_LEFT, 
                DEVICE_ID_X+1, DEVICE_ID_Y-1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges + tile_input_width_original) : tile_input_width_original;
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
            get_forward_group_boundry_data(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_bottom_left, 
                bottom_boundry_edges, left_boundry_edges, TOP_RIGHT, 
                DEVICE_ID_X-1, DEVICE_ID_Y+1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int top_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges + tile_input_height_original) : tile_input_height_original;
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
            get_forward_group_boundry_data(SHARED_NETWORKS, SHARED_INPUT_IMAGES,
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_bottom_right, 
                bottom_boundry_edges, right_boundry_edges, TOP_LEFT, 
                DEVICE_ID_X+1, DEVICE_ID_Y+1,
                DEVICE_ID_X, DEVICE_ID_Y);

            int top_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges + tile_input_height_original) : tile_input_height_original;
            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges + tile_input_width_original) : tile_input_width_original;
            for (int i = 0; i < bottom_boundry_edges; ++i)
            {
                for (int j = 0; j < right_boundry_edges; ++j)
                {
                    net->input[(i+top_write_offset)*tile_total_input_width + (j+left_write_offset)] = boundry_bottom_right[(i*right_boundry_edges) + j];
                }
            }
            free(boundry_bottom_right);
        }

        for (int i = 0; i < start_layer.featuremap_in_h_with_boundry; ++i)
        {
            for (int j = 0; j < start_layer.featuremap_in_w_with_boundry; ++j)
            {
                printf("%.2f ", net->input[(i*start_layer.featuremap_in_w_with_boundry) + j]);
            }
            printf("\n");
        }

        printf("\n");

        if(group.layer_start_idx > 0){
            for (int i = 0; i < start_layer.featuremap_in_h_with_boundry; ++i)
            {
                for (int j = 0; j < start_layer.featuremap_in_w_with_boundry; ++j)
                {
                    net->layers[group.layer_start_idx - 1].output[(i*start_layer.featuremap_in_w_with_boundry) + j] = net->input[(i*start_layer.featuremap_in_w_with_boundry) + j];
                }
            }           
        }


}


void assemble_backward_group_data(network*** SHARED_NETWORKS, 
                                float***SHARED_EXP_DELTAS,
                                int NUM_TILES_X, int NUM_TILES_Y,
                                 group_profile_backward group,
                                 int DEVICE_ID_X, int DEVICE_ID_Y,
                                 int num_layers
                                 ){
        network* net = SHARED_NETWORKS[DEVICE_ID_Y][DEVICE_ID_X];

        layer end_layer = net->layers[group.layer_end_idx];

        int top_boundry_edges = end_layer.top_boundry_edges_delta;
        int bottom_boundry_edges = end_layer.bottom_boundry_edges_delta;
        int right_boundry_edges = end_layer.right_boundry_edges_delta;
        int left_boundry_edges = end_layer.left_boundry_edges_delta;

        //int delta_input_height_original = end_layer.original_featuremap_in_h;
        //int delta_input_width_original = end_layer.original_featuremap_in_w;
        int tile_delta_in_height = end_layer.delta_in_h_without_boundry;
        int tile_delta_in_width = end_layer.delta_in_w_without_boundry;
        int tile_total_delta_in_height = end_layer.delta_in_h_with_boundry;
        int tile_total_delta_in_width = end_layer.delta_in_w_with_boundry;

        int current_layer_idx = group.layer_end_idx;

        //Core tile delta
        for (int i = 0; i < (tile_delta_in_height); ++i)
        {
            for (int j = 0; j < (tile_delta_in_width); ++j)
            {
                net->layers[current_layer_idx].delta_with_boundry[(i+top_boundry_edges)*tile_total_delta_in_width + j+left_boundry_edges] = net->layers[current_layer_idx].delta_without_boundry[(i)*tile_delta_in_width + (j)];
            }
        }


        //Top
        if(top_boundry_edges > 0){
            //receive top edges
            float* boundry_top;
            get_backward_group_boundry_data(SHARED_NETWORKS, SHARED_EXP_DELTAS,
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
            get_backward_group_boundry_data(SHARED_NETWORKS, SHARED_EXP_DELTAS,
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
            get_backward_group_boundry_data(SHARED_NETWORKS, SHARED_EXP_DELTAS,
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
            get_backward_group_boundry_data(SHARED_NETWORKS, SHARED_EXP_DELTAS,
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
            get_backward_group_boundry_data(SHARED_NETWORKS, SHARED_EXP_DELTAS,
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
            get_backward_group_boundry_data(SHARED_NETWORKS, SHARED_EXP_DELTAS,
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
            get_backward_group_boundry_data(SHARED_NETWORKS, SHARED_EXP_DELTAS,
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
            get_backward_group_boundry_data(SHARED_NETWORKS, SHARED_EXP_DELTAS,
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

        for (int i = 0; i < end_layer.delta_in_h_with_boundry; ++i)
        {
            for (int j = 0; j < end_layer.delta_in_w_with_boundry; ++j)
            {
                printf("%.2f ", net->layers[current_layer_idx].delta_with_boundry[(i*end_layer.delta_in_w_with_boundry) + j]);
            }
            printf("\n");
        }

        // printf("\n");

}



void zero_out_edges_featuremap(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int DEVICE_ID_Y, int DEVICE_ID_X){
    if(layer_idx > 0){

        if(DEVICE_ID_Y == 0){
            for (int m = 0; m < net->layers[layer_idx].top_boundry_edges_featuremap; ++m)
            {
                for (int n = 0; n < net->layers[layer_idx].featuremap_in_w_with_boundry; ++n)
                {
                    net->layers[layer_idx-1].output[m*net->layers[layer_idx].featuremap_in_w_with_boundry + n] = 0.0;
                }
            }
        }

        if(DEVICE_ID_X == 0){
            for (int m = 0; m < net->layers[layer_idx].featuremap_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[layer_idx].left_boundry_edges_featuremap; ++n)
                {
                    net->layers[layer_idx-1].output[(m*net->layers[layer_idx].featuremap_in_w_with_boundry) + n] = 0.0;
                }
            }
        }

        if(DEVICE_ID_Y == (NUM_TILES_Y - 1)){
            for (int m = 0; m < net->layers[layer_idx].bottom_boundry_edges_featuremap; ++m)
            {
                for (int n = 0; n < net->layers[layer_idx].featuremap_in_w_with_boundry; ++n)
                {
                    net->layers[layer_idx-1].output[(m+net->layers[layer_idx].featuremap_in_h_without_boundry+net->layers[layer_idx].top_boundry_edges_featuremap)*net->layers[layer_idx].featuremap_in_w_with_boundry + n] = 0.0;
                }
            }
        }

        if(DEVICE_ID_X == (NUM_TILES_X - 1)){
            for (int m = 0; m < net->layers[layer_idx].featuremap_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[layer_idx].right_boundry_edges_featuremap; ++n)
                {
                    net->layers[layer_idx-1].output[(m)*net->layers[layer_idx].featuremap_in_w_with_boundry + n + net->layers[layer_idx].featuremap_in_w_without_boundry+net->layers[layer_idx].left_boundry_edges_featuremap] = 0.0;
                }
            }
        }

    }    
}



void zero_out_edges_delta(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int DEVICE_ID_Y, int DEVICE_ID_X){
    if(layer_idx > 0){

        if(DEVICE_ID_Y == 0){
            for (int m = 0; m < net->layers[layer_idx].top_boundry_edges_delta; ++m)
            {
                for (int n = 0; n < net->layers[layer_idx].delta_in_w_with_boundry; ++n)
                {
                    net->layers[layer_idx].delta_with_boundry[m*net->layers[layer_idx].delta_in_w_with_boundry + n] = 0.0;
                }
            }
        }

        if(DEVICE_ID_X == 0){
            for (int m = 0; m < net->layers[layer_idx].delta_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[layer_idx].left_boundry_edges_delta; ++n)
                {
                    net->layers[layer_idx].delta_with_boundry[(m*net->layers[layer_idx].delta_in_w_with_boundry) + n] = 0.0;
                }
            }
        }

        if(DEVICE_ID_Y == (NUM_TILES_Y - 1)){
            for (int m = 0; m < net->layers[layer_idx].bottom_boundry_edges_delta; ++m)
            {
                for (int n = 0; n < net->layers[layer_idx].delta_in_w_with_boundry; ++n)
                {
                    net->layers[layer_idx].delta_with_boundry[(m+net->layers[layer_idx].delta_in_h_without_boundry+net->layers[layer_idx].top_boundry_edges_delta)*net->layers[layer_idx].delta_in_w_with_boundry + n] = 0.0;
                }
            }
        }

        if(DEVICE_ID_X == (NUM_TILES_X - 1)){
            for (int m = 0; m < net->layers[layer_idx].delta_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[layer_idx].right_boundry_edges_delta; ++n)
                {
                    net->layers[layer_idx].delta_with_boundry[(m)*net->layers[layer_idx].delta_in_w_with_boundry + n + net->layers[layer_idx].delta_in_w_without_boundry+net->layers[layer_idx].left_boundry_edges_delta] = 0.0;
                }
            }
        }

    }    
}


void execute_backward(){}
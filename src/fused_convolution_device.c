#include "fused_convolution_device.h"
#include "fused_device.h"
#include "transport.h"


void get_forward_group_boundry_data_device(
    int NUM_TILES_X, int NUM_TILES_Y,
    int current_layer_idx, 
    float** device_data, 
    int rows, int cols, int depth, orientation region,
    int device_src_id_x, int device_src_id_y, 
    int device_dst_id_x, int device_dst_id_y){

    float* boundry_data = calloc(rows*cols*depth, sizeof(float));
    *device_data = boundry_data;
    //TODO: ASSERT CHECK DIM OVER/UNDERFLOW

    int x_dim;
    int y_dim; 

    float* boundry_src_data;

    if((device_src_id_x >= NUM_TILES_X) && 
        (region == BOTTOM_LEFT || region == LEFT || region == TOP_LEFT)){
        fill_cpu(depth*rows*cols, 0, *device_data, 1);
        return;
    }
    if((device_src_id_x < 0) && 
        (region == TOP_RIGHT || region == RIGHT || region == BOTTOM_RIGHT)){
        fill_cpu(depth*rows*cols, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y >= NUM_TILES_Y) && 
        (region == TOP_LEFT || region == TOP || region == TOP_RIGHT)){
        fill_cpu(depth*rows*cols, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y < 0) && 
        (region == BOTTOM_LEFT || region == BOTTOM || region == BOTTOM_RIGHT)){
        fill_cpu(depth*rows*cols, 0, *device_data, 1);
        return;
    }

    receive_boundry(boundry_data, depth*rows*cols, device_src_id_x, device_src_id_y);

}

// void send_backward_group_boundry_data_device(network* net, float* OUTPUT_DELTA,
//     int NUM_TILES_X, int NUM_TILES_Y,
//     int current_layer_idx, int num_layers,
//     int device_id_x, int device_id_y){


















// }


void get_backward_group_boundry_data_device(
    int NUM_TILES_X, int NUM_TILES_Y,
    int current_layer_idx, int num_layers,
    float** device_data, 
    int rows, int cols, int depth, orientation region,
    int device_src_id_x, int device_src_id_y, 
    int device_dst_id_x, int device_dst_id_y) {

    float* boundry_data = calloc(rows*cols*depth, sizeof(float));
    *device_data = boundry_data;
    //TODO: ASSERT CHECK DIM OVER/UNDERFLOW

    int x_dim;
    int y_dim; 

    float* boundry_src_data;

    if((device_src_id_x >= NUM_TILES_X) && 
        (region == BOTTOM_LEFT || region == LEFT || region == TOP_LEFT)){
        fill_cpu(rows*cols*depth, 0, *device_data, 1);
        return;
    }
    if((device_src_id_x < 0) && 
        (region == TOP_RIGHT || region == RIGHT || region == BOTTOM_RIGHT)){
        fill_cpu(rows*cols*depth, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y >= NUM_TILES_Y) && 
        (region == TOP_LEFT || region == TOP || region == TOP_RIGHT)){
        fill_cpu(rows*cols*depth, 0, *device_data, 1);
        return;
    }
    if((device_src_id_y < 0) && 
        (region == BOTTOM_LEFT || region == BOTTOM || region == BOTTOM_RIGHT)){
        fill_cpu(rows*cols*depth, 0, *device_data, 1);
        return;
    }

    receive_boundry(boundry_data, rows*cols*depth, device_src_id_x, device_src_id_y);

}


void assemble_forward_group_data_device(network* net, 
                                float* INPUT_IMAGE,
                                int NUM_TILES_X, int NUM_TILES_Y,
								 group_profile_forward group,
                                 int device_id_x, int device_id_y
								 ){


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

        int depth = start_layer.c;


        // send_forward_group_boundry_data_device(net, INPUT_IMAGE,
        //     NUM_TILES_X, NUM_TILES_Y,
        //     current_layer_idx, 
        //     device_id_x, device_id_y);

        int featuremap_width = net->layers[current_layer_idx].featuremap_in_w_without_boundry;
        int featuremap_height = net->layers[current_layer_idx].featuremap_in_h_without_boundry;

        int x_dim = net->layers[current_layer_idx].original_featuremap_in_w;
        int y_dim = net->layers[current_layer_idx].original_featuremap_in_h;
        int z_dim = net->layers[current_layer_idx].c;

        float* transmit_data;
        int transmit_size;

        float* src_structure = (current_layer_idx == 0) ? INPUT_IMAGE : (net->layers[current_layer_idx - 1].output_without_boundry);


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
            group_initial_featuremap = INPUT_IMAGE;
        }
        else{
            group_initial_featuremap = net->layers[group.layer_start_idx - 1].output;
        }


        //Core tile image

        for (int c = 0; c < depth; ++c)
        {
            for (int i = 0; i < num_core_img_elements_y; ++i)
            {
                for (int j = 0; j < num_core_img_elements_x; ++j)
                {
                    net->input[(c*tile_total_input_width*tile_total_input_height) + (i+core_img_write_start_offset_y)*tile_total_input_width + (j+core_img_write_start_offset_x)] = group_initial_featuremap[(c*tile_input_width_original*tile_input_height_original) + ((i+core_img_read_start_offset_y)*tile_input_width_original) + (j+core_img_read_start_offset_x)];
                }
            }
        }

        // //Top left
        if((top_boundry_edges > 0) && (left_boundry_edges > 0)){

            float* boundry_top_left;
            get_forward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_top_left, 
                top_boundry_edges, left_boundry_edges, depth, BOTTOM_RIGHT, 
                device_id_x-1, device_id_y-1,
                device_id_x, device_id_y);

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < top_boundry_edges; ++i)
                {
                    for (int j = 0; j < left_boundry_edges; ++j)
                    {
                        net->input[(c*tile_total_input_width*tile_total_input_height) + (i*tile_total_input_width) + j] = boundry_top_left[(c*top_boundry_edges*left_boundry_edges) + (i*left_boundry_edges) + j];
                    }
                }
            }
           // free(boundry_top_left);

            //SEND TOP LEFT
            if((device_id_y > 0) && (device_id_x > 0)){
                int rows = top_boundry_edges;
                int cols = left_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[c*(rows*cols) + i*cols + j] = src_structure[(c*x_dim*y_dim) + (i)*x_dim + j];
                        }
                    }       
                }
         
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x-1, device_id_y-1);
                free(transmit_data);

            }

        }


        //Top
        if(top_boundry_edges > 0){

            //receive top edges
            float* boundry_top;
            get_forward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_top, 
                top_boundry_edges, tile_input_width_original, depth, BOTTOM, 
                device_id_x, device_id_y-1,
                device_id_x, device_id_y);

            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges) : 0;

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < top_boundry_edges; ++i)
                {
                    for (int j = 0; j < tile_input_width_original; ++j)
                    {
                        net->input[(c*tile_total_input_width*tile_total_input_height) + (i*tile_total_input_width) + (j+left_write_offset)] = boundry_top[(c*top_boundry_edges*tile_input_width_original) + (i*tile_input_width_original) + j];
                    }
                }
            }
           // free(boundry_top);

            //SEND TOP
            if(device_id_y > 0){
                int rows = top_boundry_edges;
                int cols = featuremap_width;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[c*(rows*cols) + i*cols + j] = src_structure[(c*x_dim*y_dim) + (i)*x_dim + j];
                        }
                    } 
                }
               
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x, device_id_y-1);
                free(transmit_data);
            }
        } 

        // //Top right
        if((top_boundry_edges > 0) && (right_boundry_edges > 0)){
            float* boundry_top_right;
            get_forward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_top_right, 
                top_boundry_edges, right_boundry_edges, depth, BOTTOM_LEFT, 
                device_id_x+1, device_id_y-1,
                device_id_x, device_id_y);

            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges + tile_input_width_original) : tile_input_width_original;
            
            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < top_boundry_edges; ++i)
                {
                    for (int j = 0; j < right_boundry_edges; ++j)
                    {
                        net->input[(c*tile_total_input_width*tile_total_input_height) + (i*tile_total_input_width) + (j+left_write_offset)] = boundry_top_right[(c*top_boundry_edges*right_boundry_edges) + (i*right_boundry_edges) + j];
                    }
                }
            }
           // free(boundry_top_right);

            //SEND TOP RIGHT
            if((device_id_y > 0) && (device_id_x < (NUM_TILES_X-1))){
                int rows = top_boundry_edges;
                int cols = right_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[c*(rows*cols) + i*cols + j] = src_structure[(c*x_dim*y_dim) + (i)*x_dim + j+x_dim-cols];
                        }
                    }  
                }
              
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x+1, device_id_y-1);
                free(transmit_data);
            }
        }

        //Left
        if(left_boundry_edges > 0){
            //receive left edges

            float* boundry_left;
            get_forward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_left, 
                tile_input_height_original, left_boundry_edges, depth, RIGHT, 
                device_id_x-1, device_id_y,
                device_id_x, device_id_y);

            int top_write_offset = (top_boundry_edges >= 0) ? top_boundry_edges : 0;

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < tile_input_height_original; ++i)
                {
                    for (int j = 0; j < left_boundry_edges; ++j)
                    {
                        net->input[(c*tile_total_input_width*tile_total_input_height) + (i+top_write_offset)*tile_total_input_width + j] = boundry_left[(c*left_boundry_edges*tile_input_height_original) + (i*left_boundry_edges) + j];
                    }
                }
            }
           // free(boundry_left);

            //SEND LEFT
            if(device_id_x > 0){

                int rows = featuremap_height;
                int cols = left_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[c*(rows*cols) + i*cols + j] = src_structure[(c*x_dim*y_dim) + (i)*x_dim + j];
                        }
                    }        
                }

                send_boundry(transmit_data, z_dim*rows*cols, device_id_x-1, device_id_y);
                free(transmit_data);

            }
        } 

        // //Right
        if(right_boundry_edges > 0){
            //receive right edges


            //SEND RIGHT
            if(device_id_x < (NUM_TILES_X-1)){

                int rows = featuremap_height;
                int cols = right_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[c*(rows*cols) + i*cols + j] = src_structure[(c*x_dim*y_dim) + (i)*x_dim + j+x_dim-cols];
                        }
                    }  
                }
              
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x+1, device_id_y);
                free(transmit_data);

            }

            float* boundry_right;
            get_forward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_right, 
                tile_input_height_original, right_boundry_edges, depth, LEFT, 
                device_id_x+1, device_id_y,
                device_id_x, device_id_y);

            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges + tile_input_width_original) : tile_input_width_original;
            int top_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges) : 0;

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < tile_input_height_original; ++i)
                {
                    for (int j = 0; j < right_boundry_edges; ++j)
                    {
                        net->input[(c*tile_total_input_width*tile_total_input_height) + (i+top_write_offset)*tile_total_input_width + (j+left_write_offset)] = boundry_right[(c*right_boundry_edges*tile_input_height_original) + (i*right_boundry_edges) + j];
                    }
                }
            }
          // free(boundry_right);
        } 

        // //Bottom left
        if((bottom_boundry_edges > 0) && (left_boundry_edges > 0)){

            //SEND BOTTOM LEFT
            if((device_id_y < (NUM_TILES_Y-1)) && (device_id_x > 0)){
                int rows = bottom_boundry_edges;
                int cols = left_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[c*(rows*cols) + i*cols + j] = src_structure[(c*x_dim*y_dim) + (i+y_dim-rows)*x_dim + j];
                        }
                    }  
                }
              
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x-1, device_id_y+1);
                free(transmit_data);
            }

            float* boundry_bottom_left;
            get_forward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_bottom_left, 
                bottom_boundry_edges, left_boundry_edges, depth, TOP_RIGHT, 
                device_id_x-1, device_id_y+1,
                device_id_x, device_id_y);

            int top_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges + tile_input_height_original) : tile_input_height_original;
            
            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < bottom_boundry_edges; ++i)
                {
                    for (int j = 0; j < left_boundry_edges; ++j)
                    {
                        net->input[(c*tile_total_input_width*tile_total_input_height) + (i+top_write_offset)*(tile_total_input_width) + j] = boundry_bottom_left[(c*bottom_boundry_edges*left_boundry_edges) + (i*left_boundry_edges) + j];
                    }
                }
            }
           // free(boundry_bottom_left);
        }


        // //Bottom 
        if(bottom_boundry_edges > 0){

            //SEND BOTTOM
            if(device_id_y < (NUM_TILES_Y-1)){

                int rows = bottom_boundry_edges;
                int cols = featuremap_width;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[c*(rows*cols) + i*cols + j] = src_structure[(c*x_dim*y_dim) + (i+y_dim-rows)*x_dim + j];
                        }
                    }
                }        
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x, device_id_y+1);
                free(transmit_data);

            }

            //receive bottom edges
            float* boundry_bottom;
            get_forward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_bottom, 
                bottom_boundry_edges, tile_input_width_original, depth, TOP, 
                device_id_x, device_id_y+1,
                device_id_x, device_id_y);

            int bottom_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges + tile_input_height_original) : tile_input_height_original;
            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges) : 0;

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < bottom_boundry_edges; ++i)
                {
                    for (int j = 0; j < tile_input_width_original; ++j)
                    {
                        net->input[(c*tile_total_input_width*tile_total_input_height) + (i+bottom_write_offset)*tile_total_input_width + (j+left_write_offset)] = boundry_bottom[(c*bottom_boundry_edges*tile_input_width_original) + (i*tile_input_width_original) + j];
                    }
                }
            }
           // free(boundry_bottom);
        }   










        // //Bottom right
        if((bottom_boundry_edges > 0) && (right_boundry_edges > 0)){

            //SEND BOTTOM RIGHT
            if((device_id_y < (NUM_TILES_Y-1)) && (device_id_x < (NUM_TILES_X-1))){
                int rows = bottom_boundry_edges;
                int cols = right_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[c*cols*rows + i*cols + j] = src_structure[(c*x_dim*y_dim) + (i+y_dim-rows)*x_dim + j+x_dim-cols];
                        }
                    }    
                }
            
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x+1, device_id_y+1);
                free(transmit_data);
            }

            float* boundry_bottom_right;
            get_forward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, 
                &boundry_bottom_right, 
                bottom_boundry_edges, right_boundry_edges, depth, TOP_LEFT, 
                device_id_x+1, device_id_y+1,
                device_id_x, device_id_y);

            int top_write_offset = (top_boundry_edges >= 0) ? (top_boundry_edges + tile_input_height_original) : tile_input_height_original;
            int left_write_offset = (left_boundry_edges >= 0) ? (left_boundry_edges + tile_input_width_original) : tile_input_width_original;
            
            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < bottom_boundry_edges; ++i)
                {
                    for (int j = 0; j < right_boundry_edges; ++j)
                    {
                        net->input[(c*tile_total_input_width*tile_total_input_height) + (i+top_write_offset)*tile_total_input_width + (j+left_write_offset)] = boundry_bottom_right[(c*bottom_boundry_edges*right_boundry_edges) + (i*right_boundry_edges) + j];
                    }
                }
            }
           // free(boundry_bottom_right);
        }











        // for (int i = 0; i < start_layer.featuremap_in_h_with_boundry; ++i)
        // {
        //     for (int j = 0; j < start_layer.featuremap_in_w_with_boundry; ++j)
        //     {
        //         printf("%.2f ", net->input[(i*start_layer.featuremap_in_w_with_boundry) + j]);
        //     }
        //     printf("\n");
        // }

        // printf("\n");

        if(group.layer_start_idx > 0){

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < start_layer.featuremap_in_h_with_boundry; ++i)
                {
                    for (int j = 0; j < start_layer.featuremap_in_w_with_boundry; ++j)
                    {
                        net->layers[group.layer_start_idx - 1].output[(c*tile_total_input_width*tile_total_input_height) + (i*start_layer.featuremap_in_w_with_boundry) + j] = 
                            net->input[(c*tile_total_input_width*tile_total_input_height) + (i*start_layer.featuremap_in_w_with_boundry) + j];
                    }
                }  
            }         
        }


}


void assemble_backward_group_data_device(network* net, 
                                float* OUTPUT_DELTA,
                                int NUM_TILES_X, int NUM_TILES_Y,
                                 group_profile_backward group,
                                 int device_id_x, int device_id_y,
                                 int num_layers
                                 ){

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

        int depth = end_layer.c;

        int delta_width = net->layers[current_layer_idx].delta_in_w_without_boundry;
        int delta_height = net->layers[current_layer_idx].delta_in_h_without_boundry;

        int x_dim = net->layers[current_layer_idx].delta_in_w_without_boundry;
        int y_dim = net->layers[current_layer_idx].delta_in_h_without_boundry;
        int z_dim = net->layers[current_layer_idx].c;

        float* transmit_data;
        int transmit_size;

        float* src_structure = (current_layer_idx == (num_layers-1)) ? OUTPUT_DELTA : (net->layers[current_layer_idx].delta_without_boundry);


        // send_backward_group_boundry_data_device(net, OUTPUT_DELTA,
        //     NUM_TILES_X, NUM_TILES_Y,
        //     current_layer_idx, num_layers,
        //     device_id_x, device_id_y);

        //Core tile delta

        for (int c = 0; c < depth; ++c)
        {
            for (int i = 0; i < (tile_delta_in_height); ++i)
            {
                for (int j = 0; j < (tile_delta_in_width); ++j)
                {
                    net->layers[current_layer_idx].delta_with_boundry[(c*tile_total_delta_in_width*tile_total_delta_in_height) + (i+top_boundry_edges)*tile_total_delta_in_width + j+left_boundry_edges] = net->layers[current_layer_idx].delta_without_boundry[(c*tile_delta_in_width*tile_delta_in_height) + (i)*tile_delta_in_width + (j)];
                }
            }
        }


        // //Top left
        if((top_boundry_edges > 0) && (left_boundry_edges > 0)){

            float* boundry_top_left;
            get_backward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, num_layers,
                &boundry_top_left, 
                top_boundry_edges, left_boundry_edges, depth, BOTTOM_RIGHT, 
                device_id_x-1, device_id_y-1,
                device_id_x, device_id_y);

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < top_boundry_edges; ++i)
                {
                    for (int j = 0; j < left_boundry_edges; ++j)
                    {
                        net->layers[current_layer_idx].delta_with_boundry[(c*tile_total_delta_in_width*tile_total_delta_in_height) + (i*tile_total_delta_in_width) + j] = boundry_top_left[(c*left_boundry_edges*top_boundry_edges) + (i*left_boundry_edges) + j];
                    }
                }
            }
           // free(boundry_top_left);

            //SEND TOP LEFT
            if((device_id_y > 0) && (device_id_x > 0)){
                int rows = top_boundry_edges;
                int cols = left_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[(c*rows*cols) + i*cols + j] = src_structure[(c*rows*cols) + (i)*x_dim + j];
                        }
                    }    
                }
            
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x-1, device_id_y-1);
                //free(transmit_data);

            }
        }

        //Top
        if(top_boundry_edges > 0){
            //receive top edges
            float* boundry_top;
            get_backward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, num_layers,
                &boundry_top, 
                top_boundry_edges, tile_delta_in_width, depth, BOTTOM, 
                device_id_x, device_id_y-1,
                device_id_x, device_id_y);

            int left_write_offset = left_boundry_edges;

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < top_boundry_edges; ++i)
                {
                    for (int j = 0; j < tile_delta_in_width; ++j)
                    {
                        net->layers[current_layer_idx].delta_with_boundry[(c*tile_total_delta_in_width*tile_total_delta_in_height) + (i*tile_total_delta_in_width) + (j+left_boundry_edges)] = boundry_top[(c*top_boundry_edges*tile_delta_in_width) + (i*tile_delta_in_width) + j];
                    }
                }
            }
               // free(boundry_top);



            //SEND TOP
            if(device_id_y > 0){
                int rows = top_boundry_edges;
                int cols = delta_width;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[(c*rows*cols) + i*cols + j] = src_structure[(c*x_dim*y_dim) + (i)*x_dim + j];
                        }
                    } 
                }
               
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x, device_id_y-1);
               // free(transmit_data);
            }
        } 

        // //Top right
        if((top_boundry_edges > 0) && (right_boundry_edges > 0)){
            float* boundry_top_right;
            get_backward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, num_layers,
                &boundry_top_right, 
                top_boundry_edges, right_boundry_edges, depth, BOTTOM_LEFT, 
                device_id_x+1, device_id_y-1,
                device_id_x, device_id_y);

            int left_write_offset = (left_boundry_edges + tile_delta_in_width);

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < top_boundry_edges; ++i)
                {
                    for (int j = 0; j < right_boundry_edges; ++j)
                    {
                        net->layers[current_layer_idx].delta_with_boundry[(c*tile_total_delta_in_width*tile_total_delta_in_height) + (i*tile_total_delta_in_width) + (j+left_write_offset)] = boundry_top_right[(c*top_boundry_edges*right_boundry_edges) + (i*right_boundry_edges) + j];
                    }
                }
            }
           // free(boundry_top_right);

            //SEND TOP RIGHT
            if((device_id_y > 0) && (device_id_x < (NUM_TILES_X-1))){
                int rows = top_boundry_edges;
                int cols = right_boundry_edges;

                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[(c*rows*cols) + i*cols + j] = src_structure[(c*rows*cols) + (i)*x_dim + j+x_dim-cols];
                        }
                    }     
                }
           
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x+1, device_id_y-1);
                //free(transmit_data);
            }
        }

        //Left
        if(left_boundry_edges > 0){
            //receive left edges

            float* boundry_left;
            get_backward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, num_layers,
                &boundry_left, 
                tile_delta_in_height, left_boundry_edges, depth, RIGHT, 
                device_id_x-1, device_id_y,
                device_id_x, device_id_y);

            int top_write_offset = top_boundry_edges;

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < tile_delta_in_height; ++i)
                {
                    for (int j = 0; j < left_boundry_edges; ++j)
                    {
                        net->layers[current_layer_idx].delta_with_boundry[(c*tile_total_delta_in_width*tile_total_delta_in_height) + (i+top_write_offset)*tile_total_delta_in_width + j] = boundry_left[(c*left_boundry_edges*tile_delta_in_height) + (i*left_boundry_edges) + j];
                    }
                }
            }
            //free(boundry_left);


            //SEND LEFT
            if(device_id_x > 0){

                int rows = delta_height;
                int cols = left_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));
                //printf("Transmitting left\n");

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            //printf("%.2f ", transmit_data[i*cols + j]);
                            transmit_data[(c*rows*cols) + i*cols + j] = src_structure[(c*rows*cols) + (i)*x_dim + j];
                        }
                       // printf("\n");
                    }        
                }
                //printf("\n");
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x-1, device_id_y);
                //free(transmit_data);

            }
        } 



        // //Right
        if(right_boundry_edges > 0){

            //SEND RIGHT
            if(device_id_x < (NUM_TILES_X-1)){

                int rows = delta_height;
                int cols = right_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));
               // printf("Transmitting right rows = %d cols = %d \n", rows, cols);

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {

                            transmit_data[(c*rows*cols) + i*cols + j] = src_structure[(c*rows*cols) + (i)*x_dim + j+x_dim-cols];
                            //printf("%.2f ", transmit_data[i*cols + j]);
                        }
                       // printf("\n");
                    }  
                }

                //printf("\n");      
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x+1, device_id_y);
                //free(transmit_data);

            }
            //receive right edges
            float* boundry_right;
            get_backward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, num_layers,
                &boundry_right, 
                tile_delta_in_height, right_boundry_edges, depth, LEFT, 
                device_id_x+1, device_id_y,
                device_id_x, device_id_y);

            int left_write_offset = left_boundry_edges + tile_delta_in_width;
            int top_write_offset = top_boundry_edges;

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < tile_delta_in_height; ++i)
                {
                    for (int j = 0; j < right_boundry_edges; ++j)
                    {
                        net->layers[current_layer_idx].delta_with_boundry[(c*tile_total_delta_in_width*tile_total_delta_in_height) + (i+top_write_offset)*tile_total_delta_in_width + (j+left_write_offset)] = boundry_right[(c*right_boundry_edges*tile_delta_in_height) + (i*right_boundry_edges) + j];
                    }
                }
            }
           // free(boundry_right);
        } 

        // //Bottom left
        if((bottom_boundry_edges > 0) && (left_boundry_edges > 0)){

            //SEND BOTTOM LEFT
            if((device_id_y < (NUM_TILES_Y-1)) && (device_id_x > 0)){
                int rows = bottom_boundry_edges;
                int cols = left_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[(c*rows*cols) + i*cols + j] = src_structure[(c*rows*cols) + (i+y_dim-rows)*x_dim + j];
                        }
                    }       
                } 
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x-1, device_id_y+1);
                //free(transmit_data);
            }
            
            float* boundry_bottom_left;
            get_backward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, num_layers,
                &boundry_bottom_left, 
                bottom_boundry_edges, left_boundry_edges, depth, TOP_RIGHT, 
                device_id_x-1, device_id_y+1,
                device_id_x, device_id_y);

            int top_write_offset = top_boundry_edges + tile_delta_in_height;

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < bottom_boundry_edges; ++i)
                {
                    for (int j = 0; j < left_boundry_edges; ++j)
                    {
                        net->layers[current_layer_idx].delta_with_boundry[(c*tile_total_delta_in_width*tile_total_delta_in_height) + (i+top_write_offset)*(tile_total_delta_in_width) + j] = boundry_bottom_left[(c*bottom_boundry_edges*left_boundry_edges) + (i*left_boundry_edges) + j];
                    }
                }
            }
            //free(boundry_bottom_left);
        }

        // //Bottom 
        if(bottom_boundry_edges > 0){

            //SEND BOTTOM
            if(device_id_y < (NUM_TILES_Y-1)){

                int rows = bottom_boundry_edges;
                int cols = delta_width;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[(c*rows*cols) + i*cols + j] = src_structure[(c*rows*cols) + (i+y_dim-rows)*x_dim + j];
                        }
                    } 
                }
               
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x, device_id_y+1);
                //free(transmit_data);

            }

            //receive bottom edges
            float* boundry_bottom;
            get_backward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, num_layers,
                &boundry_bottom, 
                bottom_boundry_edges, tile_delta_in_width, depth, TOP, 
                device_id_x, device_id_y+1,
                device_id_x, device_id_y);

            int bottom_write_offset = top_boundry_edges + tile_delta_in_height;
            int left_write_offset = left_boundry_edges;

            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < bottom_boundry_edges; ++i)
                {
                    for (int j = 0; j < tile_delta_in_width; ++j)
                    {
                        net->layers[current_layer_idx].delta_with_boundry[(c*tile_total_delta_in_width*tile_total_delta_in_height) + (i+bottom_write_offset)*tile_total_delta_in_width + (j+left_write_offset)] = boundry_bottom[(c*bottom_boundry_edges*tile_delta_in_width) + (i*tile_delta_in_width) + j];
                    }
                }
            }
           // free(boundry_bottom);
        }   

        // //Bottom right
        if((bottom_boundry_edges > 0) && (right_boundry_edges > 0)){


            //SEND BOTTOM RIGHT
            if((device_id_y < (NUM_TILES_Y-1)) && (device_id_x < (NUM_TILES_X-1))){
                int rows = bottom_boundry_edges;
                int cols = right_boundry_edges;
                transmit_data = calloc((z_dim*rows*cols), sizeof(float));

                for (int c = 0; c < z_dim; ++c)
                {
                    for (int i = 0; i < rows; ++i)
                    {
                        for (int j = 0; j < cols; ++j)
                        {
                            transmit_data[(c*rows*cols) + i*cols + j] = src_structure[(c*rows*cols) + (i+y_dim-rows)*x_dim + j+x_dim-cols];
                        }
                    }  
                }
              
                send_boundry(transmit_data, z_dim*rows*cols, device_id_x+1, device_id_y+1);
                //free(transmit_data);
            }


            float* boundry_bottom_right;
            get_backward_group_boundry_data_device(
                NUM_TILES_X, NUM_TILES_Y,
                current_layer_idx, num_layers,
                &boundry_bottom_right, 
                bottom_boundry_edges, right_boundry_edges, depth, TOP_LEFT, 
                device_id_x+1, device_id_y+1,
                device_id_x, device_id_y);

            int top_write_offset = top_boundry_edges + tile_delta_in_height;
            int left_write_offset = left_boundry_edges + tile_delta_in_width;


            for (int c = 0; c < depth; ++c)
            {
                for (int i = 0; i < bottom_boundry_edges; ++i)
                {
                    for (int j = 0; j < right_boundry_edges; ++j)
                    {
                        net->layers[current_layer_idx].delta_with_boundry[(c*tile_total_delta_in_width*tile_total_delta_in_height) + (i+top_write_offset)*tile_total_delta_in_width + (j+left_write_offset)] = boundry_bottom_right[(c*bottom_boundry_edges*right_boundry_edges) + (i*right_boundry_edges) + j];
                    }
                }
            }
           // free(boundry_bottom_right);
        }



        // for (int i = 0; i < end_layer.delta_in_h_with_boundry; ++i)
        // {
        //     for (int j = 0; j < end_layer.delta_in_w_with_boundry; ++j)
        //     {
        //         printf("%.2f ", net->layers[current_layer_idx].delta_with_boundry[(i*end_layer.delta_in_w_with_boundry) + j]);
        //     }
        //     printf("\n");
        // }

        // printf("\n");

}



void zero_out_edges_featuremap_device(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int device_id_y, int device_id_x){

    int x_dim = net->layers[layer_idx].featuremap_in_w_with_boundry;
    int y_dim = net->layers[layer_idx].featuremap_in_h_with_boundry;
    int depth = net->layers[layer_idx].c;

    if(layer_idx > 0){

        if(device_id_y == 0){
            int rows = net->layers[layer_idx].top_boundry_edges_featuremap;
            int cols = net->layers[layer_idx].featuremap_in_w_with_boundry;

            for (int c = 0; c < depth; ++c)
            {
                for (int m = 0; m < rows; ++m)
                {
                    for (int n = 0; n < cols; ++n)
                    {
                        net->layers[layer_idx-1].output[(c*x_dim*y_dim) + m*cols + n] = 0.0;
                    }
                }
            }
        }

        if(device_id_x == 0){


            for (int c = 0; c < depth; ++c)
            {
                for (int m = 0; m < net->layers[layer_idx].featuremap_in_h_with_boundry; ++m)
                {
                    for (int n = 0; n < net->layers[layer_idx].left_boundry_edges_featuremap; ++n)
                    {
                        net->layers[layer_idx-1].output[(c*x_dim*y_dim) + (m*net->layers[layer_idx].featuremap_in_w_with_boundry) + n] = 0.0;
                    }
                }
            }
        }

        if(device_id_y == (NUM_TILES_Y - 1)){

            for (int c = 0; c < depth; ++c)
            {
                for (int m = 0; m < net->layers[layer_idx].bottom_boundry_edges_featuremap; ++m)
                {
                    for (int n = 0; n < net->layers[layer_idx].featuremap_in_w_with_boundry; ++n)
                    {
                        net->layers[layer_idx-1].output[(c*x_dim*y_dim) + (m+net->layers[layer_idx].featuremap_in_h_without_boundry+net->layers[layer_idx].top_boundry_edges_featuremap)*net->layers[layer_idx].featuremap_in_w_with_boundry + n] = 0.0;
                    }
                }
            }
        }

        if(device_id_x == (NUM_TILES_X - 1)){

            for (int c = 0; c < depth; ++c)
            {
                for (int m = 0; m < net->layers[layer_idx].featuremap_in_h_with_boundry; ++m)
                {
                    for (int n = 0; n < net->layers[layer_idx].right_boundry_edges_featuremap; ++n)
                    {
                        net->layers[layer_idx-1].output[(c*x_dim*y_dim) + (m)*net->layers[layer_idx].featuremap_in_w_with_boundry + n + net->layers[layer_idx].featuremap_in_w_without_boundry+net->layers[layer_idx].left_boundry_edges_featuremap] = 0.0;
                    }
                }
            }
        }

    }    
}



void zero_out_edges_delta_device(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int device_id_y, int device_id_x){


    int x_dim = net->layers[layer_idx].delta_in_w_with_boundry;
    int y_dim = net->layers[layer_idx].delta_in_h_with_boundry;
    int depth = net->layers[layer_idx].c;

    if(layer_idx > 0){

        if(device_id_y == 0){

            for (int c = 0; c < depth; ++c)
            {
                for (int m = 0; m < net->layers[layer_idx].top_boundry_edges_delta; ++m)
                {
                    for (int n = 0; n < net->layers[layer_idx].delta_in_w_with_boundry; ++n)
                    {
                        net->layers[layer_idx].delta_with_boundry[(c*x_dim*y_dim) + m*net->layers[layer_idx].delta_in_w_with_boundry + n] = 0.0;
                    }
                }
            }
        }

        if(device_id_x == 0){

            for (int c = 0; c < depth; ++c)
            {
                for (int m = 0; m < net->layers[layer_idx].delta_in_h_with_boundry; ++m)
                {
                    for (int n = 0; n < net->layers[layer_idx].left_boundry_edges_delta; ++n)
                    {
                        net->layers[layer_idx].delta_with_boundry[(c*x_dim*y_dim) + (m*net->layers[layer_idx].delta_in_w_with_boundry) + n] = 0.0;
                    }
                }
            }
        }

        if(device_id_y == (NUM_TILES_Y - 1)){

            for (int c = 0; c < depth; ++c)
            {
                for (int m = 0; m < net->layers[layer_idx].bottom_boundry_edges_delta; ++m)
                {
                    for (int n = 0; n < net->layers[layer_idx].delta_in_w_with_boundry; ++n)
                    {
                        net->layers[layer_idx].delta_with_boundry[(c*x_dim*y_dim) + (m+net->layers[layer_idx].delta_in_h_without_boundry+net->layers[layer_idx].top_boundry_edges_delta)*net->layers[layer_idx].delta_in_w_with_boundry + n] = 0.0;
                    }
                }
            }
        }

        if(device_id_x == (NUM_TILES_X - 1)){

            for (int c = 0; c < depth; ++c)
            {
                for (int m = 0; m < net->layers[layer_idx].delta_in_h_with_boundry; ++m)
                {
                    for (int n = 0; n < net->layers[layer_idx].right_boundry_edges_delta; ++n)
                    {
                        net->layers[layer_idx].delta_with_boundry[(c*x_dim*y_dim) + (m)*net->layers[layer_idx].delta_in_w_with_boundry + n + net->layers[layer_idx].delta_in_w_without_boundry+net->layers[layer_idx].left_boundry_edges_delta] = 0.0;
                    }
                }
            }
        }

    }    
}

//TODO : Variable filter sizes

void receive_sum_broadcast_weight_updates(network* net, int NUM_TILES_Y, int NUM_TILES_X){

    int total_weights = 0;

    for (int l = 0; l < net->n; ++l){
        int num_filters = net->layers[l].n;
        int filter_size = net->layers[l].size;
        int channels = net->layers[l].c;
        total_weights += num_filters*channels*filter_size*filter_size;
    }    

    float* data = calloc(total_weights, sizeof(float));
    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        for (int j = 0; j < NUM_TILES_X; ++j){
            if((i != 0) || (j != 0)){
                receive_boundry(data, total_weights, j, i);
                //printf("Weights from device %d %d\n", j, i);
                int layer_cumulative_weights = 0;
                for (int l = 0; l < net->n; ++l){

                   // printf("Weights of layer %d\n", l);

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
                                    net->layers[l].weight_updates[(c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n] +=
                                     data[layer_cumulative_weights + ((c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n)];

                                     //printf("%.2f ", data[layer_cumulative_weights + ((c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n)]);
                                }
                                //printf("\n");
                            }

                            //printf("\n");
                        }

                        //printf("\n\n");

                    }
                    //printf("\n\n\n");
                    layer_cumulative_weights += num_filters*channels*filter_size*filter_size;
                }
                //printf("\n\n\n\n");
            }
        }
    }



    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        for (int j = 0; j < NUM_TILES_X; ++j){
            if((i != 0) || (j != 0)){

                //printf("Weights for device %d %d\n", j, i);
                int layer_cumulative_weights = 0;
                for (int l = 0; l < net->n; ++l){

                    //printf("Weights for layer %d\n", l);

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
                                    data[layer_cumulative_weights + ((c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n)] = 
                                    net->layers[l].weight_updates[(c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n];

                                    //printf("%.2f ", data[layer_cumulative_weights + ((c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n)]);
                                }
                                ///printf("\n");
                            }
                            //printf("\n");
                        }
                       // printf("\n\n");
                    }
                   // printf("\n\n\n");
                    layer_cumulative_weights += num_filters*channels*filter_size*filter_size;
                }
               // printf("\n\n\n\n");
               send_boundry(data, total_weights, j, i);
            }

            
        }
    }

    // for (int i = 0; i < NUM_TILES_Y; ++i)
    // {
    //     for (int j = 0; j < NUM_TILES_X; ++j){
    //         if((i != 0) || (j != 0)){
    //             for (int l = 0; l < net->n; ++l){
    //                 for (int m = 0; m < net->layers[l].size; ++m)
    //                 {
    //                     for (int n = 0; n < net->layers[l].size; ++n)
    //                     {
    //                         data[l*net->layers[l].size*net->layers[l].size + (m*net->layers[l].size + n)] = net->layers[l].weight_updates[m*net->layers[l].size + n];
    //                     }
    //                 }
    //             }
    //         }

    //         send_boundry(data, net->n * net->layers[0].size * net->layers[0].size, j, i);
    //     }
    // }

    free(data);


}





void sync_weight_updates(network* net, int NUM_TILES_Y, int NUM_TILES_X){

    // float* data = malloc(net->n * net->layers[0].size * net->layers[0].size * sizeof(float));

    // for (int l = 0; l < net->n; ++l){
    //     for (int m = 0; m < net->layers[l].size; ++m)
    //     {
    //         for (int n = 0; n < net->layers[l].size; ++n)
    //         {
    //             data[l*net->layers[l].size*net->layers[l].size + (m*net->layers[l].size + n)] = net->layers[l].weight_updates[m*net->layers[l].size + n];
    //         }
    //     }
    // }

    int total_weights = 0;

    for (int l = 0; l < net->n; ++l){
        int num_filters = net->layers[l].n;
        int filter_size = net->layers[l].size;
        int channels = net->layers[l].c;
        total_weights += num_filters*channels*filter_size*filter_size;
    }    

    float* data = malloc(total_weights * sizeof(float));

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
                        data[layer_cumulative_weights + ((c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n)] = 
                        net->layers[l].weight_updates[(c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n];
                    }
                }
            }
        }
        layer_cumulative_weights += num_filters*channels*filter_size*filter_size;
    }

    send_boundry(data, total_weights, 0, 0);

    receive_boundry(data, total_weights, 0, 0);
    // for (int l = 0; l < net->n; ++l){
    //     for (int m = 0; m < net->layers[l].size; ++m)
    //     {
    //         for (int n = 0; n < net->layers[l].size; ++n)
    //         {
    //             net->layers[l].weight_updates[m*net->layers[l].size + n] = data[l*net->layers[l].size*net->layers[l].size + (m*net->layers[l].size + n)];
    //         }
    //     }
    // }

    layer_cumulative_weights = 0;

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
                        net->layers[l].weight_updates[(c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n] = 
                        data[layer_cumulative_weights + ((c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n)];

                    //    printf("%.2f ", net->layers[l].weight_updates[(c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n]);
                    }
                  //  printf("\n");
                }
            }
        }
        layer_cumulative_weights += num_filters*channels*filter_size*filter_size;
    }

    free(data);


}


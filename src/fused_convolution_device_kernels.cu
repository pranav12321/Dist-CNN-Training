#include "fused_convolution_device.h"
#include "fused_device.h"
#include "transport.h"


#include "cuda.h"


extern int NUM_TILES_X;
extern int NUM_TILES_Y;
extern int DEVICE_ID_X;
extern int DEVICE_ID_Y;

extern network_config network_params_original;
extern network_config network_params_tile;
extern ftp_config ftp_params;

extern device_tile current_tile;
extern network_device current_device;
extern ftp_network ftp_cluster;

#ifdef GPU

__global__ void clear_elements_kernel(float* src, int batch, int depth,
                    int height_src, int width_src,
                    int start_x, int start_y,
                    int clear_height, int clear_width)
{
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int total_tile_sample_size = width_src*height_src*depth;

    if (n < clear_width && m < clear_height){
        for(int sample_id = 0; sample_id < batch; sample_id++){
            for (int c = 0; c < depth; ++c)
            {
                src[(sample_id*total_tile_sample_size) + (c*height_src*width_src) + ((m+start_y)*width_src) + n+start_x] = 0.0;
            }
        }
    }
}

extern "C" void clear_edges_featuremap_device_gpu(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int device_id_y, int device_id_x){

    int x_dim = net->layers[layer_idx].featuremap_in_w_with_boundry;
    int y_dim = net->layers[layer_idx].featuremap_in_h_with_boundry;
    int depth = net->layers[layer_idx].c;
    int total_tile_sample_size = x_dim*y_dim*depth;
    int batches = net->batch;

    if(layer_idx > 0){

        if(device_id_y == 0){
            int rows = net->layers[layer_idx].top_boundry_edges_featuremap;
            int cols = net->layers[layer_idx].featuremap_in_w_with_boundry;

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
            clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx-1].output_gpu, batches, depth, y_dim, x_dim, 0, 0, rows, cols);
        }

        if(device_id_x == 0){
            int rows = net->layers[layer_idx].featuremap_in_h_with_boundry;
            int cols = net->layers[layer_idx].left_boundry_edges_featuremap;

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
            clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx-1].output_gpu, batches, depth, y_dim, x_dim, 0, 0, rows, cols);
        }

        if(device_id_y == (NUM_TILES_Y - 1)){
            int rows = net->layers[layer_idx].bottom_boundry_edges_featuremap;
            int cols = net->layers[layer_idx].featuremap_in_w_with_boundry;
            int offset_y = net->layers[layer_idx].featuremap_in_h_without_boundry+net->layers[layer_idx].top_boundry_edges_featuremap;

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
            clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx-1].output_gpu, batches, depth, y_dim, x_dim, 0, offset_y, rows, cols);
        }

        if(device_id_y == (NUM_TILES_X - 1)){
            int rows = net->layers[layer_idx].featuremap_in_h_with_boundry;
            int cols = net->layers[layer_idx].right_boundry_edges_featuremap;
            int offset_x = net->layers[layer_idx].featuremap_in_w_without_boundry+net->layers[layer_idx].left_boundry_edges_featuremap;

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
            clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx-1].output_gpu, batches, depth, y_dim, x_dim, offset_x, 0, rows, cols);
        }

    }
}


extern "C" void clear_edges_delta_device_gpu(network* net, int layer_idx, int NUM_TILES_Y, int NUM_TILES_X, int device_id_y, int device_id_x){

    int x_dim = net->layers[layer_idx].delta_in_w_with_boundry;
    int y_dim = net->layers[layer_idx].delta_in_h_with_boundry;
    int depth = (net->layers[layer_idx].type == CONVOLUTIONAL) ? net->layers[layer_idx].n : net->layers[layer_idx].c;
    int total_tile_sample_size = x_dim*y_dim*depth;
    int batches = net->batch;


    if(layer_idx > 0){

        if(device_id_y == 0){
            int rows = net->layers[layer_idx].top_boundry_edges_delta;
            int cols = net->layers[layer_idx].delta_in_w_with_boundry;

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
            clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx].delta_gpu, batches, depth, y_dim, x_dim, 0, 0, rows, cols);
        }

        if(device_id_x == 0){
            int rows = net->layers[layer_idx].delta_in_h_with_boundry;
            int cols = net->layers[layer_idx].left_boundry_edges_delta;

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
            clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx].delta_gpu, batches, depth, y_dim, x_dim, 0, 0, rows, cols);
        }

        if(device_id_y == (NUM_TILES_Y - 1)){
            int rows = net->layers[layer_idx].bottom_boundry_edges_delta;
            int cols = net->layers[layer_idx].delta_in_w_with_boundry;
            int offset_y = net->layers[layer_idx].delta_in_h_without_boundry+net->layers[layer_idx].top_boundry_edges_delta;

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
            clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx].delta_gpu, batches, depth, y_dim, x_dim, 0, offset_y, rows, cols);
        }

        if(device_id_y == (NUM_TILES_X - 1)){
            int rows = net->layers[layer_idx].delta_in_h_with_boundry;
            int cols = net->layers[layer_idx].right_boundry_edges_delta;
            int offset_x = net->layers[layer_idx].delta_in_w_without_boundry+net->layers[layer_idx].left_boundry_edges_delta;

            dim3 threadsPerBlock(16, 16);
            dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
            clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx].delta_gpu, batches, depth, y_dim, x_dim, offset_x, 0, rows, cols);
        }

    }
}

extern "C" void clear_spurious_edges_featuremap_gpu(network* net, int layer_idx){

    int x_dim = net->layers[layer_idx].featuremap_in_w_with_boundry;
    int y_dim = net->layers[layer_idx].featuremap_in_h_with_boundry;
    int depth = net->layers[layer_idx].c;
    int total_tile_sample_size = x_dim*y_dim*depth;
    int batches = net->batch;

    int device_id_y = ftp_params.DEVICE_ID_Y;
    int device_id_x = ftp_params.DEVICE_ID_X;
    int NUM_TILES_X = ftp_params.NUM_TILES_X;
    int NUM_TILES_Y = ftp_params.NUM_TILES_Y;

    int start_x_coordinate = network_params_tile.spurious_blocks[layer_idx].start_x_coordinate;
    int start_y_coordinate = network_params_tile.spurious_blocks[layer_idx].start_y_coordinate;

    float* featuremap;

    // if(layer_idx > 0)
    //     featuremap = net->layers[layer_idx-1].output;
    // else
    featuremap = net->input;

    if((device_id_x == (NUM_TILES_X - 1)) && (start_x_coordinate > -1)){
        int rows = net->layers[layer_idx].featuremap_in_h_with_boundry;
        int cols = net->layers[layer_idx].featuremap_in_w_with_boundry - (net->layers[layer_idx].left_boundry_edges_featuremap + start_x_coordinate);
        int offset_x = net->layers[layer_idx].left_boundry_edges_featuremap + start_x_coordinate;

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
        clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx].delta_gpu, batches, depth, y_dim, x_dim, offset_x, 0, rows, cols);
    }

    if((device_id_y == (NUM_TILES_Y - 1)) && (start_y_coordinate > -1)){
        int rows = net->layers[layer_idx].featuremap_in_h_with_boundry - (net->layers[layer_idx].top_boundry_edges_featuremap + start_y_coordinate);
        int cols = net->layers[layer_idx].featuremap_in_w_with_boundry;
        int offset_y = net->layers[layer_idx].top_boundry_edges_featuremap + start_y_coordinate;

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
        clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx].delta_gpu, batches, depth, y_dim, x_dim, 0, offset_y, rows, cols);
    }
   
}

extern "C" void clear_spurious_edges_delta_gpu(network* net, int layer_idx){

    int x_dim = net->layers[layer_idx].delta_in_w_with_boundry;
    int y_dim = net->layers[layer_idx].delta_in_h_with_boundry;
    int depth = (net->layers[layer_idx].type == CONVOLUTIONAL) ? net->layers[layer_idx].n : net->layers[layer_idx].c;
    int total_tile_sample_size = x_dim*y_dim*depth;
    int batches = net->batch;

    int device_id_y = ftp_params.DEVICE_ID_Y;
    int device_id_x = ftp_params.DEVICE_ID_X;
    int NUM_TILES_X = ftp_params.NUM_TILES_X;
    int NUM_TILES_Y = ftp_params.NUM_TILES_Y;

    int start_x_coordinate = network_params_tile.spurious_blocks[layer_idx+1].start_x_coordinate;
    int start_y_coordinate = network_params_tile.spurious_blocks[layer_idx+1].start_y_coordinate;

    if((device_id_x == (NUM_TILES_X - 1)) && (start_x_coordinate > -1)){
        int rows = net->layers[layer_idx].delta_in_h_with_boundry;
        int cols = net->layers[layer_idx].delta_in_w_with_boundry - (net->layers[layer_idx].left_boundry_edges_delta + start_x_coordinate);
        int offset_x = net->layers[layer_idx].left_boundry_edges_delta + start_x_coordinate;

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
        clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx].delta_gpu, batches, depth, y_dim, x_dim, offset_x, 0, rows, cols);
    }

    if((device_id_y == (NUM_TILES_Y - 1)) && (start_y_coordinate > -1)){
        int rows = net->layers[layer_idx].delta_in_h_with_boundry - (net->layers[layer_idx].top_boundry_edges_delta + start_y_coordinate);
        int cols = net->layers[layer_idx].delta_in_w_with_boundry;
        int offset_y = net->layers[layer_idx].top_boundry_edges_delta + start_y_coordinate;

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
        clear_elements_kernel<<<numBlocks, threadsPerBlock>>>(net->layers[layer_idx].delta_gpu, batches, depth, y_dim, x_dim, 0, offset_y, rows, cols);
    }
   
}

    __global__ void copy_slice_kernel(float* dst, float* src, int batch, int depth,
                    int height_src, int width_src, int height_dst, int width_dst,
                    int src_start_x, int src_start_y, int dst_start_x, int dst_start_y,
                    int copy_height_src, int copy_width_src, int copy_height_dst, int copy_width_dst,
                    float* workspace)
    {
        int w = blockIdx.x * blockDim.x + threadIdx.x;
        int h = blockIdx.y * blockDim.y + threadIdx.y;
        int total_tile_sample_size = depth*copy_height_src*copy_width_src;

        if (h < copy_height_src && w < copy_width_src){

            float* src_intermediate = src;

            if(dst == src){
                src_intermediate = workspace;

                for(int b = 0; b < batch; b++){
                    for(int d = 0; d < depth; d++){
                        workspace[b*total_tile_sample_size + d*copy_height_src*copy_width_src + h*copy_width_src + w] = 
                        src[b*depth*height_src*width_src + d*height_src*width_src + (h + src_start_y)*width_src + w + src_start_x];
                    }
                }
            }

            for(int b = 0; b < batch; b++){
                for(int d = 0; d < depth; d++){
                    dst[b*total_tile_sample_size + d*height_dst*width_dst + (h + dst_start_y)*width_dst + w + dst_start_x] = 
                    src_intermediate[b*total_tile_sample_size + d*copy_height_dst*copy_width_dst + h*copy_width_dst + w];     
                }        
            }
        }
    }

    extern "C" void copy_slice_gpu(float* dst, float* src, int batch, int depth,
                    int height_src, int width_src, int height_dst, int width_dst,
                    int src_start_x, int src_start_y, int dst_start_x, int dst_start_y,
                    int copy_height_src, int copy_width_src, int copy_height_dst, int copy_width_dst,
                    float* workspace){

        int rows = copy_height_src;
        int cols = copy_width_src;

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((cols + threadsPerBlock.x -1) / threadsPerBlock.x, (rows+threadsPerBlock.y -1) / threadsPerBlock.y);
        copy_slice_kernel<<<numBlocks, threadsPerBlock>>>(dst, src, batch, depth,
                    height_src, width_src, height_dst, width_dst,
                    src_start_x, src_start_y, dst_start_x, dst_start_y,
                    copy_height_src, copy_width_src, copy_height_dst, copy_width_dst,
                    workspace);
    }



#endif

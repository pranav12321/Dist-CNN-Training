#include "darknet.h"
#include "fused_device.h"
#include "sm.h"

network_config network_params_original;
network_config network_params_tile;
ftp_config ftp_params;

device_tile current_tile;
network_device current_device;
ftp_network ftp_cluster;

void config_init(int argc, char* argv[]){

    int stride_vector[16] = {1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1};
    int filter_stack_vector[16] = {32, 32, 64, 64, 128, 64, 128, 128, 256, 128, 256, 256, 512, 256, 512, 256};
    LAYER_TYPE layer_type_vector[16] = {CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL,
                                   CONVOLUTIONAL, CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, CONVOLUTIONAL,
                                   CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, CONVOLUTIONAL, CONVOLUTIONAL, CONVOLUTIONAL};
    int filter_size_vector[16] = {3, 2, 3, 2, 3, 1, 1, 2, 3, 1, 3, 2, 3, 1, 3, 1};

    int INPUT_WIDTH = 608;
    int INPUT_HEIGHT = 608;
    int INPUT_CHANNELS = 3;

    int NUM_TILES_X = 2;
    int NUM_TILES_Y = 1;

    ftp_params.NUM_TILES_Y = atoi(argv[2]);
    ftp_params.NUM_TILES_X = atoi(argv[1]);

    int num_layers = atoi(argv[ftp_params.NUM_TILES_Y*ftp_params.NUM_TILES_X + 5 ]);

    network_params_tile.num_layers = num_layers;
    network_params_original.num_layers = num_layers;

    int batch_size = atoi(argv[ftp_params.NUM_TILES_Y*ftp_params.NUM_TILES_X + 6 ]);

    network_params_tile.batch_size = batch_size;
    network_params_original.batch_size = batch_size;

    network_params_tile.INPUT_WIDTH = INPUT_WIDTH;
    network_params_original.INPUT_WIDTH = INPUT_WIDTH;

    network_params_tile.INPUT_HEIGHT = INPUT_HEIGHT;
    network_params_original.INPUT_HEIGHT = INPUT_HEIGHT;

    network_params_original.featuremap_dim_without_boundry_vector[0].x_dim = INPUT_WIDTH;
    network_params_original.featuremap_dim_without_boundry_vector[0].y_dim = INPUT_HEIGHT;
    network_params_original.featuremap_dim_without_boundry_vector[0].depth = INPUT_CHANNELS;
    network_params_tile.featuremap_dim_without_boundry_vector[0].depth = INPUT_CHANNELS;
    network_params_tile.featuremap_dim_with_boundry_vector[0].depth = INPUT_CHANNELS;

    for (int i = 0; i < 16; ++i)
    {
        network_params_original.stride_vector[i] = stride_vector[i];
        network_params_tile.stride_vector[i] = stride_vector[i];

        network_params_original.filter_stack_vector[i] = filter_stack_vector[i];
        network_params_tile.filter_stack_vector[i] = filter_stack_vector[i];

        network_params_original.filter_size_vector[i] = filter_size_vector[i];
        network_params_tile.filter_size_vector[i] = filter_size_vector[i];

        network_params_original.layer_type_vector[i] = layer_type_vector[i];
        network_params_tile.layer_type_vector[i] = layer_type_vector[i];

        network_params_tile.spurious_blocks[i].start_x_coordinate = -1;
        network_params_tile.spurious_blocks[i].start_y_coordinate = -1;
    }

    for (int i = 1; i < num_layers; ++i)
    {
        network_params_original.featuremap_dim_without_boundry_vector[i].x_dim = network_params_original.featuremap_dim_without_boundry_vector[i-1].x_dim / (network_params_original.stride_vector[i-1]);
        network_params_original.featuremap_dim_without_boundry_vector[i].y_dim = network_params_original.featuremap_dim_without_boundry_vector[i-1].y_dim / (network_params_original.stride_vector[i-1]);
        network_params_original.featuremap_dim_without_boundry_vector[i].depth = filter_stack_vector[i-1];
        network_params_tile.featuremap_dim_without_boundry_vector[i].depth = filter_stack_vector[i-1];
        network_params_tile.featuremap_dim_with_boundry_vector[i].depth = filter_stack_vector[i-1];
    }

    network_params_original.delta_dim_without_boundry_vector[num_layers-1].x_dim = network_params_original.featuremap_dim_without_boundry_vector[num_layers-1].x_dim / (network_params_original.stride_vector[num_layers-1]);
    network_params_original.delta_dim_without_boundry_vector[num_layers-1].y_dim = network_params_original.featuremap_dim_without_boundry_vector[num_layers-1].y_dim / (network_params_original.stride_vector[num_layers-1]);;
    network_params_original.delta_dim_without_boundry_vector[num_layers-1].depth = filter_stack_vector[num_layers-1];
    network_params_tile.delta_dim_without_boundry_vector[num_layers-1].depth = filter_stack_vector[num_layers-1];
    network_params_tile.delta_dim_with_boundry_vector[num_layers-1].depth = filter_stack_vector[num_layers-1];

    for (int i = num_layers-2; i >= 0; --i)
    {
        network_params_original.delta_dim_without_boundry_vector[i].x_dim = network_params_original.featuremap_dim_without_boundry_vector[i+1].x_dim;
        network_params_original.delta_dim_without_boundry_vector[i].y_dim = network_params_original.featuremap_dim_without_boundry_vector[i+1].y_dim;
        network_params_original.delta_dim_without_boundry_vector[i].depth = network_params_original.featuremap_dim_without_boundry_vector[i+1].depth;
        network_params_tile.delta_dim_without_boundry_vector[i].depth = network_params_tile.featuremap_dim_without_boundry_vector[i+1].depth;
        network_params_tile.delta_dim_with_boundry_vector[i].depth = network_params_tile.featuremap_dim_with_boundry_vector[i+1].depth;
    }

    ftp_params.NUM_GROUPS_FORWARD = 1;//num_layers;

    ftp_params.sync_group_vector_forward[0] = 0;
    ftp_params.sync_group_vector_forward[1] = 8;
    ftp_params.sync_group_vector_forward[2] = 10;
    ftp_params.sync_group_vector_forward[3] = 12;
    ftp_params.sync_group_vector_forward[4] = 14;
    ftp_params.sync_group_vector_forward[5] = 5;
    ftp_params.sync_group_vector_forward[6] = 6;
    ftp_params.sync_group_vector_forward[7] = 7;
    ftp_params.sync_group_vector_forward[8] = 8;
    ftp_params.sync_group_vector_forward[9] = 9;
    ftp_params.sync_group_vector_forward[10] = 10;
    ftp_params.sync_group_vector_forward[11] = 11;
    ftp_params.sync_group_vector_forward[12] = 12;
    ftp_params.sync_group_vector_forward[13] = 13;
    ftp_params.sync_group_vector_forward[14] = 14;
    ftp_params.sync_group_vector_forward[15] = 15;

    ftp_params.NUM_GROUPS_BACKWARD = num_layers - 1;

    ftp_params.sync_group_vector_backward[0] = 1;
    ftp_params.sync_group_vector_backward[1] = 2;
    ftp_params.sync_group_vector_backward[2] = 3;
    ftp_params.sync_group_vector_backward[3] = 4;
    ftp_params.sync_group_vector_backward[4] = 5;
    ftp_params.sync_group_vector_backward[5] = 6;
    ftp_params.sync_group_vector_backward[6] = 7;
    ftp_params.sync_group_vector_backward[7] = 8;
    ftp_params.sync_group_vector_backward[8] = 9;
    ftp_params.sync_group_vector_backward[9] = 10;
    ftp_params.sync_group_vector_backward[10] = 11;
    ftp_params.sync_group_vector_backward[11] = 12;
    ftp_params.sync_group_vector_backward[12] = 13;
    ftp_params.sync_group_vector_backward[13] = 14;
    ftp_params.sync_group_vector_backward[14] = 15;

    ftp_params.DEVICE_ID_X = atoi(argv[ftp_params.NUM_TILES_Y*ftp_params.NUM_TILES_X + 3 ]);
    ftp_params.DEVICE_ID_Y = atoi(argv[ftp_params.NUM_TILES_Y*ftp_params.NUM_TILES_X + 4 ]);
}

void init_network(network** net_inp){

    *net_inp = calloc(1, sizeof(network));

    network* net = *net_inp;
    net->n = network_params_tile.num_layers;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    net->batch = network_params_tile.batch_size;

    for (int i = 0; i < net->n; ++i)
    {

        if(network_params_tile.layer_type_vector[i] == CONVOLUTIONAL){

            net->layers[i] = make_convolutional_layer(net->batch, network_params_tile.featuremap_dim_with_boundry_vector[i].y_dim, network_params_tile.featuremap_dim_with_boundry_vector[i].x_dim,
                                                     network_params_tile.featuremap_dim_with_boundry_vector[i].depth, network_params_tile.filter_stack_vector[i], 1,
                                                     network_params_tile.filter_size_vector[i], network_params_tile.stride_vector[i], 0, RELU, 0, 0, 0, 0);
            int total_filter_elements = net->layers[i].size*net->layers[i].size*net->layers[i].c*net->layers[i].n;

            for (int i_f = 0; i_f < (total_filter_elements); ++i_f)
            {
                    net->layers[i].weights[i_f] = 0.01;
            }

            // Get shared memory 

            float* device_weight_update_buffers;
            char shm_file[3];
            shm_file[0] = '/';
            shm_file[1] = '0' + i;
            shm_file[2] = '\0';
            if(current_tile.is_device_representative_tile){
                create_sm(shm_file, &device_weight_update_buffers, current_device.num_tiles, total_filter_elements);
                net->layers[i].weight_updates = device_weight_update_buffers;
            }
            else{
                sleep(1);
                get_sm_buffer(shm_file, &device_weight_update_buffers, current_device.num_tiles, total_filter_elements);
                net->layers[i].weight_updates = (device_weight_update_buffers + ((current_tile.device_tile_id)*total_filter_elements));
            }
        }

        else if(network_params_tile.layer_type_vector[i] == MAXPOOL){
            //(int batch, int h, int w, int c, int size, int stride, int padding)
            net->layers[i] = make_maxpool_layer(net->batch, network_params_tile.featuremap_dim_with_boundry_vector[i].y_dim, network_params_tile.featuremap_dim_with_boundry_vector[i].x_dim,
                             network_params_tile.featuremap_dim_with_boundry_vector[i].depth, network_params_tile.filter_size_vector[i], network_params_tile.stride_vector[i], 0); 
        }

        net->layers[i].featuremap_in_w_without_boundry = network_params_tile.featuremap_dim_without_boundry_vector[i].x_dim;
        net->layers[i].featuremap_in_h_without_boundry = network_params_tile.featuremap_dim_without_boundry_vector[i].y_dim;
        net->layers[i].featuremap_in_w_with_boundry = network_params_tile.featuremap_dim_with_boundry_vector[i].x_dim;
        net->layers[i].featuremap_in_h_with_boundry = network_params_tile.featuremap_dim_with_boundry_vector[i].y_dim;

        net->layers[i].original_featuremap_in_h = net->layers[i].featuremap_in_h_without_boundry;
        net->layers[i].original_featuremap_in_w = net->layers[i].featuremap_in_w_without_boundry;

        net->layers[i].delta_in_w_without_boundry = network_params_tile.delta_dim_without_boundry_vector[i].x_dim;
        net->layers[i].delta_in_h_without_boundry = network_params_tile.delta_dim_without_boundry_vector[i].y_dim;
        net->layers[i].delta_in_w_with_boundry = network_params_tile.delta_dim_with_boundry_vector[i].x_dim;
        net->layers[i].delta_in_h_with_boundry = network_params_tile.delta_dim_with_boundry_vector[i].y_dim;

        net->layers[i].right_boundry_edges_featuremap = network_params_tile.featuremap_edges_vector[i].right_boundry_edges;
        net->layers[i].left_boundry_edges_featuremap = network_params_tile.featuremap_edges_vector[i].left_boundry_edges;
        net->layers[i].bottom_boundry_edges_featuremap = network_params_tile.featuremap_edges_vector[i].bottom_boundry_edges;
        net->layers[i].top_boundry_edges_featuremap = network_params_tile.featuremap_edges_vector[i].top_boundry_edges;

        net->layers[i].right_boundry_edges_delta = network_params_tile.delta_edges_vector[i].right_boundry_edges;
        net->layers[i].left_boundry_edges_delta = network_params_tile.delta_edges_vector[i].left_boundry_edges;
        net->layers[i].bottom_boundry_edges_delta = network_params_tile.delta_edges_vector[i].bottom_boundry_edges;
        net->layers[i].top_boundry_edges_delta = network_params_tile.delta_edges_vector[i].top_boundry_edges;

    }

    int max = 0;
    for (int i = 0; i < net->n; ++i)
    {
        printf("%d\n", net->layers[i].workspace_size);
        if(net->layers[i].workspace_size > max){
            max = net->layers[i].workspace_size;
        }
    }
    printf("wsize = %d inputs = %d outputs = %d\n", max*sizeof(float), net->inputs, net->layers[0].outputs);
    net->workspace = calloc(max, sizeof(float));

    printf("%p\n", net);
}

void forward_pass(){

    int num_layers = network_params_original.num_layers;

    dim net_last_layer_dim_original = network_params_original.delta_dim_without_boundry_vector[num_layers-1];

    network_params_tile.featuremap_dim_without_boundry_vector[num_layers-1].x_dim = network_params_tile.stride_vector[num_layers-1]*((net_last_layer_dim_original.x_dim + (ftp_params.NUM_TILES_X - 1))/ftp_params.NUM_TILES_X);
    network_params_tile.featuremap_dim_without_boundry_vector[num_layers-1].y_dim = network_params_tile.stride_vector[num_layers-1]*((net_last_layer_dim_original.y_dim + (ftp_params.NUM_TILES_Y - 1))/ftp_params.NUM_TILES_Y);

    if(ftp_params.DEVICE_ID_X == (ftp_params.NUM_TILES_X - 1)){ 

        int xdim = (network_params_tile.featuremap_dim_without_boundry_vector[num_layers-1].x_dim)/(network_params_tile.stride_vector[num_layers-1]);

        int spurious_x_edges = (ftp_params.NUM_TILES_X*(xdim)) - net_last_layer_dim_original.x_dim;

        printf("%d %d %d\n", network_params_original.delta_dim_without_boundry_vector[num_layers-1].x_dim, xdim, spurious_x_edges);

        if(spurious_x_edges == xdim){
            spurious_x_edges = 0;
        }

        network_params_tile.spurious_blocks[num_layers].start_x_coordinate = (spurious_x_edges > 0) ? (xdim - spurious_x_edges) : -1;

        xdim = network_params_tile.featuremap_dim_without_boundry_vector[num_layers-1].x_dim;
        spurious_x_edges = (ftp_params.NUM_TILES_X*(network_params_tile.featuremap_dim_without_boundry_vector[num_layers-1].x_dim)) - network_params_original.featuremap_dim_without_boundry_vector[num_layers-1].x_dim;
        printf("%d %d %d\n",network_params_original.featuremap_dim_without_boundry_vector[num_layers-1].x_dim, xdim, spurious_x_edges);
        if(spurious_x_edges == xdim){
            spurious_x_edges = 0;
        }
        network_params_tile.spurious_blocks[num_layers-1].start_x_coordinate = (spurious_x_edges > 0) ? (xdim - spurious_x_edges) : -1;
    }

    if(ftp_params.DEVICE_ID_Y == (ftp_params.NUM_TILES_Y - 1)){    

        int ydim = network_params_tile.featuremap_dim_without_boundry_vector[num_layers-1].y_dim/(network_params_tile.stride_vector[num_layers-1]);

        int spurious_y_edges = (ftp_params.NUM_TILES_Y*(ydim)) - net_last_layer_dim_original.y_dim;
        printf("%d %d %d\n",network_params_original.delta_dim_without_boundry_vector[num_layers-1].y_dim, ydim, spurious_y_edges);
        if(spurious_y_edges == ydim){
            spurious_y_edges = 0;
        }

        network_params_tile.spurious_blocks[num_layers].start_y_coordinate = (spurious_y_edges > 0) ? (ydim - spurious_y_edges) : -1;

        ydim = network_params_tile.featuremap_dim_without_boundry_vector[num_layers-1].y_dim;
        spurious_y_edges = (ftp_params.NUM_TILES_Y*(network_params_tile.featuremap_dim_without_boundry_vector[num_layers-1].y_dim)) - network_params_original.featuremap_dim_without_boundry_vector[num_layers-1].y_dim;
        printf("%d %d %d\n",network_params_original.featuremap_dim_without_boundry_vector[num_layers-1].y_dim, ydim, spurious_y_edges);
        if(spurious_y_edges == ydim){
            spurious_y_edges = 0;
        }
        network_params_tile.spurious_blocks[num_layers-1].start_y_coordinate = (spurious_y_edges > 0) ? (ydim - spurious_y_edges) : -1;
    }

    printf("Spurious x = %d Spurious y = %d\n", network_params_tile.spurious_blocks[num_layers].start_x_coordinate, network_params_tile.spurious_blocks[num_layers].start_y_coordinate);
    printf("Spurious x = %d Spurious y = %d\n", network_params_tile.spurious_blocks[num_layers-1].start_x_coordinate, network_params_tile.spurious_blocks[num_layers-1].start_y_coordinate);
    // printf("FEATUREMAP H with boundry/without boundry = %d %d\n", network_params_tile.featuremap_dim_with_boundry_vector[j].y_dim, network_params_tile.featuremap_dim_without_boundry_vector[j].y_dim);
    // printf("FEATUREMAP H with boundry/without boundry = %d %d\n", network_params_tile.featuremap_dim_with_boundry_vector[j].y_dim, network_params_tile.featuremap_dim_without_boundry_vector[j].y_dim);
    // printf("FEATUREMAP H with boundry/without boundry = %d %d\n", network_params_tile.featuremap_dim_with_boundry_vector[j].y_dim, network_params_tile.featuremap_dim_without_boundry_vector[j].y_dim);

    for (int i = num_layers - 2; i >= 0; --i)
    {
        network_params_tile.featuremap_dim_without_boundry_vector[i].x_dim = network_params_tile.stride_vector[i]*network_params_tile.featuremap_dim_without_boundry_vector[i+1].x_dim;
        network_params_tile.featuremap_dim_without_boundry_vector[i].y_dim = network_params_tile.stride_vector[i]*network_params_tile.featuremap_dim_without_boundry_vector[i+1].y_dim;

        if(ftp_params.DEVICE_ID_X == (ftp_params.NUM_TILES_X - 1)){

            int xdim = network_params_tile.featuremap_dim_without_boundry_vector[i].x_dim;

            int spurious_x_edges = (ftp_params.NUM_TILES_X*(network_params_tile.featuremap_dim_without_boundry_vector[i].x_dim)) - network_params_original.featuremap_dim_without_boundry_vector[i].x_dim;
            printf("%d %d %d\n",network_params_original.featuremap_dim_without_boundry_vector[i].x_dim, xdim, spurious_x_edges);
            if(spurious_x_edges == xdim){
                spurious_x_edges = 0;
            }

            network_params_tile.spurious_blocks[i].start_x_coordinate = spurious_x_edges > 0 ? (xdim - spurious_x_edges) : -1;
        }

        if(ftp_params.DEVICE_ID_Y == (ftp_params.NUM_TILES_Y - 1)){
            
            int ydim = network_params_tile.featuremap_dim_without_boundry_vector[i].y_dim;

            int spurious_y_edges = (ftp_params.NUM_TILES_Y*(network_params_tile.featuremap_dim_without_boundry_vector[i].y_dim)) - network_params_original.featuremap_dim_without_boundry_vector[i].y_dim;
            printf("%d %d %d\n",network_params_original.featuremap_dim_without_boundry_vector[i].y_dim, ydim, spurious_y_edges);
            if(spurious_y_edges == ydim){
                spurious_y_edges = 0;
            }

            network_params_tile.spurious_blocks[i].start_y_coordinate = (spurious_y_edges > 0) ? (ydim - spurious_y_edges) : -1;
        }
        printf("Spurious x = %d Spurious y = %d\n", network_params_tile.spurious_blocks[i].start_x_coordinate, network_params_tile.spurious_blocks[i].start_y_coordinate);
    }

    //while(1);

    for (int i = 0; i < (ftp_params.NUM_GROUPS_FORWARD) ; ++i)
    {

        int group_start_idx = ftp_params.sync_group_vector_forward[i];
        int group_end_idx = (i == (ftp_params.NUM_GROUPS_FORWARD - 1)) ? (num_layers - 1) : (ftp_params.sync_group_vector_forward[i+1]-1);

        dim group_end_dim_tile = network_params_tile.delta_dim_without_boundry_vector[group_end_idx];

        coordinate_bounds group_end_bounds_tile;
        group_end_bounds_tile.start_x_coordinate = 0;
        group_end_bounds_tile.start_y_coordinate = 0;
        group_end_bounds_tile.end_x_coordinate = group_end_dim_tile.x_dim - 1;
        group_end_bounds_tile.end_y_coordinate = group_end_dim_tile.y_dim - 1;

        edges group_end_edges_tile;
        group_end_edges_tile.left_boundry_edges = 0;
        group_end_edges_tile.right_boundry_edges = 0;
        group_end_edges_tile.top_boundry_edges = 0;
        group_end_edges_tile.bottom_boundry_edges = 0;

        for (int j = group_end_idx; j >= group_start_idx; --j)
        {

            int filter_size = network_params_tile.filter_size_vector[j];
            int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
            int stride = network_params_tile.stride_vector[j];

            if(network_params_tile.layer_type_vector[j] == MAXPOOL){
                unit_boundry = 0;
            }

            //Previous layer coordinate bounds and edges
            int start_x_coordinate = (j == group_end_idx) ? group_end_bounds_tile.start_x_coordinate : network_params_tile.featuremap_bounds_vector[j+1].start_x_coordinate;
            int start_y_coordinate = (j == group_end_idx) ? group_end_bounds_tile.start_y_coordinate : network_params_tile.featuremap_bounds_vector[j+1].start_y_coordinate;
            int end_x_coordinate = (j == group_end_idx) ? group_end_bounds_tile.end_x_coordinate : network_params_tile.featuremap_bounds_vector[j+1].end_x_coordinate;
            int end_y_coordinate = (j == group_end_idx) ? group_end_bounds_tile.end_y_coordinate : network_params_tile.featuremap_bounds_vector[j+1].end_y_coordinate;

            int left_boundry_edges = (j == group_end_idx) ? group_end_edges_tile.left_boundry_edges : network_params_tile.featuremap_edges_vector[j+1].left_boundry_edges;
            int right_boundry_edges = (j == group_end_idx) ? group_end_edges_tile.right_boundry_edges : network_params_tile.featuremap_edges_vector[j+1].right_boundry_edges;
            int top_boundry_edges = (j == group_end_idx) ? group_end_edges_tile.top_boundry_edges : network_params_tile.featuremap_edges_vector[j+1].top_boundry_edges;
            int bottom_boundry_edges = (j == group_end_idx) ? group_end_edges_tile.bottom_boundry_edges : network_params_tile.featuremap_edges_vector[j+1].bottom_boundry_edges;

            left_boundry_edges = (left_boundry_edges)*stride + unit_boundry;
            top_boundry_edges = (top_boundry_edges)*stride + unit_boundry;
            bottom_boundry_edges = (bottom_boundry_edges)*stride + unit_boundry;
            right_boundry_edges = (right_boundry_edges)*stride + unit_boundry;

            int bottom_right_excess_pool = ((filter_size-1) > (stride-1)) ? ((filter_size - 1) - (stride-1)) : 0;

            if(network_params_tile.layer_type_vector[j] == MAXPOOL){
                bottom_boundry_edges += bottom_right_excess_pool;
                right_boundry_edges += bottom_right_excess_pool;
            }

            network_params_tile.featuremap_dim_with_boundry_vector[j].x_dim = network_params_tile.featuremap_dim_without_boundry_vector[j].x_dim + left_boundry_edges + right_boundry_edges; 
            network_params_tile.featuremap_dim_with_boundry_vector[j].y_dim = network_params_tile.featuremap_dim_without_boundry_vector[j].y_dim + top_boundry_edges + bottom_boundry_edges; 

            network_params_tile.featuremap_edges_vector[j].right_boundry_edges = right_boundry_edges;
            network_params_tile.featuremap_edges_vector[j].left_boundry_edges = left_boundry_edges;
            network_params_tile.featuremap_edges_vector[j].bottom_boundry_edges = bottom_boundry_edges;
            network_params_tile.featuremap_edges_vector[j].top_boundry_edges = top_boundry_edges;


            printf("Layer %d\n\n", j);
            printf("FEATUREMAP H with boundry/without boundry = %d %d\n", network_params_tile.featuremap_dim_with_boundry_vector[j].y_dim, network_params_tile.featuremap_dim_without_boundry_vector[j].y_dim);
            printf("FEATUREMAP W with boundry/without boundry = %d %d\n", network_params_tile.featuremap_dim_with_boundry_vector[j].x_dim, network_params_tile.featuremap_dim_without_boundry_vector[j].x_dim);

            printf("Top boundry edges = %d\n", top_boundry_edges);
            printf("Left boundry edges = %d\n", left_boundry_edges);
            printf("Right boundry edges = %d\n", right_boundry_edges);
            printf("Bottom boundry edges = %d\n\n", bottom_boundry_edges);

        }
    }
}



void backward_pass(){

    int num_layers = network_params_original.num_layers;

    dim net_last_layer_dim_original = network_params_original.delta_dim_without_boundry_vector[num_layers-1];

    network_params_tile.delta_dim_without_boundry_vector[num_layers-1].x_dim = (net_last_layer_dim_original.x_dim + (ftp_params.NUM_TILES_X - 1))/ftp_params.NUM_TILES_X;
    network_params_tile.delta_dim_without_boundry_vector[num_layers-1].y_dim = (net_last_layer_dim_original.y_dim + (ftp_params.NUM_TILES_Y - 1))/ftp_params.NUM_TILES_Y;

    for (int i = num_layers - 2; i >= 0; --i)
    {
        network_params_tile.delta_dim_without_boundry_vector[i].x_dim = network_params_tile.stride_vector[i+1]*network_params_tile.delta_dim_without_boundry_vector[i+1].x_dim;
        network_params_tile.delta_dim_without_boundry_vector[i].y_dim = network_params_tile.stride_vector[i+1]*network_params_tile.delta_dim_without_boundry_vector[i+1].y_dim;
    }

    for (int i = 0; i < (ftp_params.NUM_GROUPS_BACKWARD) ; ++i)
    {

        int group_start_idx = (i == 0) ? 1 : (ftp_params.sync_group_vector_backward[i-1]+1);
        int group_end_idx = ftp_params.sync_group_vector_backward[i];

        dim group_start_dim_tile = network_params_tile.delta_dim_without_boundry_vector[group_start_idx-1];

        coordinate_bounds group_start_bounds_tile;
        group_start_bounds_tile.start_x_coordinate = 0;
        group_start_bounds_tile.start_y_coordinate = 0;
        group_start_bounds_tile.end_x_coordinate = group_start_dim_tile.x_dim - 1;
        group_start_bounds_tile.end_y_coordinate = group_start_dim_tile.y_dim - 1;

        edges group_start_edges_tile;
        group_start_edges_tile.left_boundry_edges = 0;
        group_start_edges_tile.right_boundry_edges = 0;
        group_start_edges_tile.top_boundry_edges = 0;
        group_start_edges_tile.bottom_boundry_edges = 0;

        printf("Group start = %d end = %d\n", group_start_idx, group_end_idx);

        for (int j = group_start_idx; j <= group_end_idx; ++j)
        {

            int filter_size = network_params_tile.filter_size_vector[j];
            int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
            int stride = network_params_tile.stride_vector[j];

            //Previous layer edges
            int left_boundry_edges = (j == group_start_idx) ? group_start_edges_tile.left_boundry_edges : network_params_tile.delta_edges_vector[j-1].left_boundry_edges;
            int right_boundry_edges = (j == group_start_idx) ? group_start_edges_tile.right_boundry_edges : network_params_tile.delta_edges_vector[j-1].right_boundry_edges;
            int top_boundry_edges = (j == group_start_idx) ? group_start_edges_tile.top_boundry_edges : network_params_tile.delta_edges_vector[j-1].top_boundry_edges;
            int bottom_boundry_edges = (j == group_start_idx) ? group_start_edges_tile.bottom_boundry_edges : network_params_tile.delta_edges_vector[j-1].bottom_boundry_edges;

            if(network_params_tile.layer_type_vector[j] == CONVOLUTIONAL){
                left_boundry_edges = (left_boundry_edges + unit_boundry)/stride;
                top_boundry_edges = (top_boundry_edges + unit_boundry)/stride;
                bottom_boundry_edges = (bottom_boundry_edges + unit_boundry + (stride-1))/stride;
                right_boundry_edges = (right_boundry_edges + unit_boundry + (stride-1))/stride;
            }

            else if(network_params_tile.layer_type_vector[j] == MAXPOOL){
                left_boundry_edges = (left_boundry_edges + (stride-1))/stride;
                top_boundry_edges = (top_boundry_edges + (stride-1))/stride;
                bottom_boundry_edges = (bottom_boundry_edges + (stride-1))/stride;
                right_boundry_edges = (right_boundry_edges + (stride-1))/stride;
            }

            // network_params_tile.back_pad[j] = left_boundry_edges_total - left_boundry_edges;

            network_params_tile.delta_dim_with_boundry_vector[j].x_dim = network_params_tile.delta_dim_without_boundry_vector[j].x_dim + left_boundry_edges + right_boundry_edges; 
            network_params_tile.delta_dim_with_boundry_vector[j].y_dim = network_params_tile.delta_dim_without_boundry_vector[j].y_dim + top_boundry_edges + bottom_boundry_edges; 

            network_params_tile.delta_edges_vector[j].right_boundry_edges = right_boundry_edges;
            network_params_tile.delta_edges_vector[j].left_boundry_edges = left_boundry_edges;
            network_params_tile.delta_edges_vector[j].bottom_boundry_edges = bottom_boundry_edges;
            network_params_tile.delta_edges_vector[j].top_boundry_edges = top_boundry_edges;

            printf("Layer %d \n\n", j);
            printf("DELTA H with boundry/without boundry = %d %d\n", network_params_tile.delta_dim_with_boundry_vector[j].y_dim, network_params_tile.delta_dim_without_boundry_vector[j].y_dim);
            printf("DELTA W with boundry/without boundry = %d %d\n", network_params_tile.delta_dim_with_boundry_vector[j].x_dim, network_params_tile.delta_dim_without_boundry_vector[j].x_dim);
            printf("Top boundry edges = %d\n", top_boundry_edges);
            printf("Left boundry edges = %d\n", left_boundry_edges);
            printf("Right boundry edges = %d\n", right_boundry_edges);
            printf("Bottom boundry edges = %d\n\n", bottom_boundry_edges);

        }
    }

    network_params_tile.delta_dim_with_boundry_vector[0].x_dim = network_params_tile.delta_dim_without_boundry_vector[0].x_dim; 
    network_params_tile.delta_dim_with_boundry_vector[0].y_dim = network_params_tile.delta_dim_without_boundry_vector[0].y_dim; 

    network_params_tile.delta_edges_vector[0].right_boundry_edges = 0;
    network_params_tile.delta_edges_vector[0].left_boundry_edges = 0;
    network_params_tile.delta_edges_vector[0].bottom_boundry_edges = 0;
    network_params_tile.delta_edges_vector[0].top_boundry_edges = 0;

        // printf("Layer %d \n\n", i);
        // printf("DELTA H with boundry/without boundry = %d %d\n", net->layers[i].delta_in_h_with_boundry, net->layers[i].delta_in_h_without_boundry);
        // printf("DELTA W with boundry/without boundry = %d %d\n", net->layers[i].delta_in_w_with_boundry, net->layers[i].delta_in_w_without_boundry);
        // printf("Top boundry edges = %d\n", top_boundry_edges);
        // printf("Left boundry edges = %d\n", left_boundry_edges);
        // printf("Right boundry edges = %d\n", right_boundry_edges);
        // printf("Bottom boundry edges = %d\n\n", bottom_boundry_edges);
        // printf("Start x coordinate = %d\n", start_x_coordinate);
        // printf("Start y coordinte = %d\n", start_y_coordinate);
        // printf("End x coordinate = %d\n", end_x_coordinate);
        // printf("End y coordinate = %d\n\n", end_y_coordinate);

}




// void partition_forward_device(network* net,
//                         train_groups_profile* profile,
//                        group_profile_forward* group, 
//                         int start_x_forward, int start_y_forward,
//                         int end_x_forward, int end_y_forward){

//     int num_layers = net->n;

//     int left_boundry_edges = 0;
//     int top_boundry_edges = 0;

//     int right_boundry_edges = 0;
//     int bottom_boundry_edges = 0;

//     int start_x_coordinate = start_x_forward;
//     int start_y_coordinate = start_y_forward;
//     int end_x_coordinate = end_x_forward;
//     int end_y_coordinate = end_y_forward;


//     for (int j = 0; j < net->n; ++j)
//     {
//         net->layers[j].stride = stride_vector[j];
//     }

//     for (int j = 0; j < net->n; ++j)
//     {
//         net->layers[j].size = filter_size_vector[j];
//     }


//     for (int i = group->layer_end_idx; i >= group->layer_start_idx; i--)
//     {
//         int unit_boundry = ((filter_size_vector[i] & 0x1) == 1) ? ((filter_size_vector[i] - 1)/2) : (filter_size_vector[i]/2);

//         if (layer_vector[i] == MAXPOOL)
//         {
//             unit_boundry = 0;
//         }
//         int boundry_frames = unit_boundry;

//         int stride = net->layers[i].stride;
//         int filter_size = net->layers[i].size;

//         int next_layer_left_edges;
//         int next_layer_right_edges;

//         if(i == (group->layer_end_idx)){
//             next_layer_left_edges = 0;
//             next_layer_right_edges = 0;
//         }
//         else{
//             next_layer_left_edges = net->layers[i+1].left_boundry_edges_featuremap;
//             next_layer_right_edges = net->layers[i+1].right_boundry_edges_featuremap;
//         }

//         left_boundry_edges = unit_boundry + (next_layer_left_edges*stride);
//         top_boundry_edges = left_boundry_edges;

//         right_boundry_edges = unit_boundry + (next_layer_right_edges*stride);;
//         bottom_boundry_edges = right_boundry_edges;


//         start_x_coordinate = (start_x_coordinate*stride) - unit_boundry;
//         start_y_coordinate = (start_y_coordinate*stride) - unit_boundry;

//         end_x_coordinate = (end_x_coordinate*stride) + unit_boundry + stride - 1;
//         end_y_coordinate = (end_y_coordinate*stride) + unit_boundry + stride - 1;

//         int featuremap_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1; 
//         int featuremap_in_h_without_boundry = featuremap_in_h_with_boundry - (top_boundry_edges + bottom_boundry_edges);

//         int featuremap_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1;
//         int featuremap_in_w_without_boundry = featuremap_in_w_with_boundry - (left_boundry_edges + right_boundry_edges);

//         int num_channels = ((i == 0) ? 3 : filter_stack_vector[i-1]);

//         if(layer_vector[i] == CONVOLUTIONAL){
//             net->layers[i] = make_convolutional_layer(1, featuremap_in_h_with_boundry, featuremap_in_w_with_boundry, num_channels, filter_stack_vector[i], 1, filter_size_vector[i], stride, 0, RELU, 0, 0, 0, 0);

//             for (int i_f = 0; i_f < filter_size*filter_size*net->layers[i].c*net->layers[i].n; ++i_f)
//             {
//                     net->layers[i].weights[i_f] = 0.01;
//             }
//         }

//         else if(layer_vector[i] == MAXPOOL){
//             //(int batch, int h, int w, int c, int size, int stride, int padding)
//             net->layers[i] = make_maxpool_layer(1, featuremap_in_h_with_boundry, featuremap_in_w_with_boundry, num_channels, filter_size_vector[i], stride, 0); 
//         }

//         net->layers[i].left_boundry_edges_featuremap = left_boundry_edges;
//         net->layers[i].top_boundry_edges_featuremap = top_boundry_edges;
//         net->layers[i].right_boundry_edges_featuremap = right_boundry_edges;
//         net->layers[i].bottom_boundry_edges_featuremap = bottom_boundry_edges;

//         net->layers[i].featuremap_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1; 
//         net->layers[i].featuremap_in_h_without_boundry = net->layers[i].featuremap_in_h_with_boundry - (net->layers[i].top_boundry_edges_featuremap + net->layers[i].bottom_boundry_edges_featuremap);
//         net->layers[i].original_featuremap_in_h = net->layers[i].featuremap_in_h_without_boundry;

//         net->layers[i].featuremap_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1;
//         net->layers[i].featuremap_in_w_without_boundry = net->layers[i].featuremap_in_w_with_boundry - (net->layers[i].left_boundry_edges_featuremap + net->layers[i].right_boundry_edges_featuremap);
//         net->layers[i].original_featuremap_in_w = net->layers[i].featuremap_in_w_without_boundry;

//         printf("Layer %d\n\n", i);
//         printf("FEATUREMAP H with boundry/without boundry = %d %d\n", net->layers[i].featuremap_in_h_with_boundry, net->layers[i].featuremap_in_h_without_boundry);
//         printf("FEATUREMAP W with boundry/without boundry = %d %d\n", net->layers[i].featuremap_in_w_with_boundry, net->layers[i].featuremap_in_w_without_boundry);

//         printf("Top boundry edges = %d\n", top_boundry_edges);
//         printf("Left boundry edges = %d\n", left_boundry_edges);
//         printf("Right boundry edges = %d\n", right_boundry_edges);
//         printf("Bottom boundry edges = %d\n\n", bottom_boundry_edges);
//         printf("Start x coordinate = %d\n", start_x_coordinate);
//         printf("Start y coordinte = %d\n", start_y_coordinate);
//         printf("End x coordinate = %d\n", end_x_coordinate);
//         printf("End y coordinate = %d\n\n", end_y_coordinate);
        
//     }
// }

// void partition_backward_device(network* net, 
//                         group_profile_backward* profile,
//                         int start_x_backward, int start_y_backward,
//                         int end_x_backward, int end_y_backward){

//     int left_boundry_edges = 0;
//     int top_boundry_edges = 0;

//     int right_boundry_edges = 0;
//     int bottom_boundry_edges = 0;
    
//     int start_x_coordinate = start_x_backward;
//     int start_y_coordinate = start_y_backward;
//     int end_x_coordinate = end_x_backward;
//     int end_y_coordinate = end_y_backward;

//     int num_layers = net->n;

//     int start_idx = 0;
//     if(profile->layer_start_idx == 0)
//         start_idx = 0;
//     else
//         start_idx = (profile->layer_start_idx);

//     net->layers[start_idx-1].delta_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1; 
//     net->layers[start_idx-1].delta_in_h_without_boundry = net->layers[start_idx-1].delta_in_h_with_boundry;

//     net->layers[start_idx-1].delta_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1; 
//     net->layers[start_idx-1].delta_in_w_without_boundry = net->layers[start_idx-1].delta_in_w_with_boundry;

//     for (int i = start_idx; i <= (profile->layer_end_idx); ++i)
//     {
//         int filter_size = net->layers[i].size;
//         int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
//         int stride = net->layers[i].stride;

//         if(net->layers[i].type == MAXPOOL){
//             unit_boundry = 0;
//         }

//         if(i == start_idx){
//             left_boundry_edges = unit_boundry;
//             top_boundry_edges = unit_boundry;
//             right_boundry_edges = unit_boundry;
//             bottom_boundry_edges = unit_boundry;
//         }

//         else{

//             left_boundry_edges = (unit_boundry + net->layers[i-1].left_boundry_edges_delta) /(net->layers[i].stride);
//             top_boundry_edges = (unit_boundry + net->layers[i-1].top_boundry_edges_delta) /(net->layers[i].stride);

//             right_boundry_edges = (unit_boundry + net->layers[i-1].right_boundry_edges_delta + net->layers[i].stride - 1) /(net->layers[i].stride);
//             bottom_boundry_edges = (unit_boundry + net->layers[i-1].bottom_boundry_edges_delta + net->layers[i].stride - 1) /(net->layers[i].stride);
//         }

//         if(i>0){

//             int prev_stride = net->layers[i-1].stride;

//             if((start_x_coordinate - unit_boundry) > 0){
//                 start_x_coordinate = (start_x_coordinate - unit_boundry + (stride - 1))/stride;
//             }
//             else{
//                 start_x_coordinate = (start_x_coordinate - unit_boundry)/stride;         
//             }

//             if((end_x_coordinate + unit_boundry) < 0){
//                 end_x_coordinate = (end_x_coordinate + unit_boundry - (stride - 1))/stride;
//             }
//             else{
//                 end_x_coordinate = (end_x_coordinate + unit_boundry)/stride;          
//             }


//             if((start_y_coordinate - unit_boundry) > 0){
//                 start_y_coordinate = (start_y_coordinate - unit_boundry + (stride - 1))/stride;
//             }
//             else{
//                 start_y_coordinate = (start_y_coordinate - unit_boundry)/stride;          
//             }

//             if((end_y_coordinate + unit_boundry) < 0){
//                 end_y_coordinate = (end_y_coordinate + unit_boundry - (stride - 1))/stride;
//             }
//             else{
//                 end_y_coordinate = (end_y_coordinate + unit_boundry)/stride;          
//             }
//         }

//         net->layers[i].left_boundry_edges_delta = left_boundry_edges;
//         net->layers[i].top_boundry_edges_delta = top_boundry_edges;
//         net->layers[i].right_boundry_edges_delta = right_boundry_edges;
//         net->layers[i].bottom_boundry_edges_delta = bottom_boundry_edges;

//         net->layers[i].delta_in_h_with_boundry = end_y_coordinate - start_y_coordinate + 1; 
//         net->layers[i].delta_in_h_without_boundry = net->layers[i].delta_in_h_with_boundry - (net->layers[i].top_boundry_edges_delta + net->layers[i].bottom_boundry_edges_delta);

//         net->layers[i].delta_in_w_with_boundry = end_x_coordinate - start_x_coordinate + 1;
//         net->layers[i].delta_in_w_without_boundry = net->layers[i].delta_in_w_with_boundry - (net->layers[i].left_boundry_edges_delta + net->layers[i].right_boundry_edges_delta);

//         printf("Layer %d \n\n", i);
//         printf("DELTA H with boundry/without boundry = %d %d\n", net->layers[i].delta_in_h_with_boundry, net->layers[i].delta_in_h_without_boundry);
//         printf("DELTA W with boundry/without boundry = %d %d\n", net->layers[i].delta_in_w_with_boundry, net->layers[i].delta_in_w_without_boundry);
//         printf("Top boundry edges = %d\n", top_boundry_edges);
//         printf("Left boundry edges = %d\n", left_boundry_edges);
//         printf("Right boundry edges = %d\n", right_boundry_edges);
//         printf("Bottom boundry edges = %d\n\n", bottom_boundry_edges);
//         printf("Start x coordinate = %d\n", start_x_coordinate);
//         printf("Start y coordinte = %d\n", start_y_coordinate);
//         printf("End x coordinate = %d\n", end_x_coordinate);
//         printf("End y coordinate = %d\n\n", end_y_coordinate);
//     }
// }

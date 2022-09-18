#include "ftp.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

void* execute_devi(void* ptr){
     char *message;
     message = (char *) ptr;
     printf("%s \n", message);
}

extern sem_t filter_diverge;
extern sem_t filter_converge;

void* execute_dev(void* ptr){

    printf("Thread started exec_device\n\n");
    //while(1);

    device_ftp_args* ftp_args = (device_ftp_args*) ptr;

    int device_id = ftp_args->device_id;
    int im_width = ftp_args->im_width;
    int im_height = ftp_args->im_height;
    int filter_size = ftp_args->filter_size;
    float* image = ftp_args->image;
    float* delta = ftp_args->delta;
    float* boundry_top = ftp_args->boundry_top;
    float* boundry_bottom = ftp_args->boundry_bottom;
    float* boundry_left = ftp_args->boundry_left;
    float* boundry_right = ftp_args->boundry_right;
    float* boundry_top_right = ftp_args->boundry_top_right;
    float* boundry_top_left = ftp_args->boundry_top_left;
    float* boundry_bottom_right = ftp_args->boundry_bottom_right;
    float* boundry_bottom_left = ftp_args->boundry_bottom_left;
    int num_layers = ftp_args->num_layers;
    float* output = ftp_args->output;
    float*** SHARED_WEIGHT_UPDATES = ftp_args->SHARED_WEIGHT_UPDATES;

    int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
    int boundry_frames = unit_boundry*num_layers;
    int core_image_width = im_width - (2*boundry_frames);
    int core_image_height = im_height - (2*boundry_frames);

    network *net = calloc(1, sizeof(network));
    net->n = num_layers;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    net->layers[0] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[3] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[4] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);  
    net->layers[0].batch_normalize = 0;
    net->layers[1].batch_normalize = 0;
    net->layers[2].batch_normalize = 0;
    net->layers[3].batch_normalize = 0;
    net->layers[4].batch_normalize = 0;

    printf("%d, %d %d %d %d\n", im_height, im_width, core_image_height, core_image_width, boundry_frames);

    //net->layers[0].data = calloc(1, sizeof(float));
    net->input = calloc(im_width*im_height, sizeof(float));
    net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));
    net->inputs = im_width*im_height;

    fill_cpu(im_width*im_height, 0, net->layers[0].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[1].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[2].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[3].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[4].delta, 1);

    for (int i = 0; i < filter_size; ++i)
    {
        for (int j = 0; j < filter_size; ++j)
        {
            net->layers[0].weights[i*filter_size + j] = 0.1;
            net->layers[1].weights[i*filter_size + j] = 0.1;
            net->layers[2].weights[i*filter_size + j] = 0.1;
            net->layers[3].weights[i*filter_size + j] = 0.1;
            net->layers[4].weights[i*filter_size + j] = 0.1;
        }
    }

    //Core image

    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            net->input[(i+boundry_frames)*im_width + (j+boundry_frames)] = image[(i*core_image_width) + j];
        }
    }

    //Top left
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i*im_width) + j] = boundry_top_left[(i*boundry_frames) + j];
        }
    }

    //Top right
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i*im_width) + (j+im_width-boundry_frames)] = boundry_top_right[(i*boundry_frames) + j];
        }
    }

    //Bottom left
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+im_height-boundry_frames)*(im_width) + j] = boundry_bottom_left[(i*boundry_frames) + j];
        }
    }

    //Bottom right
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+im_height-boundry_frames)*im_width + (j+im_width-boundry_frames)] = boundry_bottom_right[(i*boundry_frames) + j];
        }
    }

    //Top

    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            net->input[(i*im_width) + (j+boundry_frames)] = boundry_top[(i*core_image_width) + j];
        }
    }

    //Left
    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+boundry_frames)*im_width + (j)] = boundry_left[(i*boundry_frames) + j];
        }
    }

    //Right
    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+boundry_frames)*im_width + (j+core_image_width+boundry_frames)] = boundry_right[(i*boundry_frames) + j];
        }
    }

    //Bottom
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            net->input[(i+core_image_height+boundry_frames)*im_width + (j+boundry_frames)] = boundry_bottom[(i*core_image_width) + j];
        }
    }

//###DEBUG


    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.2f ", net->input[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // printf("\n");
//###END DEBUG

    net->index = 0;
    forward_convolutional_layer(net->layers[0], *net);
    net->input = net->layers[0].output;
    net->index = 1;
    forward_convolutional_layer(net->layers[1], *net);
    net->input = net->layers[1].output;
    net->index = 2;
    forward_convolutional_layer(net->layers[2], *net);
    net->input = net->layers[2].output;
    net->index = 3;
    forward_convolutional_layer(net->layers[3], *net);
    net->input = net->layers[3].output;
    net->index = 4;
    forward_convolutional_layer(net->layers[4], *net);

//###DEBUG


    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].output[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
//###END DEBUG


    //fill_cpu(im_width*im_height, 0, net->layers[4].delta, 1);
    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[4].delta[i*im_width + j] = delta[i*im_width + j];
        }
    }
    for (int i = 0; i < im_height; ++i)
    {
        for (int j = 0; j < im_width; ++j)
        {
            net->layers[4].delta_with_boundry[i*im_width + j] = delta[i*im_width + j];
        }
    }

//###DEBUG


    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].delta[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
//###END DEBUG

    net->input = net->layers[3].output;
    net->delta = net->layers[3].delta_with_boundry;
    net->index = 4;
    backward_convolutional_layer_dist(net->layers[4], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[3].delta[i*im_width + j] = net->layers[3].delta_with_boundry[i*im_width + j];
        }
    }

//###DEBUG
    // for (int i = 0; i < 3; ++i)
    // {
    //     for (int j = 0; j < 3; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].weights[(i*3) + j]);
    //     }
    //     printf("\n");
    // }

    // for (int i = 0; i < filter_size; ++i)
    // {
    //     for (int j = 0; j < filter_size; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].weight_updates[(i*filter_size) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.4f ", net->layers[3].delta_with_boundry[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
//###END DEBUG

    net->input = net->layers[2].output;
    net->delta = net->layers[2].delta_with_boundry;
    net->index = 3;
    backward_convolutional_layer_dist(net->layers[3], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[2].delta[i*im_width + j] = net->layers[2].delta_with_boundry[i*im_width + j];
        }
    }

//###DEBUG


    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            printf("%.4f ", net->layers[3].weight_updates[(i*3) + j]);
        }
        printf("\n");
    }
    printf("\n");
//###END DEBUG

    net->input = net->layers[1].output;
    net->delta = net->layers[1].delta_with_boundry;
    net->index = 2;
    backward_convolutional_layer_dist(net->layers[2], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[1].delta[i*im_width + j] = net->layers[1].delta_with_boundry[i*im_width + j];
        }
    }

    net->input = net->layers[0].output;
    net->delta = net->layers[0].delta_with_boundry;
    net->index = 1;
    backward_convolutional_layer_dist(net->layers[1], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[0].delta[i*im_width + j] = net->layers[0].delta_with_boundry[i*im_width + j];
        }
    }

    //update_convolutional_layer(l, update_args a);

    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            output[i*core_image_width + j] = net->layers[0].delta_with_boundry[(i+boundry_frames)*im_width + j+boundry_frames];
        }
    }

    for (int i = 0; i < num_layers; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
               SHARED_WEIGHT_UPDATES [device_id][i][j*3 + k] = net->layers[i].weight_updates[(j*3) + k];
            }
        }   
    }
    sem_post(&filter_diverge);
    printf("DEVICE %d Partial sum computation complete\n", device_id);

    int sema_value;
    sem_getvalue(&filter_converge, &sema_value);
    
    sem_wait(&filter_converge);

    printf("DEVICE %d Partial sum merge complete\n", device_id);

    //Update weights here
    for (int i = 0; i < num_layers; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
              net->layers[i].weight_updates[(j*3) + k] = SHARED_WEIGHT_UPDATES [0][i][j*3 + k];
            }
        }   
    }

    update_args a = {
        1,
        0.005,
        1,
        1,
        1,
        0.5,
        0.5,
        2.0,
        1,
    };

    for (int i = 0; i < num_layers; ++i)
    {
        update_convolutional_layer(net->layers[i], a);
    }


    // for (int i = 0; i < num_layers; ++i)
    // {
    //     printf("Layer weights %d\n", i);
    //     for (int j = 0; j < 3; ++j)
    //     {
    //         for (int k = 0; k < 3; ++k)
    //         {
    //           printf("%.4f ", net->layers[i].weight_updates[(j*3) + k]);
    //         }
    //         printf("\n");
    //     }  
    //     printf("\n\n"); 
    // }

//### DEBUG
    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.2f ", net->layers[0].delta_with_boundry[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

//###END DEBUG


}

//1) layer_size_x_+1 = ceil(layer_Size_X_+1/TILES_X)
//2) pattern = filter_size + n*(stride) n = 0, 1, 2, 3....
//3) find largest n such that pattern <= layer_size_x -> this marks the last element idx in that tile
                                                        //feature map in x direction
                                                        //Rest (if any are ignored/passed onto next tile)
    //Let this n be called n_last_tile_x
    //Also, let the total elements needed for that tile filter_size + n*(stride) = total_tile_x_desired
    
//4) If tile_idx_x happens to be 0 or TILES_X - 1, subtract padding from total_tile_x_desired
        //Add 2*padding to total_tile_x_desired
        //total_tile_x_desired = total_tile_x_desired + 2*padding
    //If not an edge tile, 

//5) If device_tile_idx_x > 0,
        //get the start idx for convolution from device_tile_idx_x - 1/gateway;
        //call this conv_start_idx_tile_x (this is inclusive of the padding start)
        //if this start idx is < 0, get that many cols of edge data from device_tile_idx_x - 1 (or pad 0s if device tile idx x is 0)
            //and tile_x_desired - |device_tile_idx_x| from device_tile_idx_x + 1 (or pad 0s if last tile in X)
//  if this start idx is >= 0, get total_tile_x_desired - X_default_size - start idx from device_tile_idx_x + 1
void* execute_dev_v2(void* ptr){

    printf("Thread started exec_device\n\n");
    //while(1);

    device_ftp_args* ftp_args = (device_ftp_args*) ptr;

    int device_id = ftp_args->device_id;
    int im_width = ftp_args->im_width;
    int im_height = ftp_args->im_height;
    int filter_size = ftp_args->filter_size;
    float* image = ftp_args->image;
    float* delta = ftp_args->delta;
    float* boundry_top = ftp_args->boundry_top;
    float* boundry_bottom = ftp_args->boundry_bottom;
    float* boundry_left = ftp_args->boundry_left;
    float* boundry_right = ftp_args->boundry_right;
    float* boundry_top_right = ftp_args->boundry_top_right;
    float* boundry_top_left = ftp_args->boundry_top_left;
    float* boundry_bottom_right = ftp_args->boundry_bottom_right;
    float* boundry_bottom_left = ftp_args->boundry_bottom_left;
    int num_layers = ftp_args->num_layers;
    float* output = ftp_args->output;
    float*** SHARED_WEIGHT_UPDATES = ftp_args->SHARED_WEIGHT_UPDATES;

    int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
    int boundry_frames = unit_boundry*num_layers;
    int core_image_width = im_width - (2*boundry_frames);
    int core_image_height = im_height - (2*boundry_frames);

    network *net = calloc(1, sizeof(network));
    net->n = num_layers;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    net->layers[0] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[3] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[4] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);  
    net->layers[0].batch_normalize = 0;
    net->layers[1].batch_normalize = 0;
    net->layers[2].batch_normalize = 0;
    net->layers[3].batch_normalize = 0;
    net->layers[4].batch_normalize = 0;

    printf("%d, %d %d %d %d\n", im_height, im_width, core_image_height, core_image_width, boundry_frames);

    //net->layers[0].data = calloc(1, sizeof(float));
    net->input = calloc(im_width*im_height, sizeof(float));
    net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));
    net->inputs = im_width*im_height;

    fill_cpu(im_width*im_height, 0, net->layers[0].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[1].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[2].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[3].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[4].delta, 1);

    for (int i = 0; i < filter_size; ++i)
    {
        for (int j = 0; j < filter_size; ++j)
        {
            net->layers[0].weights[i*filter_size + j] = 0.1;
            net->layers[1].weights[i*filter_size + j] = 0.1;
            net->layers[2].weights[i*filter_size + j] = 0.1;
            net->layers[3].weights[i*filter_size + j] = 0.1;
            net->layers[4].weights[i*filter_size + j] = 0.1;
        }
    }

    //Core image

    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            net->input[(i+boundry_frames)*im_width + (j+boundry_frames)] = image[(i*core_image_width) + j];
        }
    }

    //Top left
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i*im_width) + j] = boundry_top_left[(i*boundry_frames) + j];
        }
    }

    //Top right
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i*im_width) + (j+im_width-boundry_frames)] = boundry_top_right[(i*boundry_frames) + j];
        }
    }

    //Bottom left
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+im_height-boundry_frames)*(im_width) + j] = boundry_bottom_left[(i*boundry_frames) + j];
        }
    }

    //Bottom right
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+im_height-boundry_frames)*im_width + (j+im_width-boundry_frames)] = boundry_bottom_right[(i*boundry_frames) + j];
        }
    }

    //Top

    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            net->input[(i*im_width) + (j+boundry_frames)] = boundry_top[(i*core_image_width) + j];
        }
    }

    //Left
    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+boundry_frames)*im_width + (j)] = boundry_left[(i*boundry_frames) + j];
        }
    }

    //Right
    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+boundry_frames)*im_width + (j+core_image_width+boundry_frames)] = boundry_right[(i*boundry_frames) + j];
        }
    }

    //Bottom
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            net->input[(i+core_image_height+boundry_frames)*im_width + (j+boundry_frames)] = boundry_bottom[(i*core_image_width) + j];
        }
    }

//###DEBUG


    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.2f ", net->input[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // printf("\n");
//###END DEBUG

    net->index = 0;
    forward_convolutional_layer(net->layers[0], *net);
    net->input = net->layers[0].output;
    net->index = 1;
    forward_convolutional_layer(net->layers[1], *net);
    net->input = net->layers[1].output;
    net->index = 2;
    forward_convolutional_layer(net->layers[2], *net);
    net->input = net->layers[2].output;
    net->index = 3;
    forward_convolutional_layer(net->layers[3], *net);
    net->input = net->layers[3].output;
    net->index = 4;
    forward_convolutional_layer(net->layers[4], *net);

//###DEBUG


    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].output[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
//###END DEBUG


    //fill_cpu(im_width*im_height, 0, net->layers[4].delta, 1);
    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[4].delta[i*im_width + j] = delta[i*im_width + j];
        }
    }
    for (int i = 0; i < im_height; ++i)
    {
        for (int j = 0; j < im_width; ++j)
        {
            net->layers[4].delta_with_boundry[i*im_width + j] = delta[i*im_width + j];
        }
    }

//###DEBUG


    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].delta[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
//###END DEBUG

    net->input = net->layers[3].output;
    net->delta = net->layers[3].delta_with_boundry;
    net->index = 4;
    backward_convolutional_layer_dist(net->layers[4], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[3].delta[i*im_width + j] = net->layers[3].delta_with_boundry[i*im_width + j];
        }
    }

//###DEBUG
    // for (int i = 0; i < 3; ++i)
    // {
    //     for (int j = 0; j < 3; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].weights[(i*3) + j]);
    //     }
    //     printf("\n");
    // }

    // for (int i = 0; i < filter_size; ++i)
    // {
    //     for (int j = 0; j < filter_size; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].weight_updates[(i*filter_size) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.4f ", net->layers[3].delta_with_boundry[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
//###END DEBUG

    net->input = net->layers[2].output;
    net->delta = net->layers[2].delta_with_boundry;
    net->index = 3;
    backward_convolutional_layer_dist(net->layers[3], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[2].delta[i*im_width + j] = net->layers[2].delta_with_boundry[i*im_width + j];
        }
    }

//###DEBUG


    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            printf("%.4f ", net->layers[3].weight_updates[(i*3) + j]);
        }
        printf("\n");
    }
    printf("\n");
//###END DEBUG

    net->input = net->layers[1].output;
    net->delta = net->layers[1].delta_with_boundry;
    net->index = 2;
    backward_convolutional_layer_dist(net->layers[2], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[1].delta[i*im_width + j] = net->layers[1].delta_with_boundry[i*im_width + j];
        }
    }

    net->input = net->layers[0].output;
    net->delta = net->layers[0].delta_with_boundry;
    net->index = 1;
    backward_convolutional_layer_dist(net->layers[1], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[0].delta[i*im_width + j] = net->layers[0].delta_with_boundry[i*im_width + j];
        }
    }

    //update_convolutional_layer(l, update_args a);

    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            output[i*core_image_width + j] = net->layers[0].delta_with_boundry[(i+boundry_frames)*im_width + j+boundry_frames];
        }
    }

    for (int i = 0; i < num_layers; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
               SHARED_WEIGHT_UPDATES [device_id][i][j*3 + k] = net->layers[i].weight_updates[(j*3) + k];
            }
        }   
    }
    sem_post(&filter_diverge);
    printf("DEVICE %d Partial sum computation complete\n", device_id);

    int sema_value;
    sem_getvalue(&filter_converge, &sema_value);
    
    sem_wait(&filter_converge);

    printf("DEVICE %d Partial sum merge complete\n", device_id);

    //Update weights here
    for (int i = 0; i < num_layers; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
              net->layers[i].weight_updates[(j*3) + k] = SHARED_WEIGHT_UPDATES [0][i][j*3 + k];
            }
        }   
    }

    update_args a = {
        1,
        0.005,
        1,
        1,
        1,
        0.5,
        0.5,
        2.0,
        1,
    };

    for (int i = 0; i < num_layers; ++i)
    {
        update_convolutional_layer(net->layers[i], a);
    }


    // for (int i = 0; i < num_layers; ++i)
    // {
    //     printf("Layer weights %d\n", i);
    //     for (int j = 0; j < 3; ++j)
    //     {
    //         for (int k = 0; k < 3; ++k)
    //         {
    //           printf("%.4f ", net->layers[i].weight_updates[(j*3) + k]);
    //         }
    //         printf("\n");
    //     }  
    //     printf("\n\n"); 
    // }

//### DEBUG
    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.2f ", net->layers[0].delta_with_boundry[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

//###END DEBUG


}




void* execute_dev_gateway(void* ptr){

    printf("Thread started exec_device\n\n");
    //while(1);

    device_ftp_args* ftp_args = (device_ftp_args*) ptr;

    int NUM_DEVICES = ftp_args->NUM_DEVICES;
    int device_id = ftp_args->device_id;
    int im_width = ftp_args->im_width;
    int im_height = ftp_args->im_height;
    int filter_size = ftp_args->filter_size;
    float* image = ftp_args->image;
    float* delta = ftp_args->delta;
    float* boundry_top = ftp_args->boundry_top;
    float* boundry_bottom = ftp_args->boundry_bottom;
    float* boundry_left = ftp_args->boundry_left;
    float* boundry_right = ftp_args->boundry_right;
    float* boundry_top_right = ftp_args->boundry_top_right;
    float* boundry_top_left = ftp_args->boundry_top_left;
    float* boundry_bottom_right = ftp_args->boundry_bottom_right;
    float* boundry_bottom_left = ftp_args->boundry_bottom_left;
    int num_layers = ftp_args->num_layers;
    float* output = ftp_args->output;
    float*** SHARED_WEIGHT_UPDATES = ftp_args->SHARED_WEIGHT_UPDATES;

    int unit_boundry = ((filter_size & 0x1) == 1) ? ((filter_size - 1)/2) : (filter_size/2);
    int boundry_frames = unit_boundry*num_layers;
    int core_image_width = im_width - (2*boundry_frames);
    int core_image_height = im_height - (2*boundry_frames);

    network *net = calloc(1, sizeof(network));
    net->n = num_layers;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    net->layers[0] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[3] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[4] = make_convolutional_layer(1, im_width, im_height, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);  
    net->layers[0].batch_normalize = 0;
    net->layers[1].batch_normalize = 0;
    net->layers[2].batch_normalize = 0;
    net->layers[3].batch_normalize = 0;
    net->layers[4].batch_normalize = 0;

    printf("%d, %d %d %d %d\n", im_height, im_width, core_image_height, core_image_width, boundry_frames);

    //net->layers[0].data = calloc(1, sizeof(float));
    net->input = calloc(im_width*im_height, sizeof(float));
    net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));
    net->inputs = im_width*im_height;

    fill_cpu(im_width*im_height, 0, net->layers[0].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[1].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[2].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[3].delta, 1);
    fill_cpu(im_width*im_height, 0, net->layers[4].delta, 1);

    for (int i = 0; i < filter_size; ++i)
    {
        for (int j = 0; j < filter_size; ++j)
        {
            net->layers[0].weights[i*filter_size + j] = 0.1;
            net->layers[1].weights[i*filter_size + j] = 0.1;
            net->layers[2].weights[i*filter_size + j] = 0.1;
            net->layers[3].weights[i*filter_size + j] = 0.1;
            net->layers[4].weights[i*filter_size + j] = 0.1;
        }
    }

    //Core image

    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            net->input[(i+boundry_frames)*im_width + (j+boundry_frames)] = image[(i*core_image_width) + j];
        }
    }

    //Top left
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i*im_width) + j] = boundry_top_left[(i*boundry_frames) + j];
        }
    }

    //Top right
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i*im_width) + (j+im_width-boundry_frames)] = boundry_top_right[(i*boundry_frames) + j];
        }
    }

    //Bottom left
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+im_height-boundry_frames)*(im_width) + j] = boundry_bottom_left[(i*boundry_frames) + j];
        }
    }

    //Bottom right
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+im_height-boundry_frames)*im_width + (j+im_width-boundry_frames)] = boundry_bottom_right[(i*boundry_frames) + j];
        }
    }

    //Top

    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            net->input[(i*im_width) + (j+boundry_frames)] = boundry_top[(i*core_image_width) + j];
        }
    }

    //Left
    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+boundry_frames)*im_width + (j)] = boundry_left[(i*boundry_frames) + j];
        }
    }

    //Right
    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < boundry_frames; ++j)
        {
            net->input[(i+boundry_frames)*im_width + (j+core_image_width+boundry_frames)] = boundry_right[(i*boundry_frames) + j];
        }
    }

    //Bottom
    for (int i = 0; i < boundry_frames; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            net->input[(i+core_image_height+boundry_frames)*im_width + (j+boundry_frames)] = boundry_bottom[(i*core_image_width) + j];
        }
    }

//###DEBUG


    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.2f ", net->input[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // printf("\n");
//###END DEBUG

    net->index = 0;
    forward_convolutional_layer(net->layers[0], *net);
    net->input = net->layers[0].output;
    net->index = 1;
    forward_convolutional_layer(net->layers[1], *net);
    net->input = net->layers[1].output;
    net->index = 2;
    forward_convolutional_layer(net->layers[2], *net);
    net->input = net->layers[2].output;
    net->index = 3;
    forward_convolutional_layer(net->layers[3], *net);
    net->input = net->layers[3].output;
    net->index = 4;
    forward_convolutional_layer(net->layers[4], *net);

//###DEBUG


    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].output[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
//###END DEBUG


    //fill_cpu(im_width*im_height, 0, net->layers[4].delta, 1);
    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[4].delta[i*im_width + j] = delta[i*im_width + j];
        }
    }
    for (int i = 0; i < im_height; ++i)
    {
        for (int j = 0; j < im_width; ++j)
        {
            net->layers[4].delta_with_boundry[i*im_width + j] = delta[i*im_width + j];
        }
    }

//###DEBUG


    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].delta[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
//###END DEBUG

    net->input = net->layers[3].output;
    net->delta = net->layers[3].delta_with_boundry;
    net->index = 4;
    backward_convolutional_layer_dist(net->layers[4], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[3].delta[i*im_width + j] = net->layers[3].delta_with_boundry[i*im_width + j];
        }
    }

//###DEBUG
    // for (int i = 0; i < 3; ++i)
    // {
    //     for (int j = 0; j < 3; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].weights[(i*3) + j]);
    //     }
    //     printf("\n");
    // }

    // for (int i = 0; i < filter_size; ++i)
    // {
    //     for (int j = 0; j < filter_size; ++j)
    //     {
    //         printf("%.4f ", net->layers[4].weight_updates[(i*filter_size) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.4f ", net->layers[3].delta_with_boundry[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
//###END DEBUG

    net->input = net->layers[2].output;
    net->delta = net->layers[2].delta_with_boundry;
    net->index = 3;
    backward_convolutional_layer_dist(net->layers[3], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[2].delta[i*im_width + j] = net->layers[2].delta_with_boundry[i*im_width + j];
        }
    }

//###DEBUG


    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            printf("%.4f ", net->layers[3].weight_updates[(i*3) + j]);
        }
        printf("\n");
    }
    printf("\n");
//###END DEBUG

    net->input = net->layers[1].output;
    net->delta = net->layers[1].delta_with_boundry;
    net->index = 2;
    backward_convolutional_layer_dist(net->layers[2], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[1].delta[i*im_width + j] = net->layers[1].delta_with_boundry[i*im_width + j];
        }
    }

    net->input = net->layers[0].output;
    net->delta = net->layers[0].delta_with_boundry;
    net->index = 1;
    backward_convolutional_layer_dist(net->layers[1], *net);

    for (int i = boundry_frames; i < (im_height-boundry_frames); ++i)
    {
        for (int j = boundry_frames; j < (im_width-boundry_frames); ++j)
        {
            net->layers[0].delta[i*im_width + j] = net->layers[0].delta_with_boundry[i*im_width + j];
        }
    }

    //update_convolutional_layer(l, update_args a);

    for (int i = 0; i < core_image_height; ++i)
    {
        for (int j = 0; j < core_image_width; ++j)
        {
            output[i*core_image_width + j] = net->layers[0].delta_with_boundry[(i+boundry_frames)*im_width + j+boundry_frames];
        }
    }

    for (int i = 0; i < num_layers; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
               SHARED_WEIGHT_UPDATES [device_id][i][j*3 + k] = net->layers[i].weight_updates[(j*3) + k];
            }
        }   
    }

    sem_wait(&filter_diverge);
    sem_wait(&filter_diverge);
    sem_wait(&filter_diverge);
    //sem_wait(&filter_diverge);
    printf("GATEWAY: Threads completed filter partial sum computation\n");

    for (int i = 0; i < filter_size; ++i)
    {
        for (int j = 0; j < filter_size; ++j)
        {
            for (int k = 0; k < num_layers; ++k)
            {
                for (int l = 1; l < NUM_DEVICES; ++l)
                {
                   SHARED_WEIGHT_UPDATES [0][k][i*3 + j] += SHARED_WEIGHT_UPDATES[l][k][i*3 + j];
                }
            }
        }   
    }

   // sleep(5);
    sem_post(&filter_converge);
    sem_post(&filter_converge);
    sem_post(&filter_converge);
    printf("GATEWAY: Finished summing weights\n");

    int sema_value;
    sem_getvalue(&filter_converge, &sema_value);
    //printf("GATEWAY DEBUG %d\n", sema_value);

    for (int i = 0; i < filter_size; ++i)
    {
        for (int j = 0; j < filter_size; ++j)
        {
            printf("%.4f ", SHARED_WEIGHT_UPDATES [0][3][i*3 + j]);
        }   
        printf("\n");
    }

//### DEBUG
    // for (int i = 0; i < im_height; ++i)
    // {
    //     for (int j = 0; j < im_width; ++j)
    //     {
    //         printf("%.2f ", net->layers[0].delta_with_boundry[(i*im_width) + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");

//###END DEBUG


}
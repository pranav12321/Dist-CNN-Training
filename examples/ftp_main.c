#include "ftp.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include "gemm.h"
#include "col2im.h"

void *print_message_function( void *ptr );

sem_t filter_diverge;
sem_t filter_converge;

int main(){

    int INPUT_WIDTH = 50;
    int INPUT_HEIGHT = 50;

    int NUM_TILES_X = 2;
    int NUM_TILES_Y = 2;

    int filter_size = 3;
    int num_layers = 5;

    network*** SHARED_NETWORKS = calloc(NUM_TILES_Y, sizeof(network**));

    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        SHARED_NETWORKS[i] = calloc(NUM_TILES_X, sizeof(network*));
    }

    float*** SHARED_INPUT_IMAGES = calloc(NUM_TILES_Y, sizeof(float**));

    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        SHARED_INPUT_IMAGES[i] = calloc(NUM_TILES_X, sizeof(float*));
    }

    float*** SHARED_EXP_DELTAS = calloc(NUM_TILES_Y, sizeof(float**));

    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        SHARED_EXP_DELTAS[i] = calloc(NUM_TILES_X, sizeof(float*));
    }

    for (int i = 0; i < NUM_TILES_Y; ++i)
    {
        for (int j = 0; j < NUM_TILES_X; ++j)
        {
            SHARED_NETWORKS[i][j] = calloc(1, sizeof(network));
            network* net = SHARED_NETWORKS[i][j];

            net->n = num_layers;
            net->layers = calloc(net->n, sizeof(layer));
            net->seen = calloc(1, sizeof(size_t));
            net->t    = calloc(1, sizeof(int));
            net->cost = calloc(1, sizeof(float));

            net->layers[0] = make_convolutional_layer(1, 27, 27, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0);
            net->layers[1] = make_convolutional_layer(1, 27, 27, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0);
            net->layers[2] = make_convolutional_layer(1, 27, 27, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0);
            net->layers[3] = make_convolutional_layer(1, 27, 27, 1, 1, 1, filter_size, 2, 0, RELU, 0, 0, 0, 0);
            net->layers[4] = make_convolutional_layer(1, 15, 15, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0);  

            for (int i_f = 0; i_f < filter_size; ++i_f)
            {
                for (int j_f = 0; j_f < filter_size; ++j_f)
                {
                    net->layers[0].weights[i_f*filter_size + j_f] = 0.1;
                    net->layers[1].weights[i_f*filter_size + j_f] = 0.1;
                    net->layers[2].weights[i_f*filter_size + j_f] = 0.1;
                    net->layers[3].weights[i_f*filter_size + j_f] = 0.1;
                    net->layers[4].weights[i_f*filter_size + j_f] = 0.1;
                }
            }

            // fill_cpu(729, 0, net->layers[0].delta, 1);
            // fill_cpu(729, 0, net->layers[1].delta, 1);
            // fill_cpu(729, 0, net->layers[2].delta, 1);
            // fill_cpu(225, 0, net->layers[3].delta, 1);
            // fill_cpu(169, 0, net->layers[4].delta, 1);

            net->workspace = calloc(net->layers[0].workspace_size*5, sizeof(float));
            net->inputs = 729;

            SHARED_INPUT_IMAGES[i][j] = calloc((INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), sizeof(float));
            fill_cpu((INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), 1, SHARED_INPUT_IMAGES[i][j], 1);

            SHARED_EXP_DELTAS[i][j] = calloc(net->layers[4].outputs, sizeof(float));
            fill_cpu(net->layers[4].outputs, 1, SHARED_EXP_DELTAS[i][j], 1);

        }
    }

    for (int i = 0; i < 13; ++i)
    {
        SHARED_EXP_DELTAS[0][1][i*13 + 12] = 0.0;
        SHARED_EXP_DELTAS[1][1][i*13 + 12] = 0.0;
        SHARED_EXP_DELTAS[1][0][(12)*13 + i] = 0.0;
        SHARED_EXP_DELTAS[1][1][(12)*13 + i] = 0.0;

    }

            for (size_t i = 0; i < 13; i++)
            {
                for (size_t j = 0; j < 13; j++)
                {
                    printf("%.2f ", SHARED_EXP_DELTAS[1][1][i*13 + j]);
                }
                printf("\n");
            }
    int tile_x_dim = (INPUT_WIDTH + (NUM_TILES_X - 1)) / NUM_TILES_X;
    int tile_y_dim = (INPUT_HEIGHT + (NUM_TILES_Y - 1)) / NUM_TILES_Y;
 

    //FORWARD

    for (int l = 0; l < num_layers; ++l)
    {
        for (int i = 0; i < NUM_TILES_Y; ++i)
        {
            for (int j = 0; j < NUM_TILES_X; ++j)
            {
                device_ftp_args_v2 ftp_args = {
                    NUM_TILES_X,
                    NUM_TILES_Y,

                    j,
                    i,

                    INPUT_WIDTH,
                    INPUT_HEIGHT,

                    SHARED_NETWORKS[i][j],
                    filter_size,
                    SHARED_INPUT_IMAGES[i][j],
                    NULL,
                    num_layers,
                    NULL,
                    SHARED_NETWORKS,
                    SHARED_INPUT_IMAGES,
                    SHARED_EXP_DELTAS,
                    NULL,
                };

                execute_dev_v2_forward(&ftp_args, l);
                forward_convolutional_layer(SHARED_NETWORKS[i][j]->layers[l], *SHARED_NETWORKS[i][j]);
            }
        }
    }

    //BACKWARD

    for (int l = num_layers - 1; l >= 1; --l)
    {
        for (int i = 0; i < NUM_TILES_Y; ++i)
        {
            for (int j = 0; j < NUM_TILES_X; ++j)
            {
                device_ftp_args_v2 ftp_args = {
                    NUM_TILES_X,
                    NUM_TILES_Y,

                    j,
                    i,

                    INPUT_WIDTH,
                    INPUT_HEIGHT,

                    SHARED_NETWORKS[i][j],
                    filter_size,
                    SHARED_INPUT_IMAGES[i][j],
                    NULL,
                    num_layers,
                    NULL,
                    SHARED_NETWORKS,
                    SHARED_INPUT_IMAGES,
                    SHARED_EXP_DELTAS,
                    NULL,
                };

                execute_dev_v2_backward(&ftp_args, l);
                //forward_convolutional_layer(SHARED_NETWORKS[i][j]->layers[l], *SHARED_NETWORKS[i][j]);
            }
        }
    }

    // for (int i = 0; i < 13; ++i)
    // {
    //     for (int j = 0; j < 13; ++j)
    //     {
    //         printf("%.2f ", SHARED_NETWORKS[1][0]->layers[4].output[i*13 + j]);
    //     }
    //     printf("\n");
    // }
    printf("\n");
    printf("\n");

    for (int i = 0; i < 13; ++i)
    {
        for (int j = 0; j < 13; ++j)
        {
            printf("%.2f ", SHARED_NETWORKS[0][0]->layers[3].delta[i*13 + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");

    for (int i = 0; i < 13; ++i)
    {
        for (int j = 0; j < 13; ++j)
        {
            printf("%.2f ", SHARED_NETWORKS[0][1]->layers[3].delta[i*13 + j]);
        }
        printf("\n");
    }


    printf("\n");
    printf("\n");
    for (int i = 0; i < 13; ++i)
    {
        for (int j = 0; j < 13; ++j)
        {
            printf("%.2f ", SHARED_NETWORKS[1][0]->layers[3].delta[i*13 + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");

    for (int i = 0; i < 13; ++i)
    {
        for (int j = 0; j < 13; ++j)
        {
            printf("%.2f ", SHARED_NETWORKS[1][1]->layers[3].delta[i*13 + j]);
        }
        printf("\n");
    }
    printf("\n");
    printf("\n");
    // for (int i = 0; i < 26; ++i)
    // {
    //     for (int j = 0; j < 26; ++j)
    //     {
    //         printf("%.2f ", SHARED_NETWORKS[0][0]->layers[2].delta[i*26 + j]);
    //     }
    //     printf("\n");
    // }
    // printf("\n");
    // printf("\n");
    // for (int i = 0; i < 26; ++i)
    // {
    //     for (int j = 0; j < 26; ++j)
    //     {
    //         printf("%.2f ", SHARED_NETWORKS[0][1]->layers[2].delta[i*26 + j]);
    //     }
    //     printf("\n");
    // }






















    network* net = calloc(1, sizeof(network));

    net->n = num_layers;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    net->layers[0] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[3] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 2, 1, RELU, 0, 0, 0, 0);
    net->layers[4] = make_convolutional_layer(1, 25, 25, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);  

    for (int i_f = 0; i_f < filter_size; ++i_f)
    {
        for (int j_f = 0; j_f < filter_size; ++j_f)
        {
            net->layers[0].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[1].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[2].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[3].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[4].weights[i_f*filter_size + j_f] = 0.1;
        }
    }

    fill_cpu(2500, 0, net->layers[0].delta, 1);
    fill_cpu(2500, 0, net->layers[1].delta, 1);
    fill_cpu(2500, 0, net->layers[2].delta, 1);
    fill_cpu(625, 0, net->layers[3].delta, 1);
    fill_cpu(625, 1, net->layers[4].delta, 1);

    net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));
    net->inputs = 2500;

    net->input = calloc(2500, sizeof(float));
    fill_cpu(2500, 1, net->input, 1);

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


    // for (int i = 0; i < 25; ++i)
    // {
    //     for (int j = 0; j < 25; ++j)
    //     {
    //         printf("%.2f ", net->layers[4].output[i*25 + j]);
    //     }
    //     printf("\n");
    // }

    net->input = net->layers[3].output;
    net->delta = net->layers[3].delta;
    net->index = 4;
    backward_convolutional_layer(net->layers[4], *net);

    net->input = net->layers[2].output;
    net->delta = net->layers[2].delta;
    net->index = 3;
    backward_convolutional_layer(net->layers[3], *net);


    for (int i = 0; i < 25; ++i)
    {
        for (int j = 0; j < 25; ++j)
        {
            printf("%.2f ", net->layers[3].delta[(i)*25 + j]);
        }
        printf("\n");
    }

    // printf("\n");
    // printf("\n");

    // for (int i = 0; i < 50; ++i)
    // {
    //     for (int j = 0; j < 50; ++j)
    //     {
    //         printf("%.2f ", net->layers[2].delta[(i)*50 + j]);
    //     }
    //     printf("\n");
    // }
    // for (int i = 0; i < 13; ++i)
    // {
    //     for (int j = 0; j < 13; ++j)
    //     {
    //         printf("%.2f ", net->layers[3].delta[i*25 + j]);
    //     }
    //     printf("\n");
    // }
}










#include "maxpool_layer.h"

int main_maxpool(){

    network *net = calloc(1, sizeof(network));
    net->n = 5;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    //(int batch, int h, int w, int c, int size, int stride, int padding)
    net->layers[0] = make_maxpool_layer(1, 10, 10, 1, 3, 3, 5); 
    net->layers[0].batch_normalize = 0;
    net->input = calloc(100, sizeof(float));
    net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));

    float* image_merged = calloc(10*10, sizeof(float));
    fill_cpu(100, 1, image_merged, 1);

    for (int i = 0; i < 10; ++i)
    {
        for (int j = 0; j < 10; ++j)
        {
            net->input[i*10 + j] = image_merged[i*10 + j];
        }
    }

    for (int i = 0; i < 10; ++i)
    {
        net->input[i*10 + 9] = -1;
        net->input[90 + i] = -1;
    }


    forward_maxpool_layer(net->layers[0], *net);
    int out_size = net->layers[0].out_h * net->layers[0].out_w * net->layers[0].out_c * net->layers[0].batch;

    for (int i = 0; i < net->layers[0].out_h; ++i)
    {
        for (int j = 0; j < net->layers[0].out_w; ++j)
        {
            printf("%.4f ", net->layers[0].output[(i*net->layers[0].out_w) + j]);
        }
        printf("\n");
    }
}


int main_stride(){


    int filter_size = 3;
    int num_layers = 5;
    int num_devices = 4;

    network *net = calloc(1, sizeof(network));
    net->n = 2;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    //(int batch, int h, int w, int c, int size, int stride, int padding)
    net->layers[0] = make_convolutional_layer(1, 6, 6, 1, 1, 1, filter_size, 3, 0, RELU, 1, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, 2, 2, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[0].batch_normalize = 0;
    net->layers[1].batch_normalize = 0;

    net->input = calloc(64, sizeof(float));
    net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));

    for (int i = 0; i < filter_size; ++i)
    {
        for (int j = 0; j < filter_size; ++j)
        {
            net->layers[0].weights[i*filter_size + j] = 1.0;
            net->layers[1].weights[i*filter_size + j] = 1.0;
        }
    }

    float* image_merged = calloc(8*8, sizeof(float));
    fill_cpu(64, 1, image_merged, 1);

    for (int i = 0; i < 8; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            net->input[i*8 + j] = image_merged[i*8 + j];
        }
    }

    net->index = 0;
    forward_convolutional_layer(net->layers[0], *net);

    int out_size = net->layers[0].out_h * net->layers[0].out_w * net->layers[0].out_c * net->layers[0].batch;

    for (int i = 0; i < net->layers[0].out_h; ++i)
    {
        for (int j = 0; j < net->layers[0].out_w; ++j)
        {
            printf("%.4f ", net->layers[0].output[(i*net->layers[0].out_w) + j]);
        }
        printf("\n");
    }

    // net->input = net->layers[0].output;
    // net->index = 1;
    // forward_convolutional_layer(net->layers[1], *net);

    // for (int i = 0; i < net->layers[1].out_h; ++i)
    // {
    //     for (int j = 0; j < net->layers[1].out_w; ++j)
    //     {
    //         printf("%.4f ", net->layers[1].output[(i*net->layers[1].out_w) + j]);
    //     }
    //     printf("\n");
    // }








    float* delta = calloc(8*8, sizeof(float));

    //fill_cpu(4, 0, net->layers[0].delta, 1);

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 2; ++j)
        {
            net->layers[0].delta[i*2 + j] =  1.0;
        }
    }

    // for (int i = 0; i < 4; ++i)
    // {
    //     for (int j = 0; j < 4; ++j)
    //     {
    //         printf("%.1f ", net->layers[1].delta[(i*4) + j]);
    //     }
    //     printf("\n");
    // }

    net->input = image_merged;
    net->delta = delta;
    net->index = 0;
    backward_convolutional_layer(net->layers[0], *net);

    // net->input = net->layers[0].output;
    // net->delta = delta;
    // net->index = 0;
    // backward_convolutional_layer(net->layers[0], *net);

    // for (int i = 0; i < 4; ++i)
    // {
    //     for (int j = 0; j < 4; ++j)
    //     {
    //         printf("%.1f ", net->layers[0].delta[(i*4) + j]);
    //     }
    //     printf("\n");
    // }

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            printf("%.2f ", delta[(i*6) + j]);
        }
        printf("\n");
    }

}


int main_col(){
    int m = 1;//l.n/l.groups;
    int n = 9;//l.size*l.size*l.c/l.groups;
    int k = 225;//l.out_w*l.out_h;

    float* delta = calloc(15*15, sizeof(float));
    for (int i = 0; i < 15; ++i)
    {
        for (int j = 0; j < 15; ++j)
        {
            delta[i*15 + j] = 1.0;
        }
    }

    float* weights = calloc(3*3, sizeof(float));
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            weights[i*3 + j] = 1.0;
        }
    }

    float* delta_prev = calloc(17*17, sizeof(float));

    float* a = weights;//l.weights + j*l.nweights/l.groups;
    float* b = delta;//l.delta_with_boundry + (i*l.groups + j)*m*k;
    float* c = calloc(225*15, sizeof(float));
    fill_cpu(225*15, 0, c, 1);
    //; //net.workspace

    // float *im  = net.input + (i*l.groups + j)*l.c/l.groups*l.h*l.w;
    // float *imd = net.delta + (i*l.groups + j)*l.c/l.groups*l.h*l.w;

    //gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);
    gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);
    //col2im_cpu(net.workspace, l.c/l.groups, l.h, l.w, l.size, l.stride, l.pad, imd);
    col2im_cpu_ftp_version(c, 1, 13, 13, 15, 15, 3, 1, 0, delta_prev);

    for (int i = 0; i < 13; ++i)
    {
        for (int j = 0; j < 13; ++j)
        {
            printf("%.2f ", delta_prev[(i*13) + j]);
        }
        printf("\n");
    }

}
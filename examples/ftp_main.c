#include "ftp.h"
#include "fused.h"
#include "fused_convolution.h"

#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"
#include "maxpool_layer.h"


#define LAYER_SIZE 608
#define FILTER_SIZE 32

int main_yolo(){
    //make_convolutional_layer(int batch, int h,
    // int w, int c, int n, int groups, int size, int stride, int padding,
    // ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)

    network* net = calloc(1, sizeof(network));//SHARED_NETWORKS[i][j];

    net->n = 9;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    int filter_size = 3;
    int num_layers = 16;
    int unit_boundry = 1;

    //yolo v2
    net->layers[0] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, 3, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    //(int batch, int h, int w, int c, int size, int stride, int padding)
    net->layers[1] = make_maxpool_layer(1, LAYER_SIZE, LAYER_SIZE, 32, 2, 2, 0); 
    //net->layers[1] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, 32, FILTER_SIZE, 1, 3, 2, 1, LEAKY, 0, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, LAYER_SIZE/2, LAYER_SIZE/2, 32, 64, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    net->layers[3] = make_maxpool_layer(1, LAYER_SIZE/2, LAYER_SIZE/2, 64, 2, 2, 0);
    //net->layers[3] = make_convolutional_layer(1, LAYER_SIZE/2, LAYER_SIZE/2, 64, 64, 1, 3, 2, 1, LEAKY, 0, 0, 0, 0);
    net->layers[4] = make_convolutional_layer(1, LAYER_SIZE/4, LAYER_SIZE/4, 64, 128, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);


    // net->layers[1] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    // net->layers[2] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    // net->layers[3] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    // net->layers[4] = make_convolutional_layer(1, LAYER_SIZE, LAYER_SIZE, FILTER_SIZE, FILTER_SIZE, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    net->layers[5] = make_convolutional_layer(1, LAYER_SIZE/4, LAYER_SIZE/4, 128, 64, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);
    net->layers[6] = make_convolutional_layer(1, LAYER_SIZE/4, LAYER_SIZE/4, 64, 128, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);


    net->layers[7] = make_maxpool_layer(1, LAYER_SIZE/4, LAYER_SIZE/4, 128, 2, 2, 0);
    //net->layers[7] = make_convolutional_layer(1, LAYER_SIZE/4, LAYER_SIZE/4, 128, 128, 1, 3, 2, 1, LEAKY, 0, 0, 0, 0);


    net->layers[8] = make_convolutional_layer(1, LAYER_SIZE/8, LAYER_SIZE/8, 128, 256, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[9] = make_convolutional_layer(1, LAYER_SIZE/8, LAYER_SIZE/8, 256, 128, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    net->layers[10] = make_convolutional_layer(1, LAYER_SIZE/8, LAYER_SIZE/8, 128, 256, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[11] = make_maxpool_layer(1, LAYER_SIZE/8, LAYER_SIZE/8, 256, 2, 2, 0);


    net->layers[12] = make_convolutional_layer(1, LAYER_SIZE/16, LAYER_SIZE/16, 256, 512, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[13] = make_convolutional_layer(1, LAYER_SIZE/16, LAYER_SIZE/16, 512, 256, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    net->layers[14] = make_convolutional_layer(1, LAYER_SIZE/16, LAYER_SIZE/16, 256, 512, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[15] = make_convolutional_layer(1, LAYER_SIZE/16, LAYER_SIZE/16, 512, 256, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    // net->layers[16] = make_convolutional_layer(1, 38/2, 38/2, 256, 512, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[17] = make_maxpool_layer(1, 38/2, 38/2, 512, 2, 2, 0);

    // net->layers[18] = make_convolutional_layer(1, 19/2, 19/2, 512, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[19] = make_convolutional_layer(1, 19/2, 19/2, 1024, 512, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    // net->layers[20] = make_convolutional_layer(1, 19/2, 19/2, 512, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[21] = make_convolutional_layer(1, 19/2, 19/2, 1024, 512, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    // net->layers[22] = make_convolutional_layer(1, 19/2, 19/2, 512, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[23] = make_convolutional_layer(1, 19/2, 19/2, 1024, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[24] = make_convolutional_layer(1, 19/2, 19/2, 1024, 1024, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);
    // net->layers[2] = make_convolutional_layer(1, 24, 24, 1, 1, 1, filter_size, 2, 1, RELU, 0, 0, 0, 0);
    // net->layers[3] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    // net->layers[4] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    // net->layers[5] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 2, 1, RELU, 0, 0, 0, 0);
    // net->layers[6] = make_convolutional_layer(1, 6, 6, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    // net->layers[7] = make_convolutional_layer(1, 6, 6, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0); 


    for (int l = 0; l < net->n; ++l)
    {
        if(net->layers[l].type == CONVOLUTIONAL){

            int filter_size = net->layers[l].size;
            int num_filters = net->layers[l].n;
            int num_channels = net->layers[l].c;

            for (int i = 0; i < (filter_size*filter_size*num_filters*num_channels); ++i)
            {

                net->layers[l].weights[i] = 0.01;
            }        


        }
    }



    net->workspace = calloc(net->layers[0].workspace_size*5, sizeof(float));
    net->inputs = 608*608*3;
    int outputs = 19*19*1024;
    net->input = calloc(net->inputs, sizeof(float));

    fill_cpu(net->inputs, 0.1, net->input, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);

    for (int l = 0; l < net->n; ++l)
    {
        net->index = l;
        printf("Filter stacks = %d\n", net->layers[l].n);
        net->layers[l].forward(net->layers[l], *net);
        net->input = net->layers[l].output;


        // for (size_t i = 0; i < net->layers[l].out_h; i++)
        // {
        //     for (size_t j = 0; j < net->layers[l].out_w; j++)
        //     {
        //         printf("%.4f ", net->layers[l].output[i*net->layers[l].out_w + j]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");

        // for (size_t i = 0; i < 12; i++)
        // {
        //     for (size_t j = 0; j < 12; j++)
        //     {
        //         printf("%.2f ", net->layers[l].output[144 + i*12 + j]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");

    }

    update_args a;
    a.batch = 1;
    a.learning_rate = 0.001;
    a.momentum = 0.9;
    a.decay = 0.0005;

    for (int l = (net->n - 1); l >=1; --l)
    {
        net->index = l;
        net->input = net->layers[l-1].output;
        net->delta = net->layers[l-1].delta;
        printf("Filter stacks = %d\n", net->layers[l].n);
        net->layers[l].backward(net->layers[l], *net);
        net->layers[l].learning_rate_scale = 1.0;
        update_convolutional_layer(net->layers[l], a);

        printf("Delta layer %d\n", l);

        // for (int m = 0; m < net->layers[l].out_h; ++m)
        // {
        //     for (int n = 0; n < net->layers[l].out_w; ++n)
        //     {
        //         printf("%.2f ", net->layers[l].delta[m*net->layers[l].out_w + n]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");




    //     for (int m = 0; m < 12; ++m)
    //     {
    //         for (int n = 0; n < 12; ++n)
    //         {
    //             printf("%.2f ", net->layers[l].delta[144 + m*12 + n]);
    //         }
    //         printf("\n");
            
    //     }
    //     printf("\n");
    }

        // for (int m = 0; m < net->layers[0].out_h; ++m)
        // {
        //     for (int n = 0; n < net->layers[0].out_w; ++n)
        //     {
        //         printf("%.2f ", net->layers[0].delta[m*net->layers[0].out_w + n]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");
        net->index = 0;
        net->input = calloc(net->inputs, sizeof(float));
        fill_cpu(net->inputs, 0.1, net->input, 1);
        net->delta = calloc(10000000, sizeof(float));
        printf("Filter stacks = %d\n", net->layers[0].n);
        net->layers[0].backward(net->layers[0], *net);
        net->layers[0].learning_rate_scale = 1.0;
        update_convolutional_layer(net->layers[0], a);


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

           int num;
           FILE *fptr;

           // use appropriate location if you are using MacOS or Linux
           fptr = fopen("weights_reference.txt","w");

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
                             fprintf(fptr,"%.4f ", net->layers[l].weights[(c*num_filters*filter_size*filter_size) + (f*filter_size*filter_size) + m*filter_size + n]);
                            }
                            fprintf(fptr, "\n");
                        }
                        fprintf(fptr, "\n");
                        fprintf(fptr, "\n");
                    }
                    fprintf(fptr, "\n");
                    fprintf(fptr, "\n");
                    fprintf(fptr, "\n");
                }

                fprintf(fptr, "\n");
                fprintf(fptr, "\n");
                fprintf(fptr, "\n");
                fprintf(fptr, "\n");

                layer_cumulative_weights += num_filters*channels*filter_size*filter_size;
            }

            fprintf(fptr, "\n");
            fprintf(fptr, "\n");
            fprintf(fptr, "\n");
            fprintf(fptr, "\n");
            fprintf(fptr, "\n");
            fprintf(fptr, "\n");
            fprintf(fptr, "\n");
            fprintf(fptr, "\n");

           fclose(fptr);

           return 0;
}

#define NUM_DIVSIONS 4
int main_filter_dist(){
    network* net = calloc(1, sizeof(network));//SHARED_NETWORKS[i][j];

    net->n = 13;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    int filter_size = 3;
    int num_layers = 11;
    int unit_boundry = 1;    

    net->layers[0] = make_convolutional_layer(1, 38, 38, 256, 512/NUM_DIVSIONS, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[1] = make_convolutional_layer(1, 38, 38, 512, 256/NUM_DIVSIONS, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    net->layers[2] = make_convolutional_layer(1, 38, 38, 256, 512/NUM_DIVSIONS, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[3] = make_convolutional_layer(1, 38, 38, 512, 256/NUM_DIVSIONS, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    net->layers[4] = make_convolutional_layer(1, 38, 38, 256, 512/NUM_DIVSIONS, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[5] = make_maxpool_layer(1, 38, 38, 512, 2, 2, 0);

    net->layers[6] = make_convolutional_layer(1, 19, 19, 512, 1024/NUM_DIVSIONS, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[7] = make_convolutional_layer(1, 19, 19, 1024, 512/NUM_DIVSIONS, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    net->layers[8] = make_convolutional_layer(1, 19, 19, 512, 1024/NUM_DIVSIONS, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[9] = make_convolutional_layer(1, 19, 19, 1024, 512/NUM_DIVSIONS, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    net->layers[10] = make_convolutional_layer(1, 19, 19, 512, 1024/NUM_DIVSIONS, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[11] = make_convolutional_layer(1, 19, 19, 1024, 1024/NUM_DIVSIONS, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    net->layers[12] = make_convolutional_layer(1, 19, 19, 1024, 1024/NUM_DIVSIONS, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);


    for (int l = 0; l < net->n; ++l)
    {
        if(net->layers[l].type == CONVOLUTIONAL){

            int filter_size = net->layers[l].size;
            int num_filters = net->layers[l].n;
            int num_channels = net->layers[l].c;

            for (int i = 0; i < (filter_size*filter_size*num_filters*num_channels); ++i)
            {

                net->layers[l].weights[i] = 0.01;
            }        


        }
    }



    net->workspace = calloc(net->layers[0].workspace_size*5, sizeof(float));
    net->inputs = 38*38*256;
    int outputs = 19*19*1024;
    net->input = calloc(net->inputs, sizeof(float));

    fill_cpu(net->inputs, 0.1, net->input, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);

    for (int l = 0; l < net->n; ++l)
    {
        net->index = l;
        printf("Filter stacks = %d\n", net->layers[l].n);
        net->layers[l].forward(net->layers[l], *net);
        net->input = net->layers[l].output;
    }

}

#define INPUT_SIZE 608
int layer_partition_main(int argc, char* argv[]){

    int stride_vector[16] = {1, 2, 1, 2, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1};
    int filter_stack_vector[16] = {32, 32, 64, 64, 128, 64, 128, 128, 256, 128, 256, 256, 512, 256, 512, 256};
    LAYER_TYPE layer_type_vector[16] = {CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL,
                                   CONVOLUTIONAL, CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, CONVOLUTIONAL,
                                   CONVOLUTIONAL, MAXPOOL, CONVOLUTIONAL, CONVOLUTIONAL, CONVOLUTIONAL, CONVOLUTIONAL};
    int filter_size_vector[16] = {3, 2, 3, 2, 3, 1, 1, 2, 3, 1, 3, 2, 3, 1, 3, 1};
    int layer_length_vector[16] = {INPUT_SIZE, INPUT_SIZE, INPUT_SIZE/2, INPUT_SIZE/2, INPUT_SIZE/4,
                INPUT_SIZE/4, INPUT_SIZE/4, INPUT_SIZE/4, INPUT_SIZE/8, INPUT_SIZE/8,
                INPUT_SIZE/8, INPUT_SIZE/8, INPUT_SIZE/16, INPUT_SIZE/16, INPUT_SIZE/16,
                INPUT_SIZE/16, INPUT_SIZE/16};
    int layer_depth_vector[16] = {3, 32, 32, 64, 64, 128, 64, 128, 128, 256, 128, 256, 256, 512, 256, 512};

    int num_layers = atoi(argv[1]);
    int num_devices = atoi(argv[2]);
    int device_id = atoi(argv[3]);
    int start_layer_idx = atoi(argv[4]);
    int end_layer_idx = atoi(argv[5]);

    int batch = atoi(argv[6]);

    int INPUT_WIDTH = 608;
    int INPUT_HEIGHT = 608;
    int INPUT_CHANNELS = 3;

    network* net = calloc(1, sizeof(network));

    net->n = end_layer_idx - start_layer_idx + 1;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    init_transport_common(num_devices, device_id, argv);

    for(int i = 0; i < net->n; i++){
        if(layer_type_vector[i+start_layer_idx] == CONVOLUTIONAL){

            //make_convolutional_layer(int batch, int h,
            // int w, int c, int n, int groups, int size, int stride, int padding,
            // ACTIVATION activation, int batch_normalize, int binary, int xnor, int adam)
            net->layers[i] = make_convolutional_layer(1, layer_length_vector[i+start_layer_idx], layer_length_vector[i+start_layer_idx],
                                                        layer_depth_vector[i+start_layer_idx], filter_stack_vector[i+start_layer_idx], 1,
                                                        filter_size_vector[i+start_layer_idx], stride_vector[i+start_layer_idx], filter_size_vector[i+start_layer_idx]/2, RELU, 0, 0, 0, 0);

            int total_filter_elements = net->layers[i].size*net->layers[i].size*net->layers[i].c*net->layers[i].n;

            for (int i_f = 0; i_f < (total_filter_elements); ++i_f)
            {
                    net->layers[i].weights[i_f] = 0.01;
            }
        }

        else if(layer_type_vector[i+start_layer_idx] == MAXPOOL){
            //(int batch, int h, int w, int c, int size, int stride, int padding)
            net->layers[i] = make_maxpool_layer(1, layer_length_vector[i+start_layer_idx], layer_length_vector[i+start_layer_idx],
                                layer_depth_vector[i+start_layer_idx], filter_size_vector[i+start_layer_idx], stride_vector[i+start_layer_idx], 0); 
        }
    }

    for (int l = 0; l < net->n; ++l)
    {
        if(net->layers[l].type == CONVOLUTIONAL){

            int filter_size = net->layers[l].size;
            int num_filters = net->layers[l].n;
            int num_channels = net->layers[l].c;

            for (int i = 0; i < (filter_size*filter_size*num_filters*num_channels); ++i)
            {

                net->layers[l].weights[i] = 0.01;
            }        
        }
    }

    int max = 0;
    for (int i = 0; i < net->n; ++i)
    {
        //printf("%d\n", net->layers[i].workspace_size);
        if(net->layers[i].workspace_size > max){
            max = net->layers[i].workspace_size;
        }
    }
    //printf("wsize = %d inputs = %d outputs = %d\n", max*sizeof(float), net->inputs, net->layers[0].outputs);
    net->workspace = calloc(max, sizeof(float));
    //net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));
    net->inputs = net->layers[0].h*net->layers[0].w*net->layers[0].c;

    int outputs = 0;
    
    if(net->layers[net->n - 1].type == CONVOLUTIONAL)
        outputs = net->layers[net->n - 1].out_h*net->layers[net->n - 1].out_w*net->layers[net->n - 1].n;
    else
        outputs = net->layers[net->n - 1].out_h*net->layers[net->n - 1].out_w*net->layers[net->n - 1].c;

    net->input = calloc(net->inputs, sizeof(float));
    net->delta = calloc(net->inputs, sizeof(float));

    fill_cpu(net->inputs, 0.1, net->input, 1);
    fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);

    int processed_samples = 0;

    float* inference_input = calloc(net->inputs, sizeof(float));
    fill_cpu(net->inputs, 0.1, inference_input, 1);

    while(processed_samples < batch){
        //1 get forward[processed_samples] from dev l-1
        if(device_id == 0)
            fill_cpu(net->inputs, 0.1, net->input, 1);
        else
            receive_data(net->input, net->inputs, device_id - 1);
        
        //2 forward
        for(int i = 0; i < net->n; i++){
            net->index = i;
            printf("Starting forward\n");
            net->layers[i].forward(net->layers[i], *net);
            net->input = net->layers[i].output;
        }
        
        //3 send forward to dev l+1
        printf("sending\n");
        if(device_id < (num_devices - 1))
            send_data(net->layers[0].output, outputs, device_id + 1);
        
        processed_samples += 1;
    }

    processed_samples = 0;
    while(processed_samples < batch){
        
        //4 get backward[processed_samples] from dev l+1
        if(device_id == (num_devices - 1))
            fill_cpu(net->layers[net->n - 1].outputs, 0.1, net->layers[net->n - 1].delta, 1);
        else
            receive_data(net->layers[net->n - 1].delta, outputs, device_id + 1);

        //5 backward

        for(int i = net->n - 1; i >= 1; i--){
            net->index = i;
            net->input = net->layers[i-1].output;
            net->delta = net->layers[i-1].delta;
            printf("Starting backward\n");
            net->layers[i].backward(net->layers[i], *net);
        }

        net->index = 0;
        net->input = inference_input;
        net->delta = calloc(net->inputs, sizeof(float));;
        printf("Starting backward base\n");
        net->layers[0].backward(net->layers[0], *net);
        // net->layers[0].learning_rate_scale = 1.0;
        // update_convolutional_layer(net->layers[0], a);

        //6 send backward to dev l-1
        if(device_id > 0)
            send_data(net->delta, net->inputs, device_id - 1);
        
        processed_samples += 1;
        
    }

    // update_args a;
    // a.batch = 1;
    // a.learning_rate = 0.001;
    // a.momentum = 0.9;
    // a.decay = 0.0005;

    // for (int l = (net->n - 1); l >=1; --l)
    // {
    //     net->index = l;
    //     net->input = net->layers[l-1].output;
    //     net->delta = net->layers[l-1].delta;
    //     printf("Filter stacks = %d\n", net->layers[l].n);
    //     net->layers[l].backward(net->layers[l], *net);
    //     net->layers[l].learning_rate_scale = 1.0;
    //     update_convolutional_layer(net->layers[l], a);

    //     printf("Delta layer %d\n", l);
    // }
    //     net->index = 0;
    //     net->input = calloc(net->inputs, sizeof(float));
    //     fill_cpu(net->inputs, 0.1, net->input, 1);
    //     net->delta = calloc(10000000, sizeof(float));
    //     printf("Filter stacks = %d\n", net->layers[0].n);
    //     net->layers[0].backward(net->layers[0], *net);
    //     net->layers[0].learning_rate_scale = 1.0;
    //     update_convolutional_layer(net->layers[0], a);

}


int main(int argc, char* argv[]){
    layer_partition_main(argc, argv);
}

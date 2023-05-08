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

    net->n = 11;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    int filter_size = 3;
    int num_layers = 11;
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

    // net->layers[11] = make_maxpool_layer(1, 76/2, 76/2, 256, 2, 2, 0);


    // net->layers[12] = make_convolutional_layer(1, 38/2, 38/2, 256, 512, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[13] = make_convolutional_layer(1, 38/2, 38/2, 512, 256, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

    // net->layers[14] = make_convolutional_layer(1, 38/2, 38/2, 256, 512, 1, 3, 1, 1, LEAKY, 0, 0, 0, 0);

    // net->layers[15] = make_convolutional_layer(1, 38/2, 38/2, 512, 256, 1, 1, 1, 0, LEAKY, 0, 0, 0, 0);

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


int main(int argc, char* argv[]){
    //main_distributed();
    //main_reference();
    //main_maxpool();
    // for (int i = 1; i < argc; i++)
    // {
    //     printf("argv[%u] = %s\n", i, argv[i]);
    //     if (argv[i][0] == '-')
    //     {
    //     }
    // }
    main_yolo();
}

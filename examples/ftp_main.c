#include "ftp.h"
#include "fused.h"

#include <stdio.h>
#include <stdlib.h>
#include "gemm.h"
#include "col2im.h"
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


typedef struct group_profile_forward{
    int start_x_forward;
    int start_y_forward;
    int end_x_forward;
    int end_y_forward;  
} group_profile_forward;

typedef struct group_profile_backward{
    int start_x_backward;
    int start_y_backward;
    int end_x_backward;
    int end_y_backward;
} group_profile_backward;

typedef struct train_groups_profile{
    group_profile_forward* fp;
    group_profile_backward* bp;
} train_groups_profile;


int main(){

    train_groups_profile profile;
    profile.fp = calloc(1, sizeof(group_profile_forward));
    profile.bp = calloc(1, sizeof(group_profile_backward));

    network* net = calloc(1, sizeof(network));//SHARED_NETWORKS[i][j];

    net->n = 4;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    int filter_size = 3;
    int num_layers = 4;
    int unit_boundry = 1;

    // net->layers[0] = make_convolutional_layer(1, 16, 16, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0);
    // net->layers[1] = make_convolutional_layer(1, 14, 14, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0);
    // net->layers[2] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 2, 0, RELU, 0, 0, 0, 0);
    // net->layers[3] = make_convolutional_layer(1, 5, 5, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0); 

    net->layers[0] = make_convolutional_layer(1, 16, 16, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, 14, 14, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 2, 0, RELU, 0, 0, 0, 0);
    net->layers[3] = make_convolutional_layer(1, 5, 5, 1, 1, 1, filter_size, 1, 0, RELU, 0, 0, 0, 0); 

    for (int i_f = 0; i_f < filter_size; ++i_f)
    {
        for (int j_f = 0; j_f < filter_size; ++j_f)
        {
            net->layers[0].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[1].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[2].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[3].weights[i_f*filter_size + j_f] = 0.1;
        }
    }

    // fill_cpu(729, 0, net->layers[0].delta, 1);
    // fill_cpu(729, 0, net->layers[1].delta, 1);
    // fill_cpu(729, 0, net->layers[2].delta, 1);
    // fill_cpu(225, 0, net->layers[3].delta, 1);
    // fill_cpu(169, 0, net->layers[4].delta, 1);

    net->workspace = calloc(net->layers[0].workspace_size*5, sizeof(float));
    net->inputs = 256;
    net->input = calloc(256, sizeof(float));

    // SHARED_INPUT_IMAGES[i][j] = calloc((INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), sizeof(float));
    // fill_cpu((INPUT_WIDTH/NUM_TILES_X)*(INPUT_HEIGHT/NUM_TILES_Y), 1, SHARED_INPUT_IMAGES[i][j], 1);

    // SHARED_EXP_DELTAS[i][j] = calloc(net->layers[3].outputs, sizeof(float));
    // fill_cpu(net->layers[3].outputs, 1, SHARED_EXP_DELTAS[i][j], 1);

    float* COMBINED_INPUT_IMAGES = calloc(144, sizeof(float));
    fill_cpu(144, 1, COMBINED_INPUT_IMAGES, 1);
    float* COMBINED_EXP_DELTAS = calloc(36, sizeof(float));
    fill_cpu(36, 1, COMBINED_EXP_DELTAS, 1);

    partition_forward(net, 
                            0, 0,
                            NULL,
                            COMBINED_INPUT_IMAGES,
                            3,
                            0, 0,
                            2, 2);

    partition_backward(net, 
                        0, 0,
                        NULL,
                        COMBINED_EXP_DELTAS,
                        3,
                        0, 0,
                        5, 5);
    //while(1);

    int total_edges_left = net->layers[0].left_boundry_edges_featuremap;
    int total_edges_right = net->layers[0].right_boundry_edges_featuremap;
    int total_edges_bottom = net->layers[0].bottom_boundry_edges_featuremap;
    int total_edges_top = net->layers[0].top_boundry_edges_featuremap;

    for (int l = 0; l < num_layers; ++l)
    {

        if(l > 0){
            for (int m = 0; m < net->layers[l].top_boundry_edges_featuremap; ++m)
            {
                for (int n = 0; n < net->layers[l].featuremap_in_w_with_boundry; ++n)
                {
                    net->layers[l-1].output[m*net->layers[l].featuremap_in_w_with_boundry + n] = 0.0;
                }
            }

            for (int m = 0; m < net->layers[l].featuremap_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[l].left_boundry_edges_featuremap; ++n)
                {
                    net->layers[l-1].output[(m*net->layers[l].featuremap_in_w_with_boundry) + n] = 0.0;
                }
            }

            // for (int m = 0; m < (total_edges_left - net->layers[l].top_boundry_edges_featuremap); ++m)
            // {
            //     for (int n = 0; n < net->layers[0].featuremap_in_w_with_boundry; ++n)
            //     {
            //         net->layers[l-1].output[m*net->layers[0].featuremap_in_w_with_boundry + n] = 0.0;
            //     }
            // }
            // for (size_t i = 0; i < net->layers[l].featuremap_in_h_with_boundry; i++)
            // {
            //     for (size_t j = 0; j < net->layers[l].featuremap_in_w_with_boundry; j++)
            //     {
            //         printf("%.2f ", net->layers[l-1].output[i*net->layers[l].featuremap_in_w_with_boundry + j]);
            //     }
            //     printf("\n");
                
            // }
        }
        net->index = l;
        forward_convolutional_layer(net->layers[l], *net);
        net->input = net->layers[l].output;

    }
            // for (size_t i = 0; i < 3; i++)
            // {
            //     for (size_t j = 0; j < 3; j++)
            //     {
            //         printf("%.2f ", net->layers[3].output[i*3 + j]);
            //     }
            //     printf("\n");
                
            // }
   // while(1);

        // for (size_t i = 0; i < 16; i++)
        // {
        //     for (size_t j = 0; j < 16; j++)
        //     {
        //         printf("%.2f ", net->input[i*16 + j]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");

        // for (size_t i = 0; i < net->layers[1].featuremap_in_h_with_boundry; i++)
        // {
        //     for (size_t j = 0; j < net->layers[1].featuremap_in_w_with_boundry; j++)
        //     {
        //         printf("%.2f ", net->layers[0].output_without_boundry[i*net->layers[1].featuremap_in_w_with_boundry + j]);
        //     }
        //     printf("\n");
            
        // }
        // printf("\n");

    for (int l = num_layers - 1; l >= 1; --l)
    {
                int featuremap_without_boundry_width = net->layers[l].featuremap_in_w_without_boundry + (2*unit_boundry);
                int featuremap_without_boundry_height = net->layers[l].featuremap_in_h_without_boundry + (2*unit_boundry);
                //printf("%d %d\n", featuremap_without_boundry_width, featuremap_without_boundry_height);

                for (int m = 0; m < featuremap_without_boundry_height; ++m)
                {
                    for (int n = 0; n < featuremap_without_boundry_width; ++n)
                    {
                        int left_edges = net->layers[l].left_boundry_edges_featuremap;
                        int right_edges = net->layers[l].right_boundry_edges_featuremap;
                        int top_edges = net->layers[l].top_boundry_edges_featuremap;
                        int bottom_edges = net->layers[l].bottom_boundry_edges_featuremap;
                        // printf("%d %d %d\n", net->layers[l].featuremap_start_coordinate_x, featuremap_without_boundry_width, featuremap_without_boundry_height);
                        // printf("%d %d %d %d\n", left_edges, right_edges, top_edges, bottom_edges);
                        // if(l == 2 && i ==0 && j ==1){
                        //     while(1);
                        // }
                        net->layers[l-1].output_without_boundry[m*featuremap_without_boundry_width + n] = 
                            net->layers[l-1].output[(m+top_edges - unit_boundry)*(net->layers[l].featuremap_in_w_with_boundry) + n + left_edges - unit_boundry];
                    }

                }

                    for (int m = 0; m < featuremap_without_boundry_height; ++m)
                    {
                        for (int n = 0; n < featuremap_without_boundry_width; ++n)
                        {
                            printf("%.2f ", net->layers[l-1].output_without_boundry[m*featuremap_without_boundry_width + n]);
                        }
                        printf("\n");
                        
                    }
                    printf("\n");
                    //while(1);


                for (int m = 0; m < net->layers[l].delta_in_h_without_boundry; ++m)
                {
                    for (int n = 0; n < net->layers[l].delta_in_w_without_boundry; ++n)
                    {
                        int left_edges = net->layers[l].left_boundry_edges_delta;
                        int right_edges = net->layers[l].right_boundry_edges_delta;
                        int top_edges = net->layers[l].top_boundry_edges_delta;
                        int bottom_edges = net->layers[l].bottom_boundry_edges_delta;

                        net->layers[l].delta_without_boundry[m*net->layers[l].delta_in_w_without_boundry + n] = 
                                                                    net->layers[l].delta_with_boundry[(m+top_edges)*(net->layers[l].delta_in_w_with_boundry) + n + left_edges];
                    }
                }




                net->input = net->layers[l-1].output_without_boundry;
                net->delta = net->layers[l-1].delta_with_boundry;


                int stride = net->layers[l].stride;

                net->layers[l].out_w = net->layers[l].delta_in_w_without_boundry;
                net->layers[l].out_h = net->layers[l].delta_in_h_without_boundry;
                net->layers[l].h = featuremap_without_boundry_height;
                net->layers[l].w = featuremap_without_boundry_width;

                net->layers[l].pad = 0;

                net->index = l;
                
               // while(1);

                backward_convolutional_layer_dist_filters(net->layers[l], *net);
                

                net->layers[l].out_w = net->layers[l].delta_in_w_with_boundry;
                net->layers[l].out_h = net->layers[l].delta_in_h_with_boundry;

                net->layers[l].pad = filter_size - 1;

                // if((net->layers[l].left_boundry_edges_delta) != (net->layers[l].right_boundry_edges_delta)){
                //     net->layers[l].pad -= (net->layers[l].left_boundry_edges_delta - net->layers[l].right_boundry_edges_delta);
                // }

                for (int m = 0; m < net->layers[l].top_boundry_edges_delta; ++m)
                {
                    for (int n = 0; n < net->layers[l].delta_in_w_with_boundry; ++n)
                    {
                        net->layers[l].delta_with_boundry[m*net->layers[l].delta_in_w_with_boundry + n] = 0.0;
                    }
                }

                for (int m = 0; m < net->layers[l].delta_in_h_with_boundry; ++m)
                {
                    for (int n = 0; n < net->layers[l].left_boundry_edges_delta; ++n)
                    {
                        net->layers[l].delta_with_boundry[(m*net->layers[l].delta_in_w_with_boundry) + n] = 0.0;
                    }
                }

                int dilated_delta_dim_x = net->layers[l].out_w*stride;
                int dilated_delta_dim_y = net->layers[l].out_h*stride;
                net->layers[l].w = net->layers[l-1].delta_in_w_with_boundry;
                net->layers[l].h = net->layers[l-1].delta_in_h_with_boundry;

                backward_convolutional_layer_dist_delta(net->layers[l], *net);

    }

            for (int m = 0; m < net->layers[0].delta_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[0].delta_in_w_with_boundry; ++n)
                {
                    printf("%.2f ", net->layers[0].delta_with_boundry[m*net->layers[0].delta_in_w_with_boundry + n]);
                }
                printf("\n");
                
            }
            printf("\n");


            for (int m = 0; m < net->layers[2].delta_in_h_with_boundry; ++m)
            {
                for (int n = 0; n < net->layers[2].delta_in_w_with_boundry; ++n)
                {
                    printf("%.2f ", net->layers[2].delta_with_boundry[m*net->layers[2].delta_in_w_with_boundry + n]);
                }
                printf("\n");
                
            }
            printf("\n");

}

int main_u(){
    network* net = calloc(1, sizeof(network));//SHARED_NETWORKS[i][j];

    net->n = 4;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));

    int filter_size = 3;
    int num_layers = 4;
    int unit_boundry = 1;

    net->layers[0] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 2, 1, RELU, 0, 0, 0, 0);
    net->layers[3] = make_convolutional_layer(1, 6, 6, 1, 1, 1, filter_size, 1, 1, RELU, 0, 0, 0, 0); 

    for (int i_f = 0; i_f < filter_size; ++i_f)
    {
        for (int j_f = 0; j_f < filter_size; ++j_f)
        {
            net->layers[0].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[1].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[2].weights[i_f*filter_size + j_f] = 0.1;
            net->layers[3].weights[i_f*filter_size + j_f] = 0.1;
        }
    }

    net->workspace = calloc(net->layers[0].workspace_size*5, sizeof(float));
    net->inputs = 144;
    net->input = calloc(144, sizeof(float));

fill_cpu(144, 1, net->input, 1);
fill_cpu(36, 1, net->layers[3].delta, 1);

    for (int l = 0; l < num_layers; ++l)
    {
        net->index = l;
        forward_convolutional_layer(net->layers[l], *net);
        net->input = net->layers[l].output;
    }

            for (size_t i = 0; i < 6; i++)
            {
                for (size_t j = 0; j < 6; j++)
                {
                    printf("%.2f ", net->layers[3].output[i*6 + j]);
                }
                printf("\n");
                
            }
            printf("\n");

    for (int l = num_layers-1; l >=1; --l)
    {
        net->index = l;
        net->input = net->layers[l-1].output;
        net->delta = net->layers[l-1].delta;
        backward_convolutional_layer(net->layers[l], *net);
    }

            for (size_t i = 0; i < 12; i++)
            {
                for (size_t j = 0; j < 12; j++)
                {
                    printf("%.2f ", net->layers[0].delta[i*12 + j]);
                }
                printf("\n");
                
            }
            printf("\n");


            for (size_t i = 0; i < 6; i++)
            {
                for (size_t j = 0; j < 6; j++)
                {
                    printf("%.2f ", net->layers[2].delta[i*6 + j]);
                }
                printf("\n");
                
            }
            printf("\n");
}

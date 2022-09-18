#include "ftp.h"

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
int main_u(){

//     float* image = calloc(25*25, sizeof(float));
//     float* delta = calloc(35*35, sizeof(float)); //delta is with boundries for now. Image is separated

//     float* boundry_top = calloc(25*5, sizeof(float));
//     float* boundry_bottom = calloc(25*5, sizeof(float));
//     float* boundry_left = calloc(25*5, sizeof(float));
//     float* boundry_right = calloc(25*5, sizeof(float));
//     float* boundry_top_right = calloc(5*5, sizeof(float));
//     float* boundry_top_left = calloc(5*5, sizeof(float));
//     float* boundry_bottom_right = calloc(5*5, sizeof(float));
//     float* boundry_bottom_left = calloc(5*5, sizeof(float));

//     float* output_delta1 = calloc(25*25, sizeof(float));
//     float* output_delta2 = calloc(25*25, sizeof(float));
//     float* output_delta3 = calloc(25*25, sizeof(float));
//     float* output_delta4 = calloc(25*25, sizeof(float));

//     fill_cpu(625, 1, image, 1);
//     fill_cpu(1225, 0, delta, 1);
//     for (int i = 10; i < 35; ++i)
//     {
//         for (int j = 10; j < 35; ++j)
//         {
//             delta[i*35 + j] =  1.0;
//         }
//     }

//     fill_cpu(125, 0, boundry_top, 1);
//     fill_cpu(125, 1, boundry_bottom, 1);
//     fill_cpu(125, 0, boundry_left, 1);
//     fill_cpu(125, 1, boundry_right, 1);
//     fill_cpu(25, 0, boundry_top_right, 1);
//     fill_cpu(25, 0, boundry_top_left, 1);
//     fill_cpu(25, 1, boundry_bottom_right, 1);
//     fill_cpu(25, 0, boundry_bottom_left, 1);

//     int im_width = 35; 
//     int im_height = 35; 
//     int filter_size = 3;

//     execute_device(im_width, im_height, filter_size,
//                        image, delta,
//                        boundry_top, boundry_bottom,
//                        boundry_left, boundry_right,
//                        boundry_top_right, boundry_top_left,
//                        boundry_bottom_right, boundry_bottom_left, 5, output_delta1);

//     fill_cpu(625, 1, image, 1);
//     fill_cpu(1225, 0, delta, 1);
//     for (int i = 10; i < 35; ++i)
//     {
//         for (int j = 0; j < 25; ++j)
//         {
//             delta[i*35 + j] =  1.0;
//         }
//     }

//     fill_cpu(125, 0, boundry_top, 1);
//     fill_cpu(125, 1, boundry_bottom, 1);
//     fill_cpu(125, 1, boundry_left, 1);
//     fill_cpu(125, 0, boundry_right, 1);
//     fill_cpu(25, 0, boundry_top_right, 1);
//     fill_cpu(25, 0, boundry_top_left, 1);
//     fill_cpu(25, 0, boundry_bottom_right, 1);
//     fill_cpu(25, 1, boundry_bottom_left, 1);

//     execute_device(im_width, im_height, filter_size,
//                        image, delta,
//                        boundry_top, boundry_bottom,
//                        boundry_left, boundry_right,
//                        boundry_top_right, boundry_top_left,
//                        boundry_bottom_right, boundry_bottom_left, 5, output_delta2);


//     fill_cpu(625, 1, image, 1);
//     fill_cpu(1225, 0, delta, 1);
//     for (int i = 0; i < 25; ++i)
//     {
//         for (int j = 10; j < 35; ++j)
//         {
//             delta[i*35 + j] =  1.0;
//         }
//     }

//     fill_cpu(125, 1, boundry_top, 1);
//     fill_cpu(125, 0, boundry_bottom, 1);
//     fill_cpu(125, 0, boundry_left, 1);
//     fill_cpu(125, 1, boundry_right, 1);
//     fill_cpu(25, 1, boundry_top_right, 1);
//     fill_cpu(25, 0, boundry_top_left, 1);
//     fill_cpu(25, 0, boundry_bottom_right, 1);
//     fill_cpu(25, 0, boundry_bottom_left, 1);

//     execute_device(im_width, im_height, filter_size,
//                        image, delta,
//                        boundry_top, boundry_bottom,
//                        boundry_left, boundry_right,
//                        boundry_top_right, boundry_top_left,
//                        boundry_bottom_right, boundry_bottom_left, 5, output_delta3);

//     fill_cpu(625, 1, image, 1);
//     fill_cpu(1225, 0, delta, 1);
//     for (int i = 0; i < 25; ++i)
//     {
//         for (int j = 0; j < 25; ++j)
//         {
//             delta[i*35 + j] =  1.0;
//         }
//     }

//     fill_cpu(125, 1, boundry_top, 1);
//     fill_cpu(125, 0, boundry_bottom, 1);
//     fill_cpu(125, 1, boundry_left, 1);
//     fill_cpu(125, 0, boundry_right, 1);
//     fill_cpu(25, 0, boundry_top_right, 1);
//     fill_cpu(25, 1, boundry_top_left, 1);
//     fill_cpu(25, 0, boundry_bottom_right, 1);
//     fill_cpu(25, 0, boundry_bottom_left, 1);

//     execute_device(im_width, im_height, filter_size,
//                        image, delta,
//                        boundry_top, boundry_bottom,
//                        boundry_left, boundry_right,
//                        boundry_top_right, boundry_top_left,
//                        boundry_bottom_right, boundry_bottom_left, 5, output_delta4);

//     float* kitty_cat = calloc(2500, sizeof(float));

//     for (int i = 0; i < 25; ++i)
//     {
//         for (int j = 0; j < 25; ++j)
//         {
//             kitty_cat[i*50 + j] = output_delta1[i*25 + j];
//         }
//     }
//     for (int i = 0; i < 25; ++i)
//     {
//         for (int j = 0; j < 25; ++j)
//         {
//             kitty_cat[(i)*50 + j+25] = output_delta2[i*25 + j];
//         }
//     }
//     for (int i = 0; i < 25; ++i)
//     {
//         for (int j = 0; j < 25; ++j)
//         {
//             kitty_cat[(i+25)*50 + j] = output_delta3[i*25 + j];
//         }
//     }
//     for (int i = 0; i < 25; ++i)
//     {
//         for (int j = 0; j < 25; ++j)
//         {
//             kitty_cat[(i+25)*50 + j+25] = output_delta4[i*25 + j];
//         }
//     }

// //###DEBUG
//     // for (int i = 0; i < 50; ++i)
//     // {
//     //     for (int j = 0; j < 50; ++j)
//     //     {
//     //         printf("%.2f ", kitty_cat[i*50 + j]);
//     //     }
//     //     printf("\n");
//     // }
//     // printf("\n");
// //###END DEBUG

//     float* image_merged = calloc(50*50, sizeof(float));
//     float* delta_merged= calloc(50*50, sizeof(float));
//     // float* kitty_cat_merged = calloc(2500, sizeof(float));
//     // float* output_delta_merged = calloc(50*50, sizeof(float));
//     fill_cpu(2500, 1, image_merged, 1);
//     fill_cpu(2500, 0, delta_merged, 1);
//     for (int i = 5; i < 45; ++i)
//     {
//         for (int j = 5; j < 45; ++j)
//         {
//             delta_merged[i*50 + j] =  1.0;
//         }
//     }
//     network *net = calloc(1, sizeof(network));
//     net->n = 5;
//     net->layers = calloc(net->n, sizeof(layer));
//     net->seen = calloc(1, sizeof(size_t));
//     net->t    = calloc(1, sizeof(int));
//     net->cost = calloc(1, sizeof(float));
//     net->layers[0] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
//     net->layers[1] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
//     net->layers[2] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
//     net->layers[3] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
//     net->layers[4] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);  
//     net->layers[0].batch_normalize = 0;
//     net->layers[1].batch_normalize = 0;
//     net->layers[2].batch_normalize = 0;
//     net->layers[3].batch_normalize = 0;
//     net->layers[4].batch_normalize = 0;

//     //printf("%d, %d %d %d %d\n", im_height, im_width, core_image_height, core_image_width, boundry_frames);

//     //net->layers[0].data = calloc(1, sizeof(float));
//     net->input = calloc(2500, sizeof(float));
//     net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));
//     net->inputs = 2500;

//     fill_cpu(2500, 0, net->layers[0].delta, 1);
//     fill_cpu(2500, 0, net->layers[1].delta, 1);
//     fill_cpu(2500, 0, net->layers[2].delta, 1);
//     fill_cpu(2500, 0, net->layers[3].delta, 1);
//     fill_cpu(2500, 0, net->layers[4].delta, 1);

//     for (int i = 0; i < filter_size; ++i)
//     {
//         for (int j = 0; j < filter_size; ++j)
//         {
//             net->layers[0].weights[i*filter_size + j] = 0.1;
//             net->layers[1].weights[i*filter_size + j] = 0.1;
//             net->layers[2].weights[i*filter_size + j] = 0.1;
//             net->layers[3].weights[i*filter_size + j] = 0.1;
//             net->layers[4].weights[i*filter_size + j] = 0.1;
//         }
//     }

//     for (int i = 0; i < 50; ++i)
//     {
//         for (int j = 0; j < 50; ++j)
//         {
//             net->input[i*50 + j] = image_merged[i*50 + j];
//         }
//     }
 
//     net->index = 0;
//     forward_convolutional_layer(net->layers[0], *net);
//     net->input = net->layers[0].output;
//     net->index = 1;
//     forward_convolutional_layer(net->layers[1], *net);
//     net->input = net->layers[1].output;
//     net->index = 2;
//     forward_convolutional_layer(net->layers[2], *net);
//     net->input = net->layers[2].output;
//     net->index = 3;
//     forward_convolutional_layer(net->layers[3], *net);
//     net->input = net->layers[3].output;
//     net->index = 4;
//     forward_convolutional_layer(net->layers[4], *net);

//     for (int i = 0; i < 50; ++i)
//     {
//         for (int j = 0; j < 50; ++j)
//         {
//             net->layers[4].delta[i*50 + j] = delta_merged[i*50 + j];
//             net->layers[4].delta_with_boundry[i*50 + j] = delta_merged[i*50 + j];
//         }
//     }

//     net->input = net->layers[3].output;
//     net->delta = net->layers[3].delta;
//     net->index = 4;
//     backward_convolutional_layer(net->layers[4], *net);

//     net->input = net->layers[2].output;
//     net->delta = net->layers[2].delta;
//     net->index = 3;
//     backward_convolutional_layer(net->layers[3], *net);


//     net->input = net->layers[1].output;
//     net->delta = net->layers[1].delta;
//     net->index = 2;
//     backward_convolutional_layer(net->layers[2], *net);
//     net->input = net->layers[0].output;
//     net->delta = net->layers[0].delta;
//     net->index = 1;
//     backward_convolutional_layer(net->layers[1], *net);

//     //### DEBUG
//     // for (int i = 0; i < 50; ++i)
//     // {
//     //     for (int j = 0; j < 50; ++j)
//     //     {
//     //         printf("%.2f ", net->layers[0].delta[i*50 + j]);
//     //         //printf("%.2f ", net->layers[0].output[i*50 + j]);
//     //     }
//     //     printf("\n");
//     // }
//     // printf("\n");
//     //### END DEBUG

//     for (int i = 0; i < 3; ++i)
//     {
//         for (int j = 0; j < 3; ++j)
//         {
//             printf("%.2f ", net->layers[3].weight_updates[i*3 + j]);
//             //printf("%.2f ", net->layers[0].output[i*50 + j]);
//         }
//         printf("\n");
//     }
//     printf("\n");

// //###END DEBUG
}


void *print_message_function( void *ptr );

sem_t filter_diverge;
sem_t filter_converge;

int main_unused(){
     pthread_t thread1, thread2, thread3, thread4;
     char *message1 = "Thread 1";
     char *message2 = "Thread 2";
     int  iret1, iret2, iret3, iret4;


    float* image1 = calloc(25*25, sizeof(float));
    float* delta1 = calloc(35*35, sizeof(float)); //delta is with boundries for now. Image is separated
    float* image2 = calloc(25*25, sizeof(float));
    float* delta2 = calloc(35*35, sizeof(float));
    float* image3 = calloc(25*25, sizeof(float));
    float* delta3 = calloc(35*35, sizeof(float));
    float* image4 = calloc(25*25, sizeof(float));
    float* delta4 = calloc(35*35, sizeof(float));

    float* boundry_top1 = calloc(25*5, sizeof(float));
    float* boundry_bottom1 = calloc(25*5, sizeof(float));
    float* boundry_left1 = calloc(25*5, sizeof(float));
    float* boundry_right1 = calloc(25*5, sizeof(float));
    float* boundry_top_right1 = calloc(5*5, sizeof(float));
    float* boundry_top_left1 = calloc(5*5, sizeof(float));
    float* boundry_bottom_right1 = calloc(5*5, sizeof(float));
    float* boundry_bottom_left1 = calloc(5*5, sizeof(float));

    fill_cpu(625, 1, image1, 1);
    fill_cpu(1225, 0, delta1, 1);
    for (int i = 10; i < 35; ++i)
    {
        for (int j = 10; j < 35; ++j)
        {
            delta1[i*35 + j] =  1.0;
        }
    }

    fill_cpu(125, 0, boundry_top1, 1);
    fill_cpu(125, 1, boundry_bottom1, 1);
    fill_cpu(125, 0, boundry_left1, 1);
    fill_cpu(125, 1, boundry_right1, 1);
    fill_cpu(25, 0, boundry_top_right1, 1);
    fill_cpu(25, 0, boundry_top_left1, 1);
    fill_cpu(25, 1, boundry_bottom_right1, 1);
    fill_cpu(25, 0, boundry_bottom_left1, 1);

    float* boundry_top2 = calloc(25*5, sizeof(float));
    float* boundry_bottom2 = calloc(25*5, sizeof(float));
    float* boundry_left2 = calloc(25*5, sizeof(float));
    float* boundry_right2 = calloc(25*5, sizeof(float));
    float* boundry_top_right2 = calloc(5*5, sizeof(float));
    float* boundry_top_left2 = calloc(5*5, sizeof(float));
    float* boundry_bottom_right2 = calloc(5*5, sizeof(float));
    float* boundry_bottom_left2 = calloc(5*5, sizeof(float));

    fill_cpu(625, 1, image2, 1);
    fill_cpu(1225, 0, delta2, 1);
    for (int i = 10; i < 35; ++i)
    {
        for (int j = 0; j < 25; ++j)
        {
            delta2[i*35 + j] =  1.0;
        }
    }

    fill_cpu(125, 0, boundry_top2, 1);
    fill_cpu(125, 1, boundry_bottom2, 1);
    fill_cpu(125, 1, boundry_left2, 1);
    fill_cpu(125, 0, boundry_right2, 1);
    fill_cpu(25, 0, boundry_top_right2, 1);
    fill_cpu(25, 0, boundry_top_left2, 1);
    fill_cpu(25, 0, boundry_bottom_right2, 1);
    fill_cpu(25, 1, boundry_bottom_left2, 1);

    float* boundry_top3 = calloc(25*5, sizeof(float));
    float* boundry_bottom3 = calloc(25*5, sizeof(float));
    float* boundry_left3 = calloc(25*5, sizeof(float));
    float* boundry_right3 = calloc(25*5, sizeof(float));
    float* boundry_top_right3 = calloc(5*5, sizeof(float));
    float* boundry_top_left3 = calloc(5*5, sizeof(float));
    float* boundry_bottom_right3 = calloc(5*5, sizeof(float));
    float* boundry_bottom_left3 = calloc(5*5, sizeof(float));

    fill_cpu(625, 1, image3, 1);
    fill_cpu(1225, 0, delta3, 1);
    for (int i = 0; i < 25; ++i)
    {
        for (int j = 10; j < 35; ++j)
        {
            delta3[i*35 + j] =  1.0;
        }
    }

    fill_cpu(125, 1, boundry_top3, 1);
    fill_cpu(125, 0, boundry_bottom3, 1);
    fill_cpu(125, 0, boundry_left3, 1);
    fill_cpu(125, 1, boundry_right3, 1);
    fill_cpu(25, 1, boundry_top_right3, 1);
    fill_cpu(25, 0, boundry_top_left3, 1);
    fill_cpu(25, 0, boundry_bottom_right3, 1);
    fill_cpu(25, 0, boundry_bottom_left3, 1);

    float* boundry_top4 = calloc(25*5, sizeof(float));
    float* boundry_bottom4 = calloc(25*5, sizeof(float));
    float* boundry_left4 = calloc(25*5, sizeof(float));
    float* boundry_right4 = calloc(25*5, sizeof(float));
    float* boundry_top_right4 = calloc(5*5, sizeof(float));
    float* boundry_top_left4 = calloc(5*5, sizeof(float));
    float* boundry_bottom_right4 = calloc(5*5, sizeof(float));
    float* boundry_bottom_left4 = calloc(5*5, sizeof(float));

    fill_cpu(625, 1, image4, 1);
    fill_cpu(1225, 0, delta4, 1);
    for (int i = 0; i < 25; ++i)
    {
        for (int j = 0; j < 25; ++j)
        {
            delta4[i*35 + j] =  1.0;
        }
    }

    fill_cpu(125, 1, boundry_top4, 1);
    fill_cpu(125, 0, boundry_bottom4, 1);
    fill_cpu(125, 1, boundry_left4, 1);
    fill_cpu(125, 0, boundry_right4, 1);
    fill_cpu(25, 0, boundry_top_right4, 1);
    fill_cpu(25, 1, boundry_top_left4, 1);
    fill_cpu(25, 0, boundry_bottom_right4, 1);
    fill_cpu(25, 0, boundry_bottom_left4, 1);

    float* output_delta1 = calloc(25*25, sizeof(float));
    float* output_delta2 = calloc(25*25, sizeof(float));
    float* output_delta3 = calloc(25*25, sizeof(float));
    float* output_delta4 = calloc(25*25, sizeof(float));

    int im_width = 35; 
    int im_height = 35; 
    int filter_size = 3;
    int num_layers = 5;
    int num_devices = 4;

    device_ftp_args* ftp_args1 = calloc(1, sizeof(device_ftp_args));

    //TODO: FILTER SIZE SHOULD BE GENERIC. JUST 3X3 FOR NOW ALL LAYERS
    float*** SHARED_WEIGHT_UPDATES = calloc(num_devices, sizeof(float**));
    for (int i = 0; i < num_devices; ++i)
    {
        SHARED_WEIGHT_UPDATES[i] = calloc(num_layers, sizeof(float*));
        for (int j = 0; j < num_layers; ++j)
        {
            SHARED_WEIGHT_UPDATES[i][j] = calloc(filter_size*filter_size, sizeof(float));
        }
    }

    ftp_args1->NUM_DEVICES = num_devices;
    ftp_args1->device_id = 0;
    ftp_args1->im_width = im_width;
    ftp_args1->im_height = im_height;
    ftp_args1->filter_size = filter_size;
    ftp_args1->image = image1;
    ftp_args1->delta = delta1;
    ftp_args1->boundry_top = boundry_top1;
    ftp_args1->boundry_bottom = boundry_bottom1;
    ftp_args1->boundry_left = boundry_left1;
    ftp_args1->boundry_right = boundry_right1;
    ftp_args1->boundry_top_right = boundry_top_right1;
    ftp_args1->boundry_top_left = boundry_top_left1;
    ftp_args1->boundry_bottom_right = boundry_bottom_right1;
    ftp_args1->boundry_bottom_left = boundry_bottom_left1;
    ftp_args1->num_layers = num_layers;
    ftp_args1->output = output_delta1; 
    ftp_args1->SHARED_WEIGHT_UPDATES = SHARED_WEIGHT_UPDATES;
    /* ->reate independent threads each of which will execute function */

    device_ftp_args* ftp_args2 = calloc(1, sizeof(device_ftp_args));

    ftp_args2->device_id = 1;
    ftp_args2->im_width = im_width;
    ftp_args2->im_height = im_height;
    ftp_args2->filter_size = filter_size;
    ftp_args2->image = image2;
    ftp_args2->delta = delta2;
    ftp_args2->boundry_top = boundry_top2;
    ftp_args2->boundry_bottom = boundry_bottom2;
    ftp_args2->boundry_left = boundry_left2;
    ftp_args2->boundry_right = boundry_right2;
    ftp_args2->boundry_top_right = boundry_top_right2;
    ftp_args2->boundry_top_left = boundry_top_left2;
    ftp_args2->boundry_bottom_right = boundry_bottom_right2;
    ftp_args2->boundry_bottom_left = boundry_bottom_left2;
    ftp_args2->num_layers = num_layers;
    ftp_args2->output = output_delta2; 
    ftp_args2->SHARED_WEIGHT_UPDATES = SHARED_WEIGHT_UPDATES;

    device_ftp_args* ftp_args3 = calloc(1, sizeof(device_ftp_args));

    ftp_args3->device_id = 2;
    ftp_args3->im_width = im_width;
    ftp_args3->im_height = im_height;
    ftp_args3->filter_size = filter_size;
    ftp_args3->image = image3;
    ftp_args3->delta = delta3;
    ftp_args3->boundry_top = boundry_top3;
    ftp_args3->boundry_bottom = boundry_bottom3;
    ftp_args3->boundry_left = boundry_left3;
    ftp_args3->boundry_right = boundry_right3;
    ftp_args3->boundry_top_right = boundry_top_right3;
    ftp_args3->boundry_top_left = boundry_top_left3;
    ftp_args3->boundry_bottom_right = boundry_bottom_right3;
    ftp_args3->boundry_bottom_left = boundry_bottom_left3;
    ftp_args3->num_layers = num_layers;
    ftp_args3->output = output_delta3;
    ftp_args3->SHARED_WEIGHT_UPDATES = SHARED_WEIGHT_UPDATES;

    device_ftp_args* ftp_args4 = calloc(1, sizeof(device_ftp_args)); 

    ftp_args4->device_id = 3;
    ftp_args4->im_width = im_width;
    ftp_args4->im_height = im_height;
    ftp_args4->filter_size = filter_size;
    ftp_args4->image = image4;
    ftp_args4->delta = delta4;
    ftp_args4->boundry_top = boundry_top4;
    ftp_args4->boundry_bottom = boundry_bottom4;
    ftp_args4->boundry_left = boundry_left4;
    ftp_args4->boundry_right = boundry_right4;
    ftp_args4->boundry_top_right = boundry_top_right4;
    ftp_args4->boundry_top_left = boundry_top_left4;
    ftp_args4->boundry_bottom_right = boundry_bottom_right4;
    ftp_args4->boundry_bottom_left = boundry_bottom_left4;
    ftp_args4->num_layers = num_layers;
    ftp_args4->output = output_delta4; 
    ftp_args4->SHARED_WEIGHT_UPDATES = SHARED_WEIGHT_UPDATES;


    sem_init(&filter_diverge, 0, 0);
    sem_init(&filter_converge, 0, 0);

    iret1 = pthread_create( &thread1, NULL, execute_dev_gateway, (void*) (ftp_args1));
    iret2 = pthread_create( &thread2, NULL, execute_dev, (void*) (ftp_args2));
    iret3 = pthread_create( &thread3, NULL, execute_dev, (void*) (ftp_args3));
    iret4 = pthread_create( &thread4, NULL, execute_dev, (void*) (ftp_args4));

     /* Wait till threads are complete before main continues. Unless we  */
     /* wait we run the risk of executing an exit which will terminate   */
     /* the process and all threads before the threads have completed.   */

     pthread_join( thread1, NULL);
     pthread_join( thread2, NULL); 
     pthread_join( thread3, NULL);
     pthread_join( thread4, NULL); 

     printf("Thread 1 returns: %d\n",iret1);
     printf("Thread 2 returns: %d\n",iret2);
     printf("Thread 3 returns: %d\n",iret3);
     printf("Thread 4 returns: %d\n",iret4);

     sem_destroy(&filter_diverge);
     sem_destroy(&filter_converge);








































    float* image_merged = calloc(50*50, sizeof(float));
    float* delta_merged= calloc(50*50, sizeof(float));
    // float* kitty_cat_merged = calloc(2500, sizeof(float));
    // float* output_delta_merged = calloc(50*50, sizeof(float));
    fill_cpu(2500, 1, image_merged, 1);
    fill_cpu(2500, 0, delta_merged, 1);
    for (int i = 5; i < 45; ++i)
    {
        for (int j = 5; j < 45; ++j)
        {
            delta_merged[i*50 + j] =  1.0;
        }
    }
    network *net = calloc(1, sizeof(network));
    net->n = 5;
    net->layers = calloc(net->n, sizeof(layer));
    net->seen = calloc(1, sizeof(size_t));
    net->t    = calloc(1, sizeof(int));
    net->cost = calloc(1, sizeof(float));
    net->layers[0] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[2] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[3] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[4] = make_convolutional_layer(1, 50, 50, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);  
    net->layers[0].batch_normalize = 0;
    net->layers[1].batch_normalize = 0;
    net->layers[2].batch_normalize = 0;
    net->layers[3].batch_normalize = 0;
    net->layers[4].batch_normalize = 0;

    //printf("%d, %d %d %d %d\n", im_height, im_width, core_image_height, core_image_width, boundry_frames);

    //net->layers[0].data = calloc(1, sizeof(float));
    net->input = calloc(2500, sizeof(float));
    net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));
    net->inputs = 2500;

    fill_cpu(2500, 0, net->layers[0].delta, 1);
    fill_cpu(2500, 0, net->layers[1].delta, 1);
    fill_cpu(2500, 0, net->layers[2].delta, 1);
    fill_cpu(2500, 0, net->layers[3].delta, 1);
    fill_cpu(2500, 0, net->layers[4].delta, 1);

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

    for (int i = 0; i < 50; ++i)
    {
        for (int j = 0; j < 50; ++j)
        {
            net->input[i*50 + j] = image_merged[i*50 + j];
        }
    }
 
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

    for (int i = 0; i < 50; ++i)
    {
        for (int j = 0; j < 50; ++j)
        {
            net->layers[4].delta[i*50 + j] = delta_merged[i*50 + j];
            net->layers[4].delta_with_boundry[i*50 + j] = delta_merged[i*50 + j];
        }
    }

    net->input = net->layers[3].output;
    net->delta = net->layers[3].delta;
    net->index = 4;
    backward_convolutional_layer(net->layers[4], *net);

    net->input = net->layers[2].output;
    net->delta = net->layers[2].delta;
    net->index = 3;
    backward_convolutional_layer(net->layers[3], *net);


    net->input = net->layers[1].output;
    net->delta = net->layers[1].delta;
    net->index = 2;
    backward_convolutional_layer(net->layers[2], *net);
    net->input = net->layers[0].output;
    net->delta = net->layers[0].delta;
    net->index = 1;
    backward_convolutional_layer(net->layers[1], *net);


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

    for (int i = 0; i < num_layers; ++i)
    {
        printf("Layer weights %d\n", i);
        for (int j = 0; j < 3; ++j)
        {
            for (int k = 0; k < 3; ++k)
            {
              printf("%.4f ", net->layers[i].weight_updates[(j*3) + k]);
            }
            printf("\n");
        }  
        printf("\n\n"); 
    }










     exit(0);
}

void *print_message_function( void *ptr )
{
     char *message;
     message = (char *) ptr;
     printf("%s \n", message);
}





#include "maxpool_layer.h"

int main(){

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
    net->layers[0] = make_convolutional_layer(1, 12, 12, 1, 1, 1, filter_size, 3, 1, RELU, 1, 0, 0, 0);
    net->layers[1] = make_convolutional_layer(1, 4, 4, 1, 1, 1, filter_size, 1, 1, RELU, 1, 0, 0, 0);
    net->layers[0].batch_normalize = 0;
    net->layers[1].batch_normalize = 0;

    net->input = calloc(144, sizeof(float));
    net->workspace = calloc(net->layers[0].workspace_size, sizeof(float));

    for (int i = 0; i < filter_size; ++i)
    {
        for (int j = 0; j < filter_size; ++j)
        {
            net->layers[0].weights[i*filter_size + j] = 1.0;
            net->layers[1].weights[i*filter_size + j] = 1.0;
        }
    }

    float* image_merged = calloc(12*12, sizeof(float));
    fill_cpu(144, 1, image_merged, 1);

    for (int i = 0; i < 12; ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            net->input[i*12 + j] = image_merged[i*12 + j];
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

    net->input = net->layers[0].output;
    net->index = 1;
    forward_convolutional_layer(net->layers[1], *net);

    for (int i = 0; i < net->layers[1].out_h; ++i)
    {
        for (int j = 0; j < net->layers[1].out_w; ++j)
        {
            printf("%.4f ", net->layers[1].output[(i*net->layers[1].out_w) + j]);
        }
        printf("\n");
    }








    float* delta = calloc(12*12, sizeof(float));

    fill_cpu(16, 0, net->layers[0].delta, 1);

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            net->layers[1].delta[i*4 + j] =  1.0;
        }
    }

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            printf("%.1f ", net->layers[1].delta[(i*4) + j]);
        }
        printf("\n");
    }

    net->input = net->layers[0].output;
    net->delta = net->layers[0].delta;
    net->index = 1;
    backward_convolutional_layer(net->layers[1], *net);

    net->input = net->layers[0].output;
    net->delta = delta;
    net->index = 0;
    backward_convolutional_layer(net->layers[0], *net);

    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            printf("%.1f ", net->layers[0].delta[(i*4) + j]);
        }
        printf("\n");
    }

    for (int i = 0; i < 12; ++i)
    {
        for (int j = 0; j < 12; ++j)
        {
            printf("%.1f ", delta[(i*12) + j]);
        }
        printf("\n");
    }

}
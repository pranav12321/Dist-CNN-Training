#include "transport.h"

#include <stdio.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h> // read(), write(), close()
#define MAX 800
#define PORT 8080 //SERVER PORT
#define SA struct sockaddr
  
int sockfd, connfd;

char transmit_buffer[MAX];
char receive_buffer[MAX];
// char receive_buffer1[MAX];
// char receive_buffer2[MAX];

#define DEVICE_ID_X 1
#define DEVICE_ID_Y 0

// void init_transport(){

// 	int connfd;
//     struct sockaddr_in servaddr, cli;
 
//     // socket create and verification
//     sockfd = socket(AF_INET, SOCK_STREAM, 0);
//     if (sockfd == -1) {
//         printf("socket creation failed...\n");
//         exit(0);
//     }
//     else
//         printf("Socket successfully created..\n");
//     bzero(&servaddr, sizeof(servaddr));
 
//     // assign IP, PORT
//     servaddr.sin_family = AF_INET;
//     servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
//     servaddr.sin_port = htons(PORT);
 
//     // connect the client socket to server socket
//     if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
//         != 0) {
//         printf("connection with the server failed...\n");
//         exit(0);
//     }
//     else
//         printf("connected to the server..\n");

// }


void init_transport(){
    int len;
    struct sockaddr_in servaddr, cli;
   
    // socket create and verification
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");
    bzero(&servaddr, sizeof(servaddr));
   
    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(PORT);
   
    // Binding newly created socket to given IP and verification
    if ((bind(sockfd, (SA*)&servaddr, sizeof(servaddr))) != 0) {
        printf("socket bind failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully binded..\n");
   
    // Now server is ready to listen and verification
    if ((listen(sockfd, 5)) != 0) {
        printf("Listen failed...\n");
        exit(0);
    }
    else
        printf("Server listening..\n");
    len = sizeof(cli);
   
    // Accept the data packet from client and verification
    connfd = accept(sockfd, (SA*)&cli, &len);
    if (connfd < 0) {
        printf("server accept failed...\n");
        exit(0);
    }
    else
        printf("server accept the client...\n");
}


void send_boundry(float* data, int size, int device_id_x, int device_id_y){
    transmit_buffer[0] = 1;
	transmit_buffer[1] = DEVICE_ID_X;
	transmit_buffer[2] = DEVICE_ID_Y;
	transmit_buffer[3] = device_id_x;
	transmit_buffer[4] = device_id_y;
	memcpy(transmit_buffer+5, data, size*(sizeof(float)));

    // printf("transmitting\n");

    //     for (int i = 0; i < (size*(sizeof(float)) + 5); ++i)
    //     {
    //         printf("%d\n", transmit_buffer[i]);
    //     }
    //     printf("\n");

	write(connfd, transmit_buffer, size*(sizeof(float)) + 5);
}
void receive_boundry(float* data, int size, int device_id_x, int device_id_y){

    while(1){

        if(receive_buffer[0] == 1 && (receive_buffer[1] == device_id_x) && (receive_buffer[2] == device_id_y) 
            && (receive_buffer[3] == DEVICE_ID_X) && (receive_buffer[4] == DEVICE_ID_Y) ){
            memcpy(data, receive_buffer+5, size*sizeof(float));
            receive_buffer[0] = 0;

            return;// success
        }

        int bytes = read(connfd, receive_buffer, size*sizeof(float) + 5);
        // printf("%d\n\n", bytes);
        // for (int i = 0; i < bytes; ++i)
        // {
        //     printf("%d\n", receive_buffer[i]);
        // }
        // printf("\n");

    }

//	transmit_buffer[0] = DEVICE_ID_X;
//	transmit_buffer[1] = DEVICE_ID_Y;
//	transmit_buffer[2] = device_id_x;
//	transmit_buffer[3] = device_id_y;
//	transmit_buffer[4] = size;
	// memcpy(transmit_buffer+5, size*(sizeof(float)));

	// 
}
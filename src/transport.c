#include "transport.h"

#include <stdio.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h> // read(), write(), close()
#include <fcntl.h>

#include <errno.h>
#include <arpa/inet.h> //close
#include <sys/time.h> //FD_SET, FD_ISSET, FD_ZERO macros

#define MAX 40000
#define PORT 8080 //SERVER PORT
#define SA struct sockaddr

#define NUM_TILES_X 2
#define NUM_TILES_Y 2

int z;
int o;
int t;
int tt;

client_structure* network_links[NUM_TILES_X*NUM_TILES_Y];

uint8_t receive_buffer[200000];

void get_device_ip(int device_id_x, int device_id_y, char* ip){
    if(device_id_x == 0 && device_id_y == 0){
        strcpy(ip, "127.0.0.1");
    }
    else if(device_id_x == 1 && device_id_y == 0){
        strcpy(ip, "127.0.0.1");
    }
    else if(device_id_x == 0 && device_id_y == 1){
        strcpy(ip, "127.0.0.1");
    }
    else if(device_id_x == 1 && device_id_y == 1){
        strcpy(ip, "127.0.0.1");
    }
}

void init_transport(){

    struct sockaddr_in servaddr, cli;
    int sockfd, connfd, len;
 

    //SERVER ENDPOINTS

     for (int i = (DEVICE_ID_X + NUM_TILES_X*DEVICE_ID_Y + 1) ; i < (NUM_TILES_X*NUM_TILES_Y); ++i)
    {
        client_structure* cs = calloc(1, sizeof(client_structure));
        cs->receive_buffer = calloc(MAX_BOUNDARY_SIZE_PER_DEVICE*3, sizeof(float));
        network_links[i] = cs;
        cs->endpoint_type = 1;

        for(int j = 0; j < 2; j++){
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
            servaddr.sin_port = htons(PORT + 4*(DEVICE_ID_X) + 8*(DEVICE_ID_Y) + i + 100*j);

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
                printf("Server listening for client %d %d on port %d\n", i%NUM_TILES_X, i/NUM_TILES_X, PORT + 4*(DEVICE_ID_X) + 8*(DEVICE_ID_Y) + i);
            len = sizeof(cli);

            // Accept the data packet from client and verification
            connfd = accept(sockfd, (SA*)&cli, &len);
            if (connfd < 0) {
                printf("server accept failed...\n");
                exit(0);
            }
            else
                printf("server %d %d successfully accepted the client %d %d on port %d\n", DEVICE_ID_X, DEVICE_ID_Y, i%NUM_TILES_X, i/NUM_TILES_X, PORT + 4*(DEVICE_ID_X) + 8*(DEVICE_ID_Y) + i + j*100);

            cs->socket_fd[j] = connfd;
            cs->endpoint_type = 1;
        }

        for (int j = 0; j < MAX_PACKETS_PER_DEVICE; ++j)
        {
            cs->receive_device_data_ptrs[j].valid = 0;
        }


    }   


    //CLIENT ENDPOINTS
    char ip[15];
    
    for (int i = (DEVICE_ID_X + NUM_TILES_X*DEVICE_ID_Y - 1); i >= 0 ; --i)
    {
        client_structure* cs = calloc(1, sizeof(client_structure));
        cs->receive_buffer = calloc(MAX_BOUNDARY_SIZE_PER_DEVICE, sizeof(uint8_t));
        network_links[i] = cs;
        cs->endpoint_type = 0;

        get_device_ip(i%NUM_TILES_X, i/NUM_TILES_X, ip);
        // socket create and verification

        for (int j = 0; j < 2; ++j)
        {
            int sockfd = -1;
            while(sockfd == -1){
                sockfd = socket(AF_INET, SOCK_STREAM, 0);
                if (sockfd == -1) {
                    // printf("socket creation failed...\n");
                    // exit(0);
                }
                else
                    printf("Socket successfully created\n");
            }
            bzero(&servaddr, sizeof(servaddr));
         
            // assign IP, PORT
            servaddr.sin_family = AF_INET;
            servaddr.sin_addr.s_addr = inet_addr(ip);
            servaddr.sin_port = htons(PORT + 8*((i>>1) & 0x1) + 4*(i & 0x1) + (DEVICE_ID_X + NUM_TILES_X*DEVICE_ID_Y) + j*100);

            printf("Attempting to connect to server %d %d on port %d\n", i%NUM_TILES_X, i/NUM_TILES_X, PORT + 8*((i>>1) & 0x1) + 4*(i & 0x1) + (DEVICE_ID_X + NUM_TILES_X*DEVICE_ID_Y) + j*100);
         
            // connect the client socket to server socket
            while (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
                != 0) {
                //printf("connection with the server failed...\n");
                //exit(0);
            }
            //else
            printf("client %d %d endpoint successfully connected with server device %d %d on port %d\n", DEVICE_ID_X, DEVICE_ID_Y, i%NUM_TILES_X, i/NUM_TILES_X, PORT + 8*((i>>1) & 0x1) + 4*(i & 0x1) + (DEVICE_ID_X + NUM_TILES_X*DEVICE_ID_Y));


            int flags = fcntl(sockfd, F_GETFL, 0);
            fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

            cs->socket_fd[j] = sockfd;
            cs->endpoint_type = 0;
        }

        for (int j = 0; j < MAX_PACKETS_PER_DEVICE; ++j)
        {
            cs->receive_device_data_ptrs[j].valid = 0;
        }
    }

}

#define MAX_PACKET_ELEMENTS 5000
#define ACK_SIZE 9
char* ack = "RECEIVED";


comm_entry transmit_entries[NUM_TILES_X*NUM_TILES_Y];
comm_entry receive_entries[NUM_TILES_X*NUM_TILES_Y];


void send_boundry(float* data, int size, int device_id_x, int device_id_y){


    int num_transactions = (size/MAX_PACKET_ELEMENTS + ((size%MAX_PACKET_ELEMENTS > 0) ? 1 : 0) );
    int cumulative_sent_size = 0;
    char ack[15];


    while(size > 0){
        int transaction_size = (size > MAX_PACKET_ELEMENTS ? MAX_PACKET_ELEMENTS : size);

        //printf("Client %d %d sending %d bytes to device %d %d\n", DEVICE_ID_X, DEVICE_ID_Y, transaction_size*(sizeof(float)), device_id_x, device_id_y);

        int to_send = 0;

        to_send = write(network_links[device_id_x + NUM_TILES_X*device_id_y]->socket_fd[0], data + cumulative_sent_size, transaction_size*sizeof(float));

        if(to_send < (transaction_size*sizeof(float))){
            printf("SEND FAILURE: Expected %d Actual : %d \n\n", transaction_size*sizeof(float), to_send);
            exit(0);
        }

        //printf("Client %d %d waiting for ack from device %d %d\n", DEVICE_ID_X, DEVICE_ID_Y, device_id_x, device_id_y);

        int bytes = 0;

        while(bytes < 9){
            int temp = read( network_links[device_id_x + NUM_TILES_X*device_id_y]->socket_fd[1] , network_links[device_id_x + NUM_TILES_X*device_id_y]->receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float));
            if(temp > 0)
                bytes += temp;
        }

        //printf("Client %d %d received ack from device %d %d size = %d %s\n", DEVICE_ID_X, DEVICE_ID_Y, device_id_x, device_id_y, bytes, network_links[device_id_x + NUM_TILES_X*device_id_y]->receive_buffer);

        cumulative_sent_size += transaction_size;
        size -= transaction_size;
    }

}

void receive_boundry(float* data_float, int size, int device_id_x, int device_id_y){

    size = size*sizeof(float);
    int num_transactions = (size/MAX_PACKET_ELEMENTS + ((size%MAX_PACKET_ELEMENTS > 0) ? 1 : 0) );
    int cumulative_received_size = 0;

    char ack[15];

    uint8_t* data = (uint8_t*)data_float;

    while(size > 0){

        int transaction_size = (size > (MAX_PACKET_ELEMENTS*4) ? (MAX_PACKET_ELEMENTS*4) : size);
        int temp = transaction_size;

       // printf("Size: %d Transaction size: %d\n", size, transaction_size);

        while(transaction_size > 0){

            int bytes = recv( network_links[device_id_x + NUM_TILES_X*device_id_y]->socket_fd[0] , receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float), MSG_DONTWAIT);

            // if(bytes > 0)
            //     printf("Client %d %d received %d raw bytes\n", DEVICE_ID_X, DEVICE_ID_Y, bytes);

            if(bytes > 0){     
                memcpy(data + cumulative_received_size, receive_buffer, transaction_size);
                cumulative_received_size += bytes;
                transaction_size -= bytes;
            }
        }

        size -= temp;
       // printf("%s\n", "send ACK");

        int to_send = 0;
        
        to_send = write(network_links[device_id_x + NUM_TILES_X*device_id_y]->socket_fd[1], "Received", ACK_SIZE);
        if(to_send < (ACK_SIZE)){
            printf("SEND FAILURE: Expected %d Actual : %d \n\n", ACK_SIZE, to_send);
            exit(0);
        }
    
    }   

}
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
#include <sys/un.h>

#define CLIENT_SOCK_FILE "client.sock"
#define SERVER_SOCK_FILE "server.sock"

#define MAX 40000
#define PORT 7500 //SERVER PORT
#define SA struct sockaddr

client_structure** network_links;

uint8_t receive_buffer[200000];

char** DEVICE_IPs;



void server_accept_local_tile_common(int num_nodes, int device_id, int i, int j, client_structure* cs){
    struct sockaddr_un servaddr, cli;
    int sockfd, connfd, len;
    char connection_str[4];
    connection_str[0] = '0' + device_id;
    connection_str[1] = '0' + i;
    connection_str[2] = '0' + j;
    connection_str[3] = '\0';
    // socket create and verification

    sockfd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");

    int port = PORT + (num_nodes)*(device_id) + i + 600*j;
    bzero(&servaddr, sizeof(servaddr));

    // assign IP, PORT
    servaddr.sun_family = AF_UNIX;
    // servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    // servaddr.sin_port = htons(port);
    //addr.sun_family = AF_UNIX;
    strcpy(servaddr.sun_path, connection_str);
    unlink(connection_str);

    // Binding newly created socket to given IP and verification
    if ((bind(sockfd, (SA*)&servaddr, sizeof(servaddr))) != 0) {
        printf("socket bind failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully binded..\n");

    // Now server is ready to listen and verification
    if ((listen(sockfd, 20)) != 0) {
        printf("Listen failed...\n");
        exit(0);
    }
    else
        printf("Server listening for client %d on port %d\n", i, port);
    len = sizeof(cli);

    // Accept the data packet from client and verification
    connfd = accept(sockfd, (SA*)&cli, &len);
    if (connfd < 0) {
        printf("server accept failed...\n");
        exit(0);
    }
    else
        printf("server %d successfully accepted the client %d on port %d\n", device_id, i, port);

    cs->socket_fd[j] = connfd;
    cs->endpoint_type = 1;
}

void server_accept_network_tile_common(int num_nodes, int device_id, int i, int j, client_structure* cs){
    struct sockaddr_in servaddr, cli;
    int sockfd, connfd, len;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");

    int port = PORT + (num_nodes)*(device_id) + i + 600*j;
    bzero(&servaddr, sizeof(servaddr));

    // assign IP, PORT
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servaddr.sin_port = htons(port);

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
        printf("Server listening for client %d on port %d\n", i, port);
    len = sizeof(cli);

    // Accept the data packet from client and verification
    connfd = accept(sockfd, (SA*)&cli, &len);
    if (connfd < 0) {
        printf("server accept failed...\n");
        exit(0);
    }
    else
        printf("server %d successfully accepted the client %d on port %d\n", device_id, i, port);

    cs->socket_fd[j] = connfd;
    cs->endpoint_type = 1;
}



void client_connect_local_tile_common(int num_nodes, int device_id, int i, int j, client_structure* cs){
    struct sockaddr_un servaddr, cli;
    int sockfd, connfd, len;

    char ip[15];
    //get_device_ip(i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, ip);
    strcpy(ip, DEVICE_IPs[i]);

    char connection_str[4];
    connection_str[0] = '0' + i;
    connection_str[1] = '0' + device_id;
    connection_str[2] = '0' + j;
    connection_str[3] = '\0';

    sockfd = -1;
    while(sockfd == -1){
        sockfd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
        if (sockfd == -1) {
            // printf("socket creation failed...\n");
            // exit(0);
        }
        else
            printf("Socket successfully created\n");
    }
    bzero(&servaddr, sizeof(servaddr));
 
    // assign IP, PORT
    servaddr.sun_family = AF_UNIX;
    strncpy(servaddr.sun_path, connection_str, sizeof(servaddr.sun_path) - 1);
   // servaddr.sin_addr.s_addr = inet_addr(ip);
    //servaddr.sin_port = htons(PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*i + (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y) + j*600);

    printf("Attempting to connect to server %d on port %d at ip %s\n", i, PORT + (num_nodes)*i + (device_id) + j*600, ip);
 
    // connect the client socket to server socket
    while (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
        != 0) {
        //printf("connection with the server failed...\n");
        //exit(0);
    }
    //else
    //printf("client %d %d endpoint successfully connected with server device %d %d on port %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*i + (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y)  + j*600);


    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

    cs->socket_fd[j] = sockfd;
    cs->endpoint_type = 0;
}

void client_connect_network_tile_common(int num_nodes, int device_id, int i, int j, client_structure* cs){
    struct sockaddr_in servaddr, cli;
    int sockfd, connfd, len;

    char ip[15];
    strcpy(ip, DEVICE_IPs[i]);
    //get_device_ip(i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, ip);

    sockfd = -1;
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
    servaddr.sin_port = htons(PORT + (num_nodes)*i + (device_id) + j*600);

    printf("Attempting to connect to server %d on port %d at ip %s\n", i, PORT + (num_nodes)*i + (device_id) + j*600, ip);
 
    // connect the client socket to server socket
    while (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
        != 0) {
        //printf("connection with the server failed...\n");
        //exit(0);
    }
    //else
    //printf("client %d %d endpoint successfully connected with server device %d %d on port %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*i + (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y)  + j*600);


    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

    cs->socket_fd[j] = sockfd;
    cs->endpoint_type = 0;
}





void init_transport_common(int num_nodes, int device_id, char* argv[]){

    
    int sockfd, connfd, len;
    char current_device_ip[32];

    DEVICE_IPs = calloc(32, sizeof(char*));
    for (int i = 0; i < 32; ++i)
    {
        DEVICE_IPs[i] = calloc(32, sizeof(char));
    }

    network_links = (client_structure**)calloc(num_nodes, sizeof(client_structure*));


    for (int i = 0; i < num_nodes; ++i)
    {
        strcpy(DEVICE_IPs[i], argv[i+7]);
    }

    printf("came here\n");

    //SERVER ENDPOINTS

     for (int i = (device_id + 1) ; i < num_nodes; ++i)
    {

        client_structure* cs = calloc(1, sizeof(client_structure));
        cs->receive_buffer = calloc(MAX_BOUNDARY_SIZE_PER_DEVICE*3, sizeof(float));

        network_links[i] = cs;
        cs->endpoint_type = 1;

        int connection = AF_INET;


        for(int j = 0; j < 2; j++){

            if(strcmp(DEVICE_IPs[i], DEVICE_IPs[device_id]) == 0){
                server_accept_local_tile_common(num_nodes, device_id, i, j, cs);
            }
            else{
                server_accept_network_tile_common(num_nodes, device_id, i, j, cs);
            }
        }

        for (int j = 0; j < MAX_PACKETS_PER_DEVICE; ++j)
        {
            cs->receive_device_data_ptrs[j].valid = 0;
        }
    }   
    
    for (int i = (device_id - 1); i >= 0 ; --i)
    {
        client_structure* cs = calloc(1, sizeof(client_structure));
        cs->receive_buffer = calloc(MAX_BOUNDARY_SIZE_PER_DEVICE, sizeof(uint8_t));
        network_links[i] = cs;
        cs->endpoint_type = 0;

        // socket create and verification

        for (int j = 0; j < 2; ++j)
        {
            if(strcmp(DEVICE_IPs[i], DEVICE_IPs[device_id]) == 0){
                client_connect_local_tile_common(num_nodes, device_id, i, j, cs);
            }
            else{
                client_connect_network_tile_common(num_nodes, device_id, i, j, cs);
            }
        }

        for (int j = 0; j < MAX_PACKETS_PER_DEVICE; ++j)
        {
            cs->receive_device_data_ptrs[j].valid = 0;
        }
    }

}

#define MAX_PACKET_ELEMENTS 5000
#define ACK_SIZE 9
extern char* ack;


void send_data(float* data, int size, int device_id){


    int num_transactions = (size/MAX_PACKET_ELEMENTS + ((size%MAX_PACKET_ELEMENTS > 0) ? 1 : 0) );
    int cumulative_sent_size = 0;
    char ack[15];

    printf("Client sending %d bytes to device %d \n", size*(sizeof(float)), device_id);

    while(size > 0){
        int transaction_size = (size > MAX_PACKET_ELEMENTS ? MAX_PACKET_ELEMENTS : size);

        //printf("Client sending %d bytes to device %d \n", transaction_size*(sizeof(float)), device_id);

        int to_send = 0;

        //to_send = write(network_links[device_id]->socket_fd[0], data + cumulative_sent_size, transaction_size*sizeof(float));

        int sent = 0;
        while(sent < (transaction_size*sizeof(float))){
            sent += write(network_links[device_id]->socket_fd[0], (uint8_t*)(data + cumulative_sent_size + sent), transaction_size*sizeof(float) - sent);
        }
        // if(to_send < (transaction_size*sizeof(float))){
        //     printf("SEND FAILURE: Expected %d Actual : %d \n\n", transaction_size*sizeof(float), to_send);
        //     exit(0);
        // }

        //printf("Client %d %d waiting for ack from device %d %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, device_id_x, device_id_y);

        int bytes = 0;

        while(bytes < 9){
            int temp = read( network_links[device_id]->socket_fd[1] , network_links[device_id]->receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float));
            if(temp > 0)
                bytes += temp;
        }

        //printf("Client %d %d received ack from device %d %d size = %d %s\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, bytes, network_links[ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y]->receive_buffer);

        cumulative_sent_size += transaction_size;
        size -= transaction_size;
    }

}

void receive_data(float* data_float, int size, int device_id){

    size = size*sizeof(float);
    int num_transactions = (size/MAX_PACKET_ELEMENTS + ((size%MAX_PACKET_ELEMENTS > 0) ? 1 : 0) );
    int cumulative_received_size = 0;

    char ack[15];

    uint8_t* data = (uint8_t*)data_float;

    printf("Receiving %d bytes from device %d \n", size, device_id);

    while(size > 0){

        int transaction_size = (size > (MAX_PACKET_ELEMENTS*4) ? (MAX_PACKET_ELEMENTS*4) : size);
        int temp = transaction_size;

       // printf("Size: %d Transaction size: %d\n", size, transaction_size);

        while(transaction_size > 0){

            int bytes = recv( network_links[device_id]->socket_fd[0] , receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float), MSG_DONTWAIT);

            // if(bytes > 0)
            //     printf("Client %d %d received %d raw bytes\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, bytes);

            if(bytes > 0){     
                memcpy(data + cumulative_received_size, receive_buffer, transaction_size);
                cumulative_received_size += bytes;
                transaction_size -= bytes;
            }
        }

        size -= temp;
       // printf("%s\n", "send ACK");

        int to_send = 0;
        
        to_send = write(network_links[device_id]->socket_fd[1], "Received", ACK_SIZE);
        if(to_send < (ACK_SIZE)){
            printf("SEND FAILURE: Expected %d Actual : %d \n\n", ACK_SIZE, to_send);
            exit(0);
        }
    
    }   

}
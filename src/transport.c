#include "transport.h"
#include "fused_device.h"

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

extern network_config network_params_original;
extern network_config network_params_tile;
extern ftp_config ftp_params;

extern device_tile current_tile;
extern network_device current_device;
extern ftp_network ftp_cluster;

client_structure** network_links;

uint8_t receive_buffer[200000];

char** DEVICE_IPs;

void get_device_ip(int device_id_x, int device_id_y, char* ip){
        strcpy(ip, DEVICE_IPs[device_id_y*ftp_params.NUM_TILES_X + device_id_x]);
}

void server_accept_local_tile(int i, int j, client_structure* cs){
    struct sockaddr_un servaddr, cli;
    int sockfd, connfd, len;
    char connection_str[6];
    connection_str[0] = '0' + ftp_params.DEVICE_ID_X;
    connection_str[1] = '0' + ftp_params.DEVICE_ID_Y;
    connection_str[2] = '0' + i%ftp_params.NUM_TILES_X;
    connection_str[3] = '0' + i/ftp_params.NUM_TILES_X;
    connection_str[4] = '0' + j;
    connection_str[5] = '\0';
    // socket create and verification

    sockfd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");

    int port = PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*((ftp_params.DEVICE_ID_X) + (ftp_params.NUM_TILES_X)*(ftp_params.DEVICE_ID_Y)) + i + 600*j;
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
        printf("Server listening for client %d %d on port %d\n", i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, port);
    len = sizeof(cli);

    // Accept the data packet from client and verification
    connfd = accept(sockfd, (SA*)&cli, &len);
    if (connfd < 0) {
        printf("server accept failed...\n");
        exit(0);
    }
    else
        printf("server %d %d successfully accepted the client %d %d on port %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, port);

    cs->socket_fd[j] = connfd;
    cs->endpoint_type = 1;
}

void server_accept_network_tile(int i, int j, client_structure* cs){
    struct sockaddr_in servaddr, cli;
    int sockfd, connfd, len;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");

    int port = PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*((ftp_params.DEVICE_ID_X) + (ftp_params.NUM_TILES_X)*(ftp_params.DEVICE_ID_Y)) + i + 600*j;
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
        printf("Server listening for client %d %d on port %d\n", i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, port);
    len = sizeof(cli);

    // Accept the data packet from client and verification
    connfd = accept(sockfd, (SA*)&cli, &len);
    if (connfd < 0) {
        printf("server accept failed...\n");
        exit(0);
    }
    else
        printf("server %d %d successfully accepted the client %d %d on port %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, port);

    cs->socket_fd[j] = connfd;
    cs->endpoint_type = 1;
}



void client_connect_local_tile(int i, int j, client_structure* cs){
    struct sockaddr_un servaddr, cli;
    int sockfd, connfd, len;

    char ip[15];
    get_device_ip(i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, ip);

    char connection_str[6];
    connection_str[0] = '0' + i%ftp_params.NUM_TILES_X;
    connection_str[1] = '0' + i/ftp_params.NUM_TILES_X;
    connection_str[2] = '0' + ftp_params.DEVICE_ID_X;
    connection_str[3] = '0' + ftp_params.DEVICE_ID_Y;
    connection_str[4] = '0' + j;
    connection_str[5] = '\0';

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

    printf("Attempting to connect to server %d %d on port %d at ip %s\n", i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*i + (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y) + j*600, ip);
 
    // connect the client socket to server socket
    while (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
        != 0) {
        //printf("connection with the server failed...\n");
        //exit(0);
    }
    //else
    printf("client %d %d endpoint successfully connected with server device %d %d on port %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*i + (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y)  + j*600);


    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

    cs->socket_fd[j] = sockfd;
    cs->endpoint_type = 0;
}

void client_connect_network_tile(int i, int j, client_structure* cs){
    struct sockaddr_in servaddr, cli;
    int sockfd, connfd, len;

    char ip[15];
    get_device_ip(i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, ip);

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
    servaddr.sin_port = htons(PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*i + (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y) + j*600);

    printf("Attempting to connect to server %d %d on port %d at ip %s\n", i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*i + (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y) + j*600, ip);
 
    // connect the client socket to server socket
    while (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
        != 0) {
        //printf("connection with the server failed...\n");
        //exit(0);
    }
    //else
    printf("client %d %d endpoint successfully connected with server device %d %d on port %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, i%ftp_params.NUM_TILES_X, i/ftp_params.NUM_TILES_X, PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*i + (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y)  + j*600);


    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

    cs->socket_fd[j] = sockfd;
    cs->endpoint_type = 0;
}

void init_transport(char* argv[]){

    
    int sockfd, connfd, len;
    char current_device_ip[32];

    DEVICE_IPs = calloc(32, sizeof(char*));
    for (int i = 0; i < 32; ++i)
    {
        DEVICE_IPs[i] = calloc(32, sizeof(char));
    }

    network_links = (client_structure**)calloc(ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y, sizeof(client_structure*));


    for (int i = 0; i < (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y); ++i)
    {
        strcpy(DEVICE_IPs[i], argv[i+3]);
    }

    ftp_cluster.total_devices = 0;
    ftp_cluster.total_tiles = (ftp_params.NUM_TILES_X)*(ftp_params.NUM_TILES_Y);

    for (int i = 0; i < ftp_cluster.total_devices; ++i)
    {
        ftp_cluster.devices[i].num_tiles = 0;
    }

    ftp_cluster.devices = calloc(32, sizeof(network_device));
    for (int i = 0; i < ftp_cluster.total_devices; ++i)
    {
        ftp_cluster.devices[i].device_tiles = calloc(32, sizeof(device_tile));
    }


    for (int i = 0; i < (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y); ++i)
    {
        int flag = 0;
        int tile_assigned_device_id = 0;
        for (int j = 0; j < i; ++j){
            if(strcmp(DEVICE_IPs[i], DEVICE_IPs[j]) == 0){
                flag = 1;
                break;
            }
        }

        if(flag == 0){
            ftp_cluster.devices[ftp_cluster.total_devices].representative_tile_network_id = i;
            strcpy(ftp_cluster.devices[ftp_cluster.total_devices].device_ip, DEVICE_IPs[i]);
            if(i == ((ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y) + ftp_params.DEVICE_ID_X)){
                strcpy(current_device.device_ip, DEVICE_IPs[i]);
                current_device.representative_tile_network_id = i;
                current_device.num_tiles = 0;
            }
            ftp_cluster.total_devices++;
        }
    }

    current_device.device_tiles = calloc(32, sizeof(device_tile));
    strcpy(current_device_ip, DEVICE_IPs[(ftp_params.NUM_TILES_X * ftp_params.DEVICE_ID_Y) + ftp_params.DEVICE_ID_X]);
    
    for (int i = 0; i < (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y); ++i)
    {
        if(strcmp(DEVICE_IPs[i], current_device_ip) == 0){
            current_device.device_tiles[current_device.num_tiles].network_tile_id = i;
            current_device.device_tiles[current_device.num_tiles].device_tile_id = current_device.num_tiles;
            current_device.device_tiles[current_device.num_tiles].is_device_representative_tile = (current_device.num_tiles == 0) ? 1 : 0;

            if(i == ((ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y) + ftp_params.DEVICE_ID_X)){
                current_tile = current_device.device_tiles[current_device.num_tiles];
            }
            current_device.num_tiles++;
        }
    }    

    //SERVER ENDPOINTS
    printf("IDx: %d IDY: %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y);

     for (int i = (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y + 1) ; i < (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y); ++i)
    {

        client_structure* cs = calloc(1, sizeof(client_structure));
        cs->receive_buffer = calloc(MAX_BOUNDARY_SIZE_PER_DEVICE*3, sizeof(float));

        network_links[i] = cs;
        cs->endpoint_type = 1;

        int connection = AF_INET;


        for(int j = 0; j < 2; j++){

            if(strcmp(DEVICE_IPs[i], DEVICE_IPs[ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y]) == 0){
                server_accept_local_tile(i, j, cs);
            }
            else{
                server_accept_network_tile(i, j, cs);
            }
        }

        for (int j = 0; j < MAX_PACKETS_PER_DEVICE; ++j)
        {
            cs->receive_device_data_ptrs[j].valid = 0;
        }
    }   
    
    for (int i = (ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y - 1); i >= 0 ; --i)
    {
        client_structure* cs = calloc(1, sizeof(client_structure));
        cs->receive_buffer = calloc(MAX_BOUNDARY_SIZE_PER_DEVICE, sizeof(uint8_t));
        network_links[i] = cs;
        cs->endpoint_type = 0;

        // socket create and verification

        for (int j = 0; j < 2; ++j)
        {
            if(strcmp(DEVICE_IPs[i], DEVICE_IPs[ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y]) == 0){
                client_connect_local_tile(i, j, cs);
            }
            else{
                client_connect_network_tile(i, j, cs);
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
char* ack = "RECEIVED";


void send_boundry(float* data, int size, int device_id_x, int device_id_y){


    int num_transactions = (size/MAX_PACKET_ELEMENTS + ((size%MAX_PACKET_ELEMENTS > 0) ? 1 : 0) );
    int cumulative_sent_size = 0;
    char ack[15];


    while(size > 0){
        int transaction_size = (size > MAX_PACKET_ELEMENTS ? MAX_PACKET_ELEMENTS : size);

        //printf("Client %d %d sending %d bytes to device %d %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, transaction_size*(sizeof(float)), device_id_x, device_id_y);

        int to_send = 0;

        to_send = write(network_links[device_id_x + ftp_params.NUM_TILES_X*device_id_y]->socket_fd[0], data + cumulative_sent_size, transaction_size*sizeof(float));

        if(to_send < (transaction_size*sizeof(float))){
            printf("SEND FAILURE: Expected %d Actual : %d \n\n", transaction_size*sizeof(float), to_send);
            exit(0);
        }

        //printf("Client %d %d waiting for ack from device %d %d\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, device_id_x, device_id_y);

        int bytes = 0;

        while(bytes < 9){
            int temp = read( network_links[device_id_x + ftp_params.NUM_TILES_X*device_id_y]->socket_fd[1] , network_links[device_id_x + ftp_params.NUM_TILES_X*device_id_y]->receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float));
            if(temp > 0)
                bytes += temp;
        }

        //printf("Client %d %d received ack from device %d %d size = %d %s\n", ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, ftp_params.DEVICE_ID_X, ftp_params.DEVICE_ID_Y, bytes, network_links[ftp_params.DEVICE_ID_X + ftp_params.NUM_TILES_X*ftp_params.DEVICE_ID_Y]->receive_buffer);

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

            int bytes = recv( network_links[device_id_x + ftp_params.NUM_TILES_X*device_id_y]->socket_fd[0] , receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float), MSG_DONTWAIT);

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
        
        to_send = write(network_links[device_id_x + ftp_params.NUM_TILES_X*device_id_y]->socket_fd[1], "Received", ACK_SIZE);
        if(to_send < (ACK_SIZE)){
            printf("SEND FAILURE: Expected %d Actual : %d \n\n", ACK_SIZE, to_send);
            exit(0);
        }
    
    }   

}
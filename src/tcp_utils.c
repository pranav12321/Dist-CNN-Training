#include "tcp_utils.h"

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

#define PORT 7500 //SERVER PORT
#define SA struct sockaddr

client_structure** network_links;

void static server_accept_local_client(char** IPs, int num_nodes, int node_id, int client_id, int channel_idx, client_structure* cs){
    struct sockaddr_un servaddr, cli;
    int sockfd, connfd, len;
    char connection_str[4];
    connection_str[0] = '0' + node_id;
    connection_str[1] = '0' + client_id;
    connection_str[2] = '0' + channel_idx;
    connection_str[3] = '\0';
    // socket create and verification

    sockfd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");

    int port = PORT + (num_nodes*node_id*2) + client_id*2 + channel_idx;
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
        printf("Server listening for client %d on port %d\n", client_id, port);
    len = sizeof(cli);

    // Accept the data packet from client and verification
    connfd = accept(sockfd, (SA*)&cli, &len);
    if (connfd < 0) {
        printf("server accept failed...\n");
        exit(0);
    }
    else
        printf("server %d successfully accepted the client %d on port %d\n", node_id, client_id, port);

    cs->socket_fd[channel_idx] = connfd;
    cs->endpoint_type = 1;
}

void static server_accept_network_client(char** IPs, int num_nodes, int node_id, int client_id, int channel_idx, client_structure* cs){
    struct sockaddr_in servaddr, cli;
    int sockfd, connfd, len;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd == -1) {
        printf("socket creation failed...\n");
        exit(0);
    }
    else
        printf("Socket successfully created..\n");

    int port = PORT + (num_nodes*node_id*2) + client_id*2 + channel_idx;
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
        printf("Server listening for client %d on port %d\n", client_id, port);
    len = sizeof(cli);

    // Accept the data packet from client and verification
    connfd = accept(sockfd, (SA*)&cli, &len);
    if (connfd < 0) {
        printf("server accept failed...\n");
        exit(0);
    }
    else
        printf("server %d successfully accepted the client %d on port %d\n", node_id, client_id, port);

    cs->socket_fd[channel_idx] = connfd;
    cs->endpoint_type = 1;
}



void static client_connect_local_server(char** IPs, int num_nodes, int node_id, int server_id, int channel_idx, client_structure* cs){
    struct sockaddr_un servaddr, cli;
    int sockfd, connfd, len;

    char ip[15];
    strcpy(ip, IPs[server_id]);

    char connection_str[4];
    connection_str[0] = '0' + server_id;
    connection_str[1] = '0' + node_id;
    connection_str[2] = '0' + channel_idx;
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
    //servaddr.sin_port = htons(PORT + (ftp_params.NUM_TILES_X*ftp_params.NUM_TILES_Y)*i + (ftp_params.node_id_X + ftp_params.NUM_TILES_X*ftp_params.node_id_Y) + j*600);

    printf("Attempting to connect to server %d on port %d at ip %s\n", server_id, PORT + (num_nodes*server_id*2) + (node_id*2) + channel_idx, ip);
 
    // connect the client socket to server socket
    while (connect(sockfd, (SA*)&servaddr, sizeof(servaddr)) != 0);

    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

    cs->socket_fd[channel_idx] = sockfd;
    cs->endpoint_type = 0;
}

void static client_connect_network_server(char** IPs, int num_nodes, int node_id, int server_id, int channel_idx, client_structure* cs){
    struct sockaddr_in servaddr, cli;
    int sockfd, connfd, len;

    char ip[15];
    strcpy(ip, IPs[server_id]);

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
    servaddr.sin_port = htons(PORT + (num_nodes*server_id*2) + (node_id*2) + channel_idx);

    printf("Attempting to connect to server %d on port %d at ip %s\n", server_id, PORT + (num_nodes*server_id*2) + (node_id*2) + channel_idx, ip);
 
    // connect the client socket to server socket
    while (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
        != 0) {
        //printf("connection with the server failed...\n");
        //exit(0);
    }

    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

    cs->socket_fd[channel_idx] = sockfd;
    cs->endpoint_type = 0;
}

void tcp_connect(int num_nodes, int node_id, char** IPs){
    int sockfd, connfd, len, i, j;
    char current_device_ip[32];

    network_links = (client_structure**)calloc(num_nodes, sizeof(client_structure*));

    //SERVER ENDPOINTS

    for (i = (node_id + 1) ; i < num_nodes; ++i)
    {

        client_structure* cs = calloc(1, sizeof(client_structure));
        network_links[i] = cs;
        cs->endpoint_type = 1;

        int connection = AF_INET;

        for(j = 0; j < 2; j++){

            if(strcmp(IPs[i], IPs[node_id]) == 0)
                server_accept_local_client(IPs, num_nodes, node_id, i, j, cs);
            else
                server_accept_network_client(IPs, num_nodes, node_id, i, j, cs);
        }
    }   
    //CLENT ENDPOINTS
    for (i = (node_id - 1); i >= 0 ; --i)
    {
        client_structure* cs = calloc(1, sizeof(client_structure));
        network_links[i] = cs;
        cs->endpoint_type = 0;

        // socket create and verification
        for (j = 0; j < 2; ++j)
        {
            if(strcmp(IPs[i], IPs[node_id]) == 0)
                client_connect_local_server(IPs, num_nodes, node_id, i, j, cs);
            else
                client_connect_network_server(IPs, num_nodes, node_id, i, j, cs);
        }
    }
}

#define MAX_TRANSFER_CHUNK_SIZE 20000
uint8_t receive_buffer[MAX_TRANSFER_CHUNK_SIZE];

void send_data(float* data, int size, int node_id){
    uint8_t* data8 = (uint8_t*)(data);
    size = size*sizeof(float);
    char* ack = "ACK";
    int num_transactions = (size + MAX_TRANSFER_CHUNK_SIZE - 1)/MAX_TRANSFER_CHUNK_SIZE;
    int cumulative_sent_size = 0;

    while(size > 0){
        int transaction_size = (size > MAX_TRANSFER_CHUNK_SIZE ? MAX_TRANSFER_CHUNK_SIZE : size);
        int sent = 0;
        while(sent < transaction_size)
            sent += write(network_links[node_id]->socket_fd[0], data8 + cumulative_sent_size + sent, transaction_size - sent);

        int to_read = 0;
        while(to_read < strlen(ack)){
            int temp = read( network_links[node_id]->socket_fd[1] , receive_buffer, MAX_TRANSFER_CHUNK_SIZE);
            if(temp > 0)
                to_read += temp;
        }
        cumulative_sent_size += transaction_size;
        size -= transaction_size;
    }

}

void receive_data(float* data, int size, int node_id){
    uint8_t* data8 = (uint8_t*)(data);
    size = size*sizeof(float);
    char* ack = "ACK";
    int num_transactions = (size + MAX_TRANSFER_CHUNK_SIZE - 1)/MAX_TRANSFER_CHUNK_SIZE;
    int cumulative_received_size = 0;

    while(size > 0){
        int transaction_size = (size > MAX_TRANSFER_CHUNK_SIZE) ? MAX_TRANSFER_CHUNK_SIZE : size;
        int temp = transaction_size;
        while(transaction_size > 0){

            int bytes = recv( network_links[node_id]->socket_fd[0], receive_buffer, MAX_TRANSFER_CHUNK_SIZE, MSG_DONTWAIT);
            if(bytes > 0){     
                memcpy(data8 + cumulative_received_size, receive_buffer, bytes);
                cumulative_received_size += bytes;
                transaction_size -= bytes;
            }
        }
        size -= temp;
        //printf("%d\n", size);
        int to_send = strlen(ack);
        int sent = 0;
        if(sent < strlen(ack))
            sent += write(network_links[node_id]->socket_fd[1], (uint8_t*)(ack + sent), to_send - sent);
    }
}

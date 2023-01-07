#include "transport.h"

#include <stdio.h>
#include <netdb.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h> // read(), write(), close()

#include <errno.h>
#include <arpa/inet.h> //close
#include <sys/time.h> //FD_SET, FD_ISSET, FD_ZERO macros

#define MAX 800
#define PORT 8080 //SERVER PORT
#define SA struct sockaddr
  
int sockfd, connfd;

char transmit_buffer[MAX];
char receive_buffer[MAX];
char slip_buffer[MAX*2 + 2];
char decoded_buffer[MAX*2 + 2];
char encoded_buffer[MAX*2 + 2];

char partitioned_receive_buffer[MAX*3];
// char receive_buffer1[MAX];
// char receive_buffer2[MAX];

typedef enum
{
    WAITING_FOR_START,
    PACKET_END,
    PACKET_DATA,
} slip_states;

int slip_data_index = 0;
slip_states current_state;

static uint32_t slip_encode(uint8_t* buffer, uint32_t size, uint8_t* encodedBuffer);
static uint32_t slip_decode(uint8_t* encodedBuffer, uint32_t size, uint8_t* decodedBuffer);
void slip_state_next(uint8_t* rx_packet, uint8_t data);
slip_states slip_get_state();


#ifdef SERVER
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

    slip_state_machine_init();
}

#define TRUE 1
#define FALSE 0

int master_socket , addrlen , new_socket , client_socket[30] ,
    max_clients = 30 , activity, i , valread , sd;
int max_sd;
struct sockaddr_in address;
    
char buffer[1025]; //data buffer of 1K
    int opt = TRUE; 

void init_server(){


    //set of socket descriptors
    fd_set readfds;
        
    //a message
    char *message = "ECHO Daemon v1.0 \r\n";
    
    //initialise all client_socket[] to 0 so not checked
    for (i = 0; i < max_clients; i++)
    {
        client_socket[i] = 0;
    }
        
    //create a master socket
    if( (master_socket = socket(AF_INET , SOCK_STREAM , 0)) == 0)
    {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    
    //set master socket to allow multiple connections ,
    //this is just a good habit, it will work without this
    if( setsockopt(master_socket, SOL_SOCKET, SO_REUSEADDR, (char *)&opt,
        sizeof(opt)) < 0 )
    {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    
    //type of socket created
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons( PORT );
        
    //bind the socket to localhost port 8888
    if (bind(master_socket, (struct sockaddr *)&address, sizeof(address))<0)
    {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    printf("Listener on port %d \n", PORT);
        
    //try to specify maximum of 3 pending connections for the master socket
    if (listen(master_socket, 3) < 0)
    {
        perror("listen");
        exit(EXIT_FAILURE);
    }
        
    //accept the incoming connection
    addrlen = sizeof(address);
    puts("Waiting for connections ...");

    //clear the socket set
    FD_ZERO(&readfds);

    //add master socket to set
    FD_SET(master_socket, &readfds);
    max_sd = master_socket;
        
    //add child sockets to set
    for ( i = 0 ; i < max_clients ; i++)
    {
        //socket descriptor
        sd = client_socket[i];
            
        //if valid socket descriptor then add to read list
        if(sd > 0)
            FD_SET( sd , &readfds);
            
        //highest file descriptor number, need it for the select function
        if(sd > max_sd)
            max_sd = sd;
    }

    //wait for an activity on one of the sockets , timeout is NULL ,
    //so wait indefinitely
    activity = select( max_sd + 1 , &readfds , NULL , NULL , NULL);

    if ((activity < 0) && (errno!=EINTR))
    {
        printf("select error");
    }
        
    //If something happened on the master socket ,
    //then its an incoming connection
    if (FD_ISSET(master_socket, &readfds))
    {
        if ((new_socket = accept(master_socket,
                (struct sockaddr *)&address, (socklen_t*)&addrlen))<0)
        {
            perror("accept");
            exit(EXIT_FAILURE);
        }
        
        //inform user of socket number - used in send and receive commands
        printf("New connection , socket fd is %d , ip is : %s , port : %d \n" , new_socket , inet_ntoa(address.sin_addr) , ntohs
            (address.sin_port));
    
        //send new connection greeting message
        if( send(new_socket, message, strlen(message), 0) != strlen(message) )
        {
            perror("send");
        }
            
        puts("Welcome message sent successfully");
            
        //add new socket to array of sockets
        for (i = 0; i < max_clients; i++)
        {
            //if position is empty
            if( client_socket[i] == 0 )
            {
                client_socket[i] = new_socket;
                printf("Adding to list of sockets as %d\n" , i);
                    
                break;
            }
        }
    }    
}

#else

void init_transport(){

 int connfd;
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
    servaddr.sin_addr.s_addr = inet_addr("127.0.0.1");
    servaddr.sin_port = htons(PORT);
 
    // connect the client socket to server socket
    if (connect(sockfd, (SA*)&servaddr, sizeof(servaddr))
        != 0) {
        printf("connection with the server failed...\n");
        exit(0);
    }
    else
        printf("connected to the server..\n");

    slip_state_machine_init();

}

#endif


#ifdef SERVER 

void route_client_links(float* data, int size, int device_id_x, int device_id_y){



}

#endif




void send_boundry(float* data, int size, int device_id_x, int device_id_y){
    transmit_buffer[0] = 1;
	transmit_buffer[1] = DEVICE_ID_X;
	transmit_buffer[2] = DEVICE_ID_Y;
	transmit_buffer[3] = device_id_x;
	transmit_buffer[4] = device_id_y;
	memcpy(transmit_buffer+5, data, size*(sizeof(float)));

#ifdef SERVER 
    int encoded_size = slip_encode(transmit_buffer, size*(sizeof(float)) + 5, encoded_buffer);
	write(connfd, encoded_buffer, encoded_size);
#else
    int encoded_size = slip_encode(transmit_buffer, size*(sizeof(float)) + 5, encoded_buffer);
    write(sockfd, encoded_buffer, encoded_size);
#endif
}
#define NUM_TILES_X 2
#define NUM_TILES_Y 1

void receive_boundry(float* data, int size, int device_id_x, int device_id_y){

    while(1){

        uint8_t* partition_offset = partitioned_receive_buffer + ((device_id_y*NUM_TILES_X) + device_id_x)*MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float);
        if(partition_offset[0] == 1 
            && (partition_offset[1] == device_id_x) 
            && (partition_offset[2] == device_id_y) 
            && (partition_offset[3] == DEVICE_ID_X)
            && (partition_offset[4] == DEVICE_ID_Y) ){
                memcpy(data, partition_offset+5, size*sizeof(float));
                partition_offset[0] = 0;

                return;// success
        }


        #ifdef SERVER 
            int bytes = read(connfd, receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float));
        #else
            int bytes = read(sockfd, receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float));
        #endif
        int last_size = 0;
        for (int i = 0; i < bytes; ++i)
        {
            
            slip_state_next(slip_buffer, receive_buffer[i]);
            if(slip_get_state() == PACKET_END){
                slip_decode(slip_buffer, i+1 - last_size, decoded_buffer);
                slip_state_machine_init();
                memcpy(partitioned_receive_buffer + ((decoded_buffer[2]*NUM_TILES_X) + decoded_buffer[1])*MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float), decoded_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float));
                last_size = (i+1);
            }
        }

    }

}


//FORMAT
//BYTE 0 - VALID
//BYTE 1 - SRC X 
//BYTE 2 - SRC Y
//BYTE 3 - DST X
//BYTE 4 - DST Y



/// \brief Key constants used in the SLIP protocol.
enum
{
    /// \brief The decimal END character (octal 0300).
    ///
    /// Indicates the end of a packet.
    END = 192, 

    /// \brief The decimal ESC character (octal 0333).
    ///
    /// Indicates byte stuffing.
    ESC = 219,

    /// \brief The decimal ESC_END character (octal 0334).
    ///
    /// ESC ESC_END means END data byte.
    ESC_END = 220,

    /// \brief The decimal ESC_ESC character (ocatal 0335).
    ///
    /// ESC ESC_ESC means ESC data byte.
    ESC_ESC = 221
};

/// \brief Get the maximum encoded buffer size for an unencoded buffer size.
///
/// SLIP has a start and end markers (192 and 219). Marker value is
/// replaced by 2 bytes in the encoded buffer. So in the worst case of
/// sending a buffer with only '192' or '219', the encoded buffer length
/// will be 2 * buffer.size() + 2.
///
/// \param unencodedBufferSize The size of the buffer to be encoded.
/// \returns the maximum size of the required encoded buffer.
static size_t getEncodedBufferSize(size_t unencodedBufferSize)
{
    return unencodedBufferSize * 2 + 2;
}

void slip_state_machine_init(){
    current_state = WAITING_FOR_START;
    slip_data_index = 0;
}

#define END 192
#define ESC 219
#define ESC_END 220
#define ESC_ESC 221

void slip_state_next(uint8_t* rx_packet, uint8_t data){
    switch (current_state){

        case WAITING_FOR_START:
            if(data == END){
                rx_packet[slip_data_index++] = data;
                current_state = PACKET_DATA;
            }
            break;
        case PACKET_DATA:
            rx_packet[slip_data_index++] = data;
            if(data == END){
                current_state = PACKET_END;
                //TODO: Check for buffer overflow
                //here and reset the machine in that case. Clearly it 
                //indicates invalid/bad data if it exceeds the max 
                //packet size to overflow the buffer
            }
            break;
        case PACKET_END:
            break;

    }
}

slip_states slip_get_state(){
 return current_state;
}

//Acknowledgement
// https://github.com/bakercp/PacketSerial/blob/master/s
// rc/Encoding/SLIP.h (encode modified sligtly from 
// original source to include the 192 sentinel byte at 
// both start and end of packet. Helps in easier and 
// more robust packet receiving on receiver through slip 
// state machine)
static uint32_t slip_encode(uint8_t* buffer, uint32_t size, uint8_t* encodedBuffer)
{
    if (size == 0)
        return 0;

    uint32_t read_index = 0;
    uint32_t write_index = 0;

    // Double-ENDed, flush any data that may have accumulated due to line noise.
    
    encodedBuffer[write_index++] = END;

    while (read_index < size)
    {
        if(buffer[read_index] == END)
        {
            encodedBuffer[write_index++] = ESC;
            encodedBuffer[write_index++] = ESC_END;
            read_index++;
        }
        else if(buffer[read_index] == ESC)
        {
            encodedBuffer[write_index++] = ESC;
            encodedBuffer[write_index++] = ESC_ESC;
            read_index++;
        }
        else
        {
            encodedBuffer[write_index++] = buffer[read_index++];
        }
    }
    encodedBuffer[write_index++] = END;
    return write_index;
}


/// \brief Decode a SLIP-encoded buffer.
/// \param encodedBuffer A pointer to the \p encodedBuffer to decode.
/// \param size The number of bytes in the \p encodedBuffer.
/// \param decodedBuffer The target buffer for the decoded bytes.
/// \returns The number of bytes written to the \p decodedBuffer.
/// \warning decodedBuffer must have a minimum capacity of size.
static uint32_t slip_decode(uint8_t* encodedBuffer, uint32_t size, uint8_t* decodedBuffer)
{
    if (size == 0)
        return 0;

    uint32_t read_index = 0;
    uint32_t write_index = 0;

    while (read_index < size)
    {
        if (encodedBuffer[read_index] == END)
        {
            // flush or done
            read_index++;
        }
        else if (encodedBuffer[read_index] == ESC)
        {
            if (encodedBuffer[read_index+1] == ESC_END)
            {
                decodedBuffer[write_index++] = END;
                read_index += 2;
            }
            else if (encodedBuffer[read_index+1] == ESC_ESC)
            {
                decodedBuffer[write_index++] = ESC;
                read_index += 2;
            }
            else
            {
                // This case is considered a protocol violation.
            }
        }
        else
        {
            decodedBuffer[write_index++] = encodedBuffer[read_index++];
        }
    }
    return write_index;
}


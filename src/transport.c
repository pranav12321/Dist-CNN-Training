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
  
int sockfd, connfd;

char transmit_buffer[MAX*5];
char receive_buffer[MAX*5];
char slip_buffer[MAX*4 + 2];
char decoded_buffer[MAX*4 + 2];
char encoded_buffer[MAX*4 + 2];

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



client_structure client_packets;

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


    int flags = fcntl(sockfd, F_GETFL, 0);
    fcntl(sockfd, F_SETFL, flags | O_NONBLOCK);

    for (int j = 0; j < MAX_PACKETS_PER_DEVICE; ++j)
    {
        client_packets.receive_device_data_ptrs[j].valid = 0;
    }

    slip_state_machine_init();

}





void send_boundry(float* data, int size, int device_id_x, int device_id_y){
    transmit_buffer[0] = 1;
    transmit_buffer[1] = DEVICE_ID_X;
    transmit_buffer[2] = DEVICE_ID_Y;
    transmit_buffer[3] = device_id_x;
    transmit_buffer[4] = device_id_y;
    memcpy(transmit_buffer+5, data, size*(sizeof(float)));

#ifdef SERVER 
    int dst_client_id = NUM_TILES_X*device_id_y + device_id_x - 1;
    int encoded_size = slip_encode(transmit_buffer, size*(sizeof(float)) + 5, encoded_buffer);
    printf("Server sending %d bytes to device %d %d\n", encoded_size, transmit_buffer[3], transmit_buffer[4]);
    write(client_socket[dst_client_id], encoded_buffer, encoded_size);
#else
    int encoded_size = slip_encode(transmit_buffer, size*(sizeof(float)) + 5, encoded_buffer);
    printf("Client %d %d sending %d bytes to device %d %d\n", DEVICE_ID_X, DEVICE_ID_Y, encoded_size, transmit_buffer[3], transmit_buffer[4]);
    write(sockfd, encoded_buffer, encoded_size);
#endif
}


void receive_boundry(float* data, int size, int device_id_x, int device_id_y){

    while(1){

        int bytes = 0;

        for (int j = 0; j < MAX_PACKETS_PER_DEVICE; ++j)
        {      
            uint8_t* packet_ptr = client_packets.receive_device_data_ptrs[j].packet_ptr;
            if(client_packets.receive_device_data_ptrs[j].valid == 1 
                && packet_ptr[0] == 1 
                && (packet_ptr[1] == device_id_x) 
                && (packet_ptr[2] == device_id_y) 
                && (packet_ptr[3] == DEVICE_ID_X)
                && (packet_ptr[4] == DEVICE_ID_Y) ){

                memcpy(data, packet_ptr+5, size*sizeof(float));
                packet_ptr[0] = 0;
                client_packets.receive_device_data_ptrs[j].valid = 0;
                free(packet_ptr);
                return;// success

            }
                            
        }   

        int poll_condition = 1;       

        while(poll_condition == 1){
            bytes = recv( sockfd , receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float), MSG_DONTWAIT);

            if(bytes > 0)
                printf("Client %d %d received %d raw bytes\n", DEVICE_ID_X, DEVICE_ID_Y, bytes);
            //read(sockfd, receive_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float));

            if(bytes > 0){
            
                int last_size = 0;
                for (int i = 0; i < bytes; ++i)
                {
                    
                    slip_state_next(slip_buffer, receive_buffer[i]);
                    if(slip_get_state() == PACKET_END){
                        int decoded_size = slip_decode(slip_buffer, slip_data_index, decoded_buffer);
                        slip_state_machine_init();
                        int client_rx_id = ((decoded_buffer[2]*NUM_TILES_X) + decoded_buffer[1]);
                        
                        //memcpy(partitioned_receive_buffer + ((decoded_buffer[2]*NUM_TILES_X) + decoded_buffer[1])*MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float), decoded_buffer, MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float));
                        printf("Client %d %d received %d decoded packet bytes from device %d %d\n", DEVICE_ID_X, DEVICE_ID_Y, decoded_size, decoded_buffer[1], decoded_buffer[2]);
                        for (int j = 0; j < MAX_PACKETS_PER_DEVICE; ++j)
                        {
                            if(client_packets.receive_device_data_ptrs[j].valid == 0){
                                client_packets.receive_device_data_ptrs[j].packet_ptr = calloc(decoded_size, sizeof(uint8_t));
                                memcpy(client_packets.receive_device_data_ptrs[j].packet_ptr, decoded_buffer, decoded_size);
                                client_packets.receive_device_data_ptrs[j].receive_device_data_packet_size = decoded_size;
                                client_packets.receive_device_data_packet_ctr[client_rx_id] ++;
                                client_packets.receive_device_data_ptrs[j].valid = 1;
                                break;
                            }
                        }
                        
                        last_size = (i+1);
                        if(i == (bytes - 1))
                            poll_condition = 0;
                    }
                }
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


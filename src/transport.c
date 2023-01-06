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
	write(connfd, transmit_buffer, size*(sizeof(float)) + 5);
#else
    write(sockfd, transmit_buffer, size*(sizeof(float)) + 5);
#endif
}
void receive_boundry(float* data, int size, int device_id_x, int device_id_y){

    while(1){

        if(receive_buffer[0] == 1 && (receive_buffer[1] == device_id_x) && (receive_buffer[2] == device_id_y) 
            && (receive_buffer[3] == DEVICE_ID_X) && (receive_buffer[4] == DEVICE_ID_Y) ){
            memcpy(data, receive_buffer+5, size*sizeof(float));
            receive_buffer[0] = 0;

            return;// success
        }

    #ifdef SERVER 
        int bytes = read(connfd, receive_buffer, size*sizeof(float) + 5);
    #else
        int bytes = read(sockfd, receive_buffer, size*sizeof(float) + 5);
    #endif

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


    static size_t encode(const uint8_t* buffer,
                         size_t size,
                         uint8_t* encodedBuffer)
    {
        if (size == 0)
            return 0;

        size_t read_index  = 0;
        size_t write_index = 0;

        // Double-ENDed, flush any data that may have accumulated due to line 
        // noise.
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

        return write_index;
    }

    /// \brief Decode a SLIP-encoded buffer.
    /// \param encodedBuffer A pointer to the \p encodedBuffer to decode.
    /// \param size The number of bytes in the \p encodedBuffer.
    /// \param decodedBuffer The target buffer for the decoded bytes.
    /// \returns The number of bytes written to the \p decodedBuffer.
    /// \warning decodedBuffer must have a minimum capacity of size.
    static size_t decode(const uint8_t* encodedBuffer,
                         size_t size,
                         uint8_t* decodedBuffer)
    {
        if (size == 0)
            return 0;

        size_t read_index  = 0;
        size_t write_index = 0;

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




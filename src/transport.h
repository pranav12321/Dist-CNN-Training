#ifndef TRANSPORT
#define TRANSPORT

#include <stdint.h>
#include <signal.h>

#ifdef SERVER
	void init_server();
	void * route_client_links(void *vargp);
#endif
void init_transport(char* argv[]);
void send_boundry(float*, int, int, int);
void receive_boundry(float*, int, int, int);

//#define SERVER 1
#define CLIENT 1

// #define DEVICE_ID_X 3
// #define DEVICE_ID_Y 3

// #define DEVICE_0000_IP "192.168.4.5"
// #define DEVICE_0001_IP "192.168.4.5"
// #define DEVICE_0010_IP "192.168.4.12"
// #define DEVICE_0011_IP "192.168.4.12"
// #define DEVICE_0100_IP "192.168.4.5"
// #define DEVICE_0101_IP "192.168.4.5"
// #define DEVICE_0110_IP "192.168.4.12"
// #define DEVICE_0111_IP "192.168.4.12"
// #define DEVICE_1000_IP "192.168.4.2"
// #define DEVICE_1001_IP "192.168.4.2"
// #define DEVICE_1010_IP "192.168.4.9"
// #define DEVICE_1011_IP "192.168.4.9"
// #define DEVICE_1100_IP "192.168.4.2"
// #define DEVICE_1101_IP "192.168.4.2"
// #define DEVICE_1110_IP "192.168.4.9"
// #define DEVICE_1111_IP "192.168.4.9"
// #define NUM_TILES_X 2
// #define NUM_TILES_Y 2

#define MAX_BOUNDARY_SIZE_PER_DEVICE 60000

#define MAX_PACKETS_PER_DEVICE 10

typedef struct rx_data_packet_ptrs{
	uint8_t* packet_ptr;
	int valid;
	int receive_device_data_packet_size;
} rx_data_packet_ptrs;

typedef struct client_structure{
	uint8_t* receive_buffer;//[MAX_BOUNDARY_SIZE_PER_DEVICE*sizeof(float)];
	rx_data_packet_ptrs receive_device_data_ptrs[MAX_PACKETS_PER_DEVICE];
	int socket_fd[2];
	int endpoint_type;
	int receive_device_data_packet_ctr[4];

} client_structure;

typedef struct comm_entry{
	uint8_t* data;
	int device_id_x;
	int device_id_y;
	int size; 
	int valid;
	int intent_to_read;
	int has_been_read;
	int ack_received;
} comm_entry;

void* send_boundry_thread(void *vargp);
void* receive_boundry_thread(void *vargp);

#endif
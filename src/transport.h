#ifndef TRANSPORT
#define TRANSPORT

#include <stdint.h>

#ifdef SERVER
	void init_server();
	void * route_client_links(void *vargp);
#endif
void init_transport();
void send_boundry(float*, int, int, int);
void receive_boundry(float*, int, int, int);

//#define SERVER 1
#define CLIENT 1

#define DEVICE_ID_X 0
#define DEVICE_ID_Y 0

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


#endif

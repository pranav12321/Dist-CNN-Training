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
	rx_data_packet_ptrs receive_device_data_ptrs[MAX_PACKETS_PER_DEVICE];
	//TODO: TEMP const hardcoded
	//int receive_device_data_packet_ctr[NUM_TILES_Y*NUM_TILES_X];
	int receive_device_data_packet_ctr[4];

} client_structure;


#endif
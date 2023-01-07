#ifndef TRANSPORT
#define TRANSPORT

#ifdef SERVER
	void init_server();
#endif
void init_transport();
void send_boundry(float*, int, int, int);
void receive_boundry(float*, int, int, int);

//#define SERVER 1
#define CLIENT 1

#define DEVICE_ID_X 1
#define DEVICE_ID_Y 0

#define MAX_BOUNDARY_SIZE_PER_DEVICE 250

#endif
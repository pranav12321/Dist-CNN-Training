#ifndef TRANSPORT
#define TRANSPORT

void init_transport();
void send_boundry(float*, int, int, int);
void receive_boundry(float*, int, int, int);

#define SERVER 1
//#define CLIENT 1

#define DEVICE_ID_X 0
#define DEVICE_ID_Y 0

#endif
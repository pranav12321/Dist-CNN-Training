#ifndef TCP_UTILS
#define TCP_UTILS

#include <stdint.h>
#include <signal.h>

void tcp_connect(int num_nodes, int node_id, char** IPs);
void send_data(float* data, int size, int node_id);
void receive_data(float* data, int size, int node_id);

typedef struct client_structure{
	int socket_fd[2];
	int endpoint_type;
} client_structure;

#endif

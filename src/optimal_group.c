#include "optimal_group.h"

int forward_group_cost(int start_node, int end_node, ftp_args args){
	return 1;
}

int backward_group_cost(int start_node, int end_node, ftp_args args){
	return 1;
}

int** create_forward_graph(ftp_args args){

	int** cost_graph = calloc((args.num_layers+1) * (args.num_layers+1), sizeof(int));
	for (int i = 0; i <= args.num_layers; ++i)
	{
		for (int j = 0; j <= args.num_layers; ++j)
		{
			cost_graph[i][j] = INFINITY;
		}
	}

	for (int i = 0; i <= args.num_layers; ++i)
	{
		for (int j = i+1; j <= args.num_layers; ++j)
		{
			cost_graph[i][j] = forward_group_cost(i, j, args);
		}
	}

	return cost_graph;

}

int** create_backward_graph(ftp_args args){

	int** cost_graph = calloc((args.num_layers+1) * (args.num_layers+1), sizeof(int));
	for (int i = 0; i <= args.num_layers; ++i)
	{
		for (int j = 0; j <= args.num_layers; ++j)
		{
			cost_graph[i][j] = INFINITY;
		}
	}

	for (int i = args.num_layers - 1; i >= 0; --i)
	{
		for (int j = i-1; j >= 0; --j)
		{
			cost_graph[args.num_layers - i][args.num_layers - j] = backward_group_cost(i, j, args);
		}
	}

	return cost_graph;

}

void compute_optimal_grouping(ftp_args args){

	int** forward_cost_graph = create_forward_graph(args);
	dijkstra(forward_cost_graph, args.num_layers + 1, 0);
	
	int** backward_cost_graph = create_backward_graph(args);
	dijkstra(backward_cost_graph, args.num_layers, 0);

}
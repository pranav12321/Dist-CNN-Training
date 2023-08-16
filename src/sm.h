#ifndef SM_H
#define SM_H

void create_sm(char* shm_file, float** buffer, int num_processes, int size_per_process);
void get_sm_buffer(char* shm_file, float** buffer, int num_processes, int size_per_process);

void sm_sema_wait(int num_processes);
void sm_sema_post(int num_processes);
void train_cycle_complete_sema_wait(int num_processes);
void train_cycle_complete_sema_post(int num_processes);
void init_complete_sema_wait(int num_processes);
void init_complete_sema_post(int num_processes);

#endif SM_H


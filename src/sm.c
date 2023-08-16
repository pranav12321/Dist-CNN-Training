#include "darknet.h"
#include "fused_device.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <semaphore.h>
#include <sys/mman.h>

#define SEM_MUTEX_NAME "/sem-mutex"
#define SEM_BUFFER_COUNT_NAME "/sem-buffer-count"
#define SEM_INIT_COMPLETE "/sem-init-complete-signal"
#define SEM_TRAIN_CYCLE_COMPLETE "/sem-train-signal"
#define SEM_FILTER_SYNC_COMPLETE "/filter-sync-complete-signal"
#define SHARED_MEM_NAME "/posix-shared-mem"

int fd_shm = -1;
sem_t *init_complete_mutex;
sem_t *sm_create_mutex;
sem_t *sm_train_mutex;
sem_t *sm_filter_sync_mutex;

void create_sm(char* shm_file, float** buffer, int num_processes, int size_per_process){

	printf("%d %d\n", num_processes, size_per_process);
    //  mutual exclusion semaphore, mutex_sem 
    if ((sm_create_mutex = sem_open (SEM_MUTEX_NAME, O_CREAT, 0660, 0)) == SEM_FAILED)
        error ("sem_open");

    if ((sm_train_mutex = sem_open (SEM_TRAIN_CYCLE_COMPLETE, O_CREAT, 0660, 0)) == SEM_FAILED)
        error ("sem_open");

    if ((sm_filter_sync_mutex = sem_open (SEM_FILTER_SYNC_COMPLETE, O_CREAT, 0660, 0)) == SEM_FAILED)
        error ("sem_open");

    if ((init_complete_mutex = sem_open (SEM_INIT_COMPLETE, O_CREAT, 0660, 0)) == SEM_FAILED)
        error ("sem_open");


	printf("%s\n", shm_file);
    if ((fd_shm = shm_open (shm_file, O_RDWR | O_CREAT | O_EXCL, 0660)) == -1)
        error ("shm_open");

    if (ftruncate (fd_shm,num_processes*size_per_process*sizeof(float)) == -1)
       error ("ftruncate");
    
    if ((*buffer = mmap (NULL, num_processes*size_per_process*sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED,
            fd_shm, 0)) == MAP_FAILED)
        error ("mmap");

    for (int i = 0; i < (num_processes - 1); ++i)
    {
	    if (sem_post (sm_create_mutex) == -1)
	        error ("sem_post: mutex_sem");
    }
}


void get_sm_buffer(char* shm_file, float** buffer, int num_processes, int size_per_process){
    // Get shared memory 
    int fd_shm = -1;

    printf("%s\n", "wait\n");

    while( ((sm_create_mutex = sem_open (SEM_MUTEX_NAME, 0, 0, 0)) == SEM_FAILED) );
    while( ((sm_train_mutex = sem_open (SEM_TRAIN_CYCLE_COMPLETE, 0, 0, 0)) == SEM_FAILED) );
    while( ((sm_filter_sync_mutex = sem_open (SEM_FILTER_SYNC_COMPLETE, 0, 0, 0)) == SEM_FAILED) );
    while( ((init_complete_mutex = sem_open (SEM_INIT_COMPLETE, 0, 0, 0)) == SEM_FAILED) );

        // error ("sem_open");

	if (sem_wait (sm_create_mutex) == -1)
	    error ("sem_wait: mutex_sem");

    if ((fd_shm = shm_open (shm_file, O_RDWR, 0)) == -1)
        error ("shm_open");

    if ((*buffer = mmap (NULL, num_processes*size_per_process*sizeof(float), PROT_READ | PROT_WRITE, MAP_SHARED,
            fd_shm, 0)) == MAP_FAILED)
       error ("mmap");
}

void sm_sema_wait(int num_processes){
    for (int i = 0; i < num_processes; ++i)
    {
        if (sem_wait (sm_create_mutex) == -1)
            error ("sem_wait: mutex_sem");
    }    
}

void sm_sema_post(int num_processes){
    for (int i = 0; i < num_processes; ++i)
    {
        if (sem_post (sm_create_mutex) == -1)
            error ("sem_post: mutex_sem");
    }    
}


void train_cycle_complete_sema_wait(int num_processes){
    for (int i = 0; i < num_processes; ++i)
    {
        if (sem_wait (sm_train_mutex) == -1)
            error ("sem_wait: mutex_sem");    
    }
}

void train_cycle_complete_sema_post(int num_processes){
    for (int i = 0; i < num_processes; ++i)
    {
        if (sem_post (sm_train_mutex) == -1)
            error ("sem_post: mutex_sem");
    }    
}

void filter_sync_complete_sema_wait(int num_processes){
    for (int i = 0; i < num_processes; ++i)
    {
        if (sem_wait (sm_filter_sync_mutex) == -1)
            error ("sem_wait: mutex_sem");    
    }
}

void filter_sync_complete_sema_post(int num_processes){
    for (int i = 0; i < num_processes; ++i)
    {
        if (sem_post (sm_filter_sync_mutex) == -1)
            error ("sem_post: mutex_sem");
    }    
}

void init_complete_sema_wait(int num_processes){
    for (int i = 0; i < num_processes; ++i)
    {
        if (sem_wait (init_complete_mutex) == -1)
            error ("sem_wait: mutex_sem");    
    }
}

void init_complete_sema_post(int num_processes){
    for (int i = 0; i < num_processes; ++i)
    {
        if (sem_post (init_complete_mutex) == -1)
            error ("sem_post: mutex_sem");
    }    
}
#include <pthread.h>
#include <stdio.h>

#define MAX 10000000

__thread int th_i = 0;

void* inc(void* a) {
    for (int i = 0; i < MAX; i++) th_i++;
    printf("n = %d\n", th_i);
}

int main() {
    int i;
    pthread_t thread1, thread2;
    pthread_create(&thread1, NULL, inc, (void*)(&i));
    pthread_create(&thread2, NULL, inc, (void*)(&i));
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    return 0;
}

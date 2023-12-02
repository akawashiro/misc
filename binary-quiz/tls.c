#include <threads.h>

thread_local unsigned long long tls = 0;

int main() { tls = 1; }

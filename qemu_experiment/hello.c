#include <unistd.h>

double add_two_double(double a, double b) { return a + b; }

int main(void) {
  add_two_double(1.0, 2.0);
  write(1, "Hello, World!\n", 14);
}

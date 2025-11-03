#include <stdlib.h>

#include <zmq.h>

int main(int argc, char **argv) {
  if (argc < 3) {
    printf("usage: test_has [0|1] FEATURE\n");
    return 1;
  }
  // check zmq_has output (1 or 0) for given features
  int expect_feature = atoi(argv[1]);
  printf("expect %d %s\n", expect_feature, argv[2]);
  int has_feature = zmq_has(argv[2]);
  printf("has %d %s\n", has_feature, argv[2]);
  if (has_feature != expect_feature) {
    return 1;
  }
  return 0;
}

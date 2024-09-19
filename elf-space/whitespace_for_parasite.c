// #include <stdio.h>

// void put_whitespace(char c) {
//   if (c == ' ') {
//     printf("[SP]");
//   } else if (c == '\t') {
//     printf("[TAB]");
//   } else if (c == '\n') {
//     printf("[LF]");
//   } else {
//     printf("[Not whitespace: %c]", c);
//   }
// }
//
// void dump(char *head) {
//   while (*head) {
//     put_whitespace(*head);
//     head++;
//   }
//   printf("\n");
// }

char *helloworld =
    "   	  	   \n	\n     		  	 	\n	\n     "
    "		 		  \n	\n     		 		  "
    "\n	\n     		 				\n	\n     	 "
    "		  \n	\n     	     \n	\n     			 	"
    "		\n	\n     		 				"
    "\n	\n     			  	 \n	\n     		 	"
    "	  \n	\n     		  	  \n	\n     	    	\n	\n  ";

// int parse_number(char **head_p) {
//   char *head = *head_p;
// 
//   int flag = 0;
//   {
//     char c = *head;
//     head++;
//     if (c == ' ') {
//       flag = 1;
//     } else if (c == '\t') {
//       flag = -1;
//     }
//   }
// 
//   int num = 0;
//   while (1) {
//     char c = *head;
//     head++;
// 
//     if (c == ' ') {
//       num = num * 2 + 0;
//     } else if (c == '\t') {
//       num = num * 2 + 1;
//     } else {
//       break;
//     }
//   }
// 
//   *head_p = head;
//   return num;
// }

void run_helloworld() {
  // Buffer to print without libc
  char print_buf[16];

  int stack[1024];
  int stack_ptr = 0;

  // dump(helloworld);

  char *head = helloworld;
  while (*head) {
    if (*head == ' ') {
      head++;
      // printf("IMP: Stack\n");
      if (*head == ' ') {
        head++;
        // printf("COMMAND: Push\n");
        // int num = parse_number(&head);
        int num = 0;
        {
          int flag = 0;
          {
            char c = *head;
            head++;
            if (c == ' ') {
              flag = 1;
            } else if (c == '\t') {
              flag = -1;
            }
          }

          while (1) {
            char c = *head;
            head++;

            if (c == ' ') {
              num = num * 2 + 0;
            } else if (c == '\t') {
              num = num * 2 + 1;
            } else {
              break;
            }
          }
        }
        stack[stack_ptr] = num;
        stack_ptr++;
        // printf("VALUE: %d\n", num);
      } else if (*head == '\n' && *(head + 1) == ' ') {
        head += 2;
        // printf("COMMAND: Duplicate\n");
      } else {
        // printf("Unknown stack manupulation\n");
        return;
      }
    } else if (*head == '\t' && *(head + 1) == '\n') {
      head += 2;
      // printf("IMP: IO\n");
      // printf("COMMAND: Output\n");
      stack_ptr--;
      // printf("VALUE: %c(%d)\n", stack[stack_ptr], stack[stack_ptr]);

      print_buf[0] = stack[stack_ptr];
      print_buf[1] = '\0';
      __asm__ volatile("syscall" : : "a"(1), "D"(1), "S"(print_buf), "d"(1));
    } else {
      // printf("Unknown manupulation\n");
      return;
    }
  }
  return;
}

int main() { run_helloworld(); }

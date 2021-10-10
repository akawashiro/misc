#include <sys/auxv.h>
#include <stdio.h>
#include <stdlib.h>
#include <elf.h>

void print_proc_self(){
    FILE *fptr;
  
    char filename[1024], c;
   
    // Open file
    fptr = fopen("/proc/self/maps", "r");
    if (fptr == NULL)
    {
        printf("Cannot open file \n");
        exit(0);
    }
  
    // Read contents from file
    c = fgetc(fptr);
    while (c != EOF)
    {
        printf ("%c", c);
        c = fgetc(fptr);
    }
  
    fclose(fptr);
}

int main(){
    unsigned long phdr = getauxval(AT_PHDR);
    unsigned long phnum = getauxval(AT_PHNUM);
    unsigned long phent = getauxval(AT_PHENT);
    unsigned long base = getauxval(AT_BASE);
    print_proc_self();
    printf("AT_PHDR: %lx AT_BASE: %lx\n", phdr, base);

    for(int i = 0;i<phnum;i++){
        Elf64_Phdr *p = (Elf64_Phdr*)(phdr + i * phent);
        printf("p->p_type = %x, p->p_vaddr = %lx\n", p->p_type, p->p_vaddr);
    }

    return 0;
}

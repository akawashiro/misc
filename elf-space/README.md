`run.sh` を実行すると本文中の実験が再現できます。実行には `docker` が必要です。

```
$ readelf -l hello
Elf file type is EXEC (Executable file)
Entry point 0x401650
There are 10 program headers, starting at offset 64

Program Headers:
  Type           Offset             VirtAddr           PhysAddr
                 FileSiz            MemSiz              Flags  Align
  LOAD           0x0000000000000000 0x0000000000400000 0x0000000000400000
                 0x0000000000000528 0x0000000000000528  R      0x1000
  LOAD           0x0000000000001000 0x0000000000401000 0x0000000000401000
                 0x000000000009665d 0x000000000009665d  R E    0x1000
  LOAD           0x0000000000098000 0x0000000000498000 0x0000000000498000
                 0x000000000002853c 0x000000000002853c  R      0x1000
  LOAD           0x00000000000c07b0 0x00000000004c17b0 0x00000000004c17b0
                 0x0000000000005ae0 0x000000000000b490  RW     0x1000
  NOTE           0x0000000000000270 0x0000000000400270 0x0000000000400270
                 0x0000000000000030 0x0000000000000030  R      0x8
  NOTE           0x00000000000002a0 0x00000000004002a0 0x00000000004002a0
                 0x0000000000000044 0x0000000000000044  R      0x4
  TLS            0x00000000000c07b0 0x00000000004c17b0 0x00000000004c17b0
                 0x0000000000000020 0x0000000000000068  R      0x8
  GNU_PROPERTY   0x0000000000000270 0x0000000000400270 0x0000000000400270
                 0x0000000000000030 0x0000000000000030  R      0x8
  GNU_STACK      0x0000000000000000 0x0000000000000000 0x0000000000000000
                 0x0000000000000000 0x0000000000000000  RW     0x10
  GNU_RELRO      0x00000000000c07b0 0x00000000004c17b0 0x00000000004c17b0
                 0x0000000000003850 0x0000000000003850  R      0x1

 Section to Segment mapping:
  Segment Sections...
   00     .note.gnu.property .note.gnu.build-id .note.ABI-tag .rela.plt 
   01     .init .plt .text __libc_freeres_fn .fini 
   02     .rodata .stapsdt.base .eh_frame .gcc_except_table 
   03     .tdata .init_array .fini_array .data.rel.ro .got .got.plt .data __libc_subfreeres __libc_IO_vtables __libc_atexit .bss __libc_freeres_ptrs 
   04     .note.gnu.property 
   05     .note.gnu.build-id .note.ABI-tag 
   06     .tdata .tbss 
   07     .note.gnu.property 
   08     
   09     .tdata .init_array .fini_array .data.rel.ro .got 
```

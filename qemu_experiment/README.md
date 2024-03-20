# QEMU の仕組み

- [QEMUのなかみ(QEMU internals) part1](https://rkx1209.hatenablog.com/entry/2015/11/15/214404)
- [QEMUのなかみ(QEMU internals) part2](https://rkx1209.hatenablog.com/entry/2015/11/20/203511)
- [TCG Intermediate Representation](https://www.qemu.org/docs/master/devel/tcg-ops.html)

# Function: `add_two_double`

## C言語
```
double add_two_double(double a, double b) { return a + b; }
```

## AArch64
```
IN: add_two_double
0x004006d4:  d10043ff  sub      sp, sp, #0x10
0x004006d8:  fd0007e0  str      d0, [sp, #8]
0x004006dc:  fd0003e1  str      d1, [sp]
0x004006e0:  fd4007e1  ldr      d1, [sp, #8]
0x004006e4:  fd4003e0  ldr      d0, [sp]
0x004006e8:  1e602820  fadd     d0, d1, d0
0x004006ec:  910043ff  add      sp, sp, #0x10
0x004006f0:  d65f03c0  ret      
```

## QEMU IR
```
OP:
 ld_i32 loc0,env,$0xfffffffffffffff0
 brcond_i32 loc0,$0x0,lt,$L0
 st8_i32 $0x0,env,$0xfffffffffffffff4

 ---- 00000000004006d4 0000000000000000 0000000000000000
 sub_i64 sp,sp,$0x10

 ---- 00000000004006d8 0000000000000000 0000000000000000
 mov_i64 loc3,sp
 add_i64 loc3,loc3,$0x8
 shl_i64 loc5,loc3,$0x8
 sar_i64 loc5,loc5,$0x8
 and_i64 loc5,loc5,loc3
 ld_i64 loc6,env,$0xcd0
 qemu_st_a64_i64 loc6,loc5,noat+un+leq,0

 ---- 00000000004006dc 0000000000000000 0000000000000000
 mov_i64 loc7,sp
 shl_i64 loc8,loc7,$0x8
 sar_i64 loc8,loc8,$0x8
 and_i64 loc8,loc8,loc7
 ld_i64 loc9,env,$0xdd0
 qemu_st_a64_i64 loc9,loc8,noat+un+leq,0

 ---- 00000000004006e0 0000000000000000 0000000000000000
 mov_i64 loc10,sp
 add_i64 loc10,loc10,$0x8
 shl_i64 loc11,loc10,$0x8
 sar_i64 loc11,loc11,$0x8
 and_i64 loc11,loc11,loc10
 qemu_ld_a64_i64 loc12,loc11,noat+un+leq,0
 st_i64 loc12,env,$0xdd0
 mov_vec v256,e8,tmp13,v256$0x0
 st_vec v64,e8,tmp13,env,$0xdd8
 st_vec v256,e8,tmp13,env,$0xde0
 st_vec v128,e8,tmp13,env,$0xe00

 ---- 00000000004006e4 0000000000000000 0000000000000000
 mov_i64 loc15,sp
 shl_i64 loc16,loc15,$0x8
 sar_i64 loc16,loc16,$0x8
 and_i64 loc16,loc16,loc15
 qemu_ld_a64_i64 loc17,loc16,noat+un+leq,0
 st_i64 loc17,env,$0xcd0
 mov_vec v256,e8,tmp18,v256$0x0
 st_vec v64,e8,tmp18,env,$0xcd8
 st_vec v256,e8,tmp18,env,$0xce0
 st_vec v128,e8,tmp18,env,$0xd00

 ---- 00000000004006e8 0000000000000000 0000000000000000
 add_i64 loc20,env,$0x2f88
 ld_i64 loc22,env,$0xdd0
 ld_i64 loc23,env,$0xcd0
 call vfp_addd,$0x0,$1,loc19,loc22,loc23,loc20
 st_i64 loc19,env,$0xcd0
 mov_vec v256,e8,tmp24,v256$0x0
 st_vec v64,e8,tmp24,env,$0xcd8
 st_vec v256,e8,tmp24,env,$0xce0
 st_vec v128,e8,tmp24,env,$0xd00

 ---- 00000000004006ec 0000000000000000 0000000000000000
 add_i64 sp,sp,$0x10

 ---- 00000000004006f0 0000000000000000 0000000000000000
 shl_i64 pc,lr,$0x8
 sar_i64 pc,pc,$0x8
 and_i64 pc,pc,lr
 call lookup_tb_ptr,$0x6,$1,tmp25,env
 goto_ptr tmp25
 set_label $L0
 exit_tb $0x7ffa0c055e83
```

## x86-64
```
OUT: [size=336]
  -- guest addr 0x00000000004006d4 + tb prologue
0x7ffa0c055f40:  8b 5d f0                 movl     -0x10(%rbp), %ebx
0x7ffa0c055f43:  85 db                    testl    %ebx, %ebx
0x7ffa0c055f45:  0f 8c 29 01 00 00        jl       0x7ffa0c056074
0x7ffa0c055f4b:  c6 45 f4 00              movb     $0, -0xc(%rbp)
0x7ffa0c055f4f:  48 8b 9d 38 01 00 00     movq     0x138(%rbp), %rbx
0x7ffa0c055f56:  48 83 c3 f0              addq     $-0x10, %rbx
0x7ffa0c055f5a:  48 89 9d 38 01 00 00     movq     %rbx, 0x138(%rbp)
  -- guest addr 0x00000000004006d8
0x7ffa0c055f61:  4c 8d 63 08              leaq     8(%rbx), %r12
0x7ffa0c055f65:  4d 8b ec                 movq     %r12, %r13
0x7ffa0c055f68:  49 c1 e5 08              shlq     $8, %r13
0x7ffa0c055f6c:  49 c1 fd 08              sarq     $8, %r13
0x7ffa0c055f70:  4d 23 ec                 andq     %r12, %r13
0x7ffa0c055f73:  4c 8b a5 d0 0c 00 00     movq     0xcd0(%rbp), %r12
0x7ffa0c055f7a:  4d 89 65 00              movq     %r12, (%r13)
  -- guest addr 0x00000000004006dc
0x7ffa0c055f7e:  4c 8b e3                 movq     %rbx, %r12
0x7ffa0c055f81:  49 c1 e4 08              shlq     $8, %r12
0x7ffa0c055f85:  49 c1 fc 08              sarq     $8, %r12
0x7ffa0c055f89:  4c 23 e3                 andq     %rbx, %r12
0x7ffa0c055f8c:  4c 8b ad d0 0d 00 00     movq     0xdd0(%rbp), %r13
0x7ffa0c055f93:  4d 89 2c 24              movq     %r13, (%r12)
  -- guest addr 0x00000000004006e0
0x7ffa0c055f97:  4c 8d 63 08              leaq     8(%rbx), %r12
0x7ffa0c055f9b:  4d 8b ec                 movq     %r12, %r13
0x7ffa0c055f9e:  49 c1 e5 08              shlq     $8, %r13
0x7ffa0c055fa2:  49 c1 fd 08              sarq     $8, %r13
0x7ffa0c055fa6:  4d 23 ec                 andq     %r12, %r13
0x7ffa0c055fa9:  4d 8b 65 00              movq     (%r13), %r12
0x7ffa0c055fad:  4c 89 a5 d0 0d 00 00     movq     %r12, 0xdd0(%rbp)
0x7ffa0c055fb4:  c5 f9 ef c0              vpxor    %xmm0, %xmm0, %xmm0
0x7ffa0c055fb8:  c5 f9 d6 85 d8 0d 00 00  vmovq    %xmm0, 0xdd8(%rbp)
0x7ffa0c055fc0:  c5 fe 7f 85 e0 0d 00 00  vmovdqu  %ymm0, 0xde0(%rbp)
0x7ffa0c055fc8:  c5 f9 7f 85 00 0e 00 00  vmovdqa  %xmm0, 0xe00(%rbp)
  -- guest addr 0x00000000004006e4
0x7ffa0c055fd0:  4c 8b eb                 movq     %rbx, %r13
0x7ffa0c055fd3:  49 c1 e5 08              shlq     $8, %r13
0x7ffa0c055fd7:  49 c1 fd 08              sarq     $8, %r13
0x7ffa0c055fdb:  4c 23 eb                 andq     %rbx, %r13
0x7ffa0c055fde:  49 8b 5d 00              movq     (%r13), %rbx
0x7ffa0c055fe2:  48 89 9d d0 0c 00 00     movq     %rbx, 0xcd0(%rbp)
0x7ffa0c055fe9:  c5 f9 ef c0              vpxor    %xmm0, %xmm0, %xmm0
0x7ffa0c055fed:  c5 f9 d6 85 d8 0c 00 00  vmovq    %xmm0, 0xcd8(%rbp)
0x7ffa0c055ff5:  c5 fe 7f 85 e0 0c 00 00  vmovdqu  %ymm0, 0xce0(%rbp)
0x7ffa0c055ffd:  c5 f9 7f 85 00 0d 00 00  vmovdqa  %xmm0, 0xd00(%rbp)
  -- guest addr 0x00000000004006e8
0x7ffa0c056005:  48 8d 95 88 2f 00 00     leaq     0x2f88(%rbp), %rdx
0x7ffa0c05600c:  48 8b f3                 movq     %rbx, %rsi
0x7ffa0c05600f:  49 8b fc                 movq     %r12, %rdi
0x7ffa0c056012:  ff 15 68 00 00 00        callq    *0x68(%rip)
0x7ffa0c056018:  48 89 85 d0 0c 00 00     movq     %rax, 0xcd0(%rbp)
0x7ffa0c05601f:  c5 f9 ef c0              vpxor    %xmm0, %xmm0, %xmm0
0x7ffa0c056023:  c5 f9 d6 85 d8 0c 00 00  vmovq    %xmm0, 0xcd8(%rbp)
0x7ffa0c05602b:  c5 fe 7f 85 e0 0c 00 00  vmovdqu  %ymm0, 0xce0(%rbp)
0x7ffa0c056033:  c5 f9 7f 85 00 0d 00 00  vmovdqa  %xmm0, 0xd00(%rbp)
  -- guest addr 0x00000000004006ec
0x7ffa0c05603b:  48 8b 9d 38 01 00 00     movq     0x138(%rbp), %rbx
0x7ffa0c056042:  48 83 c3 10              addq     $0x10, %rbx
0x7ffa0c056046:  48 89 9d 38 01 00 00     movq     %rbx, 0x138(%rbp)
  -- guest addr 0x00000000004006f0
0x7ffa0c05604d:  48 8b 9d 30 01 00 00     movq     0x130(%rbp), %rbx
0x7ffa0c056054:  4c 8b e3                 movq     %rbx, %r12
0x7ffa0c056057:  49 c1 e4 08              shlq     $8, %r12
0x7ffa0c05605b:  49 c1 fc 08              sarq     $8, %r12
0x7ffa0c05605f:  4c 23 e3                 andq     %rbx, %r12
0x7ffa0c056062:  4c 89 a5 40 01 00 00     movq     %r12, 0x140(%rbp)
0x7ffa0c056069:  48 8b fd                 movq     %rbp, %rdi
0x7ffa0c05606c:  ff 15 16 00 00 00        callq    *0x16(%rip)
0x7ffa0c056072:  ff e0                    jmpq     *%rax
0x7ffa0c056074:  48 8d 05 08 fe ff ff     leaq     -0x1f8(%rip), %rax
0x7ffa0c05607b:  e9 98 9f fa ff           jmp      0x7ffa0c000018
  data: [size=16]
0x7ffa0c056080:  .quad  0x000055c27a44c5dd
0x7ffa0c056088:  .quad  0x000055c27a690267
```
# `0x004006d8` 部分だけ

## AArch64
```
0x004006d8:  fd0007e0  str      d0, [sp, #8]
```

## QEMU IR
```
 ---- 00000000004006d8 0000000000000000 0000000000000000
 mov_i64 loc3,sp
 add_i64 loc3,loc3,$0x8
 shl_i64 loc5,loc3,$0x8
 sar_i64 loc5,loc5,$0x8
 and_i64 loc5,loc5,loc3
 ld_i64 loc6,env,$0xcd0
 qemu_st_a64_i64 loc6,loc5,noat+un+leq,0
```

## x86-64
```
  -- guest addr 0x00000000004006d8
0x7ffa0c055f61:  4c 8d 63 08              leaq     8(%rbx), %r12
0x7ffa0c055f65:  4d 8b ec                 movq     %r12, %r13
0x7ffa0c055f68:  49 c1 e5 08              shlq     $8, %r13           ; shl_i64 loc5,loc3,$0x8 Shift left by 8
0x7ffa0c055f6c:  49 c1 fd 08              sarq     $8, %r13           ; sar_i64 loc5,loc5,$0x8 Shift right by 8
0x7ffa0c055f70:  4d 23 ec                 andq     %r12, %r13         ; and_i64 loc5,loc5,loc3
0x7ffa0c055f73:  4c 8b a5 d0 0c 00 00     movq     0xcd0(%rbp), %r12  ; ld_i64 loc6,env,$0xcd0
0x7ffa0c055f7a:  4d 89 65 00              movq     %r12, (%r13)       ; qemu_st_a64_i64 loc6,loc5,noat+un+leq,0
```
# `0x004006e8` 部分だけ
## AArch64
```
0x004006e8:  1e602820  fadd     d0, d1, d0
```

## QEMU IR
```
 ---- 00000000004006e8 0000000000000000 0000000000000000
 add_i64 loc20,env,$0x2f88
 ld_i64 loc22,env,$0xdd0
 ld_i64 loc23,env,$0xcd0
 call vfp_addd,$0x0,$1,loc19,loc22,loc23,loc20
 st_i64 loc19,env,$0xcd0
 mov_vec v256,e8,tmp24,v256$0x0
 st_vec v64,e8,tmp24,env,$0xcd8
 st_vec v256,e8,tmp24,env,$0xce0
 st_vec v128,e8,tmp24,env,$0xd00
```

## x86-64
```
  -- guest addr 0x00000000004006e8
0x7ffa0c056005:  48 8d 95 88 2f 00 00     leaq     0x2f88(%rbp), %rdx
0x7ffa0c05600c:  48 8b f3                 movq     %rbx, %rsi
0x7ffa0c05600f:  49 8b fc                 movq     %r12, %rdi
0x7ffa0c056012:  ff 15 68 00 00 00        callq    *0x68(%rip); vfp_addd だと思われる
0x7ffa0c056018:  48 89 85 d0 0c 00 00     movq     %rax, 0xcd0(%rbp)
0x7ffa0c05601f:  c5 f9 ef c0              vpxor    %xmm0, %xmm0, %xmm0
0x7ffa0c056023:  c5 f9 d6 85 d8 0c 00 00  vmovq    %xmm0, 0xcd8(%rbp)
0x7ffa0c05602b:  c5 fe 7f 85 e0 0c 00 00  vmovdqu  %ymm0, 0xce0(%rbp)
0x7ffa0c056033:  c5 f9 7f 85 00 0d 00 00  vmovdqa  %xmm0, 0xd00(%rbp)
```

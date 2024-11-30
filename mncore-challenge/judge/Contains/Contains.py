for lm0_addr in range(0, 32, 2):
    for lm1_addr in range(0, 32, 8):
        mask_reg = lm1_addr // 8 + 1
        print(f"lxor $lm{lm0_addr} $ln{lm1_addr}v $omr{mask_reg}")
    for lm1_addr in range(0, 32, 8):
        ans_addr = lm0_addr + 1
        mask_reg = lm1_addr // 8 + 1
        print(f'imm i"1" $r{ans_addr}/$imr{mask_reg}')

print()
print("d getd $lr0n0c0b0m0p0 16")

print()
for greg_addr in range(0, 32, 8):
    l1bm_addr = greg_addr * 4 // 2
    print(f"l1bmrlbor $lr{greg_addr}v2 $lb{l1bm_addr}")

# print("d getd $lb0n0c0b0m0p0 64")

print("nop/2")
for l1bm_addr in range(0, 64, 8):
    print(f"l1bmp $llb{l1bm_addr} $llr{l1bm_addr * 2}")
for l1bm_addr in range(0, 64, 8):
    print(f"l1bmp $llb{l1bm_addr} $lls{l1bm_addr * 2}")

print("d getd $lr0n0c0b0m0p0 64")

for ans_addr in range(0, 32, 2):
    print(f"lor $lr{ans_addr * 8} $ls{ans_addr * 8 + 2} $nowrite")
    print(f"lor $aluf $lr{ans_addr * 8 + 4} $nowrite")
    print(f"lor $aluf $lr{ans_addr * 8 + 6} $ln{ans_addr + 32}")

print("d getd $ln32n0c0b0m0p0 16")

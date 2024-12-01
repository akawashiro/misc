for contents in range(0, 32768, 32 * 8):
    l2bm_addr_offset = contents // 2
    for l2b_id in range(0, 8): 
        for lw in range(0, 16):
            first = contents + l2b_id * 32 + lw * 2 + 1
            second = contents + l2b_id * 32 + lw * 2 + 2
            l2bm_addr = (l2bm_addr_offset // (32 * 4)) * 16 + lw
            print(f"d set $lc{l2bm_addr}n{l2b_id // 2}c{l2b_id % 2} 1 s{hex(first)[2:]}_{hex(second)[2:]}")

L2B_DATA_SIZE_LW = 16384 // 8
for l2bm_addr in range(0, L2B_DATA_SIZE_LW, 64 * 4):
    print(f"l2bmd $lc{l2bm_addr} $lb{l2bm_addr // 8}")

print("nop/2")

for l2bm_addr in range(0, L2B_DATA_SIZE_LW, 64 * 4):
    print(f"l2bmd $lb{l2bm_addr // 8} $lc{l2bm_addr + L2B_DATA_SIZE_LW}")
print("nop/2")

print(f"mvd/n{16384 // 8} $lc{16384 // 8} $p0@0")
print(f"mvp/n16384 $p0@0 $d0@0")
# for i in range(0, 16384):
#     print(f"d getd $p{i}n0 1")

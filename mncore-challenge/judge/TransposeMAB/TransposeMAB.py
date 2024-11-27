for mab in range(0, 16):
    for lmaddr in range(0, 32, 8):
        l1baddr = mab * 64 + lmaddr * 2
        print(f"l1bmm@{mab} $lm{lmaddr}v $lb{l1baddr}")

print("nop/2")

for l1baddr in range(0, 64 * 16, 64):
    lmaddr = l1baddr // 32
    print(f"l1bmd $lb{l1baddr} $ln{lmaddr}v")

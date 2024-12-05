def i2v(l2b: int, l1b: int, mab: int, pe: int, lw: int, addr: int):
    v = 1
    f = 1

    print(f"{f=} at 2_W:1")
    v += lw * f
    f *= 2

    print(f"{f=} at 4_PE:1")
    v += pe * f
    f *= 4

    print(f"{f=} at 2_MAB:1")
    v += (mab % 2) * f
    f *= 2

    print(f"{f=} at 2_L1B:1")
    v += (l1b % 2) * f
    f *= 2

    print(f"{f=} at 8_L2B:1")
    v += l2b * f
    f *= 8

    print(f"{f=} at 4_L1B:2")
    v += (l1b // 2) * f
    f *= 4

    print(f"{f=} at 8_MAB:2")
    v += (mab // 2) * f
    f *= 8

    print(f"{f=} at 4_Addr:1")
    v += addr * f

    print(hex(v))

i2v(0, 0, 0, 0, 0, 0)
i2v(0, 0, 0, 0, 0, 1)
i2v(2,5,11,0,0,0)
i2v(7, 7, 15, 3, 0, 3)

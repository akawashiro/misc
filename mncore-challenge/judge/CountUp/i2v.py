def i2v(l2b: int, l1b: int, mab: int, pe: int, lw: int, addr: int):
    v = 1
    f = 1

    v += lw * f
    f *= 2

    v += pe * f
    f *= 4

    v += (mab % 2) * f
    f *= 2

    v += (l1b % 2) * f
    f *= 2

    v += l2b * f
    f *= 8

    v += (l1b // 2) * f
    f *= 4

    v += (mab // 2) * f
    f *= 8

    v += addr * f

    print(hex(v))

i2v(0, 0, 0, 0, 0, 0)
i2v(0, 0, 0, 0, 0, 1)
i2v(2,5,11,0,0,0)
i2v(7, 7, 15, 3, 0, 3)

for l1addr in range(0, 1024, 8):
    print(f"l1bmp $llb{l1addr} $llr0v")
    print("nop/2")
    print(f"lpassa $lr{l1addr * 2}v $lt")
    print("ipassa $r128 $mt1000")

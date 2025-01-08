for i in range(0, 256, 8):
    if i == 0:
        print(f"dvmulu $lm{i}v $lm{i}v $nowrite")
    else:
        print(f"dvfmau $lm{i}v $lm{i}v $mauf $nowrite")
    print(f"dvfmad $lm{i}v $lm{i}v $mauf $nowrite")
print(f"lpassa $mauf $lr0v")

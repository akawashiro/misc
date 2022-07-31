import argparse
import pathlib

def parse_objdump(result, filename):
    with open(filename) as f:
        start_addr = None
        end_addr = None
        name = None
        for l in f.readlines():
            if l[0] == '0' and l[-2] == ":":
                start_addr = int("0x" + l[:16], base=16)
                name = l[18:-3]
            if l != "\n" and start_addr != None and name != None:
                end_addr = int("0x" + l[2:8], base=16)
            if l == "\n" and name != None and start_addr != None and end_addr != None:
                result[name] = end_addr - start_addr
                start_addr = None
                end_addr = None
                name = None


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('objdump_result_path', type=pathlib.Path)
    # parser.add_argument('output', type=pathlib.Path)
    # args = parser.parse_args()

    funcsize = {}
    funcsize["objdump-Oz"] = {}
    parse_objdump(funcsize["objdump-Oz"], "objdump-Oz")
    funcsize["objdump-O2"] = {}
    parse_objdump(funcsize["objdump-O2"], "objdump-O2")
    funcsize["objdump-O3"] = {}
    parse_objdump(funcsize["objdump-O3"], "objdump-O3")

    difflist = []
    for fname in funcsize["objdump-Oz"].keys():
        if fname in funcsize["objdump-O3"]:
            difflist.append((funcsize["objdump-O3"][fname] - funcsize["objdump-Oz"][fname], fname))
    difflist.sort(reverse=True)
    for d in difflist[:300]:
        print(d[0], d[1])

if __name__ == '__main__':
    main()

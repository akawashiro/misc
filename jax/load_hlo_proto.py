import sys

sys.path.append("hlo_proto")

import hlo_pb2

with open("mnist.hlo_proto", "rb") as f:
    mnist_hlo_proto_data = f.read()
    mnist_hlo_proto = hlo_pb2.HloModuleProto()
    # print(mnist_hlo_proto)
    mnist_hlo_proto.ParseFromString(mnist_hlo_proto_data)

# print(mnist_hlo_proto)
with open("mnist_hlo_proto.txt", "w") as f:
    f.write(str(mnist_hlo_proto))

print(len(mnist_hlo_proto.computations))

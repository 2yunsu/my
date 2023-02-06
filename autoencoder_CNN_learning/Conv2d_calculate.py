stride = 2
kernel_size = 4
padding = 1
dilation = 1
H_in = 28
depth = 2


for i in range(depth):
    H_out = ((H_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1
    print("H_out: {}, depth: {}" .format(H_out, i))
    H_in = H_out

H_out = H_out**2
Linear_size = H_out*20

print(H_out)
print(Linear_size)

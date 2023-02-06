def in_order(node):
    if node[2] != -1:
        in_order(node_list[int(node[2])-1])
    print(node[1], end='')
    if node[3] != -1:
        in_order(node_list[int(node[3])-1])

for k in range(10):
    N = int(input())
    node_list = [[i+1, '', -1, -1] for i in range(N)]

    for i in range(N):
        node = input()
        node = node.split(' ')
        node_list[i][0] = node[0]
        node_list[i][1] = node[1]
        if len(node) >= 3:
            node_list[i][2] = node[2]
        if len(node) >= 4:
            node_list[i][3] = node[3]

    print("#{} ".format(k+1), end='')
    in_order(node_list[0])
    print()
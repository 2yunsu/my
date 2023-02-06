
# BFS를 수행한 결과 출력
if __name__ == "__main__":
    # 2차원 리스트의 맵 정보 입력 받기
    num_node = int(input())
    node_list = [[i+1, '', -1, -1] for i in range(num_node)] # [[id, item, left, right], ...]

    for i in range(num_node):
        node = input()
        node = node.split(' ')
        node_list[i][0] = node[0] # node[0] = id
        node_list[i][1] = node[1]  # node[1] = item
        if len(node) >= 3:
            node_list[i][2] = node[2] # node[2] = left
        if len(node) >= 4:
            node_list[i][3] = node[3] # node[2] = right
    print(node_list)
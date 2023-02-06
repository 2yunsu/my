from collections import deque

# N, M을 공백을 기준으로 구분하여 입력 받기
n, m = 15, 15 #map(int, input().split())

# 2차원 리스트의 맵 정보 입력 받기
all_graph = []
no = 0
for i in range(170): #args 어떻게 넣나?
    all_graph.append(list(map(int, input())))

# 이동할 네 가지 방향 정의 (상, 하, 좌, 우)
dx = [-1, 1, 0, 0]
dy = [0, 0, -1, 1]

# BFS 소스코드 구현
def bfs(x, y):
    # 큐(Queue) 구현을 위해 deque 라이브러리 사용
    queue = deque()
    queue.append((x, y))
    end = 0
    # 큐가 빌 때까지 반복하기
    while queue:
        x, y = queue.popleft()
        # 현재 위치에서 4가지 방향으로의 위치 확인
        for i in range(4):
            nx = x + dx[i]
            ny = y + dy[i]
            # 미로 찾기 공간을 벗어난 경우 무시
            if nx < 0 or nx >= n or ny < 0 or ny >= m:
                continue
            # 벽인 경우 무시
            if graph[nx][ny] == 1: #0을 1로 변경
                continue
            # if graph[nx][ny] > 3:
            #     continue
            # 해당 노드를 처음 방문하는 경우에만 최단 거리 기록
            if graph[nx][ny] == 0: #1을 0으로 변경
                graph[nx][ny] = 1 #지나온 길을 1로 변경
                queue.append((nx, ny))

            if graph[nx][ny] == 3:
                end = 1
                break

    # 가장 오른쪽 아래까지의 최단 거리 반환
    return end

# BFS를 수행한 결과 출력

for i in range(10):
    graph = []
    if len(all_graph[0]) == 1:
        no = all_graph[0][0]
    if len(all_graph[0]) >= 2:
        no = all_graph[0][0]*10+all_graph[0][1]
    del all_graph[0]
    for j in range(16):
        new_item = all_graph[j]
        graph.append(new_item)
    print("#{}" .format(no), bfs(1,1))
    del all_graph[0:16]
from collections import deque
# 이동할 세 가지 방향 정의 (우, 좌, 상)
dx = [1, -1, 0]
dy = [0, 0, -1]

# BFS 소스코드 구현
def bfs(_graph):
    # 큐(Queue) 구현을 위해 deque 라이브러리 사용
    width = 10
    exit = _graph[9].index(2)
    x, y = exit, 9
    queue = deque()
    queue.append((x, y))

    # 큐가 빌 때까지 반복하기
    while queue:
        x, y = queue.popleft()
        # 현재 위치에서 4가지 방향으로의 위치 확인

        for i in range(3):
            nx = x + dx[i]
            ny = y + dy[i]
            # 미로 찾기 공간을 벗어난 경우 무시
            if nx < 0 or nx >= width or ny < 0 or ny >= width:
                continue

            if _graph[ny][nx] == 1:
                _graph[ny][nx] = 0
                queue.append((nx, ny))

            if ny == 0:
                return nx

# BFS를 수행한 결과 출력
if __name__ == "__main__":
    for _ in range(1):
        graph = []
        test_no = int(input())
        for _ in range(10):
            graph.append(list(map(int, input().split())))

        print(bfs(graph))

from collections import deque
import argparse

# N, M을 공백을 기준으로 구분하여 입력 받기
# global variable
n, m = 15, 15

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



parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--N', type=int, default='16',
                    help='an integer for the accumulator')
parser.add_argument('--test_case', type=int, default='10',
                    help='an integer for the accumulator')
args = parser.parse_args()

# BFS를 수행한 결과 출력
if __name__ == "__main__":
    # 2차원 리스트의 맵 정보 입력 받기
    for i in range(args.test_case):
        no = int(input())
        graph = []
        for i in range(args.N):
            graph.append(list(map(int, input())))
        print("#{}" .format(no), bfs(1,1))
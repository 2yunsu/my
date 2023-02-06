import random
from pandas import DataFrame

#승리 조건
def win(z):
    dol=sorted(z)
    k=0
    while k<len(dol):
        nx=(dol[k])[0]
        ny=(dol[k])[1]
        if (nx,ny+1) in dol and (nx,ny+2) in dol and (nx,ny+3) in dol and (nx,ny+4) in dol:
            return 1
        elif (nx+1,ny) in dol and (nx+2,ny) in dol and (nx+3,ny) in dol and (nx+4,ny) in dol:
            return 1
        elif (nx+1,ny+1) in dol and (nx+2,ny+2) in dol and (nx+3,ny+3) in dol and (nx+4,ny+4) in dol:
            return 1
        elif (nx+1,ny-1) in dol and (nx+2,ny-2) in dol and (nx+3,ny-3) in dol and (nx+4,ny-4) in dol:
            return 1
        else:
            k += 1
    return 0

#오목판 만들기
row_list=[]
col_list=[]

for j in range(10):
    row_list.append(".")

for i in range(10):
    col_list.append(row_list)

board=DataFrame(col_list)

#돌 놓기
you=[]
com=[]

while True:
    # 1. 사람이 돌 놓기
    x=int(input())
    y=int(input())
    if x>9 or y>9 or board.iat[x,y]=="O" or board.iat[x,y]=="X":
        print("다른 좌표를 입력해야 합니다.")
        print(board)
        continue

    board.iat[x,y]="X"
    you.append((x,y))

    # 2. 사람이 이겼는지 확인
    if win(you) == True:
        print(board)
        print("You WIN")
        break #break은 어디로 탈출하는지

    # 3. 컴퓨터가 돌 놓기
    while True:
        cx = random.randint(0, 9)
        cy = random.randint(0, 9)
        if board.iat[cx,cy]=="X" or board.iat[cx,cy]=="O":
            continue
        break
    board.iat[cx,cy]="O"
    com.append((cx, cy))

    # 4. 컴퓨터가 이겼는지 확인
    if win(com)==True: #random이라 실험이 힘듦
        print(board)
        print("You Lose")
        break

    print(board)
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
            break
        elif (nx+1,ny) in dol and (nx+2,ny) in dol and (nx+3,ny) in dol and (nx+4,ny) in dol:
            return 1
            break
        elif (nx+1,ny+1) in dol and (nx+2,ny+2) in dol and (nx+3,ny+3) in dol and (nx+4,ny+4) in dol:
            return 1
            break
        elif (nx+1,ny-1) in dol and (nx+2,ny-2) in dol and (nx+3,ny-3) in dol and (nx+4,ny-4) in dol:
            return 1
            break
        else:
            k += 1

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
    x=int(input())
    y=int(input())
    if x>9 or y>9 or board.iat[x,y]=="O" or board.iat[x,y]=="X":
        print("다른 좌표를 입력해야 합니다.")
        print(board)
    else:
        board.iat[x,y]="X"
        you.append((x,y))
        if win(you) == True:
            print(board)
            print("You WIN")
            break
        while True:
            cx = random.randint(0, 9)
            cy = random.randint(0, 9)
            if board.iat[cx,cy]=="X" or board.iat[cx,cy]=="O":
                cx = random.randint(0, 9)
                cy = random.randint(0, 9)
            else:
                board.iat[cx,cy]="O"
                if win(com)==True: #random이라 실험이 힘듦
                    print(board)
                    print("You Lose")
                break
        print(board)
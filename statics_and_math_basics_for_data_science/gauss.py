import numpy as np
import random


def random_matrix(rows, cols, max_no=20):
	row_list = []
	for r in range(rows):
		col_list = []
		for c in range(cols):
			col_list.append(float(random.randint(0, max_no)))  # float형으로 변환함
		row_list.append(col_list)

	return np.array(row_list)  # 넘파이 배열 형태로 변환


def gauss_jordan_elimination(matrix):
	print("가우스-조르당 소거법 예시 \n")

	# 행렬의 가로 세로
	mcol = len(matrix[0])
	mrow = len(matrix)

	print("최초상태")
	print(matrix, "\n")

	# 입력 행렬의 1차 검증 : 값이 모두 0인 열이 있는지 -> 변수의 값을 구할 수 없음
	idx = 0
	for c in range(mcol):  # 열
		for r in range(mrow):  # 행
			idx += matrix[r][c]

		# if idx == 0:  # 첫 열이 모두 0
		# 	print(c + 1, "열 의 값이 모두 0입니다.")
		# 	return

		# 2차 검증 : 행/열의 갯수
		# if mrow != mcol - 1:  # 변수의 갯수(결괏값을 제외한 열의 갯수)와 행의 갯수가 일치하여야 함
		# 	print("해가 무수히 많거나 없습니다.")
		# 	return

		# 1. 가우스 소거법 : Leading 1(선행원소 1)을 찾기 위한 알고리즘
	row_count = 0  # 연산 중인 열을 지시
	for col_count in range(mcol - 1):  # 선행 원소 아래의 값(열 하단)이 0이 될때까지 반복 수행
		comp_num = np.inf  # 비교할 숫자의 기본값을 무한대로 지정(보다 작은 수)

		# 첫 열이 1이거나, 0이 아닌 수 중 가장 작은 행을 추출
		for i in range(row_count, mrow):
			if (0 < abs(matrix[i][col_count])) & (abs(matrix[i][col_count]) < abs(comp_num)):
				comp_num = matrix[i][col_count]  # 절댓값이 가장 작은 첫 행의 값

				first_row = i  # 첫 행으로 사용할 행
				if matrix[i][0] == 1:  # 1을 찾은 경우
					break

		matrix[first_row] = matrix[first_row] / comp_num  # 첫 행의 첫 열을 1로 변환
		print(first_row + 1, "행이", row_count + 1, "행과 교환")
		matrix[first_row], matrix[row_count] = matrix[row_count].copy(), matrix[first_row].copy()  # 대상 인덱스와 첫 행을 교환

		# 상수배 후 덧셈연산
		for j in range(row_count + 1, mrow):
			if matrix[j][col_count] != 0:
				con_no = matrix[j][col_count]
				matrix[j] = matrix[j] - con_no * matrix[row_count]

		# 이미 Leading 1을 찾은 행을 연산에서 제외
		row_count += 1
		print(matrix, "\n")

	# 2. Back Substitution(가우스-조르당 소거법) : 기약행사다리꼴 형태로 변환
	for col in range(mcol - 2, 0, -1):
		for row in range(col - 1, -1, -1):  # col/row를 파이썬 인덱스에 맞게 적용
			const = matrix[row][col] * -1  # constraint
			print(col + 1, "행을", const, "만큼 상수배하여", row + 1, "행의", col + 1, "열을 소거")  #
			matrix[row] += const * matrix[col]

			print(matrix, "\n")

	print("방정식의 해는")
	print(matrix[:, -1])
	return  # matrix <- 기약사다리꼴 형태를 반환

matrix = np.array([[1,-1,0,0,1,3],[1,1,0,-3,0,6], [2,-1,0,1,-1,5], [-1,2,0,-2,-1,-1]])
gauss_jordan_elimination(matrix)
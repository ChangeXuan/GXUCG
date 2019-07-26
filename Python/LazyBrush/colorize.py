import build_graph
import cv2
import numpy as np

def func(I, M, B, C, mix, verbose):
	J = None
	print('Color: Starting...')
	nrow, ncol = I.shape[0], I.shape[1]
	# 论文中测算出来的超参数
	lambda_value = 0.95
	color_count = len(C)
	# 周长
	K_value = 2*(nrow+ncol)
	# 连接终端的边权重
	D = K_value*(1-lambda_value)
	# 颜色计数器
	count = 0
	print(C)
	while count < len(C):
		print('Color: processing new color')
		# 获取颜色标号
		c = C[count,3]
		if verbose:
			print(C[0,:])

		print('Color: detecting empty areas.')
		print('Color: building graph (first-type edge).')

		G, indices, S, T = build_graph.func(M,I,mix)

		print('Color: adding second-type edges.')

		s, t, w, count_S, count_T = [], [], [], 0, 0

		# 添加两个终端结点
		for k in range(len(indices)):
			x, y = indices[k][0], indices[k][1]
			if B[x, y] == c:
				# maxflow隐藏了两个终端结点
				G.add_tedge(k, 0, D)
				count_S += 1
			# 不等于当前的c和不等于0即为其他颜色
			elif B[x, y] != 0:
				G.add_tedge(k, D, 0)
				count_T += 1

		print('Color: performing min-cut')

		if G:
			flow = G.maxflow()
			if verbose:
				print(count_S, count_T)
				print(flow)

			for k in range(len(indices)):
				x, y = indices[k][0], indices[k][1]
				if G.get_segment(k):
					M[x,y] = c
			# cv2.imshow('test', M)
			# cv2.waitKey(0)

			print('Color: saveing new attribution')

		count += 1

	J = M
	
	return J



if __name__ == '__main__':
	pass
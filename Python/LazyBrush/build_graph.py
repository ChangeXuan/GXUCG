import maxflow
import numpy as np

def func(M, I, mix):
	G, indices, S, T = None, None, None, None
	mix1 = mix
	mix2 = 1-mix
	nrow, ncol = I.shape[0], I.shape[1]

	# 建立索引
	i_mask, j_mask = np.where(M == 0)
	# 初始化图信息
	s, t, w = [], [], []
	# 这里的是逐行扫描，matlab里的是逐列
	indices = np.column_stack((i_mask, j_mask))
	indices_len = len(indices)
	K0 = np.zeros((nrow, ncol))
	
	# 构建索引矩阵，这里是先行后列
	# 建立矩阵是为了用来较好的判断像素邻接点
	for k in range(indices_len):
		K0[indices[k][0],indices[k][1]] = k

	# 初始化终端结点
	S = indices_len + 1
	T = indices_len + 1

	# 构建一个图G
	G = maxflow.GraphFloat()
	node_idx = G.add_grid_nodes(len(indices))

	for k in range(indices_len):
		x, y = indices[k][0], indices[k][1]
		if x < nrow-1:
			if M[x+1,y] == 0:
				w = I[x+1,y]
				G.add_edge(k, K0[x+1,y], w, w)
				# s.append(k)
				# t.append(K0[x+1,y])
				# # 这里的权重于matlab的不同是因为计算灰度的公式不同
				# w.append(mix2*I[x+1,y])

		if y < ncol-1:
			if M[x, y+1] == 0:
				w = I[x,y+1]
				G.add_edge(k, K0[x,y+1], w, w)
				# s.append(k)
				# t.append(K0[x,y+1])
				# w.append(mix2*I[x,y+1])

	return G, indices, S, T



if __name__ == '__main__':
	# x = np.array([0,1,2])
	# y = np.array([4,5,6,7])
	# xs,ys = np.meshgrid(x,y)
	# print(xs, ys)
	s = []
	ks = [1,2,3,4,5]
	for k in ks:
		s = np.hstack((s, k))
	print(s)
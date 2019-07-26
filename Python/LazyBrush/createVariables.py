import cv2
import numpy as np
import operator
from functools import reduce

def rgb2gray(rgb):
	# rgb_re = np.reshape(rgb, (-1,3))
	# rgb_sum = np.sum(rgb_re, axis=1)/3
	# rgb_resu = np.reshape(rgb_sum, (rgb.shape[:2]))
	# return rgb_resu
	return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

# 取得画笔的颜色种类
def get_colors(img):
	colors = None
	white_flag = np.array([255,255,255])

	img_re = np.reshape(img, (-1,3))
	colors = np.unique(img_re, axis=0)

	index = np.where((colors == white_flag).all(1))[0]
	colors = list(colors)
	del colors[index[0]]
	colors = np.array(colors)

	return colors, img_re


# 外部调用的方法
def func(imagePath, brushPath, scaling, verbose):
	I, M, B, C, intensity, im_overlay = None, None, None, None, None, None
	print('Init: Loading from the files.')
	## 读入图片
	im = cv2.imread(imagePath)
	im_overlay = im.copy()
	# 连透明度也读取
	# im_b = cv2.imread(brushPath, cv2.IMREAD_UNCHANGED)
	im_b = cv2.imread(brushPath)
	nrow, ncol = im.shape[0], im.shape[1]

	## 寻找颜色
	print('Init: Finding colors.')

	## 得到需要返回的变量
	# 求I二次非线性权重图
	im_gray = rgb2gray(im)
	img = im_gray/255
	perimeter = 2*(nrow+ncol)
	im_scaled = (perimeter)*(img*img)+1
	I = im_scaled
	# print(im_scaled[100,:])
	# cv2.imshow('122',im_scaled)
	# cv2.waitKey(0)
	# 初始化蒙版M
	M = np.zeros((nrow, ncol))
	# M[0,0] = 1
	# 复制img得到intensity
	intensity = img.copy()
	# 求C颜色种类矩阵
	# column_stack将一维向量组合到二维
	colors, im_b_2 = get_colors(im_b)
	C = np.column_stack((colors, range(1,len(colors)+1)))

	# 求B笔刷矩阵
	im_b_3 = np.zeros((nrow*ncol,1))
	for index in range(1,len(C)+1):
		im_b_3[np.where((C[index-1,:3] == im_b_2).all(1))[0]] = index
	B = np.reshape(im_b_3, (nrow, ncol))

	return I, M, B, C, intensity, im_overlay

if __name__ == '__main__':
	a = np.array([[1,1,1], [2,2,2]])
	print(a[0])
	b = np.sum(a, axis=1)
	print(b)

	# a = np.array([[1,2], [3,4]])
	# b = np.array([[5,6], [7,8]])
	# print(a)
	# print(b)
	# print(a*b)
	# a = [1,2,5]
	# b = np.zeros((6,1))
	# print(b)
	# b[a] = 1
	# print(b)
	# c = np.array(((1, 2 ,1), (3, 4,1), (5, 6,1), (1, 3,1), (3, 4,1), (7, 6,1)))
	# print(c)
	# # s = set() #创建空集合
	# # for t in c:
	# # 	s.add(tuple(t)) #将数组转为元祖tuple，保证不被修改；再把元祖加入到集合中，完成去重
	# # g = np.array(list(s)) # 将集合转换为列表，最后转为二维数组
	# g = np.array(list(set(tuple(t) for t in c)))
	# print(g)

	# a = np.array([[1,2,3],[2,2,2],[0,0,0]])
	# b = np.array([[5,7,4],[0,0,0]])
	# c = np.in1d(a,b)
	# # for item in a:
	# # 	if item in b:
	# # 		print('1')
	# # 	else:
	# # 		print('0')
	# # M = g.reshape(g.shape+(1,)) - a.T.reshape((1,a.shape[1],a.shape[0]))
	# # c = np.any(np.any(M == 0, axis=0), axis=1)
	# print(c)

	# a = np.array([[255, 255, 255],
	# 	 [255, 255 ,255],
	# 	 [255, 255, 255],
	# 	 [1,2,3]])
	# b = np.array([[1,2,3,1],
	# 	[1,2,3,1]])
	# print(b[:,:3])
	# for item in a:
	# 	if item in b[:,:3]:
	# 		print('1')
	# 	else:
	# 		print('0')
	# def findByRow(mat, row):
	# 	return np.where((mat == row).all(1))[0]
	# print(np.arange(18, 27))
	# print(findByRow(np.arange(270).reshape((30, 9)), np.arange(18, 27)))
	# str=np.array([1,2,3,4,5,2,6])
	# del list(str)[1]
	# print(str)
	pass
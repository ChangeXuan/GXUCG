#######
# 图片预处理
# 把一张带有场景得物体切分成目标物体和背景
# 其中，目标为所需要的图片
#######
import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

# 交互创建白色空洞
class ChooseMask:
	def __init__(self, img, img2, mask, neetRect):
		self.img = img
		self.img2 = img2
		self.mask = mask
		self.rectangle = False 
		self.ix, self.iy = 0, 0
		self.rect = (0,0,1,1)
		self.rect_or_mask = 100
		self.true_break = False
		self.rect_over = not neetRect
		self.font = 0

	def draw(self, event, x, y, flags, param):
		# -------动态绘制矩形框
		if event == cv2.EVENT_RBUTTONDOWN:
			self.rectangle = True
			self.ix,self.iy = x,y
		elif event == cv2.EVENT_MOUSEMOVE:
			if self.rectangle == True:
				self.img = self.img2.copy()
				cv2.rectangle(self.img,(self.ix,self.iy),(x,y),(0, 255, 0),2)
				self.rect = (self.ix, self.iy,abs(self.ix-x),abs(self.iy-y))
				self.rect_or_mask = 0
		elif event == cv2.EVENT_RBUTTONUP:
			self.rectangle = False
			self.rect_over = True
			cv2.rectangle(self.img,(self.ix,self.iy),(x,y),(0, 255, 0),2)
			self.rect = (self.ix,self.iy,abs(self.ix-x),abs(self.iy-y))
			self.rect_or_mask = 0
			print(" Now use mouse left to select mask \n")
		# -------绘制mask区域(背景)
		if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON)):
			if self.rect_over:
				self.true_break = True
				cv2.circle(self.img, (x, y), 2, (255, 255, 255), 2)
				cv2.circle(self.mask, (x, y), 2, 255, 2)
			else:
				print('first use mouse right to select rect')

	def drawMake(self):
		cv2.namedWindow('draw')
		cv2.setMouseCallback('draw', self.draw)
		# 因为这里的鼠标回调会改变self.img
		# 所以需要使用循环来刷新self.img的显示
		while 1:
			cv2.imshow('draw', self.img)
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				if self.true_break:
					break
				else:
					print("plean use left button select mask roi \n")
			if k == ord('q'):
				return self.mask, self.rect, self.font, 0
			if k == ord('f'):
				self.font = 1
				print()
			if k == ord('b'):
				self.font = 0
		cv2.destroyWindow('draw')
		return self.mask, self.rect, self.font, 1

def stand_img(img):
	w = img.shape[1]
	if w > 700:
		img = cv2.resize(img, (img.shape[1]//10,img.shape[0]//10))
	return img

def main(img):
	neetRect = True
	font = 0
	while 1:
		bg_model = np.zeros((1, 65), np.float64)
		fg_model = np.zeros((1, 65), np.float64)
		mask = np.zeros(img.shape[:2], np.uint8)
		# -------new_mask由鼠标绘制给出
		CM = ChooseMask(img.copy(), img.copy(), mask, neetRect)
		mask, rect, font, flag = CM.drawMake()
		if flag == 0:
			return img
		# -------先用rect
		if neetRect:
			mask_rect = np.zeros(img.shape[:2],np.uint8)
			cv2.grabCut(img,mask_rect,rect,bg_model,fg_model,1,cv2.GC_INIT_WITH_RECT)
		# -------再用mask
		if font:
			mask_rect[mask == 255] = 1
			mask,_,_ = cv2.grabCut(img,mask_rect,None,bg_model,fg_model,5,cv2.GC_INIT_WITH_MASK)
			mask = np.where((mask==1)|(mask==3),1,0).astype('uint8')
		else:
			mask_rect[mask == 255] = 0
			mask,_,_ = cv2.grabCut(img,mask_rect,None,bg_model,fg_model,5,cv2.GC_INIT_WITH_MASK)
			mask = np.where((mask==0)|(mask==2),0,1).astype('uint8')
		img = img*mask[:,:,np.newaxis]
		# -------显示图片
		cv2.imshow('show', img)
		neetRect = False

'''
步骤：
1. 右键框选出目标物体
2. 左键去除背面区域(默认)，点击f表示左键去除正面，点击b表示左键去除背面
3. 按下q键保存图片并进行下一张图片的操作

'''
if __name__ == '__main__':
	src_dir = "src_img"
	tar_dir = "pre_img"
	for filename in os.listdir(r"./"+src_dir):
		print(filename) #just for test
		img = cv2.imread(src_dir+"/"+filename)
		img = stand_img(img)
		pre_img = main(img)
		cv2.imwrite(tar_dir+"/"+filename, pre_img)
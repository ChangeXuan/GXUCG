#######
# 形态矫正
# 对目标进行仿射变换
# 把目标顶满图片
# 使用最小包裹矩形来进行矫正
#######
import cv2
import numpy as np
import os, math

# 取得最小包裹矩形
def cal_rect_angle(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (7, 7), 0)
	thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY)[1]
	kernel = np.ones((9, 9), np.uint8)
	thresh = cv2.erode(thresh, kernel)  # 腐蚀
	# coords = np.column_stack(np.where(thresh > 0))
	image, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	return cv2.minAreaRect(contours[-1])

# 矫正角度
def correct_angle(angle, box):
	min_x = sorted(box.tolist())[:2]
	for index in range(len(box)):
		box[index] = box[index][::-1]
	min_y = sorted(box.tolist())[:2]
	dis_x = abs(min_y[0][1] - min_y[1][1])
	dis_y = abs(min_x[0][1] - min_x[1][1])
	angle = abs(angle)
	# 逆时针旋转
	if angle > 45:
		angle = 90-angle
	else:
		angle = -angle
	return angle

# 矫正图片
def correct_image(angle, temp_img, img):
	# 定义一个旋转矩阵
	# 逆时针旋转
	mat_rotate = cv2.getRotationMatrix2D(temp_img.shape[:2][::-1], angle, 1)
	temp_img = cv2.warpAffine(temp_img, mat_rotate, temp_img.shape[:2][::-1])
	img = cv2.warpAffine(img, mat_rotate, img.shape[:2][::-1])
	return temp_img, img

# 剪切图片
def cut_img(img, box):
	left_top = box[1]
	box_w = abs(box[1][0] - box[2][0])
	box_h = abs(box[0][1] - box[1][1])
	s_y, s_x = max(0,box[1][1]), max(0,box[1][0])
	img = img[s_y:s_y+box_h, s_x:s_x+box_w]
	return img

# 把目标图片填充满整张图片
def full_image(img, temp_img):
	rect_info = cal_rect_angle(temp_img)
	angle = rect_info[-1]
	box = cv2.boxPoints(rect_info)
	box = np.int0(box)
	img = cut_img(img, box)
	return img

def main(img_name):
	print(img_name)
	img = cv2.imread(img_name)
	rect_info = cal_rect_angle(img)
	angle = rect_info[-1]
	box = cv2.boxPoints(rect_info)
	box = np.int0(box)
	temp_img = img.copy()
	cv2.drawContours(temp_img, [box], 0, (0,0, 255), 3)
	angle = correct_angle(angle, box)
	temp_img, img = correct_image(angle, temp_img, img)
	img = full_image(img, temp_img)
	return img

if __name__ == '__main__':
	src_dir = "pre_img"
	tar_dir = "morph_img"
	for image_name in os.listdir(r"./"+src_dir):
		morph_img = main(image_name)
		cv2.imwrite(tar_dir+"/"+image_name, morph_img)
	cv2.waitKey(0)
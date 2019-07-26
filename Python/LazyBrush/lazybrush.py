import createVariables, colorize
import numpy as np
import cv2

def func(base_name, mode, save_flag):
	verbose = 0
	I, M, B, C, intensity, im_overlay = createVariables.func(base_name+'.png', base_name+'_brushes.png', 
						mode, verbose)

	mix = 0.0
	color_map = colorize.func(I,M,B,C,mix,verbose)

	result = np.zeros((I.shape[0], I.shape[1], 3), np.uint8)
	for x in range(I.shape[0]):
		for y in range(I.shape[1]):
			# print(im_overlay[x, y,:])
			if im_overlay[x, y, 0] > 200 and im_overlay[x, y, 1] > 200 and im_overlay[x, y, 2] > 200:
				im_overlay[x, y,:] = C[int(color_map[x, y])-1][:3]


	cv2.imshow('1', im_overlay)
	cv2.waitKey(0)

	return result

if __name__ == '__main__':
	result = np.zeros((100,100,3), np.uint8)
	print(result)
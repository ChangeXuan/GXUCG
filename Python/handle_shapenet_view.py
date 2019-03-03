from pynput import mouse,keyboard
from pynput.keyboard import Key, Controller
import os, time
import numpy as np

kc = Controller()
mc = mouse.Controller();

# 取得模型的文件路径
def get_model_path(fix_path):
	model_path_ary = []
	# 第一层
	for one_file in os.listdir(fix_path):
		# 第二层
		for two_file in os.listdir(fix_path+'/'+one_file):
			model_path_ary.append('\\%s\\%s\\'%(one_file, two_file))
		break
	return model_path_ary


# 0-主视图,1-左视图,2-俯视图
# 俯视图需要沿y轴旋转180度
def view_key(flag):
	if flag == 0:
		kc.press(keyboard.Key.ctrl)
		kc.press(keyboard.Key.end)
		kc.release(keyboard.Key.ctrl)
		kc.release(keyboard.Key.end)
	elif flag == 1:
		kc.press(keyboard.Key.ctrl)
		kc.press(keyboard.Key.page_down)
		kc.release(keyboard.Key.ctrl)
		kc.release(keyboard.Key.page_down)
	else:
		kc.press(keyboard.Key.home)
		kc.release(keyboard.Key.home)

# 保存视图
def save_view():
	# 鼠标定位
	# 不同的电脑需要自己定位
	mc.position = (177, 104)
	# 左键单击
	mc.click(mouse.Button.left,1)
	time.sleep(0.5)
	# 回车确定
	kc.press(keyboard.Key.enter)
	kc.release(keyboard.Key.enter)

'''
主要步骤分为一下几步
{
	- 打开程序
	- 导入模型
	{
		- 切换视角
		- 屏幕截图(177,104)
	}
	- 关闭程序
}
'''
def main():
	# -导入模型1
	app_name = 'MeshLab.lnk'
	fix_path = 'D:\\QzxSpace\\Model\\ShapeNetCore.v2'
	model_name = 'models\\model_normalized.obj'
	model_path_ary = get_model_path(fix_path)
	compute_flag = 0
	for model_path in model_path_ary:
		# 打开程序
		# D:\QzxSpace\Model\ShapeNetCore.v2\02691156\10155655850468db78d106ce0a280f87\models\model_normalized.obj
		os.popen(r'%s %s%s%s'%(app_name, fix_path, model_path, model_name))
		time.sleep(2)
		for index in range(3):
			view_key(index)
			save_view()
			time.sleep(1)
		# 关闭程序
		os.popen('taskkill /f /im meshlab.exe')
		time.sleep(2)
		compute_flag += 1
		if compute_flag >= 10:
			time.sleep(10)
			compute_flag = 0
		


def on_press(key):
	try:
		if key.char=='1':
			# 开始运行
			start_time = time.time()
			main()
			end_time = time.time()
			print(end_time-start_time)
	except AttributeError:
		# print(key)11
		pass

def on_release(key):
	if key == keyboard.Key.esc:
		return False

print('running...')
# 键盘监听
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
	listener.join()





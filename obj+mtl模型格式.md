```
OBJ
# obj对应的材质文件
mtllib testvt.mtl
# 组名称
g default
# o 对象名称(Object name)
o testvt.obj
# 顶点
v -0.5 -0.5 0.1
v -0.5 -0.5 -0.1
v 0 0.5 0.1
v 0 0.5 -0.1
v 0.5 -0.5 0.1
v 0.5 -0.5 -0.1
# 纹理坐标
vt 0 1
vt 1 1
vt 0.5 0
# 顶点法线
vn 0 0 1
vn 0 0 -1
# 当前图元所用材质
usemtl Default
# s Smooth shading across polygons is enabled by smoothing groups.
# Smooth shading can be disabled as well.
s off
# v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3(索引起始于1)    
f 1/1/1 5/2/1 3/3/1
f 6/2/2 2/1/2 4/3/2
```
```
MTL
# 定义一个名为 'xxx'的材质
newmtl xxx
# 材质的环境光（ambient color）
Ka 0 0 0
# 散射光（diffuse color）用Kd
Kd 0.784314 0.784314 0.784314
# 镜面光（specular color）用Ks
Ks 0 0 0
# 折射值 可在0.001到10之间进行取值。若取值为1.0，光在通过物体的时候不发生弯曲。玻璃的折射率为1.5。
Ni 1
# 反射指数 定义了反射高光度。该值越高则高光越密集，一般取值范围在0~1000。
Ns 400
# 滤光透射率
Tf 1 1 1
# 渐隐指数描述 参数factor表示物体融入背景的数量，取值范围为0.0~1.0，取值为1.0表示完全不透明，取值为0.0时表示完全透明。
d 1
# 为漫反射指定颜色纹理文件
map_Kd test_vt.bmp
```
来自https://www.jianshu.com/p/b52e152d44a9

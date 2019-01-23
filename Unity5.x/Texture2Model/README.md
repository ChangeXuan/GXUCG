# 图片贴到模型上
思路如下：
- 使用射线检测来判断模型中的那些顶点是目前可见的。
- 使用可见的顶点构建uv坐标。
- 为了使一个模型包含两种及以上的图片贴图，所以把图片贴图整合到一张图片中
- 最后使用uv完成贴图。

未完成：
- 模型自动旋转，然后把该时刻的可见顶点坐标存入数据结构中。
- 通过数据结构中的顶点坐标来构建拥有不同贴图的uv坐标
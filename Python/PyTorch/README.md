# Window10 下使用Anaconda3(python3.7.1)安装 PyTorch1.0
## follow
- step1: 下载[Anaconda3](https://www.anaconda.com/distribution/#windows)。
- step2: 安装Anaconda3。在安装过程的Advanced Options，可选择第一个。
- step3: 安装好后，打开cmd，输入python，看看是否能够正常运行，查看完后输入'ctrl+z'和敲击回车退出python输入界面。
- step4: 去到[PyTroch](https://pytorch.org/get-started/locally/#anaconda)选择相应的版本。
```
我使用的是：Your OS=windows, Package=Conda, Language=Python 3.7, CUDA=9.0.
官网提示的命令为：conda install pytorch torchvision cudatoolkit=9.0 -c pytorch
```
- step5: 在cmd中输入官网提供的命令即可完成下载安装
```
验证：
$ python
>> import torch
>> print(torch.__version__)
1.0.1
>> torch.cuda.is_availiable()
```

## 安装pydensecrf
```
可以使用pip install pydensecrf, 但是只限于python3.5.
因为我使用的时Anaconda3自带的python3.7.1故可以使用conda install pydensecrf,
但是直接使用conda install pydensecrf会出现找不到包的错误出现.
按照Anaconda官网的提示，可以使用conda install -c conda-forge pydensecrf,
但是直接使用这条命令会更新整个Anaconda，并且会使python的版本降为3.6.7，
而且之前下载的Pytorch等也会被移除。
故推荐的下载顺序如下
```
- 安装Anaconda3
- conda install -c conda-forge pydensecrf(如果一直卡在solv，可以试试conda update --all)
- conda install pytorch torchvision cudatoolkit=9.0 -c pytorch(如果教慢可以添加[清华的源](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/))


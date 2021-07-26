### pretreat.py

   用于提取关节坐标的特征，形成关于特征的灰度图，并保存为npy特征向量。要运行此py文件，需要再该py文件下新建文件夹。如果用vgg19提取特征，需要建一个vgg19的文件夹，在vgg19文件夹下新建train、test文件夹下，train、test文件夹下分别新建000，001，002，003，004文件夹，保存的文件在分别的文件夹下。如果用googlenet，需要一样的操作。

### train.py

​	在提取完特征后，运行train.py文件即可训练，在文件前面几行可以修改参数，如果提取的特征网络为vgg19时，train.py的参数backbone需要修改为vgg19。

### utils.py

​	里面包含了提取特征需要的各种函数。也可以输出灰度图，查看效果。

### net.py

​	里面包含了网络的定义，特征提取网络和训练网络都在该py文件有定义

### test.py

​	用该py文件测试模型结果，需要用pretreat.py预先提取test集的特征并保存，然后再运行test.py文件查看模型效果

# 参考文献
[1] A New Representation of Skeletion Sequences for 3D Action Recognition
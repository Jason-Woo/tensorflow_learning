# tensorflow_learning
####基本概念
* 使用图graphs表示计算任务
* 在被称为会话session的上下文context种执行图
* 使用tensor表示数据
* 通过变量variable维护状态
* feed和fetch为任意操作赋值或从中获取数据

tf使用graph表示计算任务，graph中的节点称为operation，一个op获得0或多个tensor执行计算，产生0或多个tensor。tensor可以看作一个n维数组。graph必须在session中被启动

![image-20200730225353840](E:\Jason\学术\Numerical Analysis\tf_learning\img\image-20200730225353840.png)
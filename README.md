# 项目来源

本项目是借鉴学习的github上的一个opencv的项目。链接：

[opencv_projects/project5 at master · wenhaoli-xmu/opencv_projects](https://github.com/wenhaoli-xmu/opencv_projects/tree/master/project5)

用到的相关网站资料链接：

[使用 Anaconda 创建 Python 虚拟环境_conda创建python虚拟环境-CSDN博客](https://blog.csdn.net/u011385476/article/details/105277426)

[PyCharm 导入 cv2_pycharm cv2-CSDN博客](https://blog.csdn.net/sazass/article/details/104021454)

[OpenCV——Canny边缘检测（cv2.Canny()）_opencv canny-CSDN博客](https://blog.csdn.net/m0_51402531/article/details/121066693)

[Python opencv膨胀函数cv2.dilate()-CSDN博客](https://blog.csdn.net/RicardoHuang/article/details/107746290)

[opencv学习—cv2.findContours()函数讲解（python）-CSDN博客](https://blog.csdn.net/weixin_44690935/article/details/109008946)

[opencv透视变换：GetPerspectiveTransform、warpPerspective函数的使用-CSDN博客](https://blog.csdn.net/jndingxin/article/details/109335687)

[【OpenCV】透视变换——cv2.getPerspectiveTransform()与cv2.warpPerspective()详解_cv2.perspectivetransform-CSDN博客](https://blog.csdn.net/AI_dataloads/article/details/133933702)

[python-opencv第四期：threshold函数详解-CSDN博客](https://blog.csdn.net/m0_55320151/article/details/127192801)

[opencv学习笔记十：使用cv2.morphologyEx()实现开运算，闭运算，礼帽与黑帽操作以及梯度运算-CSDN博客](https://blog.csdn.net/qq_39507748/article/details/104539673)

# 环境部署

用到的主要是python。然后用anaconda来配置环境。

## python和pycharm

1. 在python官网中下载：https://www.python.org/downloads/，可以选择特定版本下载，本次实验使用3.10。
2. 根据电脑操作系统下载对应版本。

![image-20250620104126033](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620104126033.png)

3. 一路next安装，记得加到环境变量中。

4. 最后打开命令行输入python出现下图表示安装成功。

![image-20250620104236485](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620104236485.png)

5. 然后下载pycharm，因为我在之前的课程中使用过pycharm，下载的是老师给的版本，当然直接在官网下载安装也是可以的，这个版本可以使用破解版。

![image-20250620104450987](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620104450987.png)

6. 不破解使用免费的也是可以的。安装很简单就不展开了。

## anaconda安装使用

Anaconda包括Conda、Python以及一大堆安装好的工具包，比如：numpy、pandas等。

因此安装Anaconda的好处主要为以下几点：

1）包含conda：conda是一个环境管理器，其功能依靠conda包来实现，该环境管理器与pip类似。

2）安装大量工具包：Anaconda会自动安装一个基本的python，该python的版本Anaconda的版本有关。

3）可以创建使用和管理多个不同的Python版本：比如想要新建一个新框架或者使用不同于Anoconda装的基本Python版本，Anoconda就可以实现同时多个python版本的管理。

1. 官网下载网速很慢，可以到镜像站下载：https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-5.2.0-Windows-x86_64.exe。
2. 因为我已经提前装好了，就不再展开了，如果有困难可以查看网站：[史上最全最详细的Anaconda安装教程-CSDN博客](https://blog.csdn.net/wq_ocean_/article/details/103889237)
3. 安装好后命令行查看：

![image-20250620105022939](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620105022939.png)

## 实验环境安装

安装好python和anaconda之后就可以开始本次实验具体内容的部署。

1. 打开anaconda。

![image-20250620105211271](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620105211271.png)

2. 可以先添加一下清华镜像源，下载更快。

```
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud//pytorch/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --set show_channel_urls yes
```

3. 输入命令

   ```
   conda create -n DigitalImageProcess python=3.10
   ```

   创建一个版本为3.10的虚拟环境。

   随后输入

   ```
   conda info -e
   ```

   可以查看到下图：

   ![image-20250620105703890](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620105703890.png)

4. 输入指令激活环境。

   ```
   conda avtivate DigitalImageProcess
   ```

5. 然后下载需要的包，numpy和opencv。输入指令：

   ```
   conda install numpy
   conda install opencv-python
   ```

6. 输入指令：

   ```
   conda list
   ```

   可以查看当前环境下的包：

   ![image-20250620110129856](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620110129856.png)

7. 最后在pycharm中使用部署的环境即可。
8. 进入pycharm中，点击右下角的解释器。

![image-20250620110308749](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620110308749.png)

9. 添加解释器，选择现有环境。

![image-20250620110510327](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620110510327.png)

10. 点击旁边的三个点，找到目录 `D:\Anaconda3\envs\DigitalImageProcess\python.exe`，前面是你自己安装的anaconda的目录，后面envs\projectname\python.exe是一样的。

![image-20250620110639782](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620110639782.png)

11. 确定即可。

# 解决过程

## Step 1 Canny边缘检测

```
edges = cv.Canny( image, threshold1, threshold2[, apertureSize[, L2gradient]])
```

阈值使用80~150，获得结果图片如下：

![image-20250620090849865](https://github.com/Xavier0624/DigitalImageProcess/blob/main/pic/image-20250620090849865.png)

然后进行膨胀操作，膨胀操作的目的在于，如果纸张的外轮廓不是很明显，Canny边缘检测后纸张外轮廓不连续有小洞，使用膨胀操作填充小洞。结果图如下：

![image-20250620091055344](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620091055344.png)

```
cv2.dilate(img, kernel, iteration)
```

kernel为碰撞核，iteration为膨胀次数。使用3x3膨胀核，膨胀1次。

## Step 2 轮廓检测与近似

```
contours, hierarchy = cv2.findContours(image, mode, method)
```

1. mode选择模式，有cv2.RETR_EXTERNAL只检测外轮廓；cv2.RETR_LIST检测的轮廓不建立等级关系；cv2.RETR_CCOMP建立两个等级的轮廓，上一层为外边界，内层为内孔的边界。如果内孔内还有连通物体，则这个物体的边界也在顶层；cv2.RETR_TREE建立一个等级树结构的轮廓。
2. method：轮廓的近似方法。cv2.CHAIN_APPROX_NOME存储所有的轮廓点，相邻的两个点的像素位置差不超过1；cv2.CHAIN_APPROX_SIMPLE压缩水平方向、垂直方向、对角线方向的元素，只保留该方向的终点坐标，例如一个矩形轮廓只需要4个点来保存轮廓信息；cv2.CHAIN_APPROX_TC89_L1，cv2.CV_CHAIN_APPROX_TC89_KCOS。

```
cv2.drawContours(image, contours, contourIdx, color, thickness=None, lineType=None, hierarchy=None, maxLevel=None, offset=None)
```

排序获得最大的轮廓并绘制在源图像上，结果如图：

![image-20250620092037691](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620092037691.png)

自适应轮廓近似，自适应轮廓近似中取`epsilon = 0.0001 * 周长`，直到结果轮廓中只有四个点，结果如图：

![image-20250620092252113](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620092252113.png)

## Step 3 透视变幻

透视变换前需要先进性预处理，把轮廓的四个点按照`左上、右上、右下、左下`的顺序进行排序，排序部分代码如下：

```
#将四个轮廓点排序
pts = np.zeros((4, 2), np.float32)

res = np.sum(points, axis=1)
pts[0] = points[np.argmin(res)]
pts[2] = points[np.argmax(res)]

res = np.diff(points, axis=1)
pts[1] = points[np.argmin(res)]
pts[3] = points[np.argmax(res)]
```

然后找到最大宽和最大高，具体代码如下：

```
#计算边长
w1 = np.sqrt((pts[0][0] - pts[1][0]) ** 2 + (pts[0][1] - pts[1][1]) ** 2)
w2 = np.sqrt((pts[2][0] - pts[3][0]) ** 2 + (pts[2][1] - pts[3][1]) ** 2)
w = int(max(w1, w2))

h1 = np.sqrt((pts[1][0] - pts[2][0]) ** 2 + (pts[1][1] - pts[2][1]) ** 2)
h2 = np.sqrt((pts[0][0] - pts[3][0]) ** 2 + (pts[0][1] - pts[3][1]) ** 2)
h = int(max(h1, h2))
```

进行完所有预处理之后，就可以开始我们最后也是最重要的一步——透视变换了，具体的代码如下：

```
#目标四个点
dst = np.array([
    [0, 0],
    [w - 1, 0],
    [w - 1, h - 1],
    [0, h - 1]
], np.float32)

#透视变换
mat = cv2.getPerspectiveTransform(pts, dst)
paper1 = org1.copy()
paper1 = cv2.warpPerspective(paper1, mat, (w, h))
if show_process:
    imshow(paper1)
```

```
retval = cv2.getPerspectiveTransform(src, dst)
dst = warpPerspective(src, M, dsize[, flags[, borderMode[, borderValue]]])
```

第一个函数获得变换矩阵，第二个函数应用该矩阵到目标图像。

得到结果如下图：

![image-20250620093746864](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620093746864.png)

## Step 4 预处理

首先为了消除不同图片曝光程度不同的影响，需要先对图片进行*自适应直方图均衡化*。

```
cv2.createCLAHE([, clipLimit[, tileGridSize]]) ->retval
dst=retval.apply(src)
```

结果如下：

![image-20250620094129978](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620094129978.png)

 然后进行自适应二值化：

```
thresh,result=cv2.threshold (src, thresh, maxval, type)
```

结果如下图：

![image-20250620094352496](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620094352496.png)

但进行完二值化的图片还有一个问题，就是在涂答题卡的时候，如果没有涂的饱满， 就可能会造成检测结果不准确，所以为了使检测结果更加准确，还需要进行*闭运算*操作：

```
cv2.morphologyEx(img, op, kernel)
```

运算核5x5，采用闭操作cv2.MORPH_CLOSE，先膨胀再腐蚀。

运行结果如下：

![image-20250620094638058](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620094638058.png)

## Step 5 轮廓检测与过滤

上面提到过这里用到的函数，运行结果如下图：

![image-20250620094759300](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620094759300.png)

可以看到提取到了很多轮廓，其中很多都是我们不需要的轮廓，于是我们需要使用一些过滤算法，把我们需要的轮廓（25个椭圆）保留下来 这里的过滤算法步骤如下所示：

- 首先获得待检测轮廓的外接图形，如果是圆，则获得轮廓的外接圆。
- 然后可以按照面积过滤，当 轮廓面积 / 外接图形面积 的比值`ratio`满足：`ratio > 0.8 and ratio < 1.2`时符合要求。
- 然后可以按照周长过滤，当 轮廓周长 / 外接图形周长 的比值`ratio`满足：`ratio > 0.8 and ratio < 1.2`时符合要求。

结果如下：

![image-20250620095343522](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620095343522.png)

在此之后我们就得到了所有比较像椭圆的轮廓，但是这还不够，因为有一些用于装订的椭圆也被保留了下来，可以观察到 这些用于装订的椭圆的特征是他们的面积比答题的椭圆要小得多，于是我们对所有轮廓进行排序，`key = 轮廓的面积` 然后将面积比较小的通过特定算法过滤掉。结果如下：

![image-20250620095523334](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620095523334.png)

## Step 6 排序+检测

然后需要按照从上到下，从左到右的顺序对轮廓进行排序，本程序在排序的同时完成检测，具体的代码如下：

```
#对多个轮廓按照从上到下的顺序排序
cnts = sorted(cnts, key=lambda x: x[0][0][1])

rows = int(len(cnts) / 5)

TAB = ['A', 'B', 'C', 'D', 'E']
ANS = []

#检查每一行（即每一题）的答案
for i in range(rows):
    subcnts = cnts[i*5:(i+1)*5]
    subcnts = sorted(subcnts, key=lambda x: x[0][0][0])

    total = []

    for (j, cnt) in enumerate(subcnts):
        mask = np.zeros(paper1.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1) #-1表示填充

        mask = cv2.bitwise_and(paper1, paper1, mask=mask)
        total.append(cv2.countNonZero(mask))

    idx = np.argmax(np.array(total))
    ANS.append(TAB[idx])

print(ANS)
```

算法大致过程为先按照轮廓的最高点排序，然后默认每行5个选项得到每一道题，按行检测，按最左边的点的横坐标排序，然后创建一个该轮廓的掩码，按位与得到该轮廓内的灰度值的和并存到total数组，最后灰度值最高的为该题的选项。

运行结果如下图：

![image-20250620100010658](C:\Users\32471\AppData\Roaming\Typora\typora-user-images\image-20250620100010658.png)

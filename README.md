#test.py 测试
#train_rfcn.py  训练rfcn

#依赖
pytorch
tensorboardX


#数据集准备
1.groundtruth   真值图
2.prior map  本次实验中，先用fcn跑出来一次结果，把这些显著图结果当先验图
3.原图
其中真值图和原图一般数据集中都有，先验图需要自己准备
4. 改代码里的数据集路径。ptag改成先验图文件夹名字。（需要修改的有train里面的和dataset里面的
5.本文中使用的数据集 ：链接：https://pan.baidu.com/s/1GtYoWkzqspzLf7WTBdm9iw 密码：gjxb


#需要tensorboard
tensorboard --logdir runs

4. 运行
python train_rfcn.py

参考论文 Wang, Linzhao, et al. "Saliency detection with recurrent fully convolutional networks." European Conference on Computer Vision. Springer International Publishing, 2016.
# recentlywork-rfcn
# recentlywork-rfcn
# recentlywork-rfcn

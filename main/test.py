import random

import numpy as np
import paddle


def create_grid_indices_array(num_rows):
    rows = paddle.arange(num_rows).reshape((-1, 1))
    return rows
print(create_grid_indices_array(5))
#
#
# def soft_max(x):
#     prob=np.exp(x-np.max(x))
#     prob/=np.sum(prob)
#     return prob
#
# box=[[[10,20],
#       [30,40],
#       [50,60],
#       [70,80]],
#      [[10,20],
#       [250,40],
#       [50,60],
#       [70,80]]]
# index=[[1.65616,0.848],[3.15448,1.8154]]
# anchor=[[20,20],[40,30],[60,60]]
# def wh_iou(wh1, wh2):
#     x1, y1 = wh1[...,0],wh1[...,1]
#     s1 = x1 * y1
#
#     x2, y2 = wh2[...,0], wh2[...,1]
#     s2 = x2 * y2
#
#     inter_w = paddle.minimum(x1, x2)
#     inter_h = paddle.minimum(y1, y2)
#     intersection = inter_h * inter_w
#
#     union = s1 + s2 - intersection
#     iou = intersection / union
#     return iou
#
#
# box=paddle.to_tensor(box)
# x,y=box[...,0],box[...,1]
# x[[0],[0]]=0
# print(box[...,0])
# # print(y)
#
#
# # box=paddle.to_tensor(box)
# # index=paddle.to_tensor(index)
# # index = paddle.floor(index)
# # i,j=index[...,0].cast(dtype='int64'),index[...,1].cast(dtype='int64')
# # box[0,i,j]=0
# #
# # print(i)
# # print(j)
# # print(box)
#
# # anchor=paddle.to_tensor(anchor)
# # print(anchor[0][0])
# # rs=paddle.stack([wh_iou(an,box) for an in anchor])
# # best_iou,best_n=rs.max(axis=0),rs.argmax(axis=0)
# # test=rs.transpose([1,2,0])
# # data=paddle.to_tensor([[3,2,0],
# #                        [4,5,4],
# #                        [6,5,2]])
# # ret=paddle.where(data > 4)
# # test[0,ret[0][0],0]=0
# # print(ret[0][0])
# # x,y=box[...,0],box[...,1]
# # x[[0],[0]]=0
# # print(x[0,ret[0][0]])
# # print(y)
# # print(test)
# # print(best_iou)
# # print(best_n)
#
# # boo=paddle.zeros(shape=[2,2],dtype='bool')
# # boo[1,0]=1
# # print(boo)

x=random.randint(0,0)
print(x)
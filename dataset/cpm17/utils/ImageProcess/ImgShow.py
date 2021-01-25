import matplotlib.pyplot as plt

'''
显示多张图片
'''


def showLineImg(imgList,figsize=(10,6)):
    length = len(imgList)
    fig,ax = plt.subplots(1,length,figsize = figsize)
    for i in range(length):
       ax[i].imshow(imgList[i])





#encoding:UTF-8

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

path = os.getcwd() + '/test'
print path

def extendBoard(img, axis, b, count):
    #获取边界值
    if axis:
        board = img[:, b].reshape((-1, 1, 3))
    else:
        board = img[b].reshape((1, -1, 3))
    #计算新边界
    if axis:
        ave = np.average(board, axis=0)
    else:
        ave = np.average(board, axis=1)
    board[:, :] = ave
    newboard = np.concatenate([board] * count, axis=axis)
    #
    if not b:
        newimg = np.concatenate([newboard, img], axis=axis)
    else:
        newimg = np.concatenate([img, newboard], axis=axis)
    return newimg

def getExtendedImg(img, ext):
    newimg = extendBoard(img, 0, 0, ext)
    newimg = extendBoard(newimg, 0, -1, ext)
    newimg = extendBoard(newimg, 1, 0, ext)
    newimg = extendBoard(newimg, 1, -1, ext)
    return newimg

def createMoveImages(path, move):
    for root, dirs, files in os.walk(path):
        for f in files:
            img = cv2.imread(root + '/' + f)
            m, n, _ = img.shape
            #extend
            newimg = extendBoard(img, 0, 0, move)
            newimg = extendBoard(newimg, 0, -1, move)
            newimg = extendBoard(newimg, 1, 0, move)
            newimg = extendBoard(newimg, 1, -1, move)
            for i in range(move):
                for j in range(move):
                    result = newimg[i:i+m, j:j+n]
                    cv2.imwrite(root+'/'+f[:-4]+'_m%d%d.png' %(i, j), result)
                    

def genColorImg(img, delta, thresh=230):
    th = img < thresh
    a = np.zeros(img.shape, dtype='uint8')
    a[th] = delta
    return img + a

def createColorImages(path, color):
    for root, dirs, files in os.walk(path):
        for f in files:
            img = cv2.imread(root + '/' + f)
            for i in range(-color, color):
                newimg = genColorImg(img, i)
                cv2.imwrite(root+'/'+f[:-4]+'_c%d.png' %(i), newimg)


def randomMoveListGen(length, maxthresh):
    done = False
    while not done:
        tmp = np.random.choice([-1, 0, 1], [length])
        a = 0
        ret = []
        for i in tmp:
            a += i
            ret.append(a)
        ret = np.array(ret)
        ret[ret>maxthresh] = maxthresh
        ret[ret<-maxthresh] = -maxthresh
        cnt = len(ret[ret==maxthresh]) + len(ret[ret==-maxthresh])
        if cnt < length/3:
            done = True
    return ret

def getTransformImg(img, maxthresh):
    m, n, _ = img.shape
    retimg = np.zeros(img.shape, dtype='uint8')
    newimg = getExtendedImg(img, maxthresh)
    rows = randomMoveListGen(m+2*maxthresh, maxthresh) + maxthresh
    cols = randomMoveListGen(n+2*maxthresh, maxthresh) + maxthresh
    for i, j in enumerate(rows[maxthresh:maxthresh+m]):
        retimg[i, :] = newimg[i, j:j+n]
    for i, j in enumerate(cols[maxthresh:maxthresh+n]):
        retimg[:, i] = newimg[j:j+m, i]
    return retimg

def createTransformImages(path, count, maxthresh=5):
    for root, dirs, files in os.walk(path):
        for f in files:
            img = cv2.imread(root + '/' + f)
            for i in range(count):
                newimg = getTransformImg(img, maxthresh)
                cv2.imwrite(root+'/'+f[:-4]+'_t%d.png' %(i), newimg)


if __name__ == '__main__':
    createMoveImages(path, 3)
    createColorImages(path, 10)
    createTransformImages(path, 100)
    print 'done'    
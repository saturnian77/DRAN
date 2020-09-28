import tensorflow as tf
import model
import cv2
import numpy as np
import math
import imageio
from glob import glob

def imread(path):
    img = imageio.imread(path).astype(np.float64)
    return img/255.

def rgb2y(x):
    if x.dtype==np.uint8:
        x=np.float64(x)
        y=65.481/255.*x[:,:,0]+128.553/255.*x[:,:,1]+24.966/255.*x[:,:,2]+16
        y=np.round(y).astype(np.uint8)
    else:
        y = 65.481 / 255. * x[:, :, 0] + 128.553 / 255. * x[:, :, 1] + 24.966 / 255. * x[:, :, 2] + 16 /255.
    return y

def psnr_(img1, img2):
    img1 = np.float64(img1)
    img2 = np.float64(img2)

    mse = np.mean((img1-img2)**2)
    if mse==0:
        return 'error'
    if np.max(img1)<=1.0:
        MAX_VAL = 1.0
    else:
        MAX_VAL = 255.0
    return 20.0*math.log10(MAX_VAL/math.sqrt(mse))

def imgcut(x, xN):
    h,w,c= x.shape
    x = x[xN:h-xN,xN:w-xN,:]
    return x

def count_param():
    param = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
    print("Model Parameter: %02.2f M" % (param/1000000.0))


def imagewrite(img, iter, scale, dataset):
    img = img.astype(np.uint8)
    img = img[0, :, :, :]
    imageio.imsave('./Results/' + dataset + '_' + scale + '_' + ('%03d' % (iter+1)) + '.png', img)

def load_testimg(dataset, xN):
    lab_path = './psnrtest/' + dataset + '/highres/' + xN + '/*.png'
    data_path = './psnrtest/' + dataset +'/lowres/' + xN + '/*.png'
    img_list =  np.sort(np.asarray(glob(data_path)))
    lab_list = np.sort(np.asarray(glob(lab_path)))
    k = len(img_list)
    imgs = {}
    labs = {}
    for i in range(k):
        img = imread(img_list[i])
        lab = imread(lab_list[i])

        if len(img.shape)<3:
            img = np.expand_dims(img, axis=2)
            lab = np.expand_dims(lab, axis=2)
            img = np.concatenate((np.concatenate((img,img), axis=2),img), axis=2)
            lab = np.concatenate((np.concatenate((lab,lab), axis=2),lab), axis=2)
        imgs[i] = img
        labs[i] = lab

    return imgs, labs, k

class Build(object):
    def __init__(self, ckpt_path, scale, dataset):
        self.savefolder = ckpt_path
        self.scale = scale
        self.dataset = dataset
        self.conf = tf.ConfigProto()
        self.input = tf.placeholder(tf.float32, [None, None, None, 3])
        self.MODEL = model.DRAN(self.input, scale)


    def test(self):
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        with tf.Session(config = self.conf) as sess:

            sess.run(self.init)

            print("Searching Checkpoint...")
            ckpt = tf.train.get_checkpoint_state(self.savefolder)
            if ckpt:
                ckpt_list = ckpt.all_model_checkpoint_paths
                self.saver.restore(sess, ckpt_list[0])
                print("Checkpoint Restored")
                count_param()

                ######## DATASET IMAGES COUNT
                timg, labs, dataset_len = load_testimg(self.dataset, self.scale)  # BSDS100 Set5 Set14 Urban100
                avg_psnr = 0.0

                for i in range(dataset_len):
                    test_img = timg[i]
                    tlab = labs[i]
                    output_ = sess.run([self.MODEL.output],feed_dict={self.input: test_img[np.newaxis,:,:,:]})
                    output_ = np.round(255 * np.clip(output_[0], 0.0, 1.0))
                    imagewrite(output_, i, self.scale, self.dataset)
                    output_ = output_[0, :, :, :]/255.0
                    if self.scale == 'x2':
                        cutedge = 2
                    elif self.scale == 'x3':
                        cutedge = 3
                    elif self.scale == 'x4':
                        cutedge = 4
                    output_ = imgcut(output_, cutedge)
                    tlab = imgcut(tlab, cutedge)
                    psnr_mean_ = psnr_(rgb2y(output_),rgb2y(tlab))
                    avg_psnr = avg_psnr + psnr_mean_

                print("Average PSNR: %02.2f dB" % (avg_psnr/(dataset_len*1.0)))
            else:
                print("Checkpoint does not exist")


    
    




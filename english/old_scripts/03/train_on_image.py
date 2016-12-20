import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from images2gif import writeGif
from scipy.stats import truncnorm

class Network:

    def __init__(self, width, height, c_count, iterations=-1, batch_size=128, img2=None, img1=None):
        self.img2 = img2
        self.img1 = img1
        self.w = width
        self.h = height
        self.c_count = c_count
        self.iter = iterations
        self.batch_size = batch_size

        self.X  = tf.placeholder("float", [None, 1])
        self.Y  = tf.placeholder("float", [None, 1])
        self.Z  = tf.placeholder("float", [None, 1])
        self.c_ = tf.placeholder("float", [None, c_count])

        self.sess = tf.Session()

    def initNet(self, numNR=32, numLR=3, reuse=False):
        self.Gen, self.Cost, self.Train  = self.createNet(self.X, self.Y, self.Z, self.c_, self.w, self.h,
                self.c_count, numNR, numLR, reuse)

        tf.initialize_all_variables().run(session=self.sess)

    @staticmethod
    def initConnection(input_, outSize, scope="FC", with_bias=True, stddev=0.5):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(scope):
            matrix = tf.get_variable("Matrix", [shape[1], outSize], tf.float32,
                    tf.truncated_normal_initializer(mean=0, stddev=stddev))

            result = tf.matmul(input_, matrix)

            if with_bias:
                bias = tf.get_variable("Bias", [1, outSize],
                    initializer=tf.constant_initializer(0.0))
                result += bias #* tf.ones([shape[0], 1], dtype=tf.float32)

            return result

    @staticmethod
    def createNet(X, Y, Z, c_, w, h, c_count, numNR, numLR, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        I = Network.initConnection(X, numNR, "X_in", False) + \
            Network.initConnection(Y, numNR, "Y_in", False) + \
            Network.initConnection(Z, numNR, "Z_in", True)


        # tanh activations
        H = tf.nn.tanh(I)
        for i in range(numLR):
            H = tf.nn.tanh( Network.initConnection(H, numNR, "H_"+str(i)))

        # relu activations
        """
        H = tf.nn.relu(I)
        for i in range(numLR):
            H = tf.nn.relu(Network.initConnection(H, numNR, "H_"+str(i)))
        """

        output = tf.nn.sigmoid(Network.initConnection(H, c_count, "C_out"))

        cost = tf.nn.l2_loss( c_ - output)

        # train_op = tf.train.RMSPropOptimizer(0.002, 0.9).minimize(cost)
        train_op = tf.train.AdamOptimizer(0.001, 0.9, 0.8, 1e-08).minimize(cost)
        return output, cost, train_op

    @staticmethod
    def _coordinates(w, h, scale=1.0):
        x_range = scale*np.linspace(-0.5,0.5, w)
        y_range = scale*np.linspace(-0.5,0.5, h)

        x_mat = np.matmul( np.ones((h,1)), x_range.reshape((1,w)) )
        y_mat = np.matmul( y_range.reshape(h,1), np.ones((1,w)))
        r_mat = np.sqrt( x_mat*x_mat + y_mat*y_mat )

        x_mat = x_mat.reshape(w,h, 1)
        y_mat = y_mat.reshape(w,h, 1)
        r_mat = r_mat.reshape(w,h, 1)

        return x_mat, y_mat, r_mat


    def render_gif(self, w, h, steps, save=True, interval=800 ):
        imgs = []
        x, y, r = self._coordinates(w, h)

        for i in range(-steps/2, steps/2):
            img = self.sess.run(self.Gen, feed_dict={self.X: x.reshape(w*h, 1), \
                        self.Y:y.reshape(w*h,1), self.Z:np.ones((w*h, 1))*(i*2.0/steps)})
            img = img.reshape(h, w, self.c_count)

            if(self.c_count==1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            imgs.append(img)

        if save:
            writeGif("animation.gif", imgs, duration=interval/1000.0)
        return imgs

    def showAni(self, interval=1, w=100, h=100):
        x,y,r = self._coordinates(w, h)
        for i in range(-50, 50):
            img = self.sess.run( self.Gen, feed_dict={self.X: x.reshape(w*h, 1), \
                net.Y: y.reshape(w*h,1), net.Z: np.ones((w*h,1))*(i/50.0)})
            cv2.imshow("frame", img.reshape(h,w,self.c_count))
            if cv2.waitKey(interval) != -1:
                break;

    def startLearning(self, img0=None, img05=None, img1=None, live=False):
        #if img3==None:
        #    #img2 = self.img2
        #    #img1 = self.img1
        #    if img3==None:
        #        raise ValueError("Need image to train with must be (w x h x c) %d x %d x %d"%(self.w, self.h, self.c_count))

        img = np.array([img0, img05, img1])

        batch_size = self.batch_size

        x, y, r = self._coordinates(self.w, self.h, scale=1.0)
        i = self.iter

        w, h = self.w, self.h

        histo = []

        cv2.imshow("orig0", img[0]) #cv2.resize(img[0], (w/2, h/2)))
        cv2.imshow("orig05", img[1])
        cv2.imshow("orig1", img[2]) #cv2.resize(, (w/2, h/2)))


        cv2.waitKey(1)
        print x.shape, y.shape, r.shape
        print img.shape

        a,b = (0-1)/1, (2-1)/1

        while i!=0:
            i -= 1

            for j in range(batch_size):
                x_cor = np.random.randint(0, self.w-1, size=(batch_size))
                y_cor = np.random.randint(0, self.h-1, size=(batch_size))
                z_cor = truncnorm.rvs(a,b, loc=1, scale=1, size=(batch_size))

                x_op = ((x_cor-w/2.0)/w).reshape(batch_size, 1)
                y_op = ((y_cor-h/2.0)/h).reshape(batch_size, 1)
                z_op = (z_cor-1).reshape(batch_size,1)

                self.sess.run(self.Train, feed_dict={self.X: x_op,
                    self.Y: y_op , self.Z: z_op,
                    self.c_: img[np.round(z_cor).astype(int), y_cor, x_cor].reshape(batch_size,self.c_count)})
                '''else:
                    self.sess.run(self.Train, feed_dict={self.X: x_op, \
                        self.Y: y_op , self.Z: np.zeros_like(y_op), self.c_: img1[y_cor, x_cor]}) '''

            if live:
                ret = self.sess.run(self.Gen, feed_dict={self.X: x.reshape(w*h,1),
                    self.Y: y.reshape(w*h,1), self.Z: np.ones((w*h,1)) })
                cv2.imshow("current", cv2.resize(ret.reshape(self.h,self.w, self.c_count), (w/3, h/3)))

                ret = self.sess.run(self.Gen, feed_dict={self.X: x.reshape(w*h,1),
                    self.Y: y.reshape(w*h,1), self.Z: np.ones((w*h,1))*0 })
                cv2.imshow("current0.5", cv2.resize(ret.reshape(self.h,self.w, self.c_count), (w/3, h/3)))

                ret = self.sess.run(self.Gen, feed_dict={self.X: x.reshape(w*h,1),
                    self.Y: y.reshape(w*h,1), self.Z: np.ones((w*h,1))*-1 })
                cv2.imshow("current1", cv2.resize(ret.reshape(self.h,self.w, self.c_count), (w/3, h/3)))

                if cv2.waitKey(1) != -1:
                    i = 0

                cost_val0 = self.sess.run(self.Cost, feed_dict={self.X: x.reshape(w*h,1),
                    self.Y: y.reshape(w*h,1), self.Z: np.ones((w*h,1))*0,
                    self.c_: img[0].reshape(w*h,self.c_count)})
                cost_val1 = self.sess.run(self.Cost, feed_dict={self.X: x.reshape(w*h,1),
                    self.Y: y.reshape(w*h,1), self.Z: np.ones((w*h,1)),
                    self.c_: img[1].reshape(w*h,self.c_count)})
                histo.append( [cost_val0, cost_val1] )
                plt.clf()
                plt.plot(histo)
                plt.pause(0.001)
                print i, cost_val0, cost_val1


# Gray
"""
img = cv2.imread("flower_100.png", flags=cv2.IMREAD_GRAYSCALE)

net = Network(img.shape[1], img.shape[0], 1, batch_size=128)
img = (img/255.0).astype("float32")
"""
img = cv2.imread("img1_edit.jpg", flags=cv2.IMREAD_GRAYSCALE)
img0 = cv2.imread("img2_edit.jpg", flags=cv2.IMREAD_GRAYSCALE)
img1 = cv2.imread("img12_blend.jpg", flags=cv2.IMREAD_GRAYSCALE)

img = np.round( cv2.resize(img, (200, 200)) )
img0 = np.round( cv2.resize(img0, (200, 200)) )
img1 = np.round( cv2.resize(img1, (200, 200)) )

net = Network(img.shape[1], img.shape[0], 1, batch_size=500)
img = (img/255.0).astype("float32")
img0 = (img0/255.0).astype("float32")
img1 = (img1/255.0).astype("float32")

net.initNet(numNR=64, numLR=10)
net.startLearning(img0=img, img05=img1, img1=img0, live=True)

# saver = Saver()
# saver.save(net.sess, "HR_NN_save.ckpt")

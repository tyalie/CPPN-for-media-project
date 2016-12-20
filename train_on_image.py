import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Network:

    def __init__(self, width, height, c_count, iterations=-1, batch_size=128, img=None):
        self.img = img
        self.w = width
        self.h = height
        self.c_count = c_count
        self.iter = iterations
        self.batch_size = batch_size

        self.X  = tf.placeholder("float", [None, 1])
        self.Y  = tf.placeholder("float", [None, 1])
        self.R  = tf.placeholder("float", [None, 1])
        self.c_ = tf.placeholder("float", [None, c_count])

        self.sess = tf.Session()

    def initNet(self, numNR=32, numLR=3, reuse=False):
        self.Gen, self.Cost, self.Train  = self.createNet(self.X, self.Y, self.R, self.c_, self.w, self.h, \
                self.c_count, numNR, numLR, reuse)

        tf.initialize_all_variables().run(session=self.sess)

    @staticmethod
    def initConnection(input_, outSize, scope="FC", with_bias=True, stddev=0.5):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(scope):
            matrix = tf.get_variable("Matrix", [shape[1], outSize], tf.float32, \
                    tf.random_normal_initializer(stddev=stddev))

            result = tf.matmul(input_, matrix)

            if with_bias:
                bias = tf.get_variable("Bias", [1, outSize], \
                    initializer=tf.constant_initializer(0.0))
                result += bias #* tf.ones([shape[0], 1], dtype=tf.float32)

            return result

    @staticmethod
    def createNet(X, Y, R, c_, w, h, c_count, numNR, numLR, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        I = Network.initConnection(X, numNR, "X_in", False) + \
            Network.initConnection(Y, numNR, "Y_in", False)
            #Network.initConnection(R, numNR, "R_in", False)

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

        cost = tf.nn.l2_loss(c_ - output)

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

    def startLearning(self, img=None, live=False):
        if img==None:
            img = self.img
            if img==None:
                raise ValueError("Need image to train with must be (w x h x c) %d x %d x %d"%(self.w, self.h, self.c_count))

        batch_size = self.batch_size

        x, y, r = self._coordinates(self.w, self.h, scale=1.0)
        i = self.iter

        w, h = self.w, self.h

        histo = []

        cv2.imshow("orig", cv2.resize(img, (w/2, h/2)))
        cv2.waitKey(1)
        print x.shape, y.shape, r.shape
        print img.shape

        while i!=0:
            i -= 1

            for j in range(batch_size):
                x_cor = np.random.randint(0, self.w-1, size=(batch_size))
                y_cor = np.random.randint(0, self.h-1, size=(batch_size))


                x_op = ((x_cor-w/2.0)/w).reshape(batch_size, 1)
                y_op = ((y_cor-h/2.0)/h).reshape(batch_size, 1)

                self.sess.run(self.Train, feed_dict={self.X: x_op, \
                    self.Y: y_op, self.R: np.sqrt(x_op*x_op + y_op*y_op) ,self.c_: img[y_cor, x_cor]})

            if live:
                ret = self.sess.run(self.Gen, feed_dict={self.X: x.reshape(w*h,1), \
                    self.Y: y.reshape(w*h,1), self.R: r.reshape(w*h,1)})
                cv2.imshow("current", cv2.resize(ret.reshape(self.h,self.w, self.c_count), (w/3, h/3)))
                if cv2.waitKey(1) != -1:
                    i = 0

                cost_val = self.sess.run(self.Cost, feed_dict={self.X: x.reshape(w*h,1), \
                    self.Y: y.reshape(w*h,1), self.R: r.reshape(w*h,1), self.c_: img.reshape(w*h,self.c_count)})
                histo.append(cost_val)
                plt.clf()
                plt.plot(histo)
                plt.pause(0.001)
                print i, cost_val



# Gray
"""
img = cv2.imread("flower_100.png", flags=cv2.IMREAD_GRAYSCALE)

net = Network(img.shape[1], img.shape[0], 1, batch_size=128)
img = (img/255.0).astype("float32")
"""
img = cv2.imread("flower.png")
net = Network(img.shape[1], img.shape[0], 3, batch_size=500)
img = (img/255.0).astype("float32")


net.initNet(numNR=64, numLR=10)
net.startLearning(img=img, live=True)

# saver = Saver()
# saver.save(net.sess, "HR_NN_save.ckpt")

import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import matplotlib.animation as animation
from images2gif import writeGif

# Note: What if I first train the z=0.5 position with the average of the two images.
#  Than I use that network to specify myself further on the edges



class Net:
    def __init__(self, c_count, batch_size, width, height, iter):
        self.c_count = c_count
        self.batch_size = batch_size
        self.w = width
        self.h = height

        self.X = tf.placeholder("float", [None, 1])
        self.Y = tf.placeholder("float", [None, 1])
        self.Z = tf.placeholder("float", [None, 1])
        self.C = tf.placeholder("float", [None, c_count])
        self.iter = iter

        self.sess = tf.Session()

    # The network consists of 3 inputs
    #  X, Y, Z
    #
    # With that my image function is now:
    #    f(x,y,z) = r,g,b
    # Z is for clarity a float with |Z|<=0.9
    # -0.9 is one image, 0.9 is the other extreme
    #
    # The network includes also multiple layers:
    # The structure can be defined with two variables:
    #   num_n = number of neurons p.L.
    #   num_l = number of layers
    # The activation function of all neurons will be a
    #    ReLU at the beginngin
    #
    # The outputs are 3 values:
    #   r, g, b in range 0.0-1.0
    # The last layer so needs to be a sigmoid activation

    def netConstructor(self, num_l=4, num_n=16, reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        I = self.fullConnection(self.X, num_n, "I_X", False) + \
            self.fullConnection(self.Y, num_n, "I_Y", False) + \
            self.fullConnection(self.Z, num_n, "I_Z", False)


        '''
        H = tf.nn.relu(I)
        for i in range(num_l):
            H = tf.nn.relu( self.fullConnection(H, num_l, "H_%d"%(i+1), True) )
        '''
        H = tf.nn.tanh(I)
        for i in range(num_l):
            H = tf.nn.tanh( self.fullConnection(H, num_l, "H_%d"%(i+1), True) )


        O = tf.nn.sigmoid( self.fullConnection(H, self.c_count, "O_%d"%(self.c_count)) )

        #loss = tf.reduce_mean( tf.square(O-self.C) )
        loss = tf.square( O - self.C )
        #loss = tf.nn.l2_loss(O-self.C)
        #train_op = tf.train.AdamOptimizer(0.001, 0.9, 0.8, 1e-08).minimize(loss)
        train_op = tf.train.AdamOptimizer(0.001, 0.8, 0.8).minimize(loss) # ONLY TEST
        #train_op = tf.train.RMSPropOptimizer(0.01, 0.9, 0.1).minimize(loss)

        return O, loss, train_op

    def fullConnection(self, input_, outSize, scope="FC", with_bias=True, stddev=0.2):
        shape = input_.get_shape().as_list()
        with tf.variable_scope(scope):
            matrix = tf.get_variable("Matrix", [shape[1], outSize], tf.float32, \
                tf.truncated_normal_initializer(mean=0, stddev=stddev))
            result = tf.matmul( input_, matrix)

            if with_bias:
                bias = tf.get_variable("Bias", [1, outSize], \
                    initializer=tf.constant_initializer(0.0))
                result += bias
            return result;


    def _coordinates(self, w, h, scale=1.0):
        x_range = scale*np.linspace(-0.5,0.5, w)
        y_range = scale*np.linspace(-0.5,0.5, h)

        x_mat = np.matmul( np.ones((h,1)), x_range.reshape((1,w)) )
        y_mat = np.matmul( y_range.reshape(h,1), np.ones((1,w)))
        # r_mat = np.sqrt( x_mat*x_mat + y_mat*y_mat )

        x_mat = x_mat.reshape(w,h, 1)
        y_mat = y_mat.reshape(w,h, 1)
        # r_mat = r_mat.reshape(w,h, 1)

        return x_mat, y_mat #, r_mat

    def initNet(self, num_l=4, num_n=16):
        self.net, self.cost, self.train = self.netConstructor(num_l, num_n)
        tf.initialize_all_variables().run(session=self.sess)



    def startLearning(self, img1, img2, live):
        x,y = self._coordinates(self.w, self.h)
        x1,y1 = self._coordinates(100, 100)
        img = np.array([img1,img2])
        dim = self.w*self.h

        i = self.iter
        histo = []

        while i!=0:
            i-=1

            if(self.batch_size > 0):
                for j in range(10):
                    x_cor = np.random.randint( 0, self.w-1, size=(self.batch_size))
                    y_cor = np.random.randint( 0, self.h-1, size=(self.batch_size))

                    x_op = ((x_cor-self.w/2.0)/self.w).reshape(self.batch_size, 1)
                    y_op = ((y_cor-self.h/2.0)/self.h).reshape(self.batch_size, 1)
                    z_op = np.random.randint(0, 2, (self.batch_size, 1))

                    self.sess.run(self.train, feed_dict={self.X: x_op, \
                        self.Y: y_op, self.Z: z_op ,self.C: \
                        img[z_op.reshape(self.batch_size), y_cor, x_cor].reshape(self.batch_size, self.c_count)})
            else:
                #self.sess.run(self.train, feed_dict={self.X: x.reshape(dim, 1), \
                #            self.Y:y.reshape(dim,1), self.Z:np.ones((dim, 1)), \
                #            self.C: img[1].reshape(dim, self.c_count)})
                #self.sess.run(self.train, feed_dict={self.X: x.reshape(dim, 1), \
                #            self.Y:y.reshape(dim,1), self.Z:np.zeros((dim, 1)), \
                #            self.C: img[0].reshape(dim, self.c_count)})
                if( i%2==0 ):
                    x_op = np.repeat( x , 2,0).reshape(dim*2,1)
                    y_op = np.repeat( y.reshape(1,dim) , 2,0).reshape(dim*2, 1)
                    z_op = np.ones((dim*2, 1))
                    z_op[0:dim, 0] = 0


                    self.sess.run(self.train, feed_dict={self.X: x_op,  \
                                self.Y: y_op, self.Z: z_op, \
                                self.C: img.reshape(dim*2, self.c_count)})
                else:
                    self.sess.run(self.train, feed_dict={self.X: x.reshape(dim, 1), \
                                self.Y:y.reshape(dim,1), self.Z:np.ones((dim, 1)), \
                                self.C: img[1].reshape(dim, self.c_count)})

            if (live and i%2==0):
                ret0 = self.sess.run(self.net, feed_dict={self.X: x1.reshape(-1, 1), \
                            self.Y:y1.reshape(-1,1), self.Z:np.zeros((100**2, 1))})
                ret1 = self.sess.run(self.net, feed_dict={self.X: x1.reshape(-1, 1), \
                            self.Y:y1.reshape(-1,1), self.Z:np.ones((100**2, 1))})
                ret05 = self.sess.run(self.net, feed_dict={self.X: x1.reshape(-1, 1), \
                            self.Y:y1.reshape(-1,1), self.Z:np.ones((100**2, 1))*0.5})
                ret = np.zeros( (100,100*3, self.c_count) )
                ret[:,0:100,:] = ret0.reshape(100,100,self.c_count)
                ret[:,100:100*2,:] = ret05.reshape(100,100,self.c_count)
                ret[:,100*2:100*3,:] = ret1.reshape(100,100,self.c_count)

                cv2.imshow("current0", (ret))
                # cv2.imshow("current0", ret0.reshape(self.h,self.w, self.c_count))
                # cv2.imshow("current1", ret1.reshape(self.h,self.w, self.c_count))
                # cv2.imshow("current0.5", ret05.reshape(self.h,self.w, self.c_count))


                if cv2.waitKey(1) != -1:
                    i = 0

                cost_val0 = self.sess.run(self.cost, feed_dict={self.X: \
                        x.reshape(-1, 1), self.Y: y.reshape(-1,1), \
                        self.Z: np.zeros((dim, 1)), self.C: img[0].reshape(dim, self.c_count)})
                cost_val1 = self.sess.run(self.cost, feed_dict={self.X: \
                        x.reshape(dim, 1), self.Y: y.reshape(dim,1), \
                        self.Z: np.ones((dim, 1)), self.C: img[1].reshape(dim, self.c_count)})
                histo.append([ cost_val0.mean(), cost_val1.mean() ])
                plt.clf()
                plt.plot(histo)
                plt.pause(0.001)
                print i, cost_val0.mean(), cost_val1.mean()


    def render_gif(self, w, h, steps, save=True, interval=800 ):
        imgs = []
        x, y = self._coordinates(w, h)

        for i in range(0, steps):
            img = self.sess.run(self.net, feed_dict={self.X: x.reshape(w*h, 1), \
                        self.Y:y.reshape(w*h,1), self.Z:np.ones((w*h, 1))*(i*1.0/steps)})
            img = img.reshape(h, w, self.c_count)

            if(self.c_count==1):
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            imgs.append(img)

        if save:
            writeGif("animation.gif", imgs, duration=interval/1000.0)
        return imgs

    def showAni(self, interval=1, w=100, h=100):
        x,y = self._coordinates(w, h)
        for i in range(100):
            img = self.sess.run( self.net, feed_dict={self.X: x.reshape(w*h, 1), \
                net.Y: y.reshape(w*h,1), net.Z: np.ones((w*h,1))*(i/100.0)})
            cv2.imshow("frame", img.reshape(h,w,self.c_count))
            if cv2.waitKey(interval) != -1:
                break;


img1 = cv2.imread("img1_edit.jpg", cv2.IMREAD_GRAYSCALE)
img1 = (img1/255.0).astype("float32")
img1 = cv2.resize(img1, (200, 200))


img2 = cv2.imread("img2_edit.jpg", cv2.IMREAD_GRAYSCALE)
img2 = (img2/255.0).astype("float32")
img2 = cv2.resize(img2, (200, 200))


net = Net(1, 500, width=200, height=200, iter=1000)
net.initNet(5, 128)
net.startLearning(img2, img1, True)
# plt.savefig('graph.png')

# saver = tf.train.Saver()
# saver.save(net.sess, "HR_NN_save.ckpt")

# net.render_gif(200, 200, 100, save=True, interval=100 )

import tensorflow as tf
import numpy as np
import cv2, math

def _coordinates(w=512, h=512, scale=8, batch_size=1):
    x_range = scale*np.linspace(-1,1, w)
    y_range = scale*np.linspace(-1,1, h)

    x_mat = np.matmul( np.ones((h,1)), x_range.reshape((1,w)) )
    y_mat = np.matmul( y_range.reshape(h,1), np.ones((1,w)))
    r_mat = np.sqrt( x_mat*x_mat + y_mat*y_mat )

    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, w*h, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, w*h, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, w*h, 1)

    return x_mat, y_mat, r_mat

def initConnection( _input,output_size, scope="FC", with_bias=True, stddev=1.0 ):
    shape = _input.get_shape().as_list()
    with tf.variable_scope(scope):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,\
            tf.random_normal_initializer(stddev=stddev))

        result = tf.matmul(_input, matrix)

        if with_bias:
            bias = tf.get_variable("Bias", [1, output_size], \
                initializer=tf.random_normal_initializer(stddev=stddev))
            result += bias * tf.ones([shape[0], 1], dtype=tf.float32)

        return result

def initNet(X,Y,Z,r, w=512, h=512, netsize=32, Z_dim=1, scale=8, reuse=False, c_mode=1):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    Z_scaled = tf.reshape(Z, [1, Z_dim]) * \
        tf.ones([w*h, 1], dtype=tf.float32) * scale

    Xm = tf.reshape(X, [w*h, 1])
    Ym = tf.reshape(Y, [w*h, 1])
    rm = tf.reshape(r, [w*h, 1])
    Zm = tf.reshape(Z_scaled, [w*h, Z_dim])

    U = initConnection(Xm, netsize, "X_in", False) + \
        initConnection(Ym, netsize, "Y_in", False) + \
        initConnection(rm, netsize, "r_in", False) + \
        initConnection(Zm, netsize, "Z_in", True)

    H = tf.nn.tanh(U)
    for i in range(3):
        H = tf.nn.tanh(initConnection(H, netsize, "hid%d"%(i)))
    output = tf.nn.sigmoid(initConnection(H, c_mode, "out"))

    result = tf.reshape(output, [h, w, c_mode])
    return result

def test(w=500, h=1000, Z_dim=8, scale=10.0, reuse=False):
    X = tf.placeholder("float", [1, None, 1])
    Y = tf.placeholder("float", [1, None, 1])
    r = tf.placeholder("float", [1, None, 1])
    Z = tf.placeholder("float", [1, None, Z_dim])

    x_mat, y_mat, r_mat = _coordinates(w, h, scale=scale)
    z_mat = np.random.uniform(-1.0, 1.0, size=(1, Z_dim)).astype(np.float32).reshape(1, 1, Z_dim)
    G = initNet(X, Y, Z, r, w, h, 32, Z_dim, scale=scale, reuse=reuse)
    sess = tf.Session()
    tf.initialize_all_variables().run(session=sess)
    image = sess.run( G, feed_dict={X: x_mat, Y: y_mat, r: r_mat, Z: z_mat})

    #cv2.imwrite("out.png", (image*255).astype("uint8"))
    return image

def ani(w,h, scale, c_mode=1, reuse=False, scl=4):
    Z_dim = 1

    X = tf.placeholder("float", [1, None, 1])
    Y = tf.placeholder("float", [1, None, 1])
    r = tf.placeholder("float", [1, None, 1])
    Z = tf.placeholder("float", [1, None, Z_dim])

    z_mat = np.ones([1,1,1])*-1
    x_mat, y_mat, r_mat = _coordinates(w, h, scale=scale)
    G = initNet(X, Y, Z, r, w, h, 32, Z_dim, scale=scale, reuse=reuse, c_mode=c_mode)
    sess = tf.Session()
    tf.initialize_all_variables().run(session=sess)

    i= 0
    p = 180.0
    while True:
        z_mat[0,0,0] = math.sin(math.pi/p*i)
        #z_mat[0,0,0] = abs( ((i/p)%2)-1 )*2-1
        i = (i+1)%(p*2)
        img = sess.run( G, feed_dict={X: x_mat, Y: y_mat, r: r_mat, Z: z_mat})
        cv2.imshow("ani", cv2.resize(img, (w*scl, h*scl), interpolation=cv2.INTER_CUBIC))
        if cv2.waitKey(1) != -1:
            break

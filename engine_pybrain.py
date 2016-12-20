from pybrain.structure import FeedForwardNetwork, LinearLayer, TanhLayer, SigmoidLayer, FullConnection
from pybrain.datasets import UnsupervisedDataSet
import numpy as np
import cv2

RGB_COLOR = 0x03
GRAY_COLOR = 0x01

def initNet(network_structure=[5,5,5], c_mode=GRAY_COLOR ):
    n = FeedForwardNetwork()
    inLayer      = LinearLayer(4, name="In_w4")
    n.addInputModule( inLayer )

    last = inLayer
    for i in range(len(network_structure)):
        hiddenLayer = TanhLayer(network_structure[i], name="HiddenLayer%d_w%d"%(i+1, network_structure[i]) )
        n.addModule( hiddenLayer )
        n.addConnection( FullConnection(last, hiddenLayer, name="FullConnection_%d"%(i+1)))

        last = hiddenLayer

    out_layer    = SigmoidLayer(c_mode, name="Out_w%d"%(c_mode))
    n.addOutputModule( out_layer )
    n.addConnection( FullConnection(last, out_layer, name="FullConnection_OUT") )

    n.sortModules()
    return n

def createImgCordinates(w=500, h=500, scale=1, batch_size = 1):
    x_range = scale*np.linspace(-1,1, w)*scale
    y_range = scale*np.linspace(-1,1, h)*scale

    x_mat = np.matmul( np.ones((h,1)), x_range.reshape((1,w)) )
    y_mat = np.matmul( y_range.reshape(h,1), np.ones((1,w)))
    r_mat = np.sqrt( x_mat*x_mat + y_mat*y_mat )

    x_mat = np.tile(x_mat.flatten(), batch_size).reshape(batch_size, w*h, 1)
    y_mat = np.tile(y_mat.flatten(), batch_size).reshape(batch_size, w*h, 1)
    r_mat = np.tile(r_mat.flatten(), batch_size).reshape(batch_size, w*h, 1)

    return x_mat, y_mat, r_mat

def activateNetwork(network, x_mat, y_mat, r_mat, size, z=[1], batch_size = 1):
    w = size[0]
    h = size[1]
    if len(z) != batch_size:
        raise ValueError("Length of Z array must be equal to the batch_size")
    z_mat = np.matmul( z, np.ones((1, w*h)) ).reshape(batch_size, w*h, 1)

    for i in range(batch_size):
        ds = UnsupervisedDataSet(4)
        for m in zip(z_mat[i], x_mat[i], y_mat[i], r_mat[i]):
            ds.addSample( m )

        ret = network.activateOnDataset(ds)
        ret = ret.reshape( h, w)
        cv2.imwrite("out%d.png"%(i), (ret*0xFF).astype("uint8"))




def createImg(w=512, h=512, scale=1, z=1, c_mode=GRAY_COLOR, network_structure=[5,5,5]):
    x_mat, y_mat, r_mat = createImgCordinates(w=w,h=h, scale=scale)
    n = initNet(network_structure=network_structure, c_mode=c_mode)
    activateNetwork(n, x_mat, y_mat, r_mat, (w,h))

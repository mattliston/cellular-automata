# CUDA_VISIBLE_DEVICES='0' python train.py
import argparse
import numpy as np ; print 'numpy ' + np.__version__
import tensorflow as tf ; print 'tensorflow ' + tf.__version__
import cv2 ; print 'cv2 ' + cv2.__version__
np.set_printoptions(threshold='nan')
np.set_printoptions(linewidth=250)
np.set_printoptions(formatter={'float': '{:12.8f}'.format, 'int': '{:4d}'.format})

# parse command line arguments
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--rule',help='cellular rule number',default='110')
parser.add_argument('--width', help='ring size', default=256, type=int)
parser.add_argument('--model',help='tensorflow graph file',default='model.proto')
parser.add_argument('--lr', help='learning rate', default=0.001, type=float)
parser.add_argument('--batch', help='batch size', default=1000, type=int)
parser.add_argument('--epochs', help='training epochs', default=1000000, type=int)
parser.add_argument('--n', help='synthetic training batches per epoch', default=10, type=int)
parser.add_argument('--height', help='number of steps to simulate forward and backward', default=256, type=int)
parser.add_argument('--scale', help='scale factor for display', default=1, type=int)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

def rule110(x0):
    xm = x0
    xl = np.roll(x0,1)
    xr = np.roll(x0,-1)
    x1 = np.bitwise_and(np.bitwise_and(xl,xm),np.bitwise_not(xr))
    x1 = np.bitwise_or(x1,np.bitwise_and(np.bitwise_and(xl,xr),np.bitwise_not(xm)))
    x1 = np.bitwise_or(x1,np.bitwise_and(np.bitwise_and(xm,xr),np.bitwise_not(xl)))
    x1 = np.bitwise_or(x1,np.bitwise_and(np.bitwise_and(np.bitwise_not(xl),np.bitwise_not(xr)),xm))
    x1 = np.bitwise_or(x1,np.bitwise_and(np.bitwise_and(np.bitwise_not(xl),np.bitwise_not(xm)),xr))
    return x1

def genbatch(args):
    x = np.random.binomial(1,0.5,size=[args.batch,args.width])
    if args.rule=='110':
        return x, rule110(x)

x = tf.placeholder('float32', [None,args.width],name='x') ; print x
y = tf.placeholder('float32', [None,args.width],name='y') ; print y

n = tf.layers.conv1d(inputs=tf.expand_dims(x,-1),filters=32,kernel_size=7,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=32,kernel_size=7,padding='same',dilation_rate=2,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=32,kernel_size=7,padding='same',dilation_rate=4,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=32,kernel_size=7,padding='same',dilation_rate=8,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=32,kernel_size=7,padding='same',dilation_rate=4,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=32,kernel_size=7,padding='same',dilation_rate=2,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=32,kernel_size=7,padding='same',dilation_rate=1,activation=tf.nn.elu) ; print n
n = tf.layers.conv1d(inputs=n,filters=1,kernel_size=7,padding='same',dilation_rate=1,activation=None) ; print n

pred = tf.sigmoid(n)
loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.expand_dims(y,-1),logits=n)
opt = tf.train.AdamOptimizer(learning_rate=args.lr)
grads = opt.compute_gradients(loss)
train = opt.apply_gradients(grads)
norm = tf.global_norm([i[0] for i in grads])
init = tf.global_variables_initializer()

# draw blank visualization window and move it to a nice location on the screen
fimg = np.zeros([args.height,args.width],dtype=np.uint8)
bimg = np.zeros([args.height,args.width],dtype=np.uint8)
cv2.imshow('vis',cv2.resize(np.concatenate([fimg,bimg],axis=1),dsize=(0,0),fx=args.scale,fy=args.scale,interpolation=cv2.INTER_LANCZOS4))
cv2.moveWindow('ca', 0,0)
cv2.waitKey(10)

with tf.Session() as sess:
    sess.run(init)
    for i in range(args.epochs):
        # TRAIN
        larr=[] # losses
        narr=[] # gradients
        for j in range(args.n):
            y_,x_ = genbatch(args)
            _,l_,n_ = sess.run([train,loss,norm],feed_dict={x:x_,y:y_})
            larr.append(l_)
            narr.append(n_)

        # TEST
        y_,x_ = genbatch(args)
        p = np.squeeze(sess.run(pred,feed_dict={x:x_}))
        s = np.random.binomial(1,p)

        print 'epoch {:6d} loss {:12.8f} grad {:12.4f} accuracy {:12.8f}'.format(i,np.mean(larr),np.mean(narr),1-np.mean(np.abs(s-y_)))
        #print p[0,0:10]

        # VISUALIZE FORWARD AND BACKWARD
        x_ = np.random.binomial(1,0.5,size=args.width)
        for j in range(args.height):
            fimg[j] = x_*255
            x_ = rule110(x_)
        for j in range(args.height-1,0,-1):
            bimg[j] = x_*255
            p = np.squeeze(sess.run(pred,feed_dict={x:[x_]}))
            x_ = np.random.binomial(1,p)
        cv2.imshow('vis',cv2.resize(np.concatenate([fimg,bimg],axis=1),dsize=(0,0),fx=args.scale,fy=args.scale,interpolation=cv2.INTER_LANCZOS4))
        cv2.waitKey(10)

import argparse
import numpy as np ; print 'numpy ' + np.__version__
import cv2 ; print 'cv2 ' + cv2.__version__

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--init', help='initialization (left,middle,right,random)', default='random')
parser.add_argument('--width', help='ring size', default=512, type=int)
parser.add_argument('--height', help='steps to display', default=512, type=int)
parser.add_argument('--steps', help='steps to simulate', default=1000000, type=int)
parser.add_argument('--scale', help='scale factor for display', default=1, type=int)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

x = np.zeros(args.width,dtype=np.int)
if args.init=='left':
    x[0] = 1
if args.init=='middle':
    x[args.width/2] = 1
if args.init=='right':
    x[-1] = 1
if args.init=='random':
    x = np.random.binomial(1,0.5,size=x.shape)

xl = np.empty(args.width,dtype=np.int)
xm = np.empty(args.width,dtype=np.int)
xr = np.empty(args.width,dtype=np.int)

img = np.zeros([args.height,args.width],dtype=np.uint8)
cv2.imshow('ca',cv2.resize(img,dsize=(0,0),fx=args.scale,fy=args.scale,interpolation=cv2.INTER_LANCZOS4))
cv2.moveWindow('ca', 0,0)
cv2.waitKey(10)

for i in range(args.steps):
    #print x
    xm = x
    xl = np.roll(x,1)
    xr = np.roll(x,-1)

    y = np.bitwise_and(np.bitwise_and(xl,xm),np.bitwise_not(xr))
    y = np.bitwise_or(y,np.bitwise_and(np.bitwise_and(xl,xr),np.bitwise_not(xm)))
    y = np.bitwise_or(y,np.bitwise_and(np.bitwise_and(xm,xr),np.bitwise_not(xl)))
    y = np.bitwise_or(y,np.bitwise_and(np.bitwise_and(np.bitwise_not(xl),np.bitwise_not(xr)),xm))
    y = np.bitwise_or(y,np.bitwise_and(np.bitwise_and(np.bitwise_not(xl),np.bitwise_not(xm)),xr))

    x = y
    img[-1] = x*255
    cv2.imshow('ca',cv2.resize(img,dsize=(0,0),fx=args.scale,fy=args.scale,interpolation=cv2.INTER_LANCZOS4))
    cv2.waitKey(10)
    img = np.roll(img,-1,axis=0)


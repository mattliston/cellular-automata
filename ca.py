import argparse
import numpy as np ; print 'numpy ' + np.__version__

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--width', help='ring size', default=30, type=int)
parser.add_argument('--steps', help='steps to simulate', default=30, type=int)
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print args

x = np.zeros(args.width,dtype=np.int)
x[-1] = 1

xl = np.empty(args.width,dtype=np.int)
xm = np.empty(args.width,dtype=np.int)
xr = np.empty(args.width,dtype=np.int)

for i in range(args.steps):
    print x
    xm = x
    xl = np.roll(x,1)
    xr = np.roll(x,-1)

    y = np.bitwise_and(np.bitwise_and(xl,xm),np.bitwise_not(xr))
    y = np.bitwise_or(y,np.bitwise_and(np.bitwise_and(xl,xr),np.bitwise_not(xm)))
    y = np.bitwise_or(y,np.bitwise_and(np.bitwise_and(xm,xr),np.bitwise_not(xl)))
    y = np.bitwise_or(y,np.bitwise_and(np.bitwise_and(np.bitwise_not(xl),np.bitwise_not(xr)),xm))
    y = np.bitwise_or(y,np.bitwise_and(np.bitwise_and(np.bitwise_not(xl),np.bitwise_not(xm)),xr))

    x = y


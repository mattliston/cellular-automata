# python life.py --search --size 200 --init patch --sigma 0.5 --w 20 --bounded
# python life.py --vis --size 200 --scale 8 --pickle search_3460.pickle

import argparse
import numpy as np ; print 'numpy ' + np.__version__
import cv2 ; print 'cv2 ' + cv2.__version__
import pickle

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--pickle', help='initial seed pickle file',default='')
parser.add_argument('--size', help='size of life world', default=100, type=int)
parser.add_argument('--init', help='initialization method {uniform,linear,ring,center,fence}',default='uniform')
parser.add_argument('--sigma', help='init parameter', default=0.5, type=float)
parser.add_argument('--w', help='init parameter', default=20, type=int)
parser.add_argument('--scale', help='visualization scale factor', default=10, type=int)
parser.add_argument('--vis', default=False, action='store_true')
parser.add_argument('--search', default=False, action='store_true')
parser.add_argument('--evolve', default=False, action='store_true')
parser.add_argument('--bounded', default=False, action='store_true')
parser.add_argument('--debug', default=False, action='store_true')
args = parser.parse_args()
print(args)

def visualize(w,scale,r,g,b):
    img=np.zeros([args.size,args.size,3],dtype=np.uint8)
    img[:,:,0] = w*b
    img[:,:,1] = w*g
    img[:,:,2] = w*r
    #img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_NEAREST)
    img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
    #img = cv2.resize(img,None,fx=scale, fy=scale, interpolation = cv2.INTER_LANCZOS4)
    cv2.imshow('img',img)

def step(w):
    w0 = np.roll(w,1,axis=0) ; w0[0,:]=0 # down y
    w1 = np.roll(w,-1,axis=0) ; w1[-1,:]=0 # up y
    w2 = np.roll(w,1,axis=1) ; w2[:,-1]=0 # right x
    w3 = np.roll(w,-1,axis=1) ; w3[:,0]=0 # left x
    w4 = np.roll(w,1,axis=0) ; w4[0,:]=0 ; w4 = np.roll(w4,1,axis=1) ; w4[:,-1]=0
    w5 = np.roll(w,-1,axis=0) ; w5[-1,:]=0 ; w5 = np.roll(w5,1,axis=1) ; w5[:,-1]=0
    w6 = np.roll(w,-1,axis=0) ; w6[-1,:]=0 ; w6 = np.roll(w6,-1,axis=1) ; w6[:,0]=0
    w7 = np.roll(w,1,axis=0) ; w7[0,:]=0 ; w7 = np.roll(w7,-1,axis=1) ; w7[:,0]=0

    s = w0+w1+w2+w3+w4+w5+w6+w7
    w = np.logical_or(np.logical_and(w, s==2), s==3)
    w = w.astype(np.int)
    return w

def run(w,steps,usehash=False):
    if usehash:
        h={}
        for i in range(steps):
            if args.bounded and (np.sum(w[0:2,:])+np.sum(w[:,0:2])+np.sum(w[-2:,:])+np.sum(w[:,-2:]))>0:
                return 0
            k=hash(w.tostring())
            if k in h:
                return i
            h[k]=1
            w = step(w)
        return i
    else:
        w1=np.zeros_like(w)
        w2=np.zeros_like(w)
        for i in range(steps):
            w = step(w)
            if args.bounded and (np.sum(w[0:2,:])+np.sum(w[:,0:2])+np.sum(w[-2:,:])+np.sum(w[:,-2:]))>0:
                return 0
            if np.array_equal(w,w1) or np.array_equal(w,w2):
                return i
            w2=w1
            w1=w
        return i

def init(args,p):
    w = np.random.binomial(1,args.sigma,size=[args.size,args.size])
    if args.init=='uniform':
        pass
    if args.init=='patch':
        w[0:int(args.size*0.5-args.w*0.5),:]=0
        w[:,0:int(args.size*0.5-args.w*0.5)]=0
        w[int(args.size*0.5+args.w*0.5):,:]=0
        w[:,int(args.size*0.5+args.w*0.5):]=0
    if args.init=='linear':
        w[0:int(args.size*0.5)-1,:]=0
        w[int(args.size*0.5)-1+3,:]=0
    if args.init=='ring':
        w[1:-1,1:-1]=0
    if args.init=='center':
        w[0,:]=0
        w[:,0]=0
        w[-1,:]=0
        w[:,-1]=0
    if args.init=='fence':
        m = np.zeros([args.size,args.size],dtype=np.int)
        m[0:6,:]=1
        m[-6:,:]=1
        m[:,0:6]=1
        m[:,-6:]=1
        s = np.tile([[1,1,0],[1,1,0],[0,0,0]],[int(args.size/3+1),int(args.size/3+1)])[0:args.size,0:args.size]
        np.logical_and(w,np.logical_not(m),out=w)
        np.logical_or(w,np.logical_and(s,m),out=w)
    if args.init=='eater1':
        m = np.zeros([args.size,args.size],dtype=np.int)
        z=5 # square size of eater cell
        m[0:z,:]=1
        m[-z:,:]=1
        m[:,0:z]=1
        m[:,-z:]=1
        s = np.tile([[1,1,0,0,0],[1,0,1,0,0],[0,0,1,0,0],[0,0,1,1,0],[0,0,0,0,0]],[int(args.size/z+1),int(args.size/z+1)])[0:args.size,0:args.size]
        #np.logical_and(w,np.logical_not(m),out=w)
        #np.logical_or(w,np.logical_and(s,m),out=w)
        w=s
    return w

if args.search:
    best=0
    e=1
    while True:
        w = init(args,args.sigma)
        act = run(w,10000)
        if act==9999:
            act = run(w,1000000,True) # use hash to check for long period blickers
        if act>best:
            best=act
            pickle.dump(w, open('search_'+str(best)+'.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)

        e+=1
        print 'search {:6d} act {:12d} best {:12d}'.format(e,act,best)
 
if args.evolve:
    w = pickle.load(open(args.pickle, 'rb'))
    curr = run(w,1000000)
    e=1
    while True:
        wtry = w.copy()
        m = init(args,args.sigma)
        wtry = np.logical_xor(wtry,m).astype(np.int)
        act = run(wtry,1000000)
        if act >= curr: # keep neutral mutations
            curr = act
            w = wtry
            pickle.dump(w, open('evolve_'+str(curr)+'.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
        e+=1
        print 'evolve {:6d} act {:12d} curr {:12d}'.format(e,act,curr)

if args.vis:
    w = np.zeros([args.size,args.size],dtype=np.int)
    w0 = pickle.load(open(args.pickle, 'rb'))
    o = int(w.shape[0]*0.5-w0.shape[0]*0.5)
    w[o:o+w0.shape[0],o:o+w0.shape[1]]=w0
    visualize(w,args.scale,255,255,255)
    cv2.waitKey(0)
    e=1
    h={}
    while True:
        w = step(w)
        k=hash(w.tostring())
        if k in h:
            r=255 ; g=0 ; b=0 ;
        else:
            r=255 ; g=255 ; b=255 ;
        h[k]=1
        visualize(w,args.scale,r,g,b)
        if cv2.waitKey(10) > 0:
            break
        e+=1
        if e%1000==0:
            print 'e',e

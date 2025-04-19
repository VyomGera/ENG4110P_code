#%%
from __future__ import division
import numpy as np
import time
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from matplotlib import colors
import matplotlib.pyplot as plt
nelx= 50
nely= 50
volfrac=0.5
rmin= 2
penal= 5
ft = 0 # ft==0 -> sens, ft==1 -> dens
imp = 2 # imp==0 -> bulk modulus, imp==1 -> shear modulus
#imp==2 -> negative poisson's ratio
#Element stiffness matrix
def lk():
	E=1
	nu=0.3
	k=np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
	KE = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
[k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
[k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
[k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
[k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
[k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
[k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
[k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]] ])
	return (KE)
# Optimality criterion
def oc(nelx,nely,xflat,volfrac,dc,dv,g):
	l1=0
	l2=1e9
	if imp==0 or imp==1:
		move=0.2
	elif imp==2:
		move = 0.1
# reshape to perform vector operations
	xnew=np.zeros(nelx*nely)
	while (l2-l1)>1e-9:
		lmid=0.5*(l2+l1)
		if imp==0 or imp==1:
			xnew[:]= np.maximum(0.0,np.maximum(xflat-move,np.minimum(1.0,np.minimum(xflat+move,xflat*np.sqrt(-dc/dv/lmid)))))
		elif imp==2:
			xnew[:] = np.maximum(0.0,np.maximum(xflat-move, np.minimum(1,np.minimum(xflat+move,xflat*(-dc/dv/lmid)))))
		gt=g+np.sum((dv*(xnew-xflat)))
		if gt>0 :
			l1=lmid
		else:
			l2=lmid
	xnew = np.reshape(xnew, (nelx,nely), order='F')
	return (xnew,gt)
#%%
print(["Bulk Modulus problem", "Shear Modulus problem", "Negative Poisson's ratio problem"][imp])
print("ndes: " + str(nelx) + " x " + str(nely))
print("volfrac: " + str(volfrac) + ", rmin: " + str(rmin) + ", penal: " + str(penal))
print("Filter method: " + ["Sensitivity based","Density based"][ft])
# Max and min stiffness
Emin=1e-9
Emax=1.0
# dofs:
ndof = 2*(nelx+1)*(nely+1)
# Allocate design variables (as array), initialize and allocate sens.
g=0 # must be initialized to use the NGuyen/Paulino OC approach
dc=np.zeros((nely,nelx), dtype=float)
# FE: Build the index vectors for the for coo matrix format.
KE=lk()
edofMat=np.zeros((nelx*nely,8),dtype=int)
for elx in range(nelx):
	for ely in range(nely):
		el = ely+elx*nely
		n1=(nely+1)*elx+ely
		n2=(nely+1)*(elx+1)+ely
		edofMat[el,:]=np.array([2*n1+2, 2*n1+3, 2*n2+2, 2*n2+3,2*n2, 2*n2+1, 2*n1, 2*n1+1])
# Construct the index pointers for the coo format
iK = np.kron(edofMat,np.ones((8,1))).flatten()
jK = np.kron(edofMat,np.ones((1,8))).flatten()    
# Filter: Build (and assemble) the index+data vectors for the coo matrix format
nfilter=int(nelx*nely*((2*(np.ceil(rmin)-1)+1)**2))
iH = np.zeros(nfilter)
jH = np.zeros(nfilter)
sH = np.zeros(nfilter)
cc=0
for i in range(nelx):
	for j in range(nely):
		row=i*nely+j
		kk1=int(np.maximum(i-(np.ceil(rmin)-1),0))
		kk2=int(np.minimum(i+np.ceil(rmin),nelx))
		ll1=int(np.maximum(j-(np.ceil(rmin)-1),0))
		ll2=int(np.minimum(j+np.ceil(rmin),nely))
		for k in range(kk1,kk2):
			for l in range(ll1,ll2):
				col=k*nely+l
				fac=rmin-np.sqrt(((i-k)*(i-k)+(j-l)*(j-l)))
				iH[cc]=row
				jH[cc]=col
				sH[cc]=np.maximum(0.0,fac)
				cc=cc+1
# Finalize assembly and convert to csc format
H=coo_matrix((sH,(iH,jH)),shape=(nelx*nely,nelx*nely)).tocsc()	
Hs=H.sum(1)
#%%
# Periodic BC's 
dofs=np.arange(1, 2*(nelx+1)*(nely+1)+1)
fixed=np.union1d(dofs[0:2*(nely+1):2],np.array([2*(nelx+1)*(nely+1)-1]))
free=np.setdiff1d(dofs,fixed)
U = np.zeros(((2*(nely+1)*(nelx+1)),3)) 
nodenrs = np.arange(1, (nelx + 1) * (nely + 1)+1).reshape((nely + 1, nelx + 1), order= 'F')
n1 = np.array((nodenrs[-1, 0:(nelx+1):(nelx)],np.flip(nodenrs[0, 0:(nelx+1):(nelx)]))).flatten()
d1py = []
for i in range (0,np.size(n1)):
	n11 = (2*n1[i])-1
	d1py.append(n11)
	n12 = (2*n1[i])
	d1py.append(n12)
d1 = np.array(d1py).flatten('F')
n3 = np.hstack((nodenrs[1:nely, 0].flatten(), nodenrs[-1, 1:nelx].flatten()))
d3py = []
for i in range (0,np.size(n3)):
	n31 = (2*n3[i])-1
	d3py.append(n31)
	n32 = (2*n3[i])
	d3py.append(n32)
d3 = np.array(d3py).flatten('F')
n4 = np.hstack([nodenrs[1:nely, -1:].flatten() , nodenrs[0, 1:nelx].flatten()])
d4py = []
for i in range (0,np.size(n4)):
	n41 = (2*n4[i])-1
	d4py.append(n41)
	n42 = (2*n4[i])
	d4py.append(n42)
d4 = np.array(d4py).flatten('F')
d21 = np.setdiff1d(dofs, np.hstack((d1, d3, d4)))
d2 = np.array(d21).flatten('F')
#%%
e0 = np.eye(3)
ufixed = np.zeros((8,3)) 
for j in range(3):  
	ufixed[2:4, j] = np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]]) @ np.array([nelx, 0]) 
	ufixed[6:8, j] = np.array([[e0[0, j], e0[2, j] / 2], [e0[2, j] / 2, e0[1, j]]]) @ np.array([0, nely])
	ufixed[4:6, j] = ufixed[2:4, j] + ufixed[6:8, j]
wfixed = np.vstack((np.tile(ufixed[2:4, :], (nely - 1, 1)), np.tile(ufixed[6:8, :], (nelx - 1, 1)))) 
qe = np.ndarray(shape=(3, 3, nelx, nely)) 
Q = np.zeros((3, 3))
dQ = np.ndarray(shape=(3, 3, nelx, nely))
x = np.tile(volfrac, (nelx, nely))
for i in range (0, (nelx-1)):
	for j in range (0, (nely-1)):
		# for a different initial guess, and therefore different results
		# change np.minimum(nelx/3,nely/3) to np.minimum(nelx/6, nely/6)
		#  and try different values for more variation
		if np.sqrt((i+1-(nelx/2)-0.5)**2+(j+1-(nely/2)-0.5)**2) < np.minimum(nelx/6, nely/6):
			x[i][j] = volfrac/2
xPhys = x.copy()
xold = x.copy()
#%%
# Initialize plot and plot the initial design
plt.ion() # Ensure that redrawing is possible
fig,ax = plt.subplots()
im = ax.imshow(-xPhys.reshape((nelx,nely)).T, cmap='gray',\
interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
fig.show()
plt.show()
# Set loop counter and gradient vectors 
loop=0
change=1
test = KE.flatten('F')[np.newaxis].T
test1 = xPhys.flatten('F')[np.newaxis]
test2 = Emin+(xPhys.flatten('F')[np.newaxis])**penal*(Emax-Emin)
dv = np.ones(nely*nelx)
ce = np.ones(nely*nelx)
#%%
while change>0.01 and loop<2000:
	loop=loop+1
	# Setup and solve FE problem
	sK=((KE.flatten('F')[np.newaxis]).T*(Emin+(xPhys.flatten('F')[np.newaxis])**penal*(Emax-Emin))).flatten(order='F')
	K = coo_matrix((sK,(iK,jK)),shape=(ndof,ndof)).tocsc()
	#Reduced K matrix, Kr
	K1 = K[np.ix_(d2-1, d2-1)].toarray() #Output was in csc format, back to array to allow for concatentation
	K2 = K[np.ix_(d2-1,d3-1)]+K[np.ix_(d2-1,d4-1)].toarray()
	K3 = K[np.ix_(d3-1,d2-1)]+K[np.ix_(d4-1,d2-1)].toarray()
	K4 = K[np.ix_(d3-1,d3-1)]+K[np.ix_(d4-1,d3-1)]+K[np.ix_(d3-1,d4-1)]+K[np.ix_(d4-1,d4-1)].toarray()
	Kr1 = np.concatenate((K1,K2),axis=1)
	Kr2 = np.concatenate((K3,K4),axis=1)
	Kr = np.concatenate((Kr1,Kr2),axis=0)
	#Constrcuting displacement matrix split into U1, U2, U3 and U4
	#where U4 = U3 + W
	U[d1-1,:] = ufixed
	mat1 = np.concatenate((K[np.ix_(d2-1,d1-1)].toarray(),(K[np.ix_(d3-1,d1-1)]+K[np.ix_(d4-1,d1-1)].toarray())),axis=0)
	mat2 = np.concatenate((K[np.ix_(d2-1,d4-1)].toarray(),(K[np.ix_(d3-1,d4-1)]+K[np.ix_(d4-1,d4-1)].toarray())),axis=0)
	mat3 = -np.dot(mat1,ufixed)-np.dot(mat2,wfixed)
	U[np.hstack((d2,d3))-1,:] = np.linalg.solve(Kr, mat3) 
	U[d4-1,:] = U[(d3-1),:]+wfixed
	# Objective and sensitivity
	for i in range (0,3):
		for j in range (0,3):
			U1 = U[:,i]
			U2 = U[:,j]		
			qe[i][j] = np.reshape(np.sum(np.multiply((U1[edofMat]@KE),U2[edofMat]),axis = 1),(nelx,nely), order='F')/(nelx*nely)
			Q[i][j] = np.sum(np.sum(Emin+xPhys**(penal*(Emax-Emin))*qe[i,j]))
			dQ[i][j] = (penal*(Emax-Emin)*(xPhys**(penal-1)))*qe[i,j]
	if imp==0:
		obj= -(Q[0][0]+Q[1][1]+Q[0][1]+Q[1][0]) #denoted as c in report
		dc= -(dQ[0][0]+dQ[1][1]+dQ[0][1]+dQ[1][0]) 
	elif imp==1:
		obj = -Q[2][2]
		dc = -dQ[2][2]
	elif imp==2:
		obj = Q[0][1]-(1**loop)*(Q[0][0]+Q[1][1])
		dc = dQ[0][1]-(1**loop)*(dQ[0][0]+dQ[1][1])
	dv = np.ones(nely*nelx)
	dc = dc.flatten('F')
	dv = dv.flatten('F')
	xflat = x.flatten('F')
	# Sensitivity filtering:
	if ft==0:
		dc[:] = np.asarray((H*(xflat*dc))[np.newaxis].T/Hs)[:,0] / np.maximum(0.001,xflat)
	#density filtering
	elif ft==1:
		dc[:] = np.asarray(H*(np.reshape(dc,(-1,1),order='F')/Hs))[:,0]
		dv[:] = np.asarray(H*(np.reshape(dv,(-1,1),order='F')/Hs))[:,0]
	# Optimality criteria
	xold[:]=x
	(x[:],g)=oc(nelx,nely,xflat,volfrac,dc,dv,g)
	# Filter design variables
	if ft==0:   
		xPhys[:]=x
	elif ft==1:	
		xPhys1 =np.asarray(H*(np.reshape(x,(-1,1),order='F'))/Hs)[:,0]
		xPhys = np.reshape(xPhys1, (nelx,nely), order='F')
	# Compute the change by the inf. norm
	change=np.linalg.norm(x.flatten('F')-xflat,np.inf)
	# Plot to screen
	im.set_array(-xPhys.T)
	fig.canvas.draw()
	# Write iteration history to screen (req. Python 2.6 or newer)
	print("it.: {0} , obj.: {1:.3f} Vol.: {2:.3f}, ch.: {3:.3f}".format(\
				loop,obj,(g+volfrac*nelx*nely)/(nelx*nely),change)) #change the way Vol is displayed
	if loop%2 == 0: #without a small pause the plot cannot refresh at the same rate as the interation
		plt.show()
		plt.draw()
		plt.pause(0.05)
plt.show()
plt.draw()
print("done")
# Make sure the plot stays and that the shell remains, keep it visible for sleep(x) period of time
# where x is the seconds the plot stays visible. for using VSC and other packages 
#which close all figures upon completion
time.sleep(60) 
plt.close()
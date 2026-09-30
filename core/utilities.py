import torch

from config import *

def computeJacobian(F):
	"""
	Compute Jacobian from deformation gradient.

	_Input Arguments_

	- `F` - deformation gradient in Voigt notation

	_Output Arguments_

	- `J` - Jacobian

	---

	"""
	F11 = F[:,0:1]
	F12 = F[:,1:2]
	F21 = F[:,2:3]
	F22 = F[:,3:4]
	F13 = F[:,4:5]
	F23 = F[:,5:6]
	F31 = F[:,6:7]
	F32 = F[:,7:8]
	F33 = F[:,8:9]

	J = -F13*F22*F31 + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33
	return J

def computeCauchyGreenStrain(F):
	"""
	Compute right Cauchy-Green strain tensor from deformation gradient.

	_Input Arguments_

	- `F` - deformation gradient in Voigt notation

	_Output Arguments_

	- `C` - Cauchy-Green strain tensor in Voigt notation

	---

	"""
	F11 = F[:,0:1]
	F12 = F[:,1:2]
	F21 = F[:,2:3]
	F22 = F[:,3:4]
	F13 = F[:,4:5]
	F23 = F[:,5:6]
	F31 = F[:,6:7]
	F32 = F[:,7:8]
	F33 = F[:,8:9]

	C11 = F11**2 + F21**2 + F31**2
	C12 = F11*F12 + F21*F22 + F31*F32
	C13 = F11*F13 + F21*F23 + F31*F33
	C21 = F11*F12 + F21*F22 + F31*F32
	C22 = F12**2 + F22**2 + F32**2
	C23 = F12*F13 + F22*F23 + F32*F33
	C31 = F11*F13 + F21*F23 + F31*F33
	C32 = F12*F13 + F22*F23 + F32*F33
	C33 = F13**2 + F23**2 + F33**2

	C = torch.cat((C11,C12,C21,C22,C13,C23,C31,C32,C33),dim=1)
	return C


def computeStrainInvariants(C):
	"""
	Compute invariants of the Cauchy-Green strain tensor.
	Plane strain is assumed.

	_Input Arguments_

	- `C` - Cauchy-Green strain tensor in Voigt notation

	_Output Arguments_

	- `I1` - 1st invariant

	- `I2` - 2nd invariant

	- `I3` - 3rd invariant

	---

	"""
	C11 = C[:,0:1]
	C12 = C[:,1:2]
	C21 = C[:,2:3]
	C22 = C[:,3:4]
	C13 = C[:,4:5]
	C23 = C[:,5:6]
	C31 = C[:,6:7]
	C32 = C[:,7:8]
	C33 = C[:,8:9]

# C11 = C[828621:828622,0:1].numpy()
# C12 = C[828621:828622,1:2].numpy()
# C21 = C[828621:828622,2:3].numpy()
# C22 = C[828621:828622,3:4].numpy()
# C13 = C[828621:828622,4:5].numpy()
# C23 = C[828621:828622,5:6].numpy()
# C31 = C[828621:828622,6:7].numpy()
# C32 = C[828621:828622,7:8].numpy()
# C33 = C[828621:828622,8:9].numpy()

	I1 = C11 + C22 + C33
	I2 = C11*C22 + C11*C33 + C22*C33 - C12*C21 - C13*C31 - C23*C32
	# I3 = (-C13*C22*C31 + C12*C23*C31 + C13*C21*C32 - C11*C23*C32 - C12*C21*C33 + C11*C22*C33)
	# I3 = (-C12*C21 + C11*C22)*C33 + (-C13*C22 + C12*C23)*C31 + (C13*C21 - C11*C23)*C32
	I3 = C13*(C21*C32-C22*C31)+C12*(C23*C31-C21*C33)+C11*(C22*C33-C23*C32)
	return I1, I2, I3


def computeStrainInvariantDerivatives(F,i,secondDerivative=False):
	"""
	Compute derivatives of the invariants of the Cauchy-Green strain tensor with respect to the deformation gradient.
	Plane strain is assumed.

	_Input Arguments_

	- `F` - deformation gradient in Voigt notation

	- `i` - specify the invariant that should be differentiated- I1, I2, I3, Ia or Ib

	- `secondDerivative` - specify if second derivative should be computed

	_Output Arguments_

	- `dIdF` - derivative (note that the size of `dIdF` depends on the choice of `secondDerivative`)

	---

	"""
	F11 = F[:,0:1]
	F12 = F[:,1:2]
	F21 = F[:,2:3]
	F22 = F[:,3:4]
	F13 = F[:,4:5]
	F23 = F[:,5:6]
	F31 = F[:,6:7]
	F32 = F[:,7:8]
	F33 = F[:,8:9]

	if not(secondDerivative):
		dIdF = torch.zeros(F.shape[0],F.shape[1])
		if(i==1):
			# dI1/dF:
			dIdF = 2.0*F
		elif(i==2):
			# dI2/dF:
			dIdF1 =  -2.*F11*(np.power(F11,2) + np.power(F21,2) + np.power(F31,2)) - 2.*F12*(F11*F12 + F21*F22 + F31*F32) - 2.*F13*(F11*F13 + F21*F23 + F31*F33) + 2.*F11*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF2 =  -2.*F11*(F11*F12 + F21*F22 + F31*F32) - 2.*F12*(np.power(F12,2) + np.power(F22,2) + np.power(F32,2)) - 2.*F13*(F12*F13 + F22*F23 + F32*F33) + 2.*F12*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF3 =  -2.*F21*(np.power(F11,2) + np.power(F21,2) + np.power(F31,2)) - 2.*F22*(F11*F12 + F21*F22 + F31*F32) - 2.*F23*(F11*F13 + F21*F23 + F31*F33) + 2.*F21*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF4 =  -2.*F21*(F11*F12 + F21*F22 + F31*F32) - 2.*F22*(np.power(F12,2) + np.power(F22,2) + np.power(F32,2)) - 2.*F23*(F12*F13 + F22*F23 + F32*F33) + 2.*F22*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF5 =  -2.*F11*(F11*F13 + F21*F23 + F31*F33) - 2.*F12*(F12*F13 + F22*F23 + F32*F33) - 2.*F13*(np.power(F13,2) + np.power(F23,2) + np.power(F33,2)) + 2.*F13*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF6 =  -2.*F21*(F11*F13 + F21*F23 + F31*F33) - 2.*F22*(F12*F13 + F22*F23 + F32*F33) - 2.*F23*(np.power(F13,2) + np.power(F23,2) + np.power(F33,2)) + 2.*F23*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF7 =  -2.*F31*(np.power(F11,2) + np.power(F21,2) + np.power(F31,2)) - 2.*F32*(F11*F12 + F21*F22 + F31*F32) - 2.*F33*(F11*F13 + F21*F23 + F31*F33) + 2.*F31*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF8 =  -2.*F31*(F11*F12 + F21*F22 + F31*F32) - 2.*F32*(np.power(F12,2) + np.power(F22,2) + np.power(F32,2)) - 2.*F33*(F12*F13 + F22*F23 + F32*F33) + 2.*F32*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF9 =  -2.*F31*(F11*F13 + F21*F23 + F31*F33) - 2.*F32*(F12*F13 + F22*F23 + F32*F33) - 2.*F33*(np.power(F13,2) + np.power(F23,2) + np.power(F33,2)) + 2.*F33*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;

			dIdF = torch.cat((dIdF1,dIdF2,dIdF3,dIdF4,dIdF5,dIdF6,dIdF7,dIdF8,dIdF9),dim=1)
		elif(i==3):
			# dI3/dF:
			dIdF1 =  -2.*(F23*F32 - 1.*F22*F33)*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF2 =  2.*(F23*F31 - 1.*F21*F33)*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF3 =  2.*(F13*F32 - 1.*F12*F33)*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF4 =  -2.*(F13*F31 - 1.*F11*F33)*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF5 =  -2.*(F22*F31 - 1.*F21*F32)*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF6 =  2.*(F12*F31 - 1.*F11*F32)*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF7 =  -2.*(F13*F22 - 1.*F12*F23)*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF8 =  2.*(F13*F21 - 1.*F11*F23)*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF9 =  -2.*(F12*F21 - 1.*F11*F22)*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;

			dIdF = torch.cat((dIdF1,dIdF2,dIdF3,dIdF4,dIdF5,dIdF6,dIdF7,dIdF8,dIdF9),dim=1)
		elif(i==4):
			dIdF1 =  2*np.cos(alpha)*(F11*np.cos(alpha) + F12*np.sin(alpha)) ;


			dIdF2 =  2*np.sin(alpha)*(F11*np.cos(alpha) + F12*np.sin(alpha)) ;


			dIdF3 =  2*np.cos(alpha)*(F21*np.cos(alpha) + F22*np.sin(alpha)) ;


			dIdF4 =  2*np.sin(alpha)*(F21*np.cos(alpha) + F22*np.sin(alpha)) ;


			dIdF5 =  0.*F11 ;


			dIdF6 =  0.*F11 ;


			dIdF7 =  2*np.cos(alpha)*(F31*np.cos(alpha) + F32*np.sin(alpha)) ;


			dIdF8 =  2*np.sin(alpha)*(F31*np.cos(alpha) + F32*np.sin(alpha)) ;


			dIdF9 =  0.*F11 ;


			dIdF = torch.cat((dIdF1,dIdF2,dIdF3,dIdF4,dIdF5,dIdF6,dIdF7,dIdF8,dIdF9),dim=1)
		elif(i==5):
			dIdF1 =  2*np.cos(alpha)*(F11*np.cos(alpha) - F12*np.sin(alpha)) ;


			dIdF2 =  2*np.sin(alpha)*(-(F11*np.cos(alpha)) + F12*np.sin(alpha)) ;


			dIdF3 =  2*np.cos(alpha)*(F21*np.cos(alpha) - F22*np.sin(alpha)) ;


			dIdF4 =  2*np.sin(alpha)*(-(F21*np.cos(alpha)) + F22*np.sin(alpha)) ;


			dIdF5 =  0.*F11 ;


			dIdF6 =  0.*F11 ;


			dIdF7 =  2*np.cos(alpha)*(F31*np.cos(alpha) - F32*np.sin(alpha)) ;


			dIdF8 =  2*np.sin(alpha)*(-(F31*np.cos(alpha)) + F32*np.sin(alpha)) ;


			dIdF9 =  0.*F11 ;

			dIdF = torch.cat((dIdF1,dIdF2,dIdF3,dIdF4,dIdF5,dIdF6,dIdF7,dIdF8,dIdF9),dim=1)
		else:
			raise ValueError('Incorrect invariant index')
	if secondDerivative:
		dIdF = torch.zeros(F.shape[1],F.shape[1])
		if(i==1):
			# d(dI1/dF)/dF:
			dIdF = 2.0*torch.eye(F.shape[1])
		elif(i==2):

			dIdF[0][0]  =  -2.*np.power(F12,2) - 2.*np.power(F13,2) - 2.*(np.power(F11,2) + np.power(F21,2) + np.power(F31,2)) + 2.*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF[0][1]  =  0. - 2.*F21*F22 - 2.*F31*F32 ;


			dIdF[0][2]  =  0. - 2.*F12*F22 - 2.*F13*F23 ;


			dIdF[0][3]  =  -2.*F12*F21 + 4.*F11*F22 ;


			dIdF[0][4]  =  0. - 2.*F21*F23 - 2.*F31*F33 ;


			dIdF[0][5]  =  -2.*F13*F21 + 4.*F11*F23 ;


			dIdF[0][6]  =  0. - 2.*F12*F32 - 2.*F13*F33 ;


			dIdF[0][7]  =  -2.*F12*F31 + 4.*F11*F32 ;


			dIdF[0][8]  =  -2.*F13*F31 + 4.*F11*F33 ;


			dIdF[1][0]  =  0. - 2.*F21*F22 - 2.*F31*F32 ;


			dIdF[1][1]  =  -2.*np.power(F11,2) - 2.*np.power(F13,2) - 2.*(np.power(F12,2) + np.power(F22,2) + np.power(F32,2)) + 2.*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF[1][2]  =  4.*F12*F21 - 2.*F11*F22 ;


			dIdF[1][3]  =  0. - 2.*F11*F21 - 2.*F13*F23 ;


			dIdF[1][4]  =  0. - 2.*F22*F23 - 2.*F32*F33 ;


			dIdF[1][5]  =  -2.*F13*F22 + 4.*F12*F23 ;


			dIdF[1][6]  =  4.*F12*F31 - 2.*F11*F32 ;


			dIdF[1][7]  =  0. - 2.*F11*F31 - 2.*F13*F33 ;


			dIdF[1][8]  =  -2.*F13*F32 + 4.*F12*F33 ;


			dIdF[2][0]  =  0. - 2.*F12*F22 - 2.*F13*F23 ;


			dIdF[2][1]  =  4.*F12*F21 - 2.*F11*F22 ;


			dIdF[2][2]  =  -2.*np.power(F22,2) - 2.*np.power(F23,2) - 2.*(np.power(F11,2) + np.power(F21,2) + np.power(F31,2)) + 2.*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF[2][3]  =  0. - 2.*F11*F12 - 2.*F31*F32 ;


			dIdF[2][4]  =  4.*F13*F21 - 2.*F11*F23 ;


			dIdF[2][5]  =  0. - 2.*F11*F13 - 2.*F31*F33 ;


			dIdF[2][6]  =  0. - 2.*F22*F32 - 2.*F23*F33 ;


			dIdF[2][7]  =  -2.*F22*F31 + 4.*F21*F32 ;


			dIdF[2][8]  =  -2.*F23*F31 + 4.*F21*F33 ;


			dIdF[3][0]  =  -2.*F12*F21 + 4.*F11*F22 ;


			dIdF[3][1]  =  0. - 2.*F11*F21 - 2.*F13*F23 ;


			dIdF[3][2]  =  0. - 2.*F11*F12 - 2.*F31*F32 ;


			dIdF[3][3]  =  -2.*np.power(F21,2) - 2.*np.power(F23,2) - 2.*(np.power(F12,2) + np.power(F22,2) + np.power(F32,2)) + 2.*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF[3][4]  =  4.*F13*F22 - 2.*F12*F23 ;


			dIdF[3][5]  =  0. - 2.*F12*F13 - 2.*F32*F33 ;


			dIdF[3][6]  =  4.*F22*F31 - 2.*F21*F32 ;


			dIdF[3][7]  =  0. - 2.*F21*F31 - 2.*F23*F33 ;


			dIdF[3][8]  =  -2.*F23*F32 + 4.*F22*F33 ;


			dIdF[4][0]  =  0. - 2.*F21*F23 - 2.*F31*F33 ;


			dIdF[4][1]  =  0. - 2.*F22*F23 - 2.*F32*F33 ;


			dIdF[4][2]  =  4.*F13*F21 - 2.*F11*F23 ;


			dIdF[4][3]  =  4.*F13*F22 - 2.*F12*F23 ;


			dIdF[4][4]  =  -2.*np.power(F11,2) - 2.*np.power(F12,2) - 2.*(np.power(F13,2) + np.power(F23,2) + np.power(F33,2)) + 2.*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF[4][5]  =  0. - 2.*F11*F21 - 2.*F12*F22 ;


			dIdF[4][6]  =  4.*F13*F31 - 2.*F11*F33 ;


			dIdF[4][7]  =  4.*F13*F32 - 2.*F12*F33 ;


			dIdF[4][8]  =  0. - 2.*F11*F31 - 2.*F12*F32 ;


			dIdF[5][0]  =  -2.*F13*F21 + 4.*F11*F23 ;


			dIdF[5][1]  =  -2.*F13*F22 + 4.*F12*F23 ;


			dIdF[5][2]  =  0. - 2.*F11*F13 - 2.*F31*F33 ;


			dIdF[5][3]  =  0. - 2.*F12*F13 - 2.*F32*F33 ;


			dIdF[5][4]  =  0. - 2.*F11*F21 - 2.*F12*F22 ;


			dIdF[5][5]  =  -2.*np.power(F21,2) - 2.*np.power(F22,2) - 2.*(np.power(F13,2) + np.power(F23,2) + np.power(F33,2)) + 2.*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF[5][6]  =  4.*F23*F31 - 2.*F21*F33 ;


			dIdF[5][7]  =  4.*F23*F32 - 2.*F22*F33 ;


			dIdF[5][8]  =  0. - 2.*F21*F31 - 2.*F22*F32 ;


			dIdF[6][0]  =  0. - 2.*F12*F32 - 2.*F13*F33 ;


			dIdF[6][1]  =  4.*F12*F31 - 2.*F11*F32 ;


			dIdF[6][2]  =  0. - 2.*F22*F32 - 2.*F23*F33 ;


			dIdF[6][3]  =  4.*F22*F31 - 2.*F21*F32 ;


			dIdF[6][4]  =  4.*F13*F31 - 2.*F11*F33 ;


			dIdF[6][5]  =  4.*F23*F31 - 2.*F21*F33 ;


			dIdF[6][6]  =  -2.*(np.power(F11,2) + np.power(F21,2) + np.power(F31,2)) - 2.*np.power(F32,2) - 2.*np.power(F33,2) + 2.*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF[6][7]  =  0. - 2.*F11*F12 - 2.*F21*F22 ;


			dIdF[6][8]  =  0. - 2.*F11*F13 - 2.*F21*F23 ;


			dIdF[7][0]  =  -2.*F12*F31 + 4.*F11*F32 ;


			dIdF[7][1]  =  0. - 2.*F11*F31 - 2.*F13*F33 ;


			dIdF[7][2]  =  -2.*F22*F31 + 4.*F21*F32 ;


			dIdF[7][3]  =  0. - 2.*F21*F31 - 2.*F23*F33 ;


			dIdF[7][4]  =  4.*F13*F32 - 2.*F12*F33 ;


			dIdF[7][5]  =  4.*F23*F32 - 2.*F22*F33 ;


			dIdF[7][6]  =  0. - 2.*F11*F12 - 2.*F21*F22 ;


			dIdF[7][7]  =  -2.*np.power(F31,2) - 2.*(np.power(F12,2) + np.power(F22,2) + np.power(F32,2)) - 2.*np.power(F33,2) + 2.*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


			dIdF[7][8]  =  0. - 2.*F12*F13 - 2.*F22*F23 ;


			dIdF[8][0]  =  -2.*F13*F31 + 4.*F11*F33 ;


			dIdF[8][1]  =  -2.*F13*F32 + 4.*F12*F33 ;


			dIdF[8][2]  =  -2.*F23*F31 + 4.*F21*F33 ;


			dIdF[8][3]  =  -2.*F23*F32 + 4.*F22*F33 ;


			dIdF[8][4]  =  0. - 2.*F11*F31 - 2.*F12*F32 ;


			dIdF[8][5]  =  0. - 2.*F21*F31 - 2.*F22*F32 ;


			dIdF[8][6]  =  0. - 2.*F11*F13 - 2.*F21*F23 ;


			dIdF[8][7]  =  0. - 2.*F12*F13 - 2.*F22*F23 ;


			dIdF[8][8]  =  -2.*np.power(F31,2) - 2.*np.power(F32,2) - 2.*(np.power(F13,2) + np.power(F23,2) + np.power(F33,2)) + 2.*np.power(np.power(F11,2) + np.power(F12,2) + np.power(F13,2) + np.power(F21,2) + np.power(F22,2) + np.power(F23,2) + np.power(F31,2) + np.power(F32,2) + np.power(F33,2),1.) ;


		elif(i==3):


			dIdF[0][0]  =  2.*np.power(F23*F32 - 1.*F22*F33,2) ;


			dIdF[0][1]  =  -2.*(F23*F31 - 1.*F21*F33)*(F23*F32 - 1.*F22*F33) ;


			dIdF[0][2]  =  -2.*(F13*F32 - 1.*F12*F33)*(F23*F32 - 1.*F22*F33) ;


			dIdF[0][3]  =  -2.*(F13*F31 - 1.*F11*F33)*(-1.*F23*F32 + F22*F33) + 2.*F33*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[0][4]  =  -2.*(F22*F31 - 1.*F21*F32)*(-1.*F23*F32 + F22*F33) ;


			dIdF[0][5]  =  -2.*(F12*F31 - 1.*F11*F32)*(F23*F32 - 1.*F22*F33) - 2.*F32*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[0][6]  =  -2.*(F13*F22 - 1.*F12*F23)*(-1.*F23*F32 + F22*F33) ;


			dIdF[0][7]  =  -2.*(F13*F21 - 1.*F11*F23)*(F23*F32 - 1.*F22*F33) - 2.*F23*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[0][8]  =  -2.*(F12*F21 - 1.*F11*F22)*(-1.*F23*F32 + F22*F33) + 2.*F22*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[1][0]  =  -2.*(F23*F31 - 1.*F21*F33)*(F23*F32 - 1.*F22*F33) ;


			dIdF[1][1]  =  2.*np.power(F23*F31 - 1.*F21*F33,2) ;


			dIdF[1][2]  =  -2.*(F13*F32 - 1.*F12*F33)*(-1.*F23*F31 + F21*F33) - 2.*F33*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[1][3]  =  -2.*(F13*F31 - 1.*F11*F33)*(F23*F31 - 1.*F21*F33) ;


			dIdF[1][4]  =  -2.*(F22*F31 - 1.*F21*F32)*(F23*F31 - 1.*F21*F33) ;


			dIdF[1][5]  =  2.*(F12*F31 - 1.*F11*F32)*(F23*F31 - 1.*F21*F33) + 2.*F31*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[1][6]  =  -2.*(F13*F22 - 1.*F12*F23)*(F23*F31 - 1.*F21*F33) + 2.*F23*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[1][7]  =  -2.*(F13*F21 - 1.*F11*F23)*(-1.*F23*F31 + F21*F33) ;


			dIdF[1][8]  =  2.*(F12*F21 - 1.*F11*F22)*(-1.*F23*F31 + F21*F33) - 2.*F21*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[2][0]  =  -2.*(F13*F32 - 1.*F12*F33)*(F23*F32 - 1.*F22*F33) ;


			dIdF[2][1]  =  -2.*(F13*F32 - 1.*F12*F33)*(-1.*F23*F31 + F21*F33) - 2.*F33*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[2][2]  =  2.*np.power(F13*F32 - 1.*F12*F33,2) ;


			dIdF[2][3]  =  -2.*(F13*F31 - 1.*F11*F33)*(F13*F32 - 1.*F12*F33) ;


			dIdF[2][4]  =  -2.*(F22*F31 - 1.*F21*F32)*(F13*F32 - 1.*F12*F33) + 2.*F32*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[2][5]  =  -2.*(F12*F31 - 1.*F11*F32)*(-1.*F13*F32 + F12*F33) ;


			dIdF[2][6]  =  -2.*(F13*F22 - 1.*F12*F23)*(F13*F32 - 1.*F12*F33) ;


			dIdF[2][7]  =  2.*(F13*F21 - 1.*F11*F23)*(F13*F32 - 1.*F12*F33) + 2.*F13*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[2][8]  =  2.*(F12*F21 - 1.*F11*F22)*(-1.*F13*F32 + F12*F33) - 2.*F12*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[3][0]  =  -2.*(F13*F31 - 1.*F11*F33)*(-1.*F23*F32 + F22*F33) + 2.*F33*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[3][1]  =  -2.*(F13*F31 - 1.*F11*F33)*(F23*F31 - 1.*F21*F33) ;


			dIdF[3][2]  =  -2.*(F13*F31 - 1.*F11*F33)*(F13*F32 - 1.*F12*F33) ;


			dIdF[3][3]  =  2.*np.power(F13*F31 - 1.*F11*F33,2) ;


			dIdF[3][4]  =  2.*(F22*F31 - 1.*F21*F32)*(F13*F31 - 1.*F11*F33) - 2.*F31*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[3][5]  =  -2.*(F12*F31 - 1.*F11*F32)*(F13*F31 - 1.*F11*F33) ;


			dIdF[3][6]  =  2.*(F13*F22 - 1.*F12*F23)*(F13*F31 - 1.*F11*F33) - 2.*F13*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[3][7]  =  -2.*(F13*F21 - 1.*F11*F23)*(F13*F31 - 1.*F11*F33) ;


			dIdF[3][8]  =  -2.*(F12*F21 - 1.*F11*F22)*(-1.*F13*F31 + F11*F33) + 2.*F11*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[4][0]  =  -2.*(F22*F31 - 1.*F21*F32)*(-1.*F23*F32 + F22*F33) ;


			dIdF[4][1]  =  -2.*(F22*F31 - 1.*F21*F32)*(F23*F31 - 1.*F21*F33) ;


			dIdF[4][2]  =  -2.*(F22*F31 - 1.*F21*F32)*(F13*F32 - 1.*F12*F33) + 2.*F32*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[4][3]  =  2.*(F22*F31 - 1.*F21*F32)*(F13*F31 - 1.*F11*F33) - 2.*F31*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[4][4]  =  2.*np.power(F22*F31 - 1.*F21*F32,2) ;


			dIdF[4][5]  =  -2.*(F12*F31 - 1.*F11*F32)*(F22*F31 - 1.*F21*F32) ;


			dIdF[4][6]  =  2.*(F13*F22 - 1.*F12*F23)*(F22*F31 - 1.*F21*F32) - 2.*F22*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[4][7]  =  2.*(F13*F21 - 1.*F11*F23)*(-1.*F22*F31 + F21*F32) + 2.*F21*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[4][8]  =  -2.*(F12*F21 - 1.*F11*F22)*(-1.*F22*F31 + F21*F32) ;


			dIdF[5][0]  =  -2.*(F12*F31 - 1.*F11*F32)*(F23*F32 - 1.*F22*F33) - 2.*F32*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[5][1]  =  2.*(F12*F31 - 1.*F11*F32)*(F23*F31 - 1.*F21*F33) + 2.*F31*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[5][2]  =  -2.*(F12*F31 - 1.*F11*F32)*(-1.*F13*F32 + F12*F33) ;


			dIdF[5][3]  =  -2.*(F12*F31 - 1.*F11*F32)*(F13*F31 - 1.*F11*F33) ;


			dIdF[5][4]  =  -2.*(F12*F31 - 1.*F11*F32)*(F22*F31 - 1.*F21*F32) ;


			dIdF[5][5]  =  2.*np.power(F12*F31 - 1.*F11*F32,2) ;


			dIdF[5][6]  =  -2.*(F13*F22 - 1.*F12*F23)*(F12*F31 - 1.*F11*F32) + 2.*F12*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[5][7]  =  -2.*(F13*F21 - 1.*F11*F23)*(-1.*F12*F31 + F11*F32) - 2.*F11*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[5][8]  =  -2.*(F12*F21 - 1.*F11*F22)*(F12*F31 - 1.*F11*F32) ;


			dIdF[6][0]  =  -2.*(F13*F22 - 1.*F12*F23)*(-1.*F23*F32 + F22*F33) ;


			dIdF[6][1]  =  -2.*(F13*F22 - 1.*F12*F23)*(F23*F31 - 1.*F21*F33) + 2.*F23*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[6][2]  =  -2.*(F13*F22 - 1.*F12*F23)*(F13*F32 - 1.*F12*F33) ;


			dIdF[6][3]  =  2.*(F13*F22 - 1.*F12*F23)*(F13*F31 - 1.*F11*F33) - 2.*F13*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[6][4]  =  2.*(F13*F22 - 1.*F12*F23)*(F22*F31 - 1.*F21*F32) - 2.*F22*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[6][5]  =  -2.*(F13*F22 - 1.*F12*F23)*(F12*F31 - 1.*F11*F32) + 2.*F12*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[6][6]  =  2.*np.power(F13*F22 - 1.*F12*F23,2) ;


			dIdF[6][7]  =  -2.*(F13*F21 - 1.*F11*F23)*(F13*F22 - 1.*F12*F23) ;


			dIdF[6][8]  =  -2.*(F12*F21 - 1.*F11*F22)*(-1.*F13*F22 + F12*F23) ;


			dIdF[7][0]  =  -2.*(F13*F21 - 1.*F11*F23)*(F23*F32 - 1.*F22*F33) - 2.*F23*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[7][1]  =  -2.*(F13*F21 - 1.*F11*F23)*(-1.*F23*F31 + F21*F33) ;


			dIdF[7][2]  =  2.*(F13*F21 - 1.*F11*F23)*(F13*F32 - 1.*F12*F33) + 2.*F13*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[7][3]  =  -2.*(F13*F21 - 1.*F11*F23)*(F13*F31 - 1.*F11*F33) ;


			dIdF[7][4]  =  2.*(F13*F21 - 1.*F11*F23)*(-1.*F22*F31 + F21*F32) + 2.*F21*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[7][5]  =  -2.*(F13*F21 - 1.*F11*F23)*(-1.*F12*F31 + F11*F32) - 2.*F11*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[7][6]  =  -2.*(F13*F21 - 1.*F11*F23)*(F13*F22 - 1.*F12*F23) ;


			dIdF[7][7]  =  2.*np.power(F13*F21 - 1.*F11*F23,2) ;


			dIdF[7][8]  =  -2.*(F12*F21 - 1.*F11*F22)*(F13*F21 - 1.*F11*F23) ;


			dIdF[8][0]  =  -2.*(F12*F21 - 1.*F11*F22)*(-1.*F23*F32 + F22*F33) + 2.*F22*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[8][1]  =  2.*(F12*F21 - 1.*F11*F22)*(-1.*F23*F31 + F21*F33) - 2.*F21*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[8][2]  =  2.*(F12*F21 - 1.*F11*F22)*(-1.*F13*F32 + F12*F33) - 2.*F12*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[8][3]  =  -2.*(F12*F21 - 1.*F11*F22)*(-1.*F13*F31 + F11*F33) + 2.*F11*np.power(-(F13*F22*F31) + F12*F23*F31 + F13*F21*F32 - F11*F23*F32 - F12*F21*F33 + F11*F22*F33,1.) ;


			dIdF[8][4]  =  -2.*(F12*F21 - 1.*F11*F22)*(-1.*F22*F31 + F21*F32) ;


			dIdF[8][5]  =  -2.*(F12*F21 - 1.*F11*F22)*(F12*F31 - 1.*F11*F32) ;


			dIdF[8][6]  =  -2.*(F12*F21 - 1.*F11*F22)*(-1.*F13*F22 + F12*F23) ;


			dIdF[8][7]  =  -2.*(F12*F21 - 1.*F11*F22)*(F13*F21 - 1.*F11*F23) ;


			dIdF[8][8]  =  2.*np.power(F12*F21 - 1.*F11*F22,2) ;



		elif(i==4):


			dIdF[0][0]  =  2*np.power(np.cos(alpha),2) ;


			dIdF[0][1]  =  np.sin(2*alpha) ;


			dIdF[0][2]  =  0 ;


			dIdF[0][3]  =  0 ;


			dIdF[0][4]  =  0 ;


			dIdF[0][5]  =  0 ;


			dIdF[0][6]  =  0 ;


			dIdF[0][7]  =  0 ;


			dIdF[0][8]  =  0 ;


			dIdF[1][0]  =  np.sin(2*alpha) ;


			dIdF[1][1]  =  2*np.power(np.sin(alpha),2) ;


			dIdF[1][2]  =  0 ;


			dIdF[1][3]  =  0 ;


			dIdF[1][4]  =  0 ;


			dIdF[1][5]  =  0 ;


			dIdF[1][6]  =  0 ;


			dIdF[1][7]  =  0 ;


			dIdF[1][8]  =  0 ;


			dIdF[2][0]  =  0 ;


			dIdF[2][1]  =  0 ;


			dIdF[2][2]  =  2*np.power(np.cos(alpha),2) ;


			dIdF[2][3]  =  np.sin(2*alpha) ;


			dIdF[2][4]  =  0 ;


			dIdF[2][5]  =  0 ;


			dIdF[2][6]  =  0 ;


			dIdF[2][7]  =  0 ;


			dIdF[2][8]  =  0 ;


			dIdF[3][0]  =  0 ;


			dIdF[3][1]  =  0 ;


			dIdF[3][2]  =  np.sin(2*alpha) ;


			dIdF[3][3]  =  2*np.power(np.sin(alpha),2) ;


			dIdF[3][4]  =  0 ;


			dIdF[3][5]  =  0 ;


			dIdF[3][6]  =  0 ;


			dIdF[3][7]  =  0 ;


			dIdF[3][8]  =  0 ;


			dIdF[4][0]  =  0 ;


			dIdF[4][1]  =  0 ;


			dIdF[4][2]  =  0 ;


			dIdF[4][3]  =  0 ;


			dIdF[4][4]  =  0 ;


			dIdF[4][5]  =  0 ;


			dIdF[4][6]  =  0 ;


			dIdF[4][7]  =  0 ;


			dIdF[4][8]  =  0 ;


			dIdF[5][0]  =  0 ;


			dIdF[5][1]  =  0 ;


			dIdF[5][2]  =  0 ;


			dIdF[5][3]  =  0 ;


			dIdF[5][4]  =  0 ;


			dIdF[5][5]  =  0 ;


			dIdF[5][6]  =  0 ;


			dIdF[5][7]  =  0 ;


			dIdF[5][8]  =  0 ;


			dIdF[6][0]  =  0 ;


			dIdF[6][1]  =  0 ;


			dIdF[6][2]  =  0 ;


			dIdF[6][3]  =  0 ;


			dIdF[6][4]  =  0 ;


			dIdF[6][5]  =  0 ;


			dIdF[6][6]  =  2*np.power(np.cos(alpha),2) ;


			dIdF[6][7]  =  np.sin(2*alpha) ;


			dIdF[6][8]  =  0 ;


			dIdF[7][0]  =  0 ;


			dIdF[7][1]  =  0 ;


			dIdF[7][2]  =  0 ;


			dIdF[7][3]  =  0 ;


			dIdF[7][4]  =  0 ;


			dIdF[7][5]  =  0 ;


			dIdF[7][6]  =  np.sin(2*alpha) ;


			dIdF[7][7]  =  2*np.power(np.sin(alpha),2) ;


			dIdF[7][8]  =  0 ;


			dIdF[8][0]  =  0 ;


			dIdF[8][1]  =  0 ;


			dIdF[8][2]  =  0 ;


			dIdF[8][3]  =  0 ;


			dIdF[8][4]  =  0 ;


			dIdF[8][5]  =  0 ;


			dIdF[8][6]  =  0 ;


			dIdF[8][7]  =  0 ;


			dIdF[8][8]  =  0 ;




		elif(i==5):


			dIdF[0][0]  =  2*np.power(np.cos(alpha),2) ;


			dIdF[0][1]  =  -np.sin(2*alpha) ;


			dIdF[0][2]  =  0 ;


			dIdF[0][3]  =  0 ;


			dIdF[0][4]  =  0 ;


			dIdF[0][5]  =  0 ;


			dIdF[0][6]  =  0 ;


			dIdF[0][7]  =  0 ;


			dIdF[0][8]  =  0 ;


			dIdF[1][0]  =  -np.sin(2*alpha) ;


			dIdF[1][1]  =  2*np.power(np.sin(alpha),2) ;


			dIdF[1][2]  =  0 ;


			dIdF[1][3]  =  0 ;


			dIdF[1][4]  =  0 ;


			dIdF[1][5]  =  0 ;


			dIdF[1][6]  =  0 ;


			dIdF[1][7]  =  0 ;


			dIdF[1][8]  =  0 ;


			dIdF[2][0]  =  0 ;


			dIdF[2][1]  =  0 ;


			dIdF[2][2]  =  2*np.power(np.cos(alpha),2) ;


			dIdF[2][3]  =  -np.sin(2*alpha) ;


			dIdF[2][4]  =  0 ;


			dIdF[2][5]  =  0 ;


			dIdF[2][6]  =  0 ;


			dIdF[2][7]  =  0 ;


			dIdF[2][8]  =  0 ;


			dIdF[3][0]  =  0 ;


			dIdF[3][1]  =  0 ;


			dIdF[3][2]  =  -np.sin(2*alpha) ;


			dIdF[3][3]  =  2*np.power(np.sin(alpha),2) ;


			dIdF[3][4]  =  0 ;


			dIdF[3][5]  =  0 ;


			dIdF[3][6]  =  0 ;


			dIdF[3][7]  =  0 ;


			dIdF[3][8]  =  0 ;


			dIdF[4][0]  =  0 ;


			dIdF[4][1]  =  0 ;


			dIdF[4][2]  =  0 ;


			dIdF[4][3]  =  0 ;


			dIdF[4][4]  =  0 ;


			dIdF[4][5]  =  0 ;


			dIdF[4][6]  =  0 ;


			dIdF[4][7]  =  0 ;


			dIdF[4][8]  =  0 ;


			dIdF[5][0]  =  0 ;


			dIdF[5][1]  =  0 ;


			dIdF[5][2]  =  0 ;


			dIdF[5][3]  =  0 ;


			dIdF[5][4]  =  0 ;


			dIdF[5][5]  =  0 ;


			dIdF[5][6]  =  0 ;


			dIdF[5][7]  =  0 ;


			dIdF[5][8]  =  0 ;


			dIdF[6][0]  =  0 ;


			dIdF[6][1]  =  0 ;


			dIdF[6][2]  =  0 ;


			dIdF[6][3]  =  0 ;


			dIdF[6][4]  =  0 ;


			dIdF[6][5]  =  0 ;


			dIdF[6][6]  =  2*np.power(np.cos(alpha),2) ;


			dIdF[6][7]  =  -np.sin(2*alpha) ;


			dIdF[6][8]  =  0 ;


			dIdF[7][0]  =  0 ;


			dIdF[7][1]  =  0 ;


			dIdF[7][2]  =  0 ;


			dIdF[7][3]  =  0 ;


			dIdF[7][4]  =  0 ;


			dIdF[7][5]  =  0 ;


			dIdF[7][6]  =  -np.sin(2*alpha) ;


			dIdF[7][7]  =  2*np.power(np.sin(alpha),2) ;


			dIdF[7][8]  =  0 ;


			dIdF[8][0]  =  0 ;


			dIdF[8][1]  =  0 ;


			dIdF[8][2]  =  0 ;


			dIdF[8][3]  =  0 ;


			dIdF[8][4]  =  0 ;


			dIdF[8][5]  =  0 ;


			dIdF[8][6]  =  0 ;


			dIdF[8][7]  =  0 ;


			dIdF[8][8]  =  0 ;

		else:
			raise ValueError('Incorrect invariant index')
	return dIdF


def computeTriangleArea(p1, p2, p3):
	"""
	Computing the area of a triangle with vertices at points p1, p2, p3 (2D).

	"""
	vec_a = p2 - p1
	vec_b = p3 - p1

	a = torch.sqrt(vec_a[:,0:1]**2 + vec_a[:,1:2]**2)

	b = torch.sqrt(vec_b[:,0:1]**2 + vec_b[:,1:2]**2)

	dotprod = vec_a[:,0:1]*vec_b[:,0:1] + vec_a[:,1:2]*vec_b[:,1:2]

	cos_theta = dotprod/(a * b)

	sin_theta = torch.sqrt( 1 - cos_theta**2 )

	area = (0.5 * a * b * sin_theta).squeeze()

	return area

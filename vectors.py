import numpy as np

#cross product
def CrossP(v1, v2):
	
	v = np.zeros(3)
	
	for i in range(3):
		v[i] = v1[(i+1)%3]*v2[(i+2)%3] - v1[(i+2)%3]*v2[(i+1)%3]
	
	return(v)

#dot product
def DotP(v1,v2):
	
	return(v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2])


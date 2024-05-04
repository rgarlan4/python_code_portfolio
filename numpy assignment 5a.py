import numpy as np

#Q1 = [‘Hello’]
#Q2 = x[3:]

##Set a random seed of 5.Create a one-dimensional array of twenty random integers from 0 to 14.○ Index the array, assigning the fourth integer to Q1.○ Slice the array to obtain the following array:Assign it to Q2.○ Slice the array to obtain the following array:Assign it to Q3.

np.random.seed(5)
arr = np.random.randint(0, 15, 20)
Q1 = arr[3]


arr1 = np.array([4, 7, 14, 11, 0, 14, 0, 7, 12, 1, 5, 7])
arr2 = np.array([14, 6, 0, 8, 7, 11, 14, 7, 1])

Q2 = np.array_split(arr1, 2)
Q3 = np.array_split(arr2, 2)


np.random.seed(6)
arr3 = np.random.randint(0, 15, (7,8))
q4 = arr[6]

arr4 = np.array([[13, 10, 0],[ 9, 10, 1]])

Q5 = np.array_spit(arr4, 2)


np.random.seed(10)
arr5 = np.random.randint(0,15 size =(4,5))

Q6 = arr5[1:4 , 1:4]

Q6[1, 1] =1

np.random.seed(8)
Q8 = np.random.randint(0,15 size =(4,5))
slice_Q8 = Q8[1:4, 1:4].copy()
slice_Q8[1,1] =5

np.random.seed(15)

Q11 = np.random.randint( 0,15, size=10)

np.random.seed(9) 
arr10 = np.random.randin(0,15, size=(5,3))

Q12 = arr10[3].copy()
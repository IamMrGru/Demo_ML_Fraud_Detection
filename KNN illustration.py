import cv2
import matplotlib.pyplot as plt
import numpy as np
import math

train = np.random.randint(0,100,(100,2))
color = np.random.randint(0,2,(100,1))
new   = np.random.randint(0,100,(1,2))
yellow = train[color.ravel()==1] 
blue =   train[color.ravel()==0]
plt.scatter(blue[:,0],blue[:,1],100,'green','o')
plt.scatter(yellow[:,0],yellow[:,1],100,'red','s')
a = []
b = []
for i in range(len(yellow)-1) :
    a.append(math.sqrt(math.pow(new[0,0]-yellow[i,0],2)+math.pow(new[0,1]-yellow[i,1],2)))
for i in range(len(blue)-1) :
    b.append(math.sqrt(math.pow(new[0,0]-blue[i,0],2)+math.pow(new[0,1]-blue[i,1],2)))
i = j = ansy = ansb = 0
a.sort()
b.sort()
c=int(input("Input for K number of the nearest elements: "))
for x in range(c) : 
    if a[i] < b[j] :
        ansy+=1
        i+=1
    else :
        ansb+=1
        j+=1

if ansy > ansb :
    plt.text(0,109,'Fraudulent Payment', bbox=dict(facecolor='red', alpha=0.5))
    plt.text(44,109,'Numberacc : ')
    plt.text(59,109,ansy)
    styley = 'solid'
    styleb = 'dotted'
else :
    plt.text(0,109,'Genuine payment', bbox=dict(facecolor='green', alpha=0.5))
    plt.text(39,109,'Numberacc  : ')
    plt.text(56,109,ansb)
    styley = 'dotted'
    styleb = 'solid'
plt.xlabel('Demo Testing ',color = 'red')

for x in range (ansy) :
    for i in range (len(yellow)-1) :
        if math.sqrt(math.pow(new[0,0]-yellow[i,0],2)+math.pow(new[0,1]-yellow[i,1],2)) == a[x] :
            plt.plot([new[0,0],yellow[i,0]],[new[0,1],yellow[i,1]],color ='red',linestyle = styley)
for x in range (ansb) :
    for i in range (len(blue)-1) :
        if math.sqrt(math.pow(new[0,0]-blue[i,0],2)+math.pow(new[0,1]-blue[i,1],2)) == b[x] :
            plt.plot([new[0,0],blue[i,0]],[new[0,1],blue[i,1]],color = 'green',linestyle=styleb)
plt.scatter(new[:,0],new[:,1],400,'black','*')
plt.show()

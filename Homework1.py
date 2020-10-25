#Exercises of the Problem 1   (77/91)

#Say "Hello, World!" With Python
print ("Hello, World!")



#Python If-Else

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(raw_input().strip())
if n%2==1:
    print("Weird")
else:
    if n>2 and n<5 :
        print("Not Weird")
    if n>=6 and n<=20:
        print("Weird")
    if n>20 :
        print ("Not Weird")



#Arithmetic Operators
        
if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())
print(a+b)
print(a-b)
print(a*b)



#Python: Division


from __future__ import division

if __name__ == '__main__':
    a = int(raw_input())
    b = int(raw_input())

print(a//b)
print(a/b)



#Loops
if __name__ == '__main__':
    n = int(raw_input())
    for i in range (0,n):
        print(i*i)


#Write a function

def is_leap(year):
    leap=False
    if year%4==0:
        leap= True
        if year%100 == 0:
            leap=False
            if year%400==0:
                leap= True
    return leap

year = int(raw_input())
print is_leap(year)
year = int(raw_input())
print is_leap(year)
#Print Function

from __future__ import print_function

if __name__ == '__main__':
    n = int(raw_input())
    a=''
    for i in range(1,n+1):
        a+=str(i)
    print(a)

#List Comprehensions
if __name__ == '__main__':
    x = int(raw_input())
    y = int(raw_input())
    z = int(raw_input())
    n = int(raw_input())
    l=[[i,j,k] for i in range(0,x+1) for j in range(0,y+1) for k in range(0,z+1)]
    s=[l[i] for i in range(0,len(l)) if sum(l[i])!=n]
    print s


#Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(raw_input())
    arr = map(int, raw_input().split())
    arr2=[arr[i] for i in range(0,len(arr)) if arr[i]!=max(arr)]
    print max(arr2)

#Nested Lists

if __name__ == '__main__':
    l=[]
    punteggi=[]
    for _ in range(int(raw_input())):
        name = raw_input()
        score = float(raw_input())
        l=l+[[name,score]]
        punteggi+=[score]
    punteggi2=[punteggi[i] for i in range(0,len(punteggi)) if punteggi[i]!=min(punteggi)]
    minimo=min(punteggi2)
    nomi=[l[i][0] for i in range (0,len(l)) if l[i][1]==minimo]
    nomi.sort()
    for n in nomi:
        print (n)

#Finding the percentage
if __name__ == '__main__':
    n = int(raw_input())
    student_marks = {}
    for _ in range(n):
        line = raw_input().split()
        name, scores = line[0], line[1:]
        scores = map(float, scores)
        student_marks[name] = scores
    query_name = raw_input()
punteggio=student_marks[query_name]
print "%.2f"%(sum(punteggio)/len(punteggio))


#Lists

if __name__ == '__main__':
    b=[]
    N = int(input())
    for _ in range(N):
       a=input().split()
       if a[0]=="insert":
        b.insert(int(a[1]),int(a[2]))
       elif a[0]=='print':
        print(b)
       elif a[0]=='remove':
        b.remove(int(a[1]))
       elif a[0]=='append':
        b.append(int(a[1]))
       elif a[0]=='sort':
        b.sort()
       elif a[0]=='pop':
        b.pop()
       elif a[0]=='reverse':
        b.reverse()

#Tuples


if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    t=tuple(integer_list)
    print(hash(t))

#sWAP cASE

def swap_case(s):
    nuovaparola=""
    for i in s:
        if i.islower()==True:
            nuovaparola+=i.upper()
        else:
            nuovaparola+=i.lower()
    return nuovaparola

if __name__ == '__main__':
    s = raw_input()
    result = swap_case(s)
    print result


#String Split and Join

def split_and_join(line):
    a=line.split(" ")
    a="-".join(a)
    return a

if __name__ == '__main__':
    line = raw_input()
    result = split_and_join(line)
    print result
    
#What's Your Name?

def print_full_name(a, b):
    print ('Hello '+a+' '+b+'! You just delved into python.')

if __name__ == '__main__':
    first_name = raw_input()
    last_name = raw_input()
    print_full_name(first_name, last_name)

    
#Mutations

def print_full_name(a, b):
    print ('Hello '+a+' '+b+'! You just delved into python.')

if __name__ == '__main__':
    first_name = raw_input()
    last_name = raw_input()
    print_full_name(first_name, last_name)

    
#Find a string

def count_substring(string, sub_string):
    count=0
    for i in range(0,len(string)):
        if string[i]==sub_string[0]:
            if string[i:i+len(sub_string)]==sub_string:
                count+=1
            

    return count

if __name__ == '__main__':
    string = raw_input().strip()
    sub_string = raw_input().strip()
    
    count = count_substring(string, sub_string)
    print count

#String Validators
if __name__ == '__main__':
    
    s = raw_input()
    a=0
    for i in s :
        a=a+1
        if i.isalnum()==True:
            print('True')
            break
        if a==len(s):
            print('False')
    a=0
    for i in s :
        a=a+1
        if i.isalpha()==True:
            print('True')
            break
        if a==len(s):
            print('False')
    a=0
    for i in s :
        a=a+1
        if i.isdigit()==True:
            print('True')
            break
        if a==len(s):
            print('False')
    a=0
    for i in s :
        a=a+1
        if i.islower()==True:
            print('True')
            break
        if a==len(s):
            print('False')
    a=0
    for i in s :
        a=a+1
        if i.isupper()==True:
            print('True')
            break
        if a==len(s):
            print('False')
        

#Text Alignment

a = int(input()) 
b = 'H'

for i in range(a):
    print((b*i).rjust(a-1)+b+(b*i).ljust(a-1))


for i in range(a+1):
    print((b*a).center(a*2)+(b*a).center(a*6))


for i in range((a+1)//2):
    print((b*a*5).center(a*6))    


for i in range(a+1):
    print((b*a).center(a*2)+(b*a).center(a*6))    


for i in range(a):
    print(((b*(a-i-1)).rjust(a)+b+(b*(a-i-1)).ljust(a)).rjust(a*6))

#Text Wrap

import textwrap

def wrap(string, max_width):
    a=''
    k=0
    for i in range(0,len(string)):
        a+=string[i]
        k+=1
        if k==max_width:
            print(a)
            a=''
            k=0
    print(a)
    return ''

if __name__ == '__main__':
    string, max_width = raw_input(), int(raw_input())
    result = wrap(string, max_width)
    print result

#Designer Door Mat

l=list(map(int,input().split()))
n=l[0]
m=l[1] 
for i in range(1,n,2): 
    print((i*'.|.').center(m,'-'))
print('WELCOME'.center(m,'-')) 
for i in range(n-2,-1,-2): 
    print((i*'.|.').center(m, '-'))

#String Formatting

def print_formatted(number):
    w=len(bin(number)[2:])
    for i in range (1,number+1):
        print(str(i).rjust(w)+' '+str(oct(i)[2:]).rjust(w)+' '+str(hex(i)[2:]).upper().rjust(w)+' '+str(bin(i)[2:]).rjust(w))

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

#Alphabet Rangoli

    def print_rangoli(size):
    
    lettere = 'abcdefghijklmnopqrstuvwxyz'
    for i in range (size-1,0,-1):
        riga=['-']*(4*size-3)
        for j in range(0, size - i):
            riga[2*(size-1+j)] = lettere[i+j]
            riga[2*(size-1-j)] = lettere[i+j]
        print("".join(riga))
    for i in range(0,size):
        riga=['-']*(4*size-3)
        for j in range(0,size-i):
            riga[2*(size-1+j)] = lettere[i+j]
            riga[2*(size-1-j)] = lettere[i+j]
        print("".join(riga))    
        
        

if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)

#Capitalize!


import math
import os
import random
import re
import sys


def solve(s):
    n=s.split(' ')
    for i in range(0,len(n)):
        n[i]=n[i].capitalize()
    s_up=' '.join(n)
    return s_up
        
        


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = raw_input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

#The Minion Game

def minion_game(string):
    s=0
    k=0
    vocali='AEIOU'
    for i in range(len(string)):
        if string[i] in vocali:
            k+=len(string)-i#oppure len(string[i:])
        else:
            s+=len(string)-i
    if s>k:
        print('Stuart',s)
    elif s<k:
        print('Kevin',k)
    else:
        print('Draw')

if __name__ == '__main__':
    s = input()
    minion_game(s)

#Merge the Tools!


def merge_the_tools(string, k):
    t=[]
    for i in range (0,len(string),k):
        t.append(string[i:i+k])
    for i in t:
        u=''
        for j in i:
            if j not in u:
                u+=j
        print (u)

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)

#collections.Counter()

from collections import Counter
n=int(input())
l=list(map(int,input().split()))
nclient=int(input())
p=0
c=Counter(l)
for _ in range (nclient):
    client=list(map(int,input().split()))
    if client[0] in c and c[client[0]]>0:
        p+=client[1]
        c[client[0]]-=1
print(p)

#Introduction to Sets


def average(array):
    s=set(array)
    m=sum(s)/len(s)
    return m
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)

#DefaultDict Tutorial

from collections import defaultdict
A=defaultdict(list)
n,m=map(int,input().split())
for i in range (n):
    A[input()].append(i+1)
for i in range(m):
    e=input()
    if e in A:
        print(' '.join(map(str,A[e])))
    else :
        print (-1)

#Calendar Module

import calendar
a=input().split(' ')
giorno=calendar.weekday(int(a[2]),int(a[0]),int(a[1]))
g=calendar.day_name[giorno]
print(g.upper())

#Exceptions

n=int(input())
for i in range(n):
    try:
        a,b=map(int,input().split())
        print (a//b)
    except ZeroDivisionError as e:
        print('Error Code:',e)
    except ValueError as v:
        print('Error Code:',v)


#Collections.namedtuple()

from collections import namedtuple
n=int(input())
somma=0
l=input().split()
stud=namedtuple('stud',l)
for _ in range (n):
    l1,l2,l3,l4=input().split()
    s=stud(l1,l2,l3,l4)
    somma+=int(s.MARKS)
print(somma/n)


#Time Delta

import math
import os
import random
import re
import sys
from datetime import datetime


def time_delta(t1, t2):
    g1=datetime.strptime(t1,'%a %d %b %Y %H:%M:%S %z')
    g2=datetime.strptime(t2,'%a %d %b %Y %H:%M:%S %z')
    differenza=int(abs((g1-g2).total_seconds()))
    return str(differenza)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


#No Idea!

n,m=map(int,input().split())
arr=list(map(int,input().split()))
A=set(map(int,input().split()))
B=set(map(int,input().split()))
happiness=0
for i in arr:
    if i in A:
        happiness+=1
    if i in B:
        happiness-=1
print(happiness)


#Collections.OrderedDict()

from collections import OrderedDict
n=int(input())
d=OrderedDict()
for _ in range (n):
    i=input().split()
    if len(i)==2:
        if i[0] not in d:
            d[i[0]]=int(i[1])
        else:
            d[i[0]]+=int(i[1])
    else:
        if i[0]+' '+i[1] not in d:
            d[i[0]+' '+i[1]]=int(i[2])
        else:
            d[i[0]+' '+i[1]]+=int(i[2])
for e in d:
    print(e,d[e])   

#Symmetric Difference

n1=input()
a=input().split(' ')
n2=input()
b=input().split(' ')
a1=list(map(int,a))
b1=list(map(int,b))
s1=set(a1)
s2=set(b1)
s3=s1.symmetric_difference(s2)
l=list(s3)
l.sort()
for elem in l  :
    print (elem)


#Set .add()

n=int(input())
s=set()
for i in range(0,n):
    s.add(input())
print(len(s))


#Word Order
from collections import OrderedDict
n=int(input())
d=OrderedDict()
for i in range(n):
    s=input()
    if s not in d:
        d[s]=1
    else:
        d[s]+=1
print (len(d))
for e in d :
    print(d[e],end=' ')
    
#Set .discard(), .remove() & .pop()

n = int(input())
s = set(map(int, input().split()))
comandi=int(input())
for i in range (0,comandi):
    a=input().split(' ')
    if a[0]=='pop':
        s.pop()
    if a[0] =='discard':
        s.discard(int(a[1]))
    if a[0] == 'remove':
        s.remove(int(a[1]))
print(sum(s))


#Collections.deque()
from collections import deque
d=deque()
for _ in range(int(input())):
    metodo,*valore=input().split()
    getattr(d, metodo)(*valore)
for elem in d:
    print(elem,end=' ')


#Company Logo

import math
import os
import random
import re
import sys

from collections import Counter

if __name__ == '__main__':
    s = sorted(input())
    c=Counter(s)
    l=c.most_common(3)
    for e in l :
        print(e[0]+' '+str(e[1]))
            

#Set .union() Operation

n1=int(input())
s1=set(map(int,input().split(' ')))
n2=int(input())
s2=set(map(int,input().split(' ')))
s3=s1.union(s2)
print(len(s3))

#Set .intersection() Operation
n1=input()
s1=set(input().split(' '))
n2=input()
s2=set(input().split(' '))
print(len(s1.intersection(s2)))

#Set .difference() Operation

n1,s1=input(),set(input().split())
n2,s2=input(),set(input().split())
print(len(s1.difference(s2)))

#Set .symmetric_difference() Operation

n1,s1= input(),set(input().split())
n2,s2= input(),set(input().split())
print(len(s1.symmetric_difference(s2)))

#Set Mutations

n,s=input(),set(map(int,input().split()))
for _ in range(int(input())):
    l,i=input().split(),set(map(int,input().split()))
    if l[0]=='update':
        s.update(i)
    if l[0]=='intersection_update':
        s.intersection_update(i)
    if l[0]=='symmetric_difference_update':
        s.symmetric_difference_update(i)
    if l[0]=='difference_update':
        s.difference_update(i)
print(sum(s))


#The Captain's Room

n=int(input())
l=input().split()
s1=set()
s2=set()
for i in l:
    if i not in s1:
        s1.add(i)
    else:
        s2.add(i)
s1.difference_update(s2)
print(list(s1)[0])

#Check Subset

for _ in range(n):
    a,s1=input(),set(map(int,input().split()))
    b,s2=input(),set(map(int,input().split()))
    if s1.intersection(s2)==s1:
        print('True')
    else:
        print('False')

#Check Strict Superset

s=set(map(int,input().split()))
n=int(input())
sup=True
for _ in range(n):
    s1=set(map(int,input().split()))
    for e in s1:
        if e not in s:
            sup=False
            exit
    if s==s1:
        sup=False
        exit
print(sup)


#Zipped!

n,x=map(int,input().split())
l=[]
for i in range (x):
    l.append(list(map(float,input().split())))
for i in (zip(*l)):
    media=sum(i)/len(i)
    print (media)

#Athlete Sort


import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    colonna=[]
    for i in range(n):
        colonna.append(arr[i][k])
    colonna.sort()
    for i in range(n):
        for j in range(n):
            if colonna[i]==arr[j][k]:
                print(*arr[j])
                arr.remove(arr[j])
                break
    

#ginortS

s=input()
p=[]
d=[]
m=[]
M=[]
for i in range(len(s)):
    if s[i].isupper():
        M.append(s[i])
    elif s[i].islower():
        m.append(s[i])
    elif int(s[i])%2==0:
        p.append(s[i])
    else:
        d.append(s[i])
M.sort()
m.sort()
p.sort()
d.sort()
print(''.join(m+M+d+p))

#Detect Floating Point Number


import re
n=int(input())
for i in range (n):
    numero=input()
    if re.match(r"^[-+]?[0-9]*\.[0-9]+$",numero):
        print(True)
    else :
        print(False)

#Map and Lambda Function


cube = lambda x :x**3 
def fibonacci(n):
    l=[0,1]
    if n<2:
        return l[:n]
    for _ in range(n-2):
        l.append(l[-1]+l[-2])
    return l

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))

#Re.split()

regex_pattern = r"[,.]"	


import re
print("\n".join(re.split(regex_pattern, input())))

#Validating phone numbers
import re
n=int(input())
for i in range(n):
    if re.match(r'[789]\d{9}$',input()):
        print('YES')
    else:
        print('NO')


#Validating and Parsing Email Addresses


import re
import email.utils
n=int(input())
for i in range(n):
    e=email.utils.parseaddr(input())
    if re.match(r'[a-z][-a-z._0-9]+@[a-z]+\.[a-z]{1,3}$',e[1]):
        print(email.utils.formataddr(e))

#Hex Color Code

import re
n=int(input())
for _ in range(n):
    color=re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})',input())
    for c in color:
        print(c)

#XML 1 - Find the Score

import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    s=0
    for child in node.iter():
        s+=len(child.attrib)
    return s

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))


#Validating UID

import re
for i in range(int(input())):
    carta=input()
    if re.match(r'^(?!.*(.).*\1)(?=(?:.*[A-Z]){2,})(?=(?:.*\d){3,})[a-zA-Z0-9]{10}$',carta):
        print('Valid')
    else:
        print('Invalid')

#XML2 - Find the Maximum Depth

import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):#bisogna usare la ricorsione perchè per ogni figlio bidogna vedere quanti figli ha a sua volta ,ogni volta aumentare il livello di 1
    global maxdepth#è una variabile globale quindi non la dobbiamo 'ritornare'
    level+=1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)

    
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

#Arrays

import numpy

def arrays(arr):
    a=numpy.array(arr,float)
    return numpy.flip(a)


arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#Shape and Reshape


import numpy
#l=list(map(int,input().split()))
#a=numpy.array(l)
#print (numpy.reshape(a,(3,3)))
l=input().split()
a=numpy.array(l,int)
print (numpy.reshape(a,(3,3)))

#Transpose and Flatten

import numpy
n,m=map(int,input().split())
l=[]
for i in range(n):
    l.append(input().split())
a=numpy.array(l,int)
print (numpy.transpose(a))
print (a.flatten())

#Concatenate

import numpy
n,m,p=map(int,input().split())
l1=[]
l2=[]
for i in range(n):
    l1.append(input().split())
for i in range (m):
    l2.append(input().split())
a=numpy.array(l1,int)
b=numpy.array(l2,int)
print(numpy.concatenate((a,b),axis=0))

#Zeros and Ones

import numpy
a,b,*c=map(int,input().split())
print (numpy.zeros((a,b,*c),dtype=numpy.int))
print (numpy.ones((a,b,*c),dtype=numpy.int))

#Eye and Identity

import numpy
r,c=map(int,input().split())
numpy.set_printoptions(sign=' ')
print (numpy.eye(r,c,k=0))


#Array Mathematics

import numpy
n,m=map(int,input().split())
l1=[]
l2=[]
for _ in range(n):
    l1.append(input().split())
for _ in range(n):
    l2.append(input().split())
a=numpy.array(l1,int)
b=numpy.array(l2,int)
print(a+b)
print(a-b)
print(a*b)
print(a//b)
print(a%b)
print(a**b)

#Floor, Ceil and Rint

import numpy
a=numpy.array(input().split(),float)
numpy.set_printoptions(sign=' ')
print(numpy.floor(a))
print(numpy.ceil(a))
print(numpy.rint(a))

#Sum and Prod

import numpy
n,m=map(int,input().split())
l=[]
for _ in range(n):
    l.append(input().split())
a=numpy.array(l,int)
s= numpy.sum(a,axis=0)
print (numpy.prod(s))

#Min and Max

import numpy
n,m=map(int,input().split())
l=[]
for _ in range(n):
    l.append(input().split())
a=numpy.array(l,int)
m=numpy.min(a,axis=1)
print(numpy.max(m))

#Mean, Var, and Std

import numpy
n,m=map(int,input().split())
l=[]
for _ in range(n):
    l.append(input().split())
a=numpy.array(l,int)
numpy.set_printoptions(legacy='1.13')
print(numpy.mean(a,axis=1))
print(numpy.var(a,0))
print(numpy.std(a))

#Dot and Cross

import numpy
n=int(input())
l1=[]
l2=[]
for _ in range(n):
    l1.append(input().split())
a=numpy.array(l1,int)
for _ in range(n):
    l2.append(input().split())
b=numpy.array(l2,int)
print(numpy.dot(a,b))

#Inner and Outer

import numpy
a=numpy.array(input().split(),int)
b=numpy.array(input().split(),int)
print(numpy.inner(a,b))
print(numpy.outer(a,b))

#Polynomials

import numpy
a=numpy.array(input().split(),float)
val=int(input())
print(numpy.polyval(a,val))

#Linear Algebra

import numpy
n=int(input())
l=[]
for _ in range(n):
    l.append(input().split())
a=numpy.array(l,float)
print(round(numpy.linalg.det(a),2))


#Exercises of the Problem 2 (6/6)
#Birthday Cake Candles

import math
import os
import random
import re
import sys



def birthdayCakeCandles(candles):
    m=max(candles)
    return candles.count(m)
 

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()

#Number Line Jumps

import math
import os
import random
import re
import sys

def kangaroo(x1, v1, x2, v2):
    if x2>x1 and v2>=v1:
        risp='NO'
        return risp
    if x1>x2 and v1>=v2:
        risp='NO'
        return risp
    if (x2-x1)%(v1-v2)==0:
        risp ='YES'
        return risp
    else :
        risp='NO'
        return risp


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    x1V1X2V2 = input().split()

    x1 = int(x1V1X2V2[0])

    v1 = int(x1V1X2V2[1])

    x2 = int(x1V1X2V2[2])

    v2 = int(x1V1X2V2[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Viral Advertising



import math
import os
import random
import re
import sys

def viralAdvertising(n):
    l=[2]
    for i in range(n-1):
        l.append(math.floor(l[-1]*3/2))
    return (sum(l))


if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Recursive Digit Sum


import math
import os
import random
import re
import sys

def superDigit(n, k):
    if len(n)==1:
     return int(n)
    l=list(map(int,n))
    p=sum(l)*k
    return superDigit(str(p),1)
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    nk = input().split()

    n = nk[0]

    k = int(nk[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort - Part 1

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    num=arr[-1]
    for i in range(2,n+1):
        if arr[n-i]>num:
            arr[n-i+1]=arr[n-i]
            print(*arr)
        else:
            arr[n-i+1]=num
            print(*arr)
            break
    if arr[0]>num:
        arr[1]=arr[0]
        arr[0]=num
        print(*arr)


if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

#Insertion Sort - Part 2

import math
import os
import random
import re
import sys

def insertionSort2(n, arr):
    for i in range (1,n):
        for j in range(0,i):
            if arr[i]<arr[j]:
                arr.remove(arr[i])
                arr.insert(j,arr[i])
                print(*arr)
        print(*arr)

if __name__ == '__main__':
    n = int(input())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)





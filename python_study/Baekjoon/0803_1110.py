import sys

A = int(input())

num = A
count = 0
while True:
      
    B = (num//10) + (num%10)
    num = (num%10)*10 + (B%10)
    count+=1
    if(num==A):
        break 

print(count)

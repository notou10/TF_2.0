a = int(input())
 
zero = [1,0,1] #피보나치에서 0 호출 회수 : 1 0 1 1 2 3 5 ... 이거 역시 피보나치
one = [0,1,1]  #피보나치에서 1 호출 회수 : 0 1 1 2 3 5 8 13..이거도 피보나치
 
def fibonacci(num):
    length = len(zero)
    if length <= num:
        for i in range(length,num+1):
            
            zero.append(zero[i-1]+zero[i-2])
            one.append(one[i-1]+one[i-2])
    
    print("%d %d"%(zero[num],one[num]))
  

 
for i in range(a):
    k = int(input())
    fibonacci(k)

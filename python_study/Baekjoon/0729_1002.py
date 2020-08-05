import math

n = int(input())
def gum():
    d = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    
    #원 좌표가 같을 떄 
    if(d==0):
        if(r1==r2):
            print(-1)
        
        if(r1!=r2):
            print(0)
    
     #원 좌표가 다를 떄 
    else:
        if(d == r1+r2 or d == abs(r2-r1)):
            print(1)
        elif(abs(r2-r1) < d < r2+r1):
            print(2)
        else:
            print(0)


for i in range(n):
    x1, y1, r1, x2, y2, r2 = map(int, input().split())
    gum()




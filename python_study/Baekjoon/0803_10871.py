import sys

N, X = map(int, sys.stdin.readline().split())


c = list(map(int, (sys.stdin.readline().split())))

for i in range(N):
    if(X>c[i]):
        print(c[i], end=' ') 
 

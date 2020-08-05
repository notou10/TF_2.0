import sys
List = []

N = int(sys.stdin.readline())

for i in range(N):
    A = int(sys.stdin.readline())
    List.append(A)

# Answer = []
count = 2

while count < max(List):
    for j in range(N):
        temp = List[0] % count
        if((List[j] % count)!= temp):
            break
        
        else:
            if(j==(N-1)):
                print(count, end= ' ')
        
            


    count+=1

# print(Answer)



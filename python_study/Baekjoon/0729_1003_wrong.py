# count_0 = 0
# count_1 = 0

# def fibonacci(N):
#     if(N==0):
#         #print("0")
#         global count_0
#         count_0 += 1
#         return 0

#     elif(N==1):
#        #print("1")
#         global count_1
#         count_1 += 1
#         return 1

#     else:
#         return(fibonacci(N-1)+ fibonacci(N-2))

# T = int(input())

# for i in range(T):
#     N = int(input())
#     fibonacci(N) 
#     print(count_0, count_1)

#     count_0 = 0
#     count_1 = 0

a = int(input())

zero = [1, 0, 1]
one = [0, 1, 1]

def fibonacci(N):
    length = len(zero)
    if(N>=length):
        for i in range(length, N+1):
            zero.append(zero(N-1)+zero(N-2))
            one.append(one(N-1)+one(N-2))

    print("%d %d"%(zero[N],one[N]))
    #print("%d %d"%(zero[N],one[N))

for i in range(a):
    k = int(input())
    fibonacci(k)




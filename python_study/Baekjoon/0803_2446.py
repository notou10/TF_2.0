A = int(input())

for i in range(A):
    print(" "*i + "*" * (2*A-2*i-1))

for j in range(A-1):
    print(" "*(A-(j+2)) + "*" * (2*j+3))
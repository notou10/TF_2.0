num = 0

def adding(a):
    a += 1
    return a

def using_global():
    global num
    num += 1



print(adding(num))
print(num)
using_global()
print(num)
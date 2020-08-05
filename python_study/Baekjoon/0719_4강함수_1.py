
def choice(type, *args):
    if type=="sum":
        sum = 0
        for i in args:
            sum += i
        return sum

    elif type == "mul":
        plus = 0
        for i in args:
            plus *= i
        return plus
    
    elif type == "minus":
        minus = 0
        for i in args:
            minus -= i
        return minus
    else:
        return 0

print(choice('sum', 1, 2, 3))
print(choice('mul', 1, 2, 3))
print(choice('minus', 1, 2, 3))
print(choice('mins', 1, 2, 3))

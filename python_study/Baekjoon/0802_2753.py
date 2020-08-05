Yun = int(input())

if ((Yun % 4 == 0) and ((Yun % 100 != 0)or( Yun % 400 == 0))):
    print(1)

else:
    print(0)
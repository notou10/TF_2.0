
a=3     #a=3을 입력하면 , 3이라는 정수형 객체가 메모리에 생성됨...a는 정수형 객체 3을 가르키는 변수
b=5
a,b = b,a
print(b)


from copy import copy
#동일한 값을 가지면서 서로 다른 객체를 만드는 방법
a = [1, 2, 3]
b = copy(a)

print(a is b)

a[1] = 10
print(a)
print(b)




#동일한 값을 가지면서 서로 같은 객체를 만드는 방법
c = [1, 2, 3]
d = a

print(c is d)
c[1] = 10
print(c)
print(d)

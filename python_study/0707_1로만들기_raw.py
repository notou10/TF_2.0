a=int(input("숫자 입력 : "))
print("입력받는 숫자 : " + str(a))

count=0

while 1:
    if (a % 3) == 0:
        a/=3
        count+=1
    if (a % 3) != 0:
        break

print(a, count)

while 1:
    if (a % 2) == 0:
        a/=2
        count+=1
    if (a % 2) != 0:
        break


print(a, count)



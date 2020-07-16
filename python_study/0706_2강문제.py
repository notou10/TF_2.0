pin = "881120-1068234"
print(pin[:6])
print(pin[7:])

a=[1,3,5,4,2]
a.sort()
a.reverse()
print(a)

b = {"LET", "IT", "GO"}
result = " ".join(b)
print(result)

c = (1, 2, 3)
c = c + (4,)
print(c)

d = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5]
aSet = set(d)
list_d = list(aSet)
print(list_d)

reverse_list_d = list_d.reverse()
print(reverse_list_d)           #왜 결과값 안나오누?


e = {'A':90,'B':80}
result = e.pop('A')
print(e)
print(result)
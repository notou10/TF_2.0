#딕셔너리
#키/value 값        key 값에는 리시트 사용 가능/ value값에는 리스트 사용 불가능
#튜블 ('a', 'b', 'c') : 안의 내용 ex)a[1] 변경 불가능.. 나머지는 리스트와 성격 동일

p = {'a' : 1}
p['b'] = 2
print(p)

del p['b'] #key 값이 'b'인 key:value 쌍 싹제
print(p)


grade = {'A':90, 'B':80, 'C':70}
grade['D'] = 60

print(grade['B'])
#print(grade.keys())
for k in grade.values():
    print(k)

print(grade.items())
print(grade['B'])
print('B'in grade) #해당 키가 딕셔너리에 있는지 조사하기 
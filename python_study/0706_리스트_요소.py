a = [1, 2, [3, 4,[5 ,6 ,7 ]]]

b = "abcde"

c = [1, 2, 4]

# c[1] = 5;
# c[1:2] = [5, 6]
c[1:2] = ['5', '6']         # c[1]시작부터, c[1]끝 까지 값을 '5', '6'으로 바꾼다는 소리. 답 : [1, '5', '6', 4]
# c[1] = ['5', '6']         #c[1]의 요소값을  ['5', '6']로. 답 : [1, ['5', '6'], 4]
print(c)
print(4654646546)
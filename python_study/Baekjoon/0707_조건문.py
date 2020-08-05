# for i in ['A', 'B', 'C']:
#     print(i)
     

# for (a, b) in [(1, 2), (3, 4), (5,6)]:
#     print(a + b)

# for i in range(1, 10):
#     for j in range(1,10):
#         print('%d * %d는 %d  입니다.' %(i, j, i*j), end=" ")          #인자 주는법 : %(i, j, i*j)      , end=" "는 인자 다음에 주는 줄바꿈 방지 
#     print("=============================")

# result = []
# for num in [1, 2, 3, 4]:
#     result.append(num*10)

# print(result)

# a = [5, 6, 7, 8]
# result_2 = [num*4 for num in a]
# print(result_2)

# result_3 = [number*3 for number in a if number%2 ==0]
# print(result_3)

result_4 = [x*y for x in range(1, 10) for y in range(1,10)]
print(result_4)
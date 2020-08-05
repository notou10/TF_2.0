N=int(input())
count=0

while 1:
    if(N>3):
        if N % 3 == 0:
            N /= 3
            count += 1
            continue
        
        elif (N % 2) == 0:
            N /= 2
            count += 1
            continue
        else:
            if N == 1:
                break
            else :
                N -= 1
                count += 1
                continue

    else:
        if N == 1:
            break

        else: 
            N -= 1
            count += 1
            continue
            
print(count)
    



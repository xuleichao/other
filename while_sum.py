#用while循环求解1-100所有奇数的和与偶数的和
i=1
sum_even=0
sum_odd=0
while(i<=100):
    if(i%2==0):
        sum_even+=i
        i+=1
    else:
        sum_odd+=i
        i+=1
print(sum_odd,sum_even)

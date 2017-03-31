#求1到100之间所有奇数和与偶数和
sum_odd=0
sum_even=0
for i in range(1,101,2):
    sum_odd=sum_odd+i
print("所有的奇数和为",sum_odd)

for i in range(2,101,2):
    sum_even=sum_even+i
print('所有的偶数和为',sum_even)

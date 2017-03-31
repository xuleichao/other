fibo1=1
fibo2=1
for i in range(3,21):
    fiboi=fibo1+fibo2
    fibo1=fibo2
    fibo2=fiboi
    print(fiboi,end=',')
    if(i%4==0):
        print('\n')

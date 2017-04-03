a=int(input('请输入一个非零数字,这个数字大于等于-2^31,小于2^31:   '))
q=[]
count=0
if a>0:    
    count=1
else:
    count=0
if (abs(a)>pow(2,31)) or ( a==pow(2,31)):
    print(0)
else:

    while(pow(2,31)>=abs(a)>=10):
        rst=abs(a)%10
        a=int(abs(a)/10)
        q.append(rst)
    q.append(a)
    sum=0
    if count==1:
        for i in range(0,len(q)):
            sum+=(q[i]*pow(10,len(q)-i-1))
    else:
        for i in range(0,len(q)):
            sum+=(q[i]*pow(10,len(q)-i-1))
        sum=-sum
    print(sum)

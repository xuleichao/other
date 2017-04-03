#输出若干行的杨辉三角
row=int(input("你要得到几行杨辉三角？"))

if row<=0:
    print("正整数啊小学没毕业吗？")
elif row==1:
    print(1)
else:
    print(' '*(row-1),end='')
    print(1,end='')
    print('\n')
    print(' '*(row-2),end='')
    print('1','1',end='')
    print('\n')
    tri=[1,1]
    for i in range(3,row+1):
        tri=tri
        tri1=[1,1]
        for j in range(1,i-1):
            tri1.insert(j,(tri[j-1]+tri[j]))
        #print(tri1)
        print(' '*(row-i),end='')
        print(tri1[0],end='')
        for k in range(1,len(tri1)):
            print(' ',end='')
            print(tri1[k],end='')
        print('\n')
        tri=tri1 
            


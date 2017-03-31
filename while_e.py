e=1
fac=1
i=1
while((1/fac)>=pow(10,-6)):
    fac=fac*i
    e=e+(1/fac)
    i+=1
print(e)

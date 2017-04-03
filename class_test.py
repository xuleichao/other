class Person:
    def say_hi(self,name=None):
        self.name=name
        if name==None:
            print('Hello')
        else:
            print('wojiao',self.name)
p=Person()
p.say_hi()
i=1
if i:
    print(i)
x=3
y=5
z=x.__add__(y)
print(z)

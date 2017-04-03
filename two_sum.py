
def two_sum(num,target):
    hash={}
    for i in range(len(num)):
        if target-num[i] in hash:
            return [hash[target-num[i]],i]
        hash[num[i]]=i
    return(hash)



two_sum([1,33,35,23,2,3],5)
                

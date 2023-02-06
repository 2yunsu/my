def factor(n):
    d=2
    list=[]
    while d<=n:
        if n%d==0:
            list.append(d)
            n=n/d
        else:
            d=d+1
    print(list.count(2), list.count(3),list.count(5),list.count(7),list.count(11))

a=int(input())
factor(a)
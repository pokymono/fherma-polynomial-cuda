"""Independent integer checks for direct tensor-sum Montgomery conversion."""
import random
R=1<<32
maximum_dot=32768*255*255
maximum_sum=maximum_dot*sum(1<<(8*i) for i in range(4))
assert maximum_dot<(1<<31)
assert maximum_sum<(1<<55)
assert maximum_sum//R<(1<<23)<(1<<30)
rng=random.Random(20261110)
checks=0
for p in [(1<<30)+1,(1<<30)+3,(1<<31)-1]+[rng.randrange(1<<29,1<<30)*2+1 for _ in range(100)]:
    inverse=pow(p,-1,R)
    values=[0,1,p-1,p,p+1,maximum_sum,p*R-1]
    values += [rng.randrange(maximum_sum+1) for _ in range(100)]
    for value in values:
        # Definition of subtractive REDC, using arbitrary-precision arithmetic.
        multiplier=((value%R)*inverse)%R
        reduced=(value-multiplier*p)//R
        if reduced<0:reduced+=p
        assert 0<=reduced<p
        assert reduced==value*pow(R,-1,p)%p
        twist=rng.randrange(p)
        scaled_twist=twist*pow(R,3,p)%p
        encoded=(reduced*scaled_twist*pow(R,-1,p))%p
        assert encoded==(value%p)*twist*R%p
        checks+=1
print(f'Passed {checks} arbitrary-precision Montgomery conversion checks; byte-sum bound <2^55')

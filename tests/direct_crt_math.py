"""Independent integer checks for the shortened CRT reconstruction proof."""
import hashlib,json,random,sys
from pathlib import Path
import gmpy2
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'tests'))
from reference import modulus
reports=[]
for n,w in [(128,128),(8192,256),(32768,868),(65536,1747)]:
 q=int(modulus(n,w));bound=n*(q-1)**2;p=((1<<31)-2)//(2*n)*(2*n)+1
 primes=[];P=1
 while P<=4*bound:
  if gmpy2.is_prime(p):primes.append(p);P*=p
  p-=2*n
 partial=[P//p for p in primes]
 inv=[pow(m,-1,p) for m,p in zip(partial,primes)]
 reciprocal=[((1<<64)-1)//p for p in primes]
 reduced=[m%q for m in partial];correction=(-P)%q
 rng=random.Random(0xC47+n+w)
 values=[-bound,-bound+1,-q,-q+1,-1,0,1,q-1,q,bound-1,bound]+[rng.randrange(-bound,bound+1) for _ in range(1000)]
 worst=0
 for x in values:
  v=[((x%p)*a)%p for p,a in zip(primes,inv)]
  true_q=(sum(a*b for a,b in zip(v,partial))-x)//P
  approximate=sum((a*b)>>32 for a,b in zip(v,reciprocal))
  exact_q=(approximate+(1<<31))>>32
  assert exact_q==true_q
  short=sum(a*b for a,b in zip(v,reduced))+exact_q*correction
  assert short%q==x%q
  assert short.bit_length()<=32*((w+31)//32+2)
 reports.append({'N':n,'W':w,'fields':len(primes),'tested_signed_integers':len(values),'product_bits':P.bit_length(),'bound_bits':bound.bit_length(),'guard_ratio_floor':P//bound,'fixed_point_error_bound_numerator':3*len(primes),'fixed_point_error_bound_denominator':2**33,'passed':True})
print(json.dumps(reports,indent=2))
(root/'evidence/direct_crt_math.json').write_text(json.dumps({'method':'Independent Python arbitrary-precision CRT, including extremal signed convolution bounds','reports':reports},indent=2)+'\n')

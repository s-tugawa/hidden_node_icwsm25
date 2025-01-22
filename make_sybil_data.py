import sys
import codecs
import random

def each_line(filename):
    with open(filename, encoding='utf-8') as f:
        for line in f:
            yield line.rstrip('\r\n')


input=sys.argv[1]

num_attack_edge=80000
net=input+"-sybil-"+str(num_attack_edge)
begin={}

f = open(net, 'w')
for line in each_line(input):
    (u,v)=line.split()
    begin[int(u)]=1
    begin[int(v)]=1
    print(u,v,sep="\t",file=f)


count=max(begin.keys())+1
print(count)

sybil={}

for line in each_line(input):
    (u,v)=line.split()
    u2=int(u)+count
    v2=int(v)+count
    sybil[u2]=1
    sybil[v2]=1
    print(u2,v2,sep="\t",file=f)

begin_list=list(begin.keys())
sybil_list=list(sybil.keys())

attack_edges={}

while len(attack_edges)<num_attack_edge:
    u=random.choice(begin_list)
    v=random.choice(sybil_list)
    edge=str(u)+"-"+str(v)
    if edge not in attack_edges:
        print(u,v,sep="\t",file=f)
        attack_edges[edge]=1

f.close()


label=input+"-sybil-"+str(num_attack_edge)+".label"
f = open(label, 'w')

for i in sybil_list:
    print(i,file=f)

f.close()

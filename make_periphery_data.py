import sys
import codecs
import random
import networkx as nx
import numpy as np

def each_line(filename):
    with open(filename, encoding='utf-8') as f:
        for line in f:
            yield line.rstrip('\r\n')


gname=sys.argv[1]
frac=0.1

g=nx.read_edgelist(gname,nodetype=int)
num_nodes=len(g.nodes())
num_target=int(num_nodes*frac)
print(num_target)

node_list=[]
p_list=[]
count=0
label=gname+"-kcore.label"
f=open(label,'w')
for u,d in sorted(nx.core_number(g).items(), key=lambda x: x[1]):
    count+=1
    print(u,d)
    print(u,file=f)
    if(count>=num_target):
        break





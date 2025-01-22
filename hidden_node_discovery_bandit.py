import lightgbm as lgb

import pandas as pd
import numpy as np
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import argparse
import time
import networkx as nx
import torch as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import graph_tool.all as gt

from deepgl import DeepGL



def each_line(filename):
    with open(filename, encoding='utf-8') as f:
        for line in f:
            yield line.rstrip('\r\n')


# Logistic regression            
def train_lr(g):
    from sklearn.linear_model import LogisticRegression
    feature=g.ndata['feat']
    label=g.ndata['label']
    train_mask=g.ndata['train_mask']
    test_mask=g.ndata['test_mask']
    train_feature=feature[train_mask]
    train_label=label[train_mask]
    clf = LogisticRegression(random_state=0).fit(train_feature, train_label)
    probs=clf.predict_proba(feature)
    probs=probs[:,1]
#    print(len(probs))
#    print(len(h.nodes()))
    for v in h.nodes():
        if v in sampled:
            continue
        lr_score[v]=probs[v]

def train_lr_deep(g):
    from sklearn.linear_model import LogisticRegression
    feature=g.ndata['feat2']
    label=g.ndata['label']
    train_mask=g.ndata['train_mask']
    test_mask=g.ndata['test_mask']
    train_feature=feature[train_mask]
    train_label=label[train_mask]
    clf = LogisticRegression(random_state=0).fit(train_feature, train_label)
    probs=clf.predict_proba(feature)
    probs=probs[:,1]
#    print(len(probs))
#    print(len(h.nodes()))
    for v in h.nodes():
        if v in sampled:
            continue
        lr_deep_score[v]=probs[v]




#Random Forest

def train_random_forest(g):
    from sklearn.ensemble import RandomForestClassifier
    feature=g.ndata['feat']
    label=g.ndata['label']
    train_mask=g.ndata['train_mask']
    test_mask=g.ndata['test_mask']
    random_forest = RandomForestClassifier()
    train_feature=feature[train_mask]
    train_label=label[train_mask]
    test_label=label[test_mask]
    random_forest.fit(train_feature, train_label)
    test_feature=feature[test_mask]
    pred_label=random_forest.predict(test_feature)
    probs=random_forest.predict_proba(feature)
    probs=probs[:,1]

    for v in h.nodes():
        if v in sampled:
            continue
        rf_score[v]=probs[v]



def train_random_forest_deep(g):
    from sklearn.ensemble import RandomForestClassifier
    feature=g.ndata['feat2']
    label=g.ndata['label']
    train_mask=g.ndata['train_mask']
    test_mask=g.ndata['test_mask']
    random_forest = RandomForestClassifier()
    train_feature=feature[train_mask]
    train_label=label[train_mask]
    test_label=label[test_mask]
    random_forest.fit(train_feature, train_label)
    test_feature=feature[test_mask]
    pred_label=random_forest.predict(test_feature)
    probs=random_forest.predict_proba(feature)
    probs=probs[:,1]

    for v in h.nodes():
        if v in sampled:
            continue
        rf_deep_score[v]=probs[v]        


# LightGBM

def train_lgbm(g):
    feature=g.ndata['feat']
    label=g.ndata['label']
    train_mask=g.ndata['train_mask']
    test_mask=g.ndata['test_mask']
    val_mask=g.ndata['val_mask']
    train_feature=feature[train_mask]
    train_label=list(label[train_mask])
    test_label=list(label[test_mask])
    x_val=feature[val_mask]
    y_val=list(label[val_mask])

    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
    "boosting_type": "gbdt"
    }
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    x_resampled, y_resampled = rus.fit_resample(train_feature, train_label)
    lgb_train = lgb.Dataset(x_resampled, y_resampled,free_raw_data=False)


    lgb_eval = lgb.Dataset(x_val, y_val,free_raw_data=False)
    gbm = lgb.train(param, lgb_train,valid_sets=lgb_eval)


    test_feature=feature[test_mask]
    pred=gbm.predict(test_feature)
    print(len(train_feature))
    print(len(pred))
    print(len(test_label))
    probs=list(gbm.predict(feature))
    pred_label=(pred>0.5) * 1
    report=classification_report(test_label, pred_label,output_dict=True)
    print(report)
#    print(report,file=model_f)
    print(gbm.feature_importance(importance_type = 'gain'))
#    print(gbm.feature_importance(importance_type = 'gain'),file=model_f)

    for v in h.nodes():
        if v in sampled:
            continue
        lgbm_score[v]=probs[v]

def train_lgbm_deep(g):
    feature=g.ndata['feat2']
    label=g.ndata['label']
    train_mask=g.ndata['train_mask']
    test_mask=g.ndata['test_mask']
    val_mask=g.ndata['val_mask']
    train_feature=feature[train_mask]
    train_label=list(label[train_mask])
    test_label=list(label[test_mask])
    x_val=feature[val_mask]
    y_val=list(label[val_mask])

    param = {
        'objective': 'binary',
        'metric': 'binary_logloss',
    "boosting_type": "gbdt"
    }
    from imblearn.under_sampling import RandomUnderSampler
    rus = RandomUnderSampler(random_state=0)
    x_resampled, y_resampled = rus.fit_resample(train_feature, train_label)
    lgb_train = lgb.Dataset(x_resampled, y_resampled,free_raw_data=False)

#    lgb_train = lgb.Dataset(train_feature, train_label,free_raw_data=False)
    lgb_eval = lgb.Dataset(x_val, y_val,free_raw_data=False)
    gbm = lgb.train(param, lgb_train,valid_sets=lgb_eval)


    test_feature=feature[test_mask]
    pred=gbm.predict(test_feature)
    print(len(train_feature))
    print(len(pred))
    print(len(test_label))
    probs=list(gbm.predict(feature))

    pred_label=(pred>0.5) * 1

    report=classification_report(test_label, pred_label,output_dict=True)
    print(report)
#    print(report,file=model_f)
    print(gbm.feature_importance(importance_type = 'gain'))
#    print(gbm.feature_importance(importance_type = 'gain'),file=model_f)


    for v in h.nodes():
        if v in sampled:
            continue
        lgbm_deep_score[v]=probs[v]        


#TN strategy        
def target_neighbor_num(g):
    for v in g.nodes():
        if v not in sampled:
            val=0
            for nbr in g[v]:
                if nbr in sampled:
#                    if T[nbr]==1:
                    if nbr in T:
                        val+=1
            nei_score[v]=val

        



def query(g,h,v):
    sampled[v]=1
    if v in T:
        discovered[v]=1

    for i in g.neighbors(v):
        h.add_edge(v,i)
        seen[i]=1
    for i in g_dir.neighbors(v):
        h_dir.add_edge(v,i)
        seen[i]=1

def query_train(g,h,v):
    sampled[v]=1
    dglh.ndata['train_mask'][v]=True
    dglh.ndata['test_mask'][v]=False
    to_update[v]=1
    if v in T:
        discovered[v]=1

    for i in g.neighbors(v):
        h.add_edge(v,i)
        seen[i]=1
        to_update[i]=1
    for i in g_dir.neighbors(v):
        h_dir.add_edge(v,i)
        seen[i]=1


def calc_base_feature(g,v,g1):
    tmp=[]
    # degree
    tmp.append(g.degree(v))

    # triangles
    val=0
    count=0
    target_tri=0
    non_target_tri=0
    for i in g.neighbors(v):
        count+=1
        if i in discovered:
            val+=1
        for j in g.neighbors(v):
            if g.has_edge(i,j):
                if i in discovered:
                    if j in discovered:
                        target_tri+=1
                if i not in discovered:
                    if j not in discovered:
                        non_target_tri+=1
                        
    tmp.append(val)
    g1.vp.target_nei[v]=val
    tmp.append(target_tri)
    g1.vp.target_tri[v]=target_tri
    tmp.append(non_target_tri)
    g1.vp.non_target_tri[v]=non_target_tri
    if count >0:
        val=float(val)/float(count)

    tmp.append(val)
    g1.vp.target_frac[v]=val
    tri=nx.triangles(g, v)
    tmp.append(tri)
    g1.vp.tri[v]=tri
    if(tri>0):
        target_tri=float(target_tri)/float(tri)
        non_target_tri=float(non_target_tri)/float(tri)
    tmp.append(target_tri)
    g1.vp.target_tri_frac[v]=target_tri
    tmp.append(non_target_tri)
    g1.vp.non_target_tri_frac[v]=non_target_tri
    two_hop_count=0
    two_hop_target=0
    two_hop_dict={}
    for i in g.neighbors(v):
        for j in g.neighbors(i):
            if j in discovered:
                two_hop_dict[j]=1
            else:
                two_hop_dict[j]=0

    two_hop_count=len(two_hop_dict)
    two_hop_target=0
    for i in two_hop_dict.values():
        two_hop_target += i
    if two_hop_count>0:
        two_hop_target=two_hop_target/two_hop_count
    tmp.append(two_hop_target)
    g1.vp.two_hop_target[v]=two_hop_target
    return(tmp)


def init(g,g_dir):
    labels=[]
    train_mask=[]
    test_mask=[]
    val_mask=[]


    for v in sorted(list(g.nodes())):
        u=g1.add_vertex()
        if u in sampled:
            if v in T:
                g1.vp.target[u]=1
                g1.vp.non_target[u]=0
                g1.vp.unknown[u]=0
            else:
                g1.vp.target[u]=0
                g1.vp.non_target[u]=1
                g1.vp.unknown[u]=0
        else:
            g1.vp.target[u]=0
            g1.vp.non_target[u]=0
            g1.vp.unknown[u]=1

    for v in g1.vertices():
        for u in g_dir.neighbors(v):
            g1.add_edge(v,u)
    print(g1)



    X1=[]
    X2=[]
    deepgl=0
    for v in sorted(list(g.nodes())):
        train_mask.append(False)
        val_mask.append(False)
        test_mask.append(False)
        if v not in T:
            labels.append(0)
        else:
            labels.append(1)


    for v in sorted(list(g.nodes())):
        X1.append(calc_base_feature(g,v,g1))
        

    import configparser

    inifile = configparser.ConfigParser()
    inifile.read(deepgl_setting, 'UTF-8')
    base=eval(inifile.get('settings', 'base_feat'))
    nbr_type=eval(inifile.get('settings', 'nbr_type'))
    deepgl = DeepGL(base_feat_defs=base,
                    ego_dist=int(inifile.get('settings', 'ego_dist')),
                        #              nbr_types=['all','in','out'],
                    nbr_types=nbr_type,
                    lambda_value=float(inifile.get('settings', 'lambda')),
                    transform_method='log_binning')
    X2 = deepgl.fit_transform(g1)
    # for nth_layer_feat_def in deepgl.feat_defs:
    #     print(nth_layer_feat_def,file=model_f)





    
    src=[]
    dst=[]
    for e in g.edges:
        src.append(e[0])
        dst.append(e[1])
        src.append(e[1])
        dst.append(e[0])
    
    edges_src = th.IntTensor(src)
    edges_dst = th.IntTensor(dst)
    dglg=dgl.graph((edges_src, edges_dst), num_nodes=len(g.nodes))
    dglg = dgl.add_self_loop(dglg)
    labels=th.tensor(labels)
    features=th.tensor(X1,dtype=th.float32)
    features2=th.tensor(X2,dtype=th.float32)
    train_mask=th.tensor(train_mask)
    test_mask=th.tensor(test_mask)
    val_mask=th.tensor(val_mask)
    dglg.ndata['label']=labels
    dglg.ndata['feat']=features
    dglg.ndata['feat2']=features2
    dglg.ndata['train_mask']=train_mask
    dglg.ndata['test_mask']=test_mask
    dglg.ndata['val_mask']=val_mask
    return dglg,deepgl



def prep_train(g,h):
    X1=[]

    if( deepgl_setting !="base"):
        for v in sorted(list(h.nodes())):
            if v in to_update:
                calc_base_feature(h,v,g1)
#            u=g1.add_vertex()
            if v in sampled:
                if v in T:
                    g1.vp.target[v]=1
                    g1.vp.non_target[v]=0
                    g1.vp.unknown[v]=0
                else:
                    g1.vp.target[v]=0
                    g1.vp.non_target[v]=1
                    g1.vp.unknown[v]=0
            else:
                g1.vp.target[v]=0
                g1.vp.non_target[v]=0
                g1.vp.unknown[v]=1
        for v in g1.vertices():
            for u in h_dir[v]:
                g1.add_edge(v,u)
        print(g1)
        X1 = deepgl.transform(g1)


    count=0
    for v in h.nodes():
        if g.ndata['train_mask'][v]:
            count+=1
            if (count % 10)==0:
                g.ndata['train_mask'][v]=False
                g.ndata['val_mask'][v]=True
        else:
            if v not in sampled:
                g.ndata['test_mask'][v]=True

                feats=calc_base_feature(h,v,g1)
                g.ndata['feat'][v]=th.tensor(feats,dtype=th.float32)
                g.ndata['feat2'][v]=th.tensor(X1[v],dtype=th.float32)
#                    dglh.ndata['feat'][v]=th.tensor(X1[v],dtype=th.float32)

                   
                     

    return g,h                      

#setting
gname=sys.argv[1]
label_file=sys.argv[2]
num_train=int(sys.argv[3])
num_init=int(sys.argv[4])
batch_size=int(sys.argv[5])
budget=int(sys.argv[6])
deepgl_setting=sys.argv[7]
algorithm='bandit'
algo_list=sys.argv[8:]

print(gname,label_file,num_train,num_init,batch_size,budget,deepgl_setting,algo_list)

#output files
basename = os.path.basename(gname)
labelname=os.path.basename(label_file)
outfile="res_unknown-"+basename+"-"+labelname+"-"+str(num_train)+"-"+str(num_init)+"-"+str(batch_size)+"-"+str(budget)+"-"+str(deepgl_setting)+"-"+"-".join(algo_list)
f=open(outfile,'a')
# model_file_name="model-"+basename+"-"+labelname+"-"+str(num_train)+"-"+str(num_init)+"-"+str(batch_size)+"-"+str(budget)+"-"+str(deepgl_setting)+"-"+"-".join(algo_list)
# model_f=open(model_file_name,'w')

                      
g=nx.read_edgelist(gname,nodetype=int)
g_dir=nx.read_edgelist(gname,nodetype=int,create_using=nx.DiGraph)



node_list=list(g.nodes())


num_sample=num_train+num_init
num_nodes=len(g.nodes())
num_target=num_nodes*0.2
T={}

# Determine target node label
for line in each_line(label_file):
    T[int(line)]=1


        
sampled={} #queried nodes
discovered={} #dicovered targets
seen={} #observed nodes


h=nx.Graph()
h_dir=nx.DiGraph()

for v in g.nodes():
    h.add_node(v)
    h_dir.add_node(v)


# obtain initial graph via random walk

candidates=list(g.nodes())
v=candidates.pop(np.random.randint(len(candidates)))
query(g,h,v)
count=1

while count<num_init:
    next_list=list(h.neighbors(v))
#    next_list=[]
    if len(next_list) ==0:
        v=node_list[np.random.randint(len(node_list))]
        count+=1
        query(g,h,v)
        continue
    u=next_list[np.random.randint(len(next_list))]
    if u not in sampled:
        count+=1
        query(g,h,u)
    v=u


g1 = gt.Graph()
g1.vp["target"] = g1.new_vertex_property("int")  
g1.vp["non_target"] = g1.new_vertex_property("int")  
g1.vp["unknown"] = g1.new_vertex_property("int")
g1.vp["target_nei"] = g1.new_vertex_property("double")  
g1.vp["target_tri"] = g1.new_vertex_property("double")  
g1.vp["non_target_tri"] = g1.new_vertex_property("double")
g1.vp["target_frac"] = g1.new_vertex_property("double")
g1.vp["tri"] = g1.new_vertex_property("double")
g1.vp["target_tri_frac"] = g1.new_vertex_property("double")
g1.vp["non_target_tri_frac"] = g1.new_vertex_property("double")
g1.vp["two_hop_target"] = g1.new_vertex_property("double")
print(g1)
#print(g1.list_properties())

dglh,deepgl=init(h,h_dir)
to_update={}
# #obtain training data

while count<num_sample:
    next_list=list(h.neighbors(v))
#    next_list=[]
    if len(next_list) ==0:
        v=node_list[np.random.randint(len(node_list))]
        count+=1
        query_train(g,h,v)
        continue
    u=next_list[np.random.randint(len(next_list))]
    if u not in sampled:
        count+=1
        query_train(g,h,u)
    v=u




nei_score={}
rf_deep_score={}
rf_score={}
lgbm_score={}
lgbm_deep_score={}
deg_score={}

lr_score={}
lr_deep_score={}
    




alpha={}
beta={}
alpha_tmp={}
beta_tmp={}



for alg in algo_list:
    alpha[alg]=1
    beta[alg]=1
    alpha_tmp[alg]=0
    beta_tmp[alg]=0
        

C=5
#batch_size=100
print(len(sampled),len(discovered),len(seen),file=f)
print(len(sampled),len(discovered),len(seen))

model_available=0
if (len(discovered)>10):
    model_available=1
    if ("lgbm_deep" in algo_list) or ("lgbm" in algo_list) or ("rf" in algo_list) or ("rf_deep" in algo_list) or ("lr" in algo_list) or ("lr_deep" in algo_list):
        dglh,h=prep_train(dglh,h)
        to_update={}

        if "rf" in algo_list:
            train_random_forest(dglh)
        if "lgbm" in algo_list:
            train_lgbm(dglh)
        if "rf_deep" in algo_list:
            train_random_forest_deep(dglh)
        if "lgbm_deep" in algo_list:
            train_lgbm_deep(dglh)
        if "lr" in algo_list:
            train_lr(dglh)
        if "lr_deep" in algo_list:
            train_lr_deep(dglh)

if "nei" in algo_list:
    target_neighbor_num(h)
if "degree" in algo_list:
    deg=dict(nx.degree(h))
    for v in sorted(list(h.nodes())):
        if v not in sampled:
            deg_score[v]=deg[v]

while count<budget:
    if((count%batch_size)==0):
        if(len(discovered)>10):
            model_available=1

            if ("lgbm_deep" in algo_list) or ("lgbm" in algo_list) or ("rf_deep" in algo_list) or ("rf" in algo_list) or ("lr" in algo_list) or ("lr_deep" in algo_list):
                dglh,h=prep_train(dglh,h)

                if "rf" in algo_list:
                    train_random_forest(dglh)
                if "rf_deep" in algo_list:
                    train_random_forest_deep(dglh)
                if "lgbm" in algo_list:
                    train_lgbm(dglh)
                if "lgbm_deep" in algo_list:
                    train_lgbm_deep(dglh)
                if "lr" in algo_list:
                    train_lr(dglh)
                if "lr_deep" in algo_list:
                    train_lr_deep(dglh)

        if "nei" in algo_list:
            target_neighbor_num(h)
        if "degree" in algo_list:
            deg=dict(nx.degree(h))
            for v in sorted(list(h.nodes())):
                if v not in sampled:
                    deg_score[v]=deg[v]




    count+=1
    max_score=-99999999
    max_alg=0
    arm=0
    if (algorithm != "bandit"):
        arm=algorithm
    else:  #select arm with D3TS
        for alg in algo_list:
            print(alg,alpha[alg],beta[alg])
            prob_val=np.random.beta(alpha[alg],beta[alg])
            print(prob_val)
            if prob_val >=max_score:
                max_score=prob_val
                max_alg=alg

        arm=max_alg

    orig_arm=arm
    if(model_available==0):
        arm="rand"



    if(arm == 'nei'):
#        max_k=target_neighbor_num(h)
        max_k = max(nei_score, key=nei_score.get)        
    elif(arm=='rf'):
#        train_random_forest(h)
        max_k = max(rf_score, key=rf_score.get)
    elif(arm=='rf_deep'):
        max_k = max(rf_deep_score, key=rf_deep_score.get)        
    elif(arm=='lgbm'):
        max_k = max(lgbm_score, key=lgbm_score.get)
    elif(arm=='lgbm_deep'):
        max_k = max(lgbm_deep_score, key=lgbm_deep_score.get)
    elif(arm=='lr'):
        max_k = max(lr_score, key=lr_score.get)
    elif(arm=='lr_deep'):
        max_k = max(lr_score, key=lr_deep_score.get)        
    elif(arm=='degree'):
#        train_random_forest(h)
        max_k = max(deg_score, key=deg_score.get)        

    if(arm=='rand'):
        candidates=list(h.nodes())
        max_k=candidates.pop(np.random.randint(len(candidates)))
        arm=orig_arm




    if max_k in rf_score:
        del rf_score[max_k]
    if max_k in rf_deep_score:
        del rf_deep_score[max_k]
    if max_k in nei_score:
        del nei_score[max_k]
    if max_k in lgbm_score:
        del lgbm_score[max_k]
    if max_k in lgbm_deep_score:
        del lgbm_deep_score[max_k]
    if max_k in deg_score:
        del deg_score[max_k]
    if max_k in lr_score:
        del lr_score[max_k]
    if max_k in lr_deep_score:
        del lr_deep_score[max_k]


    if max_k not in sampled:
        query_train(g,h,max_k)
#    else:
#        print(max_k)
    r=0
    if max_k in T:
        r+=1


# update arm reward        
    alpha[arm]+=r
    beta[arm]+=(1-r)
    if(alpha[arm]+beta[arm]>=C):
        alpha[arm]=alpha[arm]*(C/(C+1))
        beta[arm]=beta[arm]*(C/(C+1))

    print(len(sampled),len(discovered),len(seen),file=f)
    print(arm)
    print(len(sampled),len(discovered),len(seen))

f.close()
#model_f.close()

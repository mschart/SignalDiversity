import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import colorcet as cc
import os
from collections import Counter
import datashader as ds

#path to data
path_dat='/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/lucia'
s = os.listdir(path_dat + '/data')#+'/'+folder+'/'+'original')
co = Counter([x.split('_')[0] for x in s])
subjects = list(co.keys())
co2 = Counter([x.split('_')[1].split('.')[0] for x in s])
conditions = list(co2.keys())
c_ols = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
         '#911eb4', '#46f0f0', '#f032e6', '#bcf60c']
cond_cols = dict(zip(conditions,c_ols))


de = [1,4]# in Hz
th = [4,8]
al = [8,13]
be = [13,30] 



specd = {f'PSpec{k}': str([de, th, al, be][k])+' Hz' for k in range(4)}
       

def plot_metric(measure):

    # measure in ['LZs', PSpec0-4], alpha is PSpec2

    try:
        sub_res = np.load(path_dat + f'/results/{measure}.npy',
                          allow_pickle=True).flat[0]
    
    except:    
        sub_res = {}
        
        for subject in subjects:
            means = []
            stderrs = []
            for condition in conditions:
                d = np.load(path_dat + 
                            f'/results/{subject}/{measure}_{condition}.npy')
                means.append(np.mean(d[:,:-2]))               
                stderrs.append(np.std(d[:,:-2])/(d.shape[0]**0.5))               
            sub_res[subject] = [means,stderrs]
        np.save(path_dat +f'/results/{measure}.npy',sub_res, 
                allow_pickle=True)

    # plot scatter with error y bars
    fig, ax = plt.subplots(figsize=(15,8))
    
    k = 0
    for condition in conditions:
        a = np.arange(len(subjects))
        b = [sub_res[sub][0][k] for sub in sub_res]
        errs = [sub_res[sub][1][k] for sub in sub_res]
        ax.scatter(a,b, label = condition, c = cond_cols[condition])    
        ax.errorbar(a, b, yerr=errs, linewidth=0.5, c = cond_cols[condition])
        k += 1
        
    ax.set_xlabel('subjects')
    ax.set_xticks(np.arange(len(subjects)))
    ax.set_xticklabels(subjects) 
    ax.set_ylabel(measure if measure == 'LZs' else specd[measure])
    ax.legend()
    fig.tight_layout()
    fig.savefig('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
               f'lucia/plots_m/{measure}.png')    
        

def scatter_pca(subject):

               
    wsc = np.load('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
                  f'lucia/results/pca/{subject}.npy')

    assert wsc.shape[1] % len(conditions) == 0, 'something off?'
    reg_len = wsc.shape[1]//len(conditions)
    

    
    
            
    cols = []

    for c in conditions:
        cols.append([cond_cols[c]] * reg_len)        
    
    cols = list(np.concatenate(cols))
    fig, ax = plt.subplots()        
    ax.scatter(wsc[1,:], wsc[2,:], c = cols, s = 0.1)


#    # datashader image                  
#    df = pd.DataFrame(data=wsc[-2:].T, columns=["x", "y"])  # create a DF from array

#    cvs = ds.Canvas(plot_width=5000, plot_height=5000)
#    agg = cvs.points(df, 'x', 'y')
#    img = ds.tf.set_background(ds.tf.shade(agg, how="log", cmap=list(cols)),
#                               "white").to_pil()
                               
                               
    #ax.imshow(img)                          
    ax.set_title(subject)
    
    legend_elements = [Line2D([0], [0], color=c_ols[k], lw=4, 
                    label=conditions[k]) for k in range(len(conditions))]
                    
    ax.legend(handles=legend_elements, loc='upper right')                           
                               
    fig.savefig('/mnt/8cfe1683-d974-40f3-a20b-b217cad4722a/'
               f'lucia/plots_m/pca/{subject}.png')
               
    #plt.close()               
               
               
    ## check histograms on first PC
    


               
               
                                              
                               

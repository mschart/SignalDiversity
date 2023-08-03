from one.api import ONE
from brainbox.io.spikeglx import spikeglx

import numpy as np
from pathlib import Path
import random
from scipy import signal
from scipy.signal import hilbert
import time

import matplotlib.pyplot as plt

'''
Exploratory LZ analysis for LF signals of the internationalbrainlaboratory
'''


one = ONE()
pids = list(one.search_insertions(project='psychedelics',
            query_type='remote'))


def cpr(string):
    '''
    Lempel-Ziv-Welch compression of binary input string, 
    Input: e.g. string='0010101'. 
    Output: the size of the dictionary of binary words.
    '''
    
    d = {'0':'0','1':'1'} 
    w = ''
    i = 1
    for c in string:
        wc = w + c
        if wc in d:
            w = wc
        else:
            d[wc] = wc
            w = c
        i+=1
    return len(d) 
    

def LZs(x):
 
    '''
    Lempel ziv complexity of single continuous lfp/EEG timeseries
    '''
    
    co = len(x)
    x = signal.detrend((x-np.mean(x))/np.std(x), axis=0)
    r = abs(hilbert(x))
    
    # turn into binary string
    y = np.digitize(r,np.linspace(min(r),np.mean(r),2))
    s = ''.join(map(str, y-1))

    M = list(s)
    random.shuffle(M)
    w = ''.join(map(str, M))
 
    return cpr(s)/float(cpr(w))
    

'''
batch processing
'''


def LZ_all(pids=pids):

    res = {}
    for pid in pids:
    
        time00 = time.perf_counter()
        
        eid, probe = one.pid2eid(pid) 

        try:    
            # Download LFP data
            lfp_paths, _ = one.load_datasets(eid, 
                            download_only=True, datasets=[
                        '_spikeglx_ephysData_g*_t0.imec*.lf.cbin', 
                        '_spikeglx_ephysData_g*_t0.imec*.lf.meta',
                        '_spikeglx_ephysData_g*_t0.imec*.lf.ch'], 
                        collections=[f'raw_ephys_data/{probe}'] * 3)
        except:
            print('no lfp for pid:', pid)
            continue
                    
        lfp_path = lfp_paths[0]
        sr = spikeglx.Reader(lfp_path)

        print(pid)
        print('length [min]:', (sr.shape[0]//sr.fs)/60)
        print('#chans:', sr.shape[1])

        # compute LZ for each channel in non-overlapping segments
        # discard last 5 channels
        seg_l = 10  # seg length [sec]
        l = int(sr.fs * seg_l)  # seg length [samples]
        n_segs = sr.shape[0] // l  # number of segments
        seg_start_times = np.arange(n_segs) * seg_l

        downsamp = 10  # downsample signal (10 if down to 250 Hz)
        discard = 5  # discard last channels
               
        d = np.zeros((sr.shape[1] - discard, n_segs))
        for i in range(n_segs):

            s = sr.read(nsel=slice(i*l, (i+1)*l, downsamp),
                             csel=slice(0, -discard, None))[0].T

            for chan in range(s.shape[0]):
                d[chan,i] = LZs(s[chan])
              
            if i%100 == 0:
                print(i, f'of {n_segs} segs done')  
                    
        res[pid] = [d, seg_start_times]         
        
        time1 = time.perf_counter()
        td = (time1 - time00)//60           
        print(pid, f'completed in {td} min')
 
    return res 
            

'''
########
plotting
########
'''

def plot_segs():

    '''
    input: 
        d: (chans x time bins) LZs scores
        t: time bin starting times
    output:
        line plot per channel for this pid
    '''    
    res = np.load('ibl_serotonin_LZ.npy',allow_pickle=True).flat[0]
    
    for pid in res:
        fig, ax = plt.subplots()
        ax.imshow(res[pid][0])    
        ax.set_title(f'pid = {pid}')
        ax.set_ylabel('channels')
        ax.set_xlabel('time in 10 sec')

        fig.savefig(f'/home/mic/ibl-serotonin-res/{pid}.png')
        plt.close()






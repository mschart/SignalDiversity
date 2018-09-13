from numpy import *
from numpy.linalg import *
from scipy import signal
from scipy.signal import (butter,lfilter,hilbert,resample)
from scipy.stats import ranksums
from scipy.io import savemat, loadmat, whosmat
from pylab import *
import os as os
from os import listdir
from os.path import isfile, join
from random import sample
 
 
 
def Pre2(X):
 '''
 Linear-detrend and subtract mean
 '''
 ro,co=shape(X)
 Z=zeros((ro,co))
 for i in range(ro): #maybe divide by std?
  Z[i,:]=signal.detrend((X[i,:]-mean(X[i,:]))/std(X[i,:]), axis=0)
 return Z
 
##############
'''
PSpec
'''
##############
 
def PSpec(X):
 '''
 X: multidimensional time series, ch x obs
 fs: sampling rate in Hz
 '''
 
 
 def find_closest(A, target):
   '''
   helper function
   '''
   #A must be sorted
   idx = A.searchsorted(target)
   idx = np.clip(idx, 1, len(A)-1)
   left = A[idx-1]
   right = A[idx]
   idx -= target - left < right - target
   return idx
 
 
 fs=250
 
 de=[1,4]# in Hz
 th=[4,8]
 al=[8,13]
 be=[13,30]
# ga=[30,60]
# hga=[60,120]
 
 F=[de,th,al,be]#,ga,hga]
 
 ro,co=shape(X)
 Q=[]
 
 for i in range(ro):
 
  v=X[i]
  co=len(v)  
  N = co # Number of samplepoints  
  T = 1.0 / fs # sample spacing (denominator in Hz)
  y = v
  yf = fft(y)
  xf = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
  yff=2.0/N * np.abs(yf[0:int(N/2)])
  bands=zeros(len(F))
  for i in range(len(F)):
   bands[i]=sum(yff[find_closest(xf, F[i][0]):find_closest(xf, F[i][1])])
  bands=bands/sum(bands)
  Q.append(bands)
 return Q
 
 
#############
'''
frequency filter
'''
#############
 
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a
 
def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y
 
def butter_highpass(lowcut, fs, order):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a
 
def butter_highpass_filter(data, lowcut, fs, order):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
 
def notch_iir(fs,f0,data):
    '''
    fs: Sample frequency (Hz)
    f0: Frequency to be removed from signal (Hz)
    '''
 
    Q = 10.# 30.0  # Quality factor
    w0 = float(f0)/(fs/2)  # Normalized Frequency
    b, a = signal.iirnotch(w0, Q)
    return lfilter(b, a, data)
 
 
##########
'''
LZc - Lempel-Ziv Complexity, column-by-column concatenation
 
X is continuous multidimensional time series, channels x observations
'''
##########
 
def cpr(string):
 '''
 Lempel-Ziv-Welch compression of binary input string, e.g. string='0010101'. It outputs the size of the dictionary of binary words.
 '''
 d={}
 w = ''
 for c in string:
  wc = w + c
  if wc in d:
   w = wc
  else:
   d[wc]=wc   
   w = c
 return len(d)
 
def str_col(X):
 '''
 Input: Continuous multidimensional time series
 Output: One string being the binarized input matrix concatenated comlumn-by-column
 '''
 ro,co=shape(X)
 TH=zeros(ro)
 M=zeros((ro,co))
 for i in range(ro):
  M[i,:]=abs(hilbert(X[i,:]))
  TH[i]=mean(M[i,:])
 
 s=''
 for j in range(co):
  for i in range(ro):
   if M[i,j]>TH[i]:
    s+='1'
   else:
    s+='0'
 
 return s
 
 
def LZc(X):
 '''
 Compute LZc and use shuffled result as normalization
 '''
 X=Pre2(X)
 SC=str_col(X)
 M=list(SC)
 shuffle(M)
 w=''
 for i in range(len(M)):
  w+=M[i]
 return cpr(SC)/float(cpr(w))
 
 
def LZs(x):
 
 '''
 Lempel ziv complexity of single timeseries
 '''
 
 #this differes from Sitt et al as they use 32 bins, not just 2.
 co=len(x)
 x=signal.detrend((x-mean(x))/std(x), axis=0)
 s=''
 r=abs(hilbert(x))
 th=mean(r)
 
 for j in range(co):
  if r[j]>th:
   s+='1'
  else:
   s+='0'
 
 M=list(s)
 shuffle(M)
 w=''
 for i in range(len(M)):
  w+=M[i]
 
 return cpr(s)/float(cpr(w))
 
############
'''
Functions for automatic renaming/copying files
'''
############
 
 
def list_files(startpath):
    '''
    get file tree printed for given path
    '''
 
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))
 
 
#def renameS(expName):
# path='/home/mic/data/kinect/'
 
 
# #expName='MH004_post_cocaine_40_60'
# oldD=path+expName
# for filename in os.listdir(oldD):
#  if filename.startswith('RID'):
#   if 'RID__' in filename:
#    os.remove(oldD+'/RID__.int16binary')
 
#   newD=path+'/all/%s_t%s' %(expName,filename.split('_')[1])
#   if not os.path.exists(newD):   
#    os.makedirs(newD)
#   newFN='RID.int16binary'
#   os.rename('%s/%s' %(oldD,filename),'%s/%s' %(newD,newFN))  
#   shutil.copy('%s/polyroi.csv' %oldD,'%s/polyroi.csv' %newD)
#   shutil.copy('%s/BG.mat' %oldD,'%s/BG.mat' %newD)
 
def getsubsMEG(cond):
 
 if cond=='PSIL':
 
  return ['010312_1',
   '020311_50',
   '020312_50',
   '020312_51',
   '021211_51',
   '030507_1', 
   '050312_1',
   '081211_51',
   '160312_2', 
   '190311_1',
   '231109_1', 
   '240107_6',
   '241111_2',
   '271108_3']
 
 if cond=='KET':
  return ['240107_6', '310713_51' ,'270407_1', '210813_51', '210813_52' ,'280813_51', '280813_52', '110913_51', '021013_51' ,'301112_51' ,'091013_51', '161013_51' ,'161013_52', '231013_51', '221112_50', '171212_1' ,'181113_1', '040806_1', '041213_1' ]
 if cond=='LSD':
  return ['230911_1',
        '010514_1',
        '140514_1',
        '140514_2',
        '290514_2',
        '270613_1',
        '260614_1',
        '090714_2',
        '310714_1', 
        '041213_1',
        '070814_2', 
        '200814_1',
        '010813_4',
        '040914_1',
        '040914_2']
  
 else:
  print('typo?')
 
 
#def MakeOriginal_MEG(folder):
 
# #folders=['KET','LSD','PSIL','DMT','FLICK']
#  
# version='original'
 
# #path to data
# path_dat='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/EEG_DMT'
 
# path_dat2='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/rev_data'
 
# 
# subjects=getsubsMEG(folder)
# conditions=['PLA',folder] 
 
# for condition in conditions:
#  for subject in subjects:
#   d=loadmat(path_dat+'/'+folder+'/%s_%s.mat' %(subject,condition))['dat'][0]
#   d0=concatenate(d,axis=1)
#   chs,obs=shape(d0)
#   d0=resample(d0,(obs*250)/600,axis=1) # resample from 600Hz to 250Hz
#   d0=butter_lowpass_filter(d, 30, 250, order=5)   
#   d0=Pre2(d0)
 
 
#   if not os.path.exists(path_dat2+'/'+folder+'/'+'%s/%s' %(version,subject)):
#    os.makedirs(path_dat2+'/'+folder+'/'+'%s/%s' %(version,subject))
#   
#   save(path_dat2+'/'+folder+'/'+'%s/%s/%s.npy' %(version,subject,condition),d0)
#   print folder, condition, subject  
 
 
def MakeOriginal_DMT(folder):
 '''
 concatenation then downsampling to 250Hz
 '''
 
 #folders=['KET','LSD','PSIL','DMT','FLICK']
   
 version='original'
 
 #paths to data
 path_dat='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/EEG_DMT'
 path_dat2='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/rev_data'
 
 subjects=os.listdir(path_dat)
 conditions=['PLA',folder] 
 
 for condition in conditions:
  for subject in subjects:
   d=loadmat('%s/%s/%s/trialsmxmshort3s.mat' %(path_dat,subject,condition))['trialsmxmshort3s'][0]
   l=[]
   for i in range(2,7):
    l.append(d[i][0][0]['trial'][0])
 
   d0=concatenate(l)
   d0=concatenate(d0,axis=1)
   d0=resample(d0,(obs*250)/1000,axis=1) # resample from 1000Hz to 250Hz
   d0=butter_lowpass_filter(d, 30, 250, order=5) #low-pass    
   d0=Pre2(d0)
 
   chs,obs=shape(d0)
    
   if not os.path.exists(path_dat2+'/'+folder+'/'+'%s/%s' %(version,subject)):
    os.makedirs(path_dat2+'/'+folder+'/'+'%s/%s' %(version,subject))
    
   save(path_dat2+'/'+folder+'/'+'%s/%s/%s.npy' %(version,subject,condition),d0)
   print(folder, condition, subject)  
 
 
def Band_pass_Pre(folder):
 '''
 bring all data to bandpass 1-30Hz, further apply Pre2
 '''
 
 #folders=['KET','LSD','PSIL','DMT','FLICK']
   
 version='original'
 
 #paths to data
 path_dat2='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/rev_data'
 
 subjects=os.listdir(path_dat2+'/'+folder+'/'+'original')
 #conditions=['dark','flick10'] 
 #cD={'dark':'PLA','flick10':'FLICK'}
 conditions=['PLA',folder]
 
 for condition in conditions:
  for subject in subjects:
   #d=loadmat('/home/mic/'+folder+'/'+'%s/%s/%s.mat' %(version,subject,condition))['data']
   #d0=butter_lowpass_filter(d, 30, 250, order=5) 
   d=load(path_dat2+'/%s/%s/%s/%s.npy' %(folder,'original',subject,condition))
   d=butter_lowpass_filter(d, 30, 250, order=5) 
   d=Pre2(d)
    
   if not os.path.exists(path_dat2+'/'+folder+'/'+'%s/%s' %(version,subject)):
    os.makedirs(path_dat2+'/'+folder+'/'+'%s/%s' %(version,subject))
 
   save(path_dat2+'/'+folder+'/'+'%s/%s/%s.npy' %(version,subject,condition),d)
   print(folder, condition, subject) 
 
 
def CheckLengths():
 
 folders=['KET','LSD','PSIL','DMT','FLICK']
 path_dat='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/rev_data'
 
 
 for folder in folders: 
  l=[]
  conditions=['PLA',folder]
  subjects=os.listdir(path_dat+'/'+folder+'/'+'original')
  for subject in subjects:
   for condition in conditions:
    d=load(path_dat+'/%s/%s/%s/%s.npy' %(folder,'original',subject,condition))
    chs,obs=shape(d)
    l.append(obs)
  print('\hline')
  print(folder,' & ',len(subjects),' & ',chs,' & ',int(mean(l)/(250*60)),'min',' &  & ')
 
 
############
'''
Data Version saving
'''
############
 
def SaveDataVersions(folder):
 
 #folders=['DMT','KET','PSIL','LSD','FLICK']
 '''
 Take data from Version 0 and save different versions of it; data curbed to 150000 obs, else cache is full
 '''
 versions=['EqualPhase']#,'EqualPower',]#
 
 #path to mini
 path_dat='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/rev_data'
 subjects=os.listdir(path_dat+'/'+folder+'/'+'original')
  
 conditions=['PLA',folder]
 l=1500 #Segment length, all data at 250Hz; needed for EqualPower
 per_segment=1
 
 for version in versions:
  for subject in subjects:
   if version=='EqualPhase':
 
    pla=load(path_dat+'/'+folder+'/'+'%s/%s/%s.npy' %('original',subject,'PLA'))[:,:150000]
    drug=load(path_dat+'/'+folder+'/'+'%s/%s/%s.npy' %('original',subject,folder))[:,:150000]
 
    chans,obs_PLA=shape(pla)
    _,obs=shape(drug)
 
    ra=min(obs_PLA,obs)
 
    drug=drug[:,:ra]
    pla=pla[:,:ra]
 
    for condition in conditions:
     drug=load(path_dat+'/'+folder+'/'+'%s/%s/%s.npy' %('original',subject,condition))[:,:ra]
 
     if per_segment==1:
      #copy phases, keep power segment by segment
      dn=[]
      for t in range(ra/l):
 
        
       R=abs(rfft(drug[:,l*t:l*(t+1)])) #power of drug
       ft=rfft(pla[:,l*t:l*(t+1)]) #phase of placebo from that
 
       '''
       if first frequency has zero amplitude, 
       devision of zero/zero is set to 1
       '''
 
       for i in range(chans): 
        if ft[i][0]==0.00000000 +0.00000000e+00j:
         ft[i][0]=1
 
       ft=ft/abs(ft) 
       dn.append(irfft(R*ft)) #put power and phases together
 
      dn=concatenate(dn,axis=1)
 
 
     if not os.path.exists(path_dat+'/'+folder+'/'+version+'/'+subject):
      os.makedirs(path_dat+'/'+folder+'/'+version+'/'+subject)
     save(path_dat+'/%s/%s/%s/%s.npy' %(folder,version,subject,condition),dn)
     print(folder, version, subject, condition)
 
   if version=='EqualPower':
    pla=load(path_dat+'/'+folder+'/'+'%s/%s/%s.npy' %('original',subject,'PLA'))[:,:150000]
    drug=load(path_dat+'/'+folder+'/'+'%s/%s/%s.npy' %('original',subject,folder))[:,:150000]
 
    chans,obs_PLA=shape(pla)
    _,obs=shape(drug)
 
    ra=min(obs_PLA,obs)
 
    drug=drug[:,:ra]
    pla=pla[:,:ra]
 
    for condition in conditions:
     drug=load(path_dat+'/'+folder+'/'+'%s/%s/%s.npy' %('original',subject,condition))[:,:ra]
 
 
     if per_segment==1:
      #copy spectrum segment by segment
      dn=[]
      for t in range(ra/l):
        
 
       R=abs(rfft(pla[:,l*t:l*(t+1)])) #power from pla
       ft=rfft(drug[:,l*t:l*(t+1)]) # phase from drug
 
       '''
       if first frequency has zero amplitude, 
       devision of zero/zero is set to 1
       '''
 
       for i in range(chans): 
        if ft[i][0]==0.00000000 +0.00000000e+00j:
         ft[i][0]=1
 
       ft=ft/abs(ft)
       dn.append(irfft(R*ft))
 
      dn=concatenate(dn,axis=1)
 
     else:
 
      R=abs(rfft(pla))
      ft=rfft(drug)
 
      '''
      if first frequency has zero amplitude, 
      devision of zero/zero is set to 1
      '''
 
      for i in range(chans): 
       if ft[i][0]==0.00000000 +0.00000000e+00j:
        ft[i][0]=1
 
      ft=ft/abs(ft)
      dn=(irfft(R*ft))
 
     if not os.path.exists(path_dat+'/'+folder+'/'+version+'/'+subject):
      os.makedirs(path_dat+'/'+folder+'/'+version+'/'+subject)
     save(path_dat+'/%s/%s/%s/%s.npy' %(folder,version,subject,condition),dn)
     print(folder, version, subject, condition)
 
############
'''
Compute Measures
'''
############
 
def Measures_all(folder):
 
 print('NEW')
 l=1500 #Segment length, all data at 250Hz
 func_dict ={'LZs':LZs,'PSpec':PSpec}
 
 versions=['original']#''EqualPhase'EqualPower',,
 
 #path to data
 path_dat='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/rev_data'
 
 subjects=os.listdir(path_dat+'/'+folder+'/'+'original')
  
 conditions=['PLA',folder]
  
 path_res='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/rev_results'
 
 for version in versions:
  for condition in conditions:
   for subject in subjects:
    d=load(path_dat+'/%s/%s/%s/%s.npy' %(folder,version,subject,condition))
    chs,obs=shape(d)
    pla=load(path_dat+'/%s/%s/%s/%s.npy' %(folder,version,subject,'PLA'))
    _,obs_PLA=shape(pla)
 
    #analyse data of equal length between conditions
    ra=min([obs_PLA,obs,150000])
    d=d[:,:ra]
 
    for measure in func_dict:
     if measure=='LZs':
      S=[]
      for t in range(int(ra/l)):
       s0=[]
       for jj in range(chs):
        s0.append(LZs(d[jj,l*t:l*(t+1)]))
       #print(condition, subject, measure, t,'of',trs)
       S.append(s0)
 
     if measure=='LZc':
      S=[]
      for t in range(int(ra/l)):
       S.append(LZc(d[:,l*t:l*(t+1)]))
        
 
     if measure=='PSpec':
      S=[]
      for t in range(int(ra/l)):
       S.append(PSpec(d[:,l*t:l*(t+1)]))
           
      S=array(S)
 
     
     if not os.path.exists(path_res+'/%s/%s/%s' %(version,folder,subject)):
      os.makedirs(path_res+'/%s/%s/%s' %(version,folder,subject))
     if measure=='PSpec':
      for jj in range(4):
       savetxt(path_res+'/%s/%s/%s/PSpec%s_%s'%(version,folder,subject,jj,condition),S[:,:,jj])
     else:
      savetxt(path_res+'/%s/%s/%s/%s_%s' %(version,folder,subject,measure,condition),S)
     print(version, condition, subject, shape(d), measure)
 
############
'''
Compute Spectra
'''
############
 
def Spectra_all(folder):
  
 l=2500
 versions=['original']#'EqualPhase''EqualPower',,
 
 #path to data
 path_dat='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/rev_data'
 
 subjects=os.listdir(path_dat+'/'+folder+'/'+'original')
  
 conditions=['PLA',folder]
  
 path_res='/media/mic/8cfe1683-d974-40f3-a20b-b217cad4722a/rev_results'
 
 measure='welch'
 
 for version in versions:
  for condition in conditions:
   for subject in subjects:
    d=load(path_dat+'/%s/%s/%s/%s.npy' %(folder,version,subject,condition))[:,:150000]
    chs,obs=shape(d)
 
    S=[]
    for t in range(int(obs/l)):
     s0=[]
     for jj in range(chs):
      s0.append(signal.welch(d[jj,l*t:l*(t+1)],250))
     #print(condition, subject, measure, t,'of',trs)
     S.append(mean(s0,axis=0))
        
    S=mean(S,axis=0)
    if not os.path.exists(path_res+'/%s/%s/%s' %(version,folder,subject)):
     os.makedirs(path_res+'/%s/%s/%s' %(version,folder,subject))
 
    save(path_res+'/%s/%s/%s/%s_%s.npy' %(version,folder,subject,measure,condition),S)
    print(version, condition, subject, shape(d), measure)

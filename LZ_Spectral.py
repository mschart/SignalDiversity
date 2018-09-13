'''
This script defines functions LZs(X) and PSpec(X) to compute for MEG or EEG data X (channels x observations). 
For details see:

Schartner, Michael M., et al. 
"Increased spontaneous MEG signal diversity for psychoactive doses of ketamine, LSD and psilocybin." 
Scientific reports 7 (2017): 46421.
'''

from scipy import signal
from scipy.signal import (butter,lfilter,hilbert,resample)
from pylab import *
import os as os
 
 
def Pre2(X):
 '''
 Linear-detrend and subtract mean of X, a multidimensional times series (channels x observations)
 '''
 ro,co=shape(X)
 Z=zeros((ro,co))
 for i in range(ro): #maybe divide by std?
  Z[i,:]=signal.detrend((X[i,:]-mean(X[i,:]))/std(X[i,:]), axis=0)
 return Z
 
##############
'''
PSpec; compute spectral power density in canonical EEG bands
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
 

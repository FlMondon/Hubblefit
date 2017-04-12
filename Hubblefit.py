from __future__ import division
from scipy.integrate import quad
from math import *
import pyfits
import glob
import pylab as P
import numpy as N 
import scipy.optimize as opt
from scipy.optimize import curve_fit
import matplotlib.patches as mpatches
import matplotlib.path as mpath
from scipy.optimize import fmin
from operator import add
from numpy.linalg import inv
from iminuit import Minuit, describe, Struct
import cPickle
import sys
from matplotlib import *
import copy


###########
#constants
###########

clight=299792.458
H=0.000070
omgM=0.295
alpha=0.141
beta=3.101
Mb=-19.05
delta_M=-0.070
w=-0.7
############
#Fonctions
############
 
def plot_ellipse(X,Y,COLOR='b'):
 
    fig, ax = P.subplots()
    Npoint = len(X)
    Path = mpath.Path
    path_data = [
        (Path.MOVETO, (X[0],Y[0])),
        (Path.LINETO, (X[1:Npoint-1],Y[1:Npoint-1])),
        (Path.CLOSEPOLY, (X[0],Y[0])),
       ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts,codes)
    patch = mpatches.PathPatch(path, facecolor=COLOR, alpha=0.5)
    ax = plt.subplots()
    ax.add_patch(patch)
    x, y = zip(*path_data.vertices)
    line, = ax.plot(x, y)
    
def savepkl(dic,name='nomame'):
	'''
	Function that create a pkl file to export some data
	inputs:
		-dic : a dictonary containing the data
		-name : the name of the pkl file
	'''
	File = open('Results/pkl/' + name +'.pkl','w')
	cPickle.dump(dic,File)
	File.close()
		
def Sx(X):
	'''
	Function that compute Sx of the distribution given in X
	For the matematical description of Sx, check the pdf file : 'Data/Equations.pdf'
	input : the distribution X
	output : Sx(X)
	'''
	return N.sqrt(abs((1/(len(X)-1))*(sum((X-N.mean(X))**2))))
		
def RMS(X):
	'''
	Funtction that compute the RMS on a distribution
	inputs :
		-X the distribution
	output :
		-The RMS
	'''
	rms = N.sqrt(abs((1/len(X))*(sum((X-N.mean(X))**2))))
	return rms
	
def RMSerr(X):
	'''
	Funtction that compute the error on the RMS on a distribution
	inputs :
		-X the distribution
	output :
		-The error of the RMS
	'''
	rmserr = Sx(X)/(N.sqrt(2*len(X)))
	return rmserr
	
def MEANerr(X):
	'''
	Function that compute the error ont he mean of the distribution given in X
	For the matematical description of Sx, check the pdf file : 'Data/Equations.pdf'
	inputs :
		-X the distibution
	output:
		-the error on the mean
	'''
	meanerr = Sx(X)* (1./N.sqrt(len(X)))
	return meanerr
	
def Histo(x,mean,sigma_mean,std,stderr,P,orient='horizontal',xlabel='',ylabel=''):
	'''
	Function that plot the histogramm of the distribution given in x
	imputs:
	-x is the distrib itself (array)
	-mean is the mean of the distrib (float)
	-sigma_mean : the error on the average (float)
	-std : the standard deviation (RMS) of the distribution (float)
	-stderr : the errot on the RMS (float)
	-P is the figure where the histogram will be plotted.
	-xylabel and y label are the name ofthe axis.
	'''
	numBins = 20
	P.hist(x,numBins,color='blue',alpha=0.8,orientation=orient,label='average = ' + str("%.5f" % mean) + '$\pm$' + str("%.5f" % sigma_mean) + '\n' +  'rms =' + str("%.5f" % std) + '$\pm$' +str("%.5f" % stderr))
	if xlabel == '':
		P.set_xlabel('number of SNe')
	else:
		P.set_xlabel(xlabel)
	if ylabel == '':
		P.set_ylabel('number of SNe')
	else:
		P.set_ylabel(ylabel)
	P.set_title('Residuals')
	P.legend(bbox_to_anchor=(0.95, 1.0),prop={'size':10})
	

def comp_rms(residuals, dof, err=True, variance=None):
	"""                                                                                                                                                                                      
	Compute the RMS or WRMS of a given distribution.
	:param 1D-array residuals: the residuals of the fit.
	:param int dof: the number of degree of freedom of the fit.                                                                                                                              
	:param bool err: return the error on the RMS (WRMS) if set to True.                                                                                                                      
	:param 1D-aray variance: variance of each point. If given,                                                                                                                               
        return the weighted RMS (WRMS).                                                                                                                                                                                                                                                                                                                             
	:return: rms or rms, rms_err                                                                                                                                                             
	"""
	if variance is None:                # RMS                                                                                                                                                
		rms = float(N.sqrt(N.sum(residuals**2)/dof))
		rms_err = float(rms / N.sqrt(2*dof))
	else:	  # Weighted RMS                                                                                                                                       
		assert len(residuals) == len(variance)
		rms = float(N.sqrt(N.sum((residuals**2)/variance) / N.sum(1./variance)))
		#rms_err = float(N.sqrt(1./N.sum(1./variance)))                                                                                                                                      
		rms_err = N.sqrt(2.*len(residuals)) / (2*N.sum(1./variance)*rms)
	if err:
		return rms, rms_err
	else:
	    return rms


def intfun_save(x,y,z):
    """
    Function that build an array that contains theoretical mu (funtion of omgM and luminosity distance)
    imputs:
    -x: represent the redshift
    -y:represent the parameter omgM
    -z:represent the parameter w
    """
    
    return 1/sqrt(y*(1+x)**3+ 1-(y+z)+z*(1+x)**2)
#    return 1/sqrt(y*(1+x)**3 +(1-y)*(1+x)**(3*(1+z)))
def intfun(z,omgM,omgK,w):
    """
    Function that build an array that contains theoretical mu (funtion of omgM and luminosity distance)
    imputs:
    -z: represent the redshift
    -omgM:represent the parameter omgM
    -omgK: represent the parameter omgK    
    -w:represent the parameter w
    """
#    return 1/sqrt(omgM*(1+z)**3 + (1-omgM-omgK)*(1+z)**(3*(1+w)) + omgK*(1+z)**2)
    return 1/sqrt(omgM*(1+z)**3 + omgK*(1+z)**(3*(1+w)) + (1-omgM-omgK)*(1+z)**2)

    
def fitfundL(zcmb,omgM):
	"""
	Function which create the distance modulus for each supernovae
	imputs:
	-zcmb is the redshift of each SNe (array)
	-omgM = 0.295.
	outputs:
	-MU is the array containing the distance modulus of each SNe
	"""
	MU=[]
	for i in range (len(zcmb)): 
	        zz=zcmb[i]
		MU.append(dL_z(zz,zz,omgM,omgK,w)) 
	return MU  

def dL_z(zcmb,zhel,omgM,omgK,w):
    """ 
    Function that compute the integral for the comoving distance.
    imputs:
        -zcmb is the redshift in the cmb framework (array)
        -zhel the redshift in the heliocentric framework (array)
        -omgM = 0.295
    outputs:
        -mu_zz is the array contaning the distance for each SNe
    """
    
    omgKK=1-omgM-omgK
    if omgKK == 0:
        mu_zz = 5*log10((1+zcmb)*clight*(quad(intfun,0,zcmb,args=(omgM,omgK,w))[0]/(10*H)))        
    elif omgKK < 0 :
        mu_zz = 5*log10((1+zcmb)*(1/N.sqrt(N.abs(1-omgM-omgK)) *clight*(N.sin(N.sqrt(N.abs(1-omgM-omgK))*quad(intfun,0,zcmb,args=(omgM,omgK,w))[0])/(10*H))))
    elif omgKK > 0 :
        mu_zz = 5*log10((1+zcmb)*(1/N.sqrt(1-omgM-omgK)) *clight*(N.sinh(N.sqrt(1-omgM-omgK)*quad(intfun,0,zcmb,args=(omgM,omgK,w))[0])/(10*H)))
        


#    if omgK == 0:
#        mu_zz = 5*log10((1+zcmb)*clight*(quad(intfun,0,zcmb,args=(omgM,omgK,w))[0]/(10*H)))        
#    elif omgK < 0 :
#        mu_zz = 5*log10((1+zcmb)*(1/N.sqrt(N.abs(omgK)) *clight*(N.sin(N.sqrt(N.abs(omgK))*quad(intfun,0,zcmb,args=(omgM,omgK,w))[0])/(10*H))))
#    elif omgK > 0 :
#        mu_zz = 5*log10((1+zcmb)*(1/N.sqrt(omgK)) *clight*(N.sinh(N.sqrt(omgK)*quad(intfun,0,zcmb,args=(omgM,omgK,w))[0])/(10*H)))
#        
    return mu_zz

def muexp(mB,X1,C,alpha,beta,Mb,delta_M,M_stell):
	"""
	#Correction to muexp regarding the stellar mass of the host galaxy (cf Betoule et al 2014)
	imputs:
	-alpha: free parameter of the Hubble fit (factor of the stretch)
	-beta: free parameter of the Hubble fit (factor of the color)
	-delta_M is a free parameter of the Hubble fit (value of the step for the mass step correction (see Betoule et al 2014))
	-M_stell: the log_10 host stellar mass in units of solar mass
	-mB: B band peak magnitude of the SNs (array)
	-X1:  SALT2 shape parameter (array)
	-C: colour parameter (array)
	-M_stell : the stellar ;aass of each host (array)
	"""
	mu=[]
	#As M.Betoule (choose one or the other)
	for i in range(len(mB)):
		if M_stell[i]<10:
			mu.append(mB[i]-Mb+alpha*X1[i]-beta*C[i])
		else :
			mu.append(mB[i]-Mb-delta_M+alpha*X1[i]-beta*C[i])
	#With fixed delta_M (choose one or the other)
	#Without mass step correction
	'''
	for i in range(len(mB)):
		mu.append(mB[i]-Mb+alpha*X1[i]-beta*C[i])
	'''
	return mu

def dmuexp(dmB,dX1,dC,alpha,beta):
	"""
	Function that build the list of dmuexp (uncertainties propagation) 
	imputs:
	-dmB: uncertainty on mB
	-dX1: uncertainty on X1
	-dC: uncertainty on C	
	-alpha: free parameter of the Hubble fit (factor of the stretch)
	-beta: free parameter of the Hubble fit (factor of the color)
	"""
	dmu=[]
	for i in range(len(dmB)):
	        dmu.append(sqrt(dmB[i]**2+(alpha*dX1[i])**2+(beta*dC[i])**2))
	return dmu

def Remove_Matrix(Tab,ID):
	"""
	function that remove from the 'Tab' matrix all the rows and colomns except those precised in 'ID'
	"""
	#create the list with the line to be removed
	try:
		tab = N.delete(N.arange(len(Tab[0])),ID,0)
	except:
		tab = N.delete(N.arange(len(Tab)),ID,0)
		
	#remove these lines to the original matrix
	Tab = N.delete(Tab,tab,0)
	try:
		Tab = N.delete(Tab,tab,1)
	except:
		'''nothing else to do'''
	return Tab
	
def mu_cov(alpha, beta, IDJLA, cov_path, sigma_mu_path):
    """
    #Function that buil the covariance matrix as Betoule et al (2014) betzeen SNe to get the chi2
    imputs:
    -alpha: free parameter of the Hubble fit (factor of the stretch)
    -beta: free parameter of the Hubble fit (factor of the color)
    -IDJLA: is an array containing the indexing value of each SNe
    outuput:
    -Cmu : the covariance matrix (2 dimension array)
    cov_path: path to covariance matrix
    sigma_mu_path : path to sigma_mu.txt
    """
    #Assemble the full covariance matrix of distance modulus, See Betoule et al. (2014), Eq. 11-13 for reference
    #You have to acces the data which are in '/data/software/jla_likelihood_v6/covmat/C*.fits'. The C_hosts.fits has been removed from the analysis  
    #Ceta = sum([pyfits.getdata(mat) for mat in glob.glob('Data/covmat/C*.fits')])
    Ceta = sum([pyfits.getdata(mat) for mat in glob.glob(cov_path)])
    Cmu = N.zeros_like(Ceta[::3,::3])
    for i, coef1 in enumerate([1., alpha, -beta]):
        for j, coef2 in enumerate([1., alpha, -beta]):
            Cmu += (coef1 * coef2) * Ceta[i::3,j::3]
    # Add diagonal term from Eq. 13
    sigma = N.loadtxt(sigma_mu_path)
    sigma_pecvel = (5 * 150 / 3e5) / (N.log(10.) * sigma[:, 2])
    Cmu[N.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2
    #Cmu=N.diag(Cmu[N.diag_indices_from(Cmu)])
    Cmu = Remove_Matrix(Cmu,IDJLA)
    #print len(Cmu)
    return Cmu

def ecarts(zcmb,zhel,mu,omgM,omgK,w):
	"""
	Function that compute the difference between mu_exp and mu_theortical into a list (residuals)
	-zcmb is the redshift in the cmb framework (array)
	-zhel the redshift in the heliocentric framework (array)
	-omgM = 0.295
	-mu :  is the experimental value
	output:
	-ecarts5 is the array containing the residuals
	"""
	ecarts=[]
	for i in range(len(zcmb)):
		z=zcmb[i]
		zz=zhel[i]	
		ecarts.append(mu[i]-dL_z(z,zz,omgM,omgK,w))
	return ecarts

def Plotconfig():
    '''
    This function as been created to configure the style of the plots
    '''
    rcParams['font.size'] = 17.
    font = {'family': 'normal', 'size': 17}
    rc('axes', linewidth=1.2)
    rc("text", usetex=True)
    rc('font', family='serif')
    rc('font', serif='Bookman')
    rc('legend', fontsize=18)
    rc('xtick.major', size=6, width=2)
    rc('ytick.major', size=6, width=2)
    rc('xtick.minor', size=4, width=1)
    rc('ytick.minor', size=4, width=1)

class Chi2: #Class definition
	'''Class for the chi2 '''

	def __init__(self,zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,cov_path, sigma_mu_path): # Construct method
            '''Attributes'''
            self.IDJLA = IDJLA
            self.chi2tot = 0.
            self.zcmb = zcmb
            self.zhel =  zhel
            self.mB =  mB
            self.dmB =  dmB
            self.X1 = X1
            self.dX1 = dX1
            self.C =  C
            self.dC =  dC
            self.M_stell =  M_stell
            self.dL = N.zeros(shape=(len(IDJLA))) 
            self.cov_path=cov_path
            self.sigma_mu_path=sigma_mu_path
            self.cache_alpha=N.nan
            self.cache_beta=N.nan

	def chi2(self,omgM,omgL,w,alpha,beta,Mb,delta_M):
		''' Funtion that calculate the chi2 '''
		result=0.
		if alpha!=self.cache_alpha or beta!=self.cache_beta:
                 self.Mat = inv(mu_cov(alpha,beta,self.IDJLA, self.cov_path, self.sigma_mu_path))
                 self.cache_alpha=alpha
                 self.cache_beta=beta
		mu_z=muexp(self.mB,self.X1,self.C,alpha,beta,Mb,delta_M,self.M_stell)
		#loop for matrix construction
		for i in range(len(self.zcmb)):
			zz = self.zcmb[i]
			zzz = self.zhel[i]
			self.dL[i] = dL_z(zz,zzz,omgM,omgL,w)

		#contruction of the chi2 by matrix product
		result =  P.dot( (mu_z-self.dL), P.dot((self.Mat),(mu_z-self.dL)))
		self.chi2tot = result
		return result
  
def contour(mobject,varx,vary,nsigma=1,nbinX=31):
    '''
    Function which computes contours fromp a converged imnuit object
    This does not replace the mncontour from minuit
    inputs:
        - mobject: the converges minuit object
        - varX : the X variable for the ocntour. Should be included in mobject.parameters
        - varY : the Y variable for the ocntour. Should be included in mobject.parameters
    returns:
        - contourline to be used with matplotlib.fill_between
    '''
    assert varx in mobject.parameters
    assert vary in mobject.parameters
    #TBD : check that first value is valid
    xvals= N.linspace(mobject.values[varx]-mobject.errors[varx],mobject.values[varx]+mobject.errors[varx],nbinX)
    yplus=list()
    yminus=list()
    # prepare intermediate minuit results        
    chi2min=mobject.fval
    marg=copy.copy(mobject.fitarg)
    for p in mobject.parameters:
        marg['fix_'+p]=True
    marg['fix_'+vary]=False
    for x in xvals:
        #find y which realizes chi2 min
        marg[varx]=x
        m = Minuit(mobject.fcn,**marg)
        m.migrad()
        ymin=m.values[vary]
        # prepare function fo be zeroed
        def f(y):
            m.values[vary]=y
            return m.fcn(**m.values) - chi2min - nsigma**2
        if f(ymin)<0:
            # *2 : safety margin
            yplus.append(opt.brentq(f,ymin,ymin+m.errors[vary]*nsigma*2))
            yminus.append(opt.brentq(f,ymin-m.errors[vary]*nsigma*2,ymin))
        else:
            yplus.append(N.nan)
            yminus.append(N.nan)
    yplus=N.array(yplus)
    yminus=N.array(yminus)
    return xvals[N.isfinite(yplus)],yminus[N.isfinite(yplus)],yplus[N.isfinite(yplus)]
    
        

        
        
                
	
def Hubble_diagram(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,exp,results, cov_path, sigma_mu_path,fixe,omgM=0.295,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070):
    """
    Function which make the hubble fit of the given compilation.
    inputs :
    	-omgM: 1st free parameter to be fitted initialized to the 0.295 value if not precised
    	-alpha: 2nd free parameter to be fitted initialized to the 0.141 value if not precised
    	-beta: 3rd free parameter to be fitted initialized to the 3.101 value if not precised
    	-Mb: 4th free parameter to be fitted initialized to the -19.05 value if not precised
    	-delta_M: 5th free parameter to be fitted initialized to the -0.070 value if not precised
      -fixe: when fixe=0 omgK and w are fixe, fixe = 1 w is fixe, fixe =2 omgK is fixe else all parameters are fixe
    	-zcmb: an array which contains the redshifts of the SNs
    	-mB: B band peak magnitude of the SNs
    	-dmB: uncertainty on mB
    	-X1:  SALT2 shape parameter
    	-dX1: uncertainty on X1Cmu[N.diag_indices_from(Cmu)]
    	-dC: uncertainty on C	
    	-C: colour parameter
    	-M_stell: the log_10 host stellar mass in units of solar mass
    	-IDJLA: index of the SNs from the 740 of jla used for the fit
    	-results : a file where some results wiil be written
    	- m : the iminuit object (see doc of iminuit for more information).
    	-ecarts5 : the residuals of the fit
    """
    #check : need to have at least 2 SNe
    '''f2,(test,P2,P3,P4,P5) = P.subplots(5, sharex=True, sharey=False, gridspec_kw=dict(height_ratios=[3,1,1,1,1]))'''
    if len(zcmb) == 1 or len(zcmb) == 0 :
    	results.write('Not enough data \n')
    	return 0
    
    #Definition of the Chi2 object
    chi2mini=Chi2(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,cov_path, sigma_mu_path)
        
    
    #minimisation of the chi2
    '''
    the values of the free parameter can be initialized.
    '''
    # TO BE CLEANED    
    if fixe == 0 :
        m=Minuit(chi2mini.chi2,omgM=0.295,omgK=0,w=-1,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,limit_omgM=(0.2,0.4),limit_omgK=(-1,1),limit_w=(-1.4,0.4),limit_alpha=(0.1,0.2),limit_beta=(2.0,4.0),limit_Mb=(-20.,-18.),limit_delta_M=(-0.1,-0.0),fix_omgM=False,fix_omgK=True,fix_w=True, fix_alpha=False, fix_beta=False, fix_Mb=False, fix_delta_M=False, print_level=1)    
    elif fixe ==1 :
#        last_iter_args={u'fix_Mb': True, u'limit_w': None, u'fix_delta_M': True, u'error_omgL': 0.1672842193144214, u'error_omgM': 0.11284796124765262, 'delta_M': -0.07040462309885408, u'error_Mb': 0.02559618467415318, u'error_w': 1.0, u'error_alpha': 0.006593439955397084, u'fix_alpha': True, u'limit_alpha': None, 'Mb': -19.038807526119403, u'limit_omgM': None, u'limit_omgL': None, u'limit_Mb': None, 'beta': 3.0999314004965535, u'limit_delta_M': None, u'limit_beta': None, 'alpha': 0.14098769528058613, 'omgM': 0.20020038670359183, 'omgL': 0.561065865545341, u'fix_w': True, u'fix_beta': True, u'fix_omgM': False, u'fix_omgL': False, u'error_delta_M': 0.023111327883836057, u'error_beta': 0.08066293365248206, 'w': -1.0}
#        m=Minuit(chi2mini.chi2,**last_iter_args)
#        m=Minuit(chi2mini.chi2,omgM=0.2,omgL=0.55,w=-1,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,limit_omgM=(0,0.4),limit_omgL=(-1,1),limit_w=(-1.4,0.4),limit_alpha=(0.1,0.2),limit_beta=(2.0,4.0),limit_Mb=(-20.,-18.),limit_delta_M=(-0.1,-0.0),fix_omgM=False,fix_omgL=False,fix_w=True, fix_alpha=False, fix_beta=False, fix_Mb=False, fix_delta_M=False, print_level=1)    
        m=Minuit(chi2mini.chi2,omgM=0.2,omgL=0.55,w=-1,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,limit_omgM=(0,0.4),limit_omgL=(-1,1),limit_w=(-1.4,0.4),limit_alpha=(0.1,0.2),limit_beta=(2.0,4.0),limit_Mb=(-20.,-18.),limit_delta_M=(-0.1,-0.0),fix_omgM=False,fix_omgL=False,fix_w=True, fix_alpha=True, fix_beta=True, fix_Mb=True, fix_delta_M=True, print_level=1)            
        #m=Minuit(chi2mini.chi2,omgM=0.2,omgL=0.55,w=-1,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,fix_omgM=False,fix_omgL=False,fix_w=True, fix_alpha=False, fix_beta=False, fix_Mb=False, fix_delta_M=False, print_level=1)    
    elif fixe ==2 : 
        m=Minuit(chi2mini.chi2,omgM=0.2,omgK=0,w=-1,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,limit_omgM=(0,0.4),limit_omgK=(-1,1),limit_w=(-1.4,0.4),limit_alpha=(0.1,0.2),limit_beta=(2.0,4.0),limit_Mb=(-20.,-18.),limit_delta_M=(-0.1,-0.0),fix_omgM=False,fix_omgK=True,fix_w=False, fix_alpha=False, fix_beta=False, fix_Mb=False, fix_delta_M=False, print_level=1)    
    else :
        m=Minuit(chi2mini.chi2,omgM=0.2,omgK=0,w=-1,alpha=0.141,beta=3.101,Mb=-19.05,delta_M=-0.070,limit_omgM=(0,0.4),limit_omgK=(-1,1),limit_w=(-1.4,0.4),limit_alpha=(0.1,0.2),limit_beta=(2.0,4.0),limit_Mb=(-20.,-18.),limit_delta_M=(-0.1,-0.0),fix_omgM=True,fix_omgK=True,fix_w=True, fix_alpha=True, fix_beta=True, fix_Mb=True, fix_delta_M=True, print_level=1)    

    
    m.migrad()
    return m    
    m.hesse()
    chi2miniminuit=chi2mini.chi2tot #minimun chi2 value obtain by iminuit
    
    #print(m.values)
    #print(m.errors)
    omgM = m.args[0]
    omgK = m.args[1]
    w=m.args[2]
    alpha = m.args[3]
    beta = m.args[4]
    Mb = m.args[5]
    delta_M = m.args[6]
    
    #errors obtain by minuit    
    omgMerr=m.errors.get('omgM')
    omgKerr=m.errors.get('omgL')
    werr=m.errors.get('w')
    alphaerr=m.errors.get('alpha')
    betaerr=m.errors.get('beta')
    Mberr = m.errors.get('Mb')
    delta_Merr = m.errors.get('delta_M')
#    cov_omg=m.hesse()
#    cov_omg=m.matrix()
    
    
    if fixe == 1:
#        Pcov = P.plot()
#        bX,bY,contourXY = m.mncontour('omgM', 'omgL', numpoints=20, sigma=1.0)    
#        bX,bY,contourXY = m.contour('omgM', 'omgL', bins=20, bound=2)
#        contourXY -= chi2miniminuit
#        Pcov=m.draw_contour('omgM', 'omgL', bins=20, bound=1)
        Pcov=m.draw_mncontour('omgM', 'omgL')
#        X,Y=N.meshgrid(N.linspace(0,0.3,1000),N.linspace(0,0.5,1000))
#        Pcov = P.contour(bX,bY,contourXY,[1,2,9])
        
#        P.show()
    elif fixe ==2 :
        Pcov = P.plot()
#        bins_X,bins_Y,contour_X_Y = m.contour('omgM', 'w', bins=20, bound=1)
        m.draw_contour('omgM', 'w', bins=20 ,bound=1)
        P.show()

        
    #write some results in output
    results.write('\\begin{tabular}{|c|l|}'+'\n'+'\\hline'+'\n')
    results.write('chi2 / d.o.f& %.3f'%((chi2mini.chi2tot)/(len(IDJLA)-5))+' \\\\ \\hline \n')
    results.write('chi2& %.3f'%(chi2mini.chi2tot) + ' \\\\ \\hline \n')
    
    #Computation of the luminosity-distance modulus and its uncertainty for the minimized parameters
    mufinal=muexp(mB,X1,C,alpha,beta,Mb,delta_M,M_stell)
    dmufinal=dmuexp(dmB,dX1,dC,alpha,beta)
    
    #Computation of the difference between the measured luminosity-distance modulus and the theorical one (residuals)
    ecarts5=ecarts(zcmb,zhel,mufinal,omgM,omgK,w)
    
    #computation of the RMS,mean and their error of the distribution
    #std5,std5err = comp_rms(N.array(ecarts5),(len(ecarts5)))
    std5=N.std(ecarts5)
    std5err = RMSerr(ecarts5)
    mean5 = N.mean(ecarts5)
    mean5err = MEANerr(ecarts5)
    results.write('mean of the residuals& %.5f'%mean5+' \\\\ \\hline \n')
    sigma_mean5 = std5/N.sqrt(len(ecarts5))
    results.write('rms& %.4f'%(std5)+' \\\\ \\hline \n')
    
    #mean = N.mean(ecarts5)
    	
    #Computaion of the rms of the fit.
    rms,err_rms = comp_rms(N.array(ecarts5), len(mB), err=True, variance=None)
    results.write('omgM& %.4f'%(omgM)+' +/- %.4f'%(omgMerr)+' \\\\ \\hline \n')
    results.write('omgK& %.4f'%(omgK)+' +/- %.4f'%(omgKerr)+' \\\\ \\hline \n')
    results.write('w& %.4f'%(w)+' +/- %.4f'%(werr)+' \\\\ \\hline \n')
    results.write('alpha& %.4f'%(alpha)+' +/- %.4f'%(alphaerr)+' \\\\ \\hline \n')
    results.write('beta& %.4f'%(beta)+' +/- %.4f'%(betaerr)+' \\\\ \\hline \n')
    results.write('Mb& %.4f'%(Mb)+' +/- %.4f'%(Mberr)+' \\\\ \\hline \n')
    results.write('delta-M& %.4f'%(delta_M) +' +/- %.4f'%(delta_Merr)+ ' \\\\ \\hline \n')
    results.write('\\end{tabular}')
    return m    
#    return  m,mufinal, dmufinal,ecarts5,mean5,sigma_mean5,std5,std5err,chi2miniminuit,cov_omg
 
    
#######
#Main
#######
if __name__=='__main__':
    
    #path to the different input and output files 
    Data_path='/data/software/jla_likelihood_v6/data/jla_lcparams.txt'
    main_path='/users/divers/lsst/mondon/hubblefit'
    results_path=main_path+'/results/'
    sigma_mu_path='/data/software/jla_likelihood_v6/covmat/sigma_mu.txt'
    cov_path='/data/software/jla_likelihood_v6/covmat/C*.fits'          #path for all covariance matrix countain in C (stat+sys)
#    cov_path='/data/software/jla_likelihood_v6/covmat/C_stat.fits'   
    
    #First file is JLA data itself
    jlaData=N.loadtxt(Data_path,dtype='str')
    
    #Select the chosen data
    SnIDjla = N.array(jlaData[:,0])
    zcmb = N.array(jlaData[:,1],float)
    zhel = N.array(jlaData[:,2],float)
    mB = N.array(jlaData[:,4],float)
    dmB = N.array(jlaData[:,5],float)
    X1 = N.array(jlaData[:,6],float)
    dX1 = N.array(jlaData[:,7],float)
    C = N.array(jlaData[:,8],float)
    dC = N.array(jlaData[:,9],float)
    M_stell= N.array(jlaData[:,10],float)
    Rajla=N.array(jlaData[:,18],float)
    Decjla=N.array(jlaData[:,19],float)
    subsample=N.array(jlaData[:,17],int)
    IDJLA = N.arange(740)


    #IDJLA = N.loadtxt('/users/divers/lsst/henne/Desktop/index.csv',dtype='int')
    
    # Perform the JLA fit and write results
    results=open(results_path+'HubbleK.txt','w') # file used to write some results (use the one you want instead of mine).
        
        
    #Fixe is a value to choose the fixed parameters (0:w and omgK fixe, 1: w fixe, 2: omgK fixe, else: all parameters fixe)
    fixe=1
    
    
    #m,mufinal, dmufinal,ecarts5,mean5,sigma_mean5,std5,std5err,chi2miniminuit,cov_omg = Hubble_diagram(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,exp,results, cov_path, sigma_mu_path,fixe,omgM,alpha,beta,Mb,delta_M)
    m = Hubble_diagram(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,exp,results, cov_path, sigma_mu_path,fixe,omgM,alpha,beta,Mb,delta_M)
#    print cov_omg    
    results.close()
    P.show()
    
#    chi2mini=Chi2(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,cov_path, sigma_mu_path)
#    omgK1=N.arange(-0.5,0.5,0.01)
#    cfunc1=N.zeros(len(omgK1))
#    cfunc2=N.zeros(len(omgK1))
#    for i in range (0,len(cfunc1)):
#        cfunc1[i]=chi2mini.chi2(0.2491,omgK1[i],-1,0.1401,3.137,-19.04,-0.06)
#        cfunc2[i]=chi2mini.chi2(0.2,omgK1[i],-1,0.1401,3.137,-19.04,-0.06)
#    P.plot(omgK1,cfunc1,label='omgM=0.2491')
#    P.plot(omgK1,cfunc2,label='omgM=0.2')
#    P.xlabel('omgK')
#    P.ylabel('chi2')
#    P.legend()
#    P.autoscale()
#    P.show()
#    

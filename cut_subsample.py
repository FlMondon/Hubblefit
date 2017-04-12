# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:39:23 2017

@author: mondon
"""
from math import *
import pylab as P
from scipy import vectorize
import numpy as N 
from Hubblefit import *
from matplotlib.backends.backend_pdf import PdfPages
import sys

###########
#constants
###########

c=299792.458
H=0.000070
omgM=0.295
alpha=0.141
beta=3.101
Mb=-19.05
delta_M=-0.070

#############
#Functions
#############

 
def FiltreEx(values,subsample,cuts):
    '''
    Function that return the subpart of 'values' which are corresponding to the sub sample contained in float
    inputs :
    -cuts is either a value or a list of values
    '''
     
    if hasattr(cuts, '__iter__'):
         filtre = N.zeros(len(subsample),dtype=bool)
         for c in cuts: 
             filtre |= (subsample == c)
    else:
         filtre =  (subsample == cuts)
    return values[filtre]


def Fit_Cut_subsample(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,subsample, cov_path, sigma_mu_path,omgM,alpha,beta,M_b,delta_M,fixe,cuts):
    '''
    Function that compute the Hubble diagram for the considered subsample (see Hubbllefit.py for more information) 
    inputs :
    	-omgM: 1st free parameter to be fitted initialized to the 0.295 value if not precised
    	-alpha: 2nd free parameter to be fitted initialized to the 0.141 value if not precised
    	-beta: 3rd free parameter to be fitted initialized to the 3.101 value if not precised
    	-Mb: 4th free parameter to be fitted initialized to the -19.05 value if not precised
    	-delta_M: 5th free parameter to be fitted initialized to the -0.070 value if not precised
      -fixe: when fixe=0 iminuit parameters are free in the other case parameters are fixe
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
      -cut1 and cut2 are the selected subsample (1:SNSL, 2:SDSS, 3:low z, 4:HST)
    '''
    mufinal=[]
    absolute=[]
    zcmbcut = FiltreEx(zcmb,subsample,cuts)	
    zhelcut = FiltreEx(zhel,subsample,cuts)	
    mBcut = FiltreEx(mB,subsample,cuts)	
    dmBcut = FiltreEx(dmB,subsample,cuts)
    X1cut = FiltreEx(X1,subsample,cuts)
    dX1cut = FiltreEx(dX1,subsample,cuts)	
    Ccut = FiltreEx(C,subsample,cuts)
    dCcut = FiltreEx(dC,subsample,cuts)
    M_stellcut = FiltreEx(M_stell,subsample,cuts)
    IDJLAcut = FiltreEx(IDJLA,subsample,cuts)
    print len(IDJLAcut)
    mufinal, dmufinal,ecarts5,mean5,sigma_mean5,std5,std5err,chi2miniminuit = Hubble_diagram(zcmbcut,zhelcut,mBcut,dmBcut,X1cut,dX1cut,Ccut,dCcut,M_stellcut,IDJLAcut,subsample,results, cov_path, sigma_mu_path,fixe,omgM,alpha,beta,Mb,delta_M)
    return IDJLAcut,zcmbcut,mufinal, dmufinal,ecarts5,mean5,sigma_mean5,std5,std5err,chi2miniminuit,M_stellcut

def Plot_hubblefit_sbs(IDJLA,zcmb,main_path, mufinal, dmufinal,ecarts5,mean5,sigma_mean5,std5,std5err,chi2miniminuit,fixe,label='Hubble diagram'):
    '''    
    Function which plot the hubble diagram of the given compilation.
    inputs :
    -chi2miniminuit: best chi2 given by minuit
    -mufinal: list that contains the mu for each SN after the minimization given by minuit
    -ecarts5 : the residuals of the fit
    -label: string that will be the title of the hubble diagram plot
    -fixe: when fixe=0 iminuit parameters are free in the other case parameters are fixe
    -plot : a subplot that need information to get fill. (used only by 'Cut.py')
    outputs:
    -The plot of the Hubble diagram
    -main_path: main path to results files
    '''
    #cretion of the cut zcmb and mufinal associate by subsample
    IDJLAcut1,zcmbcut1,mufinal1, dmufinal1,ecarts51,mean51,sigma_mean51,std51,std5err1,chi2miniminuit1,M_stellcut1 = Fit_Cut_subsample(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,subsample, cov_path, sigma_mu_path,omgM,alpha,beta,Mb,delta_M,fixe,1)                                                                                                                                            
    IDJLAcut2,zcmbcut2,mufinal2, dmufinal2,ecarts52,mean52,sigma_mean52,std52,std5err2,chi2miniminuit2,M_stellcut2 = Fit_Cut_subsample(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,subsample, cov_path, sigma_mu_path,omgM,alpha,beta,Mb,delta_M,fixe,2)        
    IDJLAcut3,zcmbcut3,mufinal3, dmufinal3,ecarts53,mean53,sigma_mean53,std53,std5err3,chi2miniminuit3,M_stellcut3 = Fit_Cut_subsample(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,subsample, cov_path, sigma_mu_path,omgM,alpha,beta,Mb,delta_M,fixe,3)        
    IDJLAcut4,zcmbcut4,mufinal4, dmufinal4,ecarts54,mean54,sigma_mean54,std54,std5err4,chi2miniminuit4,M_stellcut4 = Fit_Cut_subsample(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,subsample, cov_path, sigma_mu_path,omgM,alpha,beta,Mb,delta_M,fixe,4)            
 
    #creation of the curve that represents the function of the fit
    xfunc = N.linspace(0.001,2,1000)
    yfunc = N.zeros(len(xfunc))
    yres = yfunc
    yfunc = fitfundL(xfunc,omgM)
    Plotconfig()#zcmb,mufinal,'$Redshift$','$\\mu = m_b - (M - \\alpha X_1 - \\beta C)$'	)
    f, (P1, P2) = P.subplots(2, sharex=True, sharey=False, gridspec_kw=dict(height_ratios=[3,1]),figsize=(15,12))
    P1.set_title(label + ' with ' + str(len(IDJLA)) + ' SNe',fontsize = 25)
    P.xticks(fontsize=12)
    
#    P1.scatter(zcmb,mufinal,color='blue',s=2,marker='.',label=('Chi2=' + str("%.2f" % chi2miniminuit) + '\n' + 'Chi2/dof= ' + str("%.2f" % ((chi2miniminuit)/(len(IDJLAcut)-3))) + '\n' 'Mean =' + str("%.3f" % mean5) + '\n' + 'RMS = ' + str("%.3f" % std5))) # + '$\pm$' + str("%.3f" % (std5/N.sqrt(740)))
    P1.scatter(zcmbcut1,mufinal1,color='red',s=2,marker='.',label=('SNLS \n')) #+'Chi2=' + str("%.2f" % chi2miniminuit1) + '\n' + 'Chi2/dof= ' + str("%.2f" % ((chi2miniminuit1)/(len(IDJLAcut1)-3))) + '\n' 'Mean =' + str("%.3f" % mean51) + '\n' + 'RMS = ' + str("%.3f" % std51))) # + '$\pm$' + str("%.3f" % (std5/N.sqrt(740)))
    P1.scatter(zcmbcut2,mufinal2,color='green',s=2,marker='.',label=('SDSS \n')) #+'Chi2=' + str("%.2f" % chi2miniminuit2) + '\n' + 'Chi2/dof= ' + str("%.2f" % ((chi2miniminuit2)/(len(IDJLAcut2)-3))) + '\n' 'Mean =' + str("%.3f" % mean52) + '\n' + 'RMS = ' + str("%.3f" % std52))) # + '$\pm$' + str("%.3f" % (std5/N.sqrt(740)))    
    P1.scatter(zcmbcut3,mufinal3,color='blue',s=2,marker='.',label=('low z \n')) #+'Chi2=' + str("%.2f" % chi2miniminuit3) + '\n' + 'Chi2/dof= ' + str("%.2f" % ((chi2miniminuit3)/(len(IDJLAcut3)-3))) + '\n' 'Mean =' + str("%.3f" % mean53) + '\n' + 'RMS = ' + str("%.3f" % std53))) # + '$\pm$' + str("%.3f" % (std5/N.sqrt(740)))        
    P1.scatter(zcmbcut4,mufinal4,color='purple',s=2,marker='.',label=('HST \n')) #+'Chi2=' + str("%.2f" % chi2miniminuit4) + '\n' + 'Chi2/dof= ' + str("%.2f" % ((chi2miniminuit4)/(len(IDJLAcut4)-3))) + '\n' 'Mean =' + str("%.3f" % mean54) + '\n' + 'RMS = ' + str("%.3f" % std54))) # + '$\pm$' + str("%.3f" % (std5/N.sqrt(740)))            
    P1.set_xscale('log')
    P1.plot(xfunc,yfunc)

    P1.set_ylim(30,48)
    P1.legend(bbox_to_anchor=(0.2,1.),fontsize=10)
    P1.set_ylabel('$\\mu = m_b^{*} - (M_{B} - \\alpha X_1 + \\beta C)$',fontsize=20)

    P.yticks(fontsize=12)
    dz = 0
    P1.errorbar(zcmbcut1,mufinal1,linestyle='',xerr=dz,yerr=dmufinal1,ecolor='black' ,alpha=1.0,zorder=0)
    P1.errorbar(zcmbcut2,mufinal2,linestyle='',xerr=dz,yerr=dmufinal2,ecolor='black' ,alpha=1.0,zorder=0)
    P1.errorbar(zcmbcut3,mufinal3,linestyle='',xerr=dz,yerr=dmufinal3,ecolor='black' ,alpha=1.0,zorder=0)    
    P1.errorbar(zcmbcut4,mufinal4,linestyle='',xerr=dz,yerr=dmufinal4,ecolor='black' ,alpha=1.0,zorder=0)

    #The x and y axis limits have to be changed manually
    P2.set_xscale('log')
    P2.set_ylabel('$\\mu - \\mu_{\\Lambda {\\rm CDM}}$',fontsize = 20)
    P2.set_ylim(-1,1)    
    P2.set_xlim(((min(zcmb)-.001)),(max(zcmb)+log10(1.14)))
    P2.plot(xfunc,yres,color='black')

    
    P2.scatter(zcmbcut1,ecarts51,c='red',s=2)
    P2.scatter(zcmbcut2,ecarts52,c='green',s=2)
    P2.scatter(zcmbcut3,ecarts53,c='blue',s=2)
    P2.scatter(zcmbcut4,ecarts54,c='purple',s=2)
    
    P2.errorbar(zcmbcut1,ecarts51, linestyle='',xerr=dz,yerr=dmufinal1,ecolor='red',alpha=1.0,zorder=0)
    P2.errorbar(zcmbcut2,ecarts52, linestyle='',xerr=dz,yerr=dmufinal2,ecolor='green',alpha=1.0,zorder=0)
    P2.errorbar(zcmbcut3,ecarts53, linestyle='',xerr=dz,yerr=dmufinal3,ecolor='blue',alpha=1.0,zorder=0)    
    P2.errorbar(zcmbcut4,ecarts54, linestyle='',xerr=dz,yerr=dmufinal4,ecolor='purple',alpha=1.0,zorder=0)
	

	
    psfile  = main_path+'/HubbleDiagram/HubbleDiagramSbS' + '(' + str(len(zcmb))+'SNe)' + '.eps'
    #P.savefig(psfile)
		
    #second plot of the residuals (separately)

    res, (P1, P2) = P.subplots(1,2, sharex=False, sharey=True,gridspec_kw=dict(width_ratios=[3,1]),figsize=(15,12))
    P1.scatter(zcmbcut1,ecarts51,c='red',s=2)
    P1.scatter(zcmbcut2,ecarts52,c='green',s=2)
    P1.scatter(zcmbcut3,ecarts53,c='blue',s=2)
    P1.scatter(zcmbcut4,ecarts54,c='purple',s=2)
    
    P1.errorbar(zcmbcut1,ecarts51, linestyle='',xerr=dz,yerr=dmufinal1,ecolor='red',alpha=1.0,zorder=0)
    P1.errorbar(zcmbcut2,ecarts52, linestyle='',xerr=dz,yerr=dmufinal2,ecolor='green',alpha=1.0,zorder=0)
    P1.errorbar(zcmbcut3,ecarts53, linestyle='',xerr=dz,yerr=dmufinal3,ecolor='blue',alpha=1.0,zorder=0)    
    P1.errorbar(zcmbcut4,ecarts54, linestyle='',xerr=dz,yerr=dmufinal4,ecolor='purple',alpha=1.0,zorder=0)

    P1.set_xscale('log')
    P1.set_xlabel('$Redshift$')
    P1.set_ylabel('$\\mu - \\mu_{\\Lambda {\\rm CDM}}$')
    P1.set_xlim((min(zcmb)-0.02),(max(zcmb)+log10(2)))
    P1.set_ylim(-1,1)
    P1.set_title(label)
    P1.plot(N.linspace(0.001,1000,1000),yres,color='black')
    res.subplots_adjust(hspace=0.30)
    Histo(ecarts5,mean5,sigma_mean5,std5,std5err,P2)
    psfile  = main_path+'/residuals/residualsSbS' + '(' + str(len(zcmb))+'SNe)' + '.eps'
    #P.savefig(psfile)
	
########
#Main
########
if __name__=='__main__':
	   
    #path to the different input and output files 
    Data_path='/data/software/jla_likelihood_v6/data/jla_lcparams.txt'
    main_path='/users/divers/lsst/mondon/hubblefit'
    results_path=main_path+'/results/'
    sigma_mu_path='/data/software/jla_likelihood_v6/covmat/sigma_mu.txt'
#    cov_path='/data/software/jla_likelihood_v6/covmat/C*.fits'          #path for all covariance matrix countain in C (stat+sys)
    cov_path='/data/software/jla_likelihood_v6/covmat/C_stat.fits'       #path for only stat covariance matrix (stat) 

    #First file is JLA data itself
    jlaData=N.loadtxt(Data_path,dtype='str')
    #First file is JLA data itself
    jlaData=N.loadtxt('/data/software/jla_likelihood_v6/data/jla_lcparams.txt',dtype='str')

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
    print len (zcmb)
    results=open(results_path+'Hubble.txt','w') # file used to write some results
    
    #fixe= 0 iminuit parameters are free else iminuit parameters are fixe
    fixe=0
    
#    IDJLAcut,zcmbcut,mufinal, dmufinal,ecarts5,mean5,sigma_mean5,std5,std5err,chi2miniminuit,M_stellcut = Fit_Cut_subsample(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,subsample, cov_path, sigma_mu_path,omgM,alpha,beta,Mb,delta_M,fixe,(1,2))
    mufinal, dmufinal,ecarts5,mean5,sigma_mean5,std5,std5err,chi2miniminuit = Hubble_diagram(zcmb,zhel,mB,dmB,X1,dX1,C,dC,M_stell,IDJLA,subsample,results, cov_path, sigma_mu_path,fixe,omgM,alpha,beta,Mb,delta_M)
    pl = Plot_hubblefit_sbs(IDJLA,zcmb,main_path, mufinal, dmufinal,ecarts5,mean5,sigma_mean5,std5,std5err,chi2miniminuit,fixe,'Hubble diagram, made with JLA dataset,')	            
    results.close()
    P.show()
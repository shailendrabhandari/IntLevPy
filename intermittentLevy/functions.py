# Necessary library imports for the momentum functions
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy import stats
from sklearn.neighbors import KernelDensity
import functools
import math
import seaborn as sns
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import random
from sklearn.linear_model import LinearRegression
import warnings
import scipy.optimize

# Configuration for warning messages
warnings.filterwarnings("error")

#The second moment and its log
###################################################################################
###################################################################################
def mom2(l_tau,v,D, l_lambdaB, l_lambdaD):
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB/(l_beta)    
    C1 = 2*((v/l_beta)**2) * (l_alpha - 1)/(l_alpha**2)
    C2 = 2*((1-l_alpha)/l_alpha) * ((v**2)/l_beta) + 4*D*l_alpha
    
    expr2 = ( C1*(1-np.exp(-l_alpha*l_beta*l_tau)) + C2*l_tau )
    return(expr2)
###################################################################################    
##log10
'''def mom2_log(l_tau,v,D, l_lambdaB, l_lambdaD):
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB/(l_beta)    
    C1_2 = 2*((v/l_beta)**2) * (l_alpha - 1)/(l_alpha**2)
    C2_2 = 2*((1-l_alpha)/l_alpha) * ((v**2)/l_beta) + 4*D*l_alpha
    
    expr2  = 2*np.log10( (C1_2*(1-np.exp(-l_alpha*l_beta*l_tau)) + C2_2*l_tau )/2)

    return(expr2)'''
    
    
##############################################################################################
#corrected second moemnts log version
##############################################################################################
def mom2_log(l_tau, v, D, l_lambdaB, l_lambdaD):
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB / l_beta    
    C1 = 2 * ((v / l_beta)**2) * (l_alpha - 1) / (l_alpha**2)
    C2 = 2 * ((1 - l_alpha) / l_alpha) * ((v**2) / l_beta) + 4 * D * l_alpha
    
    # Calculate the second moment
    moment_expr =  (C1 * (1 - np.exp(-l_alpha * l_beta * l_tau)) + C2 * l_tau)
    
    # Compute and return the log10 of the second moment
    log_expr = np.log10(moment_expr)
    return log_expr
    
##############################################################################################
##############################################################################################
# The fourth moments and its log
###################################################################################
###################################################################################
def moment4(t, v0, D, lambdaB, lambdaD):
    C1 = 3 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -2 ) * ( 2 * D * lambdaB**2 + v0**2 * lambdaD )**2
    C2 = 3 * lambdaB**( -3 ) * lambdaD * ( lambdaB + lambdaD )**( -3 ) * ( 8 * D**2 * lambdaB**4 - 8 * D * v0**2 * lambdaB**2 * ( 2 * lambdaB + lambdaD ) + v0**4 * ( 3 * lambdaB**2 - 2 * lambdaB * lambdaD - 3 * lambdaD**2 ) )
    C3 = 3* lambdaB**( -4 ) * lambdaD * ( lambdaB + lambdaD )**( -4 ) * ( -8 * D**2 * lambdaB**5 + 8 * D * v0**2 * lambdaB**2 * ( 3 * lambdaB**2 + 3 * lambdaB * lambdaD + lambdaD**2 ) + v0**4 * ( -9 * lambdaB**3 - 7 * lambdaB**2 * lambdaD + 3 * lambdaB * lambdaD**2 + 3 * lambdaD**3 ) )
    C4 = 3/2 * v0**4 * lambdaB**( -2 ) * lambdaD * ( lambdaB + lambdaD )**( -1 )
    C5 = 6 * v0**4 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -1 )
    C6 = 0.0
    C7 = -3 * v0**2 / (lambdaB**( 4 ) * lambdaD * ( lambdaB + lambdaD )) * ( 8 * D * lambdaB**2  * lambdaD  + v0**2 * ( 2 * lambdaB**2 - 6 * lambdaB * lambdaD + 3 * lambdaD**2 ) )
    C8 = 6 * lambdaB / (lambdaD * ( lambdaB + lambdaD )**( 4 )) * ( v0**2 + 2 * D * lambdaD )**2
    expr = C1 * t**2 + C2 * t + C3 + C4 * t**2 * np.e**( -lambdaB * t ) + C5 * t * np.e**( -lambdaB * t ) + C6 * t * np.e**( -( lambdaB + lambdaD ) * t ) + C7 * np.e**( -lambdaB * t ) + C8 * np.e**( -( lambdaB + lambdaD ) * t )
    #print('a',[C1,C2<,C3,C4,C5,C6,C7,C8])
    return(expr)
    
def mom22_4_diff(l_tau, v, D, l_lambdaB, l_lambdaD):
    """
        Calculate a modified version of the second moment in logarithmic form.

        Parameters:
        l_tau, v, D, l_lambdaB, l_lambdaD: Same as in mom2_serg_log.

        Returns:
        float: Modified logarithmic second moment.
        """
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB/(l_beta)    
    C1_2 = 2*((v/l_beta)**2) * (l_alpha - 1)/(l_alpha**2)
    C2_2 = 2*((1-l_alpha)/l_alpha) * ((v**2)/l_beta) + 4*D*l_alpha
    
    expr2  = 2*( (C1_2*(1-np.exp(-l_alpha*l_beta*l_tau)) + C2_2*l_tau )/2)
    
    lambdaB = l_lambdaB
    lambdaD = l_lambdaD
    v0 = v
    t = l_tau
    
    C1 = 3 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -2 ) * ( 2 * D * lambdaB**2 + v0**2 * lambdaD )**2
    C2 = 3 * lambdaB**( -3 ) * lambdaD * ( lambdaB + lambdaD )**( -3 ) * ( 8 * D**2 * lambdaB**4 - 8 * D * v0**2 * lambdaB**2 * ( 2 * lambdaB + lambdaD ) + v0**4 * ( 3 * lambdaB**2 - 2 * lambdaB * lambdaD - 3 * lambdaD**2 ) )
    C3 = 3* lambdaB**( -4 ) * lambdaD * ( lambdaB + lambdaD )**( -4 ) * ( -8 * D**2 * lambdaB**5 + 8 * D * v0**2 * lambdaB**2 * ( 3 * lambdaB**2 + 3 * lambdaB * lambdaD + lambdaD**2 ) + v0**4 * ( -9 * lambdaB**3 - 7 * lambdaB**2 * lambdaD + 3 * lambdaB * lambdaD**2 + 3 * lambdaD**3 ) )
    C4 = 3/2 * v0**4 * lambdaB**( -2 ) * lambdaD * ( lambdaB + lambdaD )**( -1 )
    C5 = 6 * v0**4 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -1 )
    C6 = 0.0
    C7 = -3 * v0**2 / (lambdaB**( 4 ) * lambdaD * ( lambdaB + lambdaD )) * ( 8 * D * lambdaB**2  * lambdaD  + v0**2 * ( 2 * lambdaB**2 - 6 * lambdaB * lambdaD + 3 * lambdaD**2 ) )
    C8 = 6 * lambdaB * lambdaD**( -1 ) * ( lambdaB + lambdaD )**( -4 ) * ( v0**2 + 2 * D * lambdaD )**2
    expr = (C1 * t**2 + C2 * t + C3 + C4 * t**2 * np.e**( -lambdaB * t ) + C5 * t * np.e**( -lambdaB * t ) + C6 * t * np.e**( -( lambdaB + lambdaD ) * t ) + C7 * np.e**( -lambdaB * t ) + C8 * np.e**( -( lambdaB + lambdaD ) * t ))

    return(expr - expr2)
def to_optimize_mom22_4_diff(params):
    try:
        model_result = mom22_4_diff(tau_list, *params)
        return np.sum(np.abs(difference - (model_result)))
    except Exception as e:
        #print("Error encountered:", e)
        return 1e10


def mom22_4_diff_log(l_tau,v,D, l_lambdaB, l_lambdaD):
    l_beta = l_lambdaB + l_lambdaD
    l_alpha = l_lambdaB/(l_beta)    
    C1_2 = 2*((v/l_beta)**2) * (l_alpha - 1)/(l_alpha**2)
    C2_2 = 2*((1-l_alpha)/l_alpha) * ((v**2)/l_beta) + 4*D*l_alpha
    
    expr2  = 2*np.log10( (C1_2*(1-np.exp(-l_alpha*l_beta*l_tau)) + C2_2*l_tau )/2)
    
    lambdaB = l_lambdaB
    lambdaD = l_lambdaD
    v0 = v
    t = l_tau
    
    C1 = 3 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -2 ) * ( 2 * D * lambdaB**2 + v0**2 * lambdaD )**2
    C2 = 3 * lambdaB**( -3 ) * lambdaD * ( lambdaB + lambdaD )**( -3 ) * ( 8 * D**2 * lambdaB**4 - 8 * D * v0**2 * lambdaB**2 * ( 2 * lambdaB + lambdaD ) + v0**4 * ( 3 * lambdaB**2 - 2 * lambdaB * lambdaD - 3 * lambdaD**2 ) )
    C3 = 3* lambdaB**( -4 ) * lambdaD * ( lambdaB + lambdaD )**( -4 ) * ( -8 * D**2 * lambdaB**5 + 8 * D * v0**2 * lambdaB**2 * ( 3 * lambdaB**2 + 3 * lambdaB * lambdaD + lambdaD**2 ) + v0**4 * ( -9 * lambdaB**3 - 7 * lambdaB**2 * lambdaD + 3 * lambdaB * lambdaD**2 + 3 * lambdaD**3 ) )
    C4 = 3/2 * v0**4 * lambdaB**( -2 ) * lambdaD * ( lambdaB + lambdaD )**( -1 )
    C5 = 6 * v0**4 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -1 )
    C6 = 0.0
    C7 = -3 * v0**2 / (lambdaB**( 4 ) * lambdaD * ( lambdaB + lambdaD )) * ( 8 * D * lambdaB**2  * lambdaD  + v0**2 * ( 2 * lambdaB**2 - 6 * lambdaB * lambdaD + 3 * lambdaD**2 ) )
    C8 = 6 * lambdaB * lambdaD**( -1 ) * ( lambdaB + lambdaD )**( -4 ) * ( v0**2 + 2 * D * lambdaD )**2
    expr = np.log10(C1 * t**2 + C2 * t + C3 + C4 * t**2 * np.e**( -lambdaB * t ) + C5 * t * np.e**( -lambdaB * t ) + C6 * t * np.e**( -( lambdaB + lambdaD ) * t ) + C7 * np.e**( -lambdaB * t ) + C8 * np.e**( -( lambdaB + lambdaD ) * t ))
    #print('a',[C1,C2<,C3,C4,C5,C6,C7,C8])
    return(expr - expr2)


def mom4_log(t,v0,D,lambdaB,lambdaD):
    C1 = 3 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -2 ) * ( 2 * D * lambdaB**2 + v0**2 * lambdaD )**2
    C2 = 3 * lambdaB**( -3 ) * lambdaD * ( lambdaB + lambdaD )**( -3 ) * ( 8 * D**2 * lambdaB**4 - 8 * D * v0**2 * lambdaB**2 * ( 2 * lambdaB + lambdaD ) + v0**4 * ( 3 * lambdaB**2 - 2 * lambdaB * lambdaD - 3 * lambdaD**2 ) )
    C3 = 3* lambdaB**( -4 ) * lambdaD * ( lambdaB + lambdaD )**( -4 ) * ( -8 * D**2 * lambdaB**5 + 8 * D * v0**2 * lambdaB**2 * ( 3 * lambdaB**2 + 3 * lambdaB * lambdaD + lambdaD**2 ) + v0**4 * ( -9 * lambdaB**3 - 7 * lambdaB**2 * lambdaD + 3 * lambdaB * lambdaD**2 + 3 * lambdaD**3 ) )
    C4 = 3/2 * v0**4 * lambdaB**( -2 ) * lambdaD * ( lambdaB + lambdaD )**( -1 )
    C5 = 6 * v0**4 * lambdaB**( -2 ) * ( lambdaB + lambdaD )**( -1 )
    C6 = 0.0
    C7 = -3 * v0**2 / (lambdaB**( 4 ) * lambdaD * ( lambdaB + lambdaD )) * ( 8 * D * lambdaB**2  * lambdaD  + v0**2 * ( 2 * lambdaB**2 - 6 * lambdaB * lambdaD + 3 * lambdaD**2 ) )
    C8 = 6 * lambdaB / (lambdaD * ( lambdaB + lambdaD )**( 4 )) * ( v0**2 + 2 * D * lambdaD )**2
    expr = C1 * t**2 + C2 * t + C3 + C4 * t**2 * np.e**( -lambdaB * t ) + C5 * t * np.e**( -lambdaB * t ) + C6 * t * np.e**( -( lambdaB + lambdaD ) * t ) + C7 * np.e**( -lambdaB * t ) + C8 * np.e**( -( lambdaB + lambdaD ) * t )
    #print('a',[C1,C2<,C3,C4,C5,C6,C7,C8])
    return(np.log10(expr))

###################################################################################
###################################################################################
def intermittent2(nt,dt,mean_bal_sac,diffusion,rate21,rate12):
    diffusion = np.sqrt(2*diffusion)
    P1 = rate21/(rate12+rate21)
    if np.random.random()<P1:
        regime=1
        waitt   = -math.log(1.0-np.random.uniform(0.0,1.0,None))/rate12
    else:
        regime=2
        angle2 = np.random.randint(2)*math.pi
        waitt   = -math.log(1.0-np.random.uniform(0.0,1.0,None))/rate21

    dts = math.sqrt(dt)
    x = np.zeros(nt)
    y = np.zeros(nt)
    time_since_last_jump = 0
    for i in range(1,nt):
        angle = random.random()*2*math.pi
        time_since_last_jump += dt        
        if regime == 1:
            #diffu = diffusion*np.random.normal(0,1)*dts
            dx = np.random.normal(0,diffusion)*dts
            dy = np.random.normal(0,diffusion)*dts
            x[i] = x[i-1] + dx
            y[i] = y[i-1] + dy
            if time_since_last_jump> waitt:
                waitt   = -math.log(1.0-np.random.uniform(0.0,1.0,None))/rate21
                regime = 2
                angle2 = angle
                time_since_last_jump =0
        if regime == 2:
            angle3 = angle2 
            bal = mean_bal_sac*dt 
            dx = bal*math.cos(angle3)
            dy = bal*math.sin(angle3)
            x[i] = x[i-1] + dx
            y[i] = y[i-1] + dy

            if time_since_last_jump> waitt:
                waitt   = -math.log(1.0-np.random.uniform(0.0,1.0,None))/rate12
                time_since_last_jump =0
                regime = 1
        
    return(x,y)
###################################################################################
###################################################################################

def levy_flight_2D_2(n_redirections,n_max,lalpha,tmin,measuring_dt):
    if lalpha <= 1:
        print("alpha should be larger than 1")
        return("alpha should be larger than 1")
    

    t_redirection= tmin*(np.ones(n_redirections) - np.random.rand(n_redirections))**(1.0/(-lalpha+1))   
    cum_t_redirection = np.cumsum(t_redirection)


    angle = np.random.rand(len(t_redirection))*2*math.pi
    x_increments = t_redirection*np.cos(angle)
    y_increments = t_redirection*np.sin(angle)
    l_x_list = np.cumsum(x_increments)
    l_y_list = np.cumsum(y_increments)
    

    if n_max*measuring_dt < cum_t_redirection[-1]:

        x_measured = np.interp(np.arange(0,n_max*measuring_dt,measuring_dt),np.cumsum(t_redirection),l_x_list)
        y_measured = np.interp(np.arange(0,n_max*measuring_dt,measuring_dt),np.cumsum(t_redirection),l_y_list)
        
    else:
        
        n_max = int(cum_t_redirection[-1]/measuring_dt)
        #print("me<zasuring time greater than simulated time. n_max becomes " + str(n_max))
        x_measured = np.interp(np.arange(0,n_max*measuring_dt,measuring_dt),np.cumsum(t_redirection),l_x_list)
        y_measured = np.interp(np.arange(0,n_max*measuring_dt,measuring_dt),np.cumsum(t_redirection),l_y_list)
        #print("measuring time greater than simulated time.")
    return x_measured,y_measured,t_redirection
###################################################################################
###################################################################################

def r_square(l_emp_points, l_emp_fit):
    """
    Calculate the coefficient of determination, R-squared, which is a statistical measure of how well
    the regression predictions approximate the real data points.

    Parameters:
    l_emp_points (list or array): The empirical data points (observed values).
    l_emp_fit (list or array): The values predicted by the regression model.

    Returns:
    float: The R-squared value, ranging from 0 to 1, where 1 indicates a perfect fit.
    """
    l_num = np.mean((np.array(l_emp_points) - np.array(l_emp_fit)) ** 2)
    l_den = np.std(np.array(l_emp_points)) ** 2
    return 1 - l_num / l_den

     
def adjusted_r_square(l_emp_points, l_emp_fit, degrees_freedom):
    """
    Calculate the adjusted R-squared, which is a modified version of R-squared that has been
    adjusted for the number of predictors in the model. It provides a more accurate measure in
    the context of multiple regression.

    Parameters:
    l_emp_points (list or array): The empirical data points (observed values).
    l_emp_fit (list or array): The values predicted by the regression model.
    degrees_freedom (int): The degrees of freedom in the model, typically the number of predictors.

    Returns:
    float: The adjusted R-squared value, which accounts for the number of predictors.
    """
    n = len(l_emp_points)
    l_num = np.mean((np.array(l_emp_points) - np.array(l_emp_fit)) ** 2)
    l_den = np.std(np.array(l_emp_points)) ** 2
    rsqu = 1 - l_num / l_den
    return 1 - (1 - rsqu) * ((n - 1) / (n - degrees_freedom))
    
def powerl_fit(l_tau,l_k,l_a):
    return(l_k*np.power(2,l_tau*l_a))
    
###################################################################################
###################################################################################




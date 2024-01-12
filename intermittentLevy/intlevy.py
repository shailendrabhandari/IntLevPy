
# Necessary library imports for the intermittent and Levy functions
import numpy as np
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
import scipy.optimize  # If optimization is required
import math  # If mathematical functions are required
import random  # If random number generation is required
# Additional imports if needed

# Intermittent function definitions

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




def HMM_first_guess(dS):
    detection_list = []
    kmin_list = []
    raw_threshold_array = np.logspace(np.log10(np.abs(np.min(dS)) + 0.0001), np.log10(np.max(dS)), 100)
    threshold_array = raw_threshold_array / max(dS)
    ldetection, ldetectionfisher, lkmin, lfishermin = form_groups(dS, threshold_array, True, 'v', 'title', '%.2f')

    lthreshold = raw_threshold_array[np.argmin(ldetection)]
    d__ss = dS
    binary_vector = np.array([max(min(int(i - lthreshold + 1), 1), 0) for i in d__ss])

    guess_dS_fix = dS[binary_vector == 0]
    guess_dS_sacc = dS[binary_vector == 1]
    guess_logdS_fix = guess_dS_fix[guess_dS_fix > 0]
    guess_logdS_sacc = guess_dS_sacc[guess_dS_sacc > 0]
    est_fix_mu = np.mean(guess_logdS_fix)
    est_fix_sigma = np.std(guess_logdS_fix)
    est_sacc_mu = np.mean(guess_logdS_sacc)
    est_sacc_sigma = np.std(guess_logdS_sacc)
    Nfix = len(binary_vector) - np.sum(binary_vector)
    Nsacc = np.sum(binary_vector)
    Ntransi = int(np.sum(np.abs(binary_vector[1:] - binary_vector[:-1])) / 2)
    est_lambda_B = -np.log(1 - Ntransi / Nsacc)
    est_lambda_D = -np.log(1 - Ntransi / Nfix)

    return (est_fix_mu, est_fix_sigma, est_lambda_B, est_lambda_D)


def HMM_first_guess_revise(dS):
    # pomegranate package changed substantially. New tutorial on https://github.com/jmschrei/pomegranate/blob/master/docs/tutorials/B_Model_Tutorial_4_Hidden_Markov_Models.ipynb

    detection_list = []
    kmin_list = []
    raw_threshold_array = np.logspace(np.log10(np.abs(np.min(dS)) + 0.0001), np.log10(np.max(dS)), 100)
    threshold_array = raw_threshold_array / max(dS)
    ldetection, ldetectionfisher, lkmin, lfishermin = form_groups(dS, threshold_array, True, 'v', 'title', '%.2f')

    lthreshold = raw_threshold_array[np.argmin(ldetection)]
    d__ss = dS
    binary_vector = np.array([max(min(int(i - lthreshold + 1), 1), 0) for i in d__ss])

    guess_dS_fix = dS[binary_vector == 0]
    guess_dS_sacc = dS[binary_vector == 1]
    guess_logdS_fix = np.log(guess_dS_fix[guess_dS_fix > 0])
    guess_logdS_sacc = np.log(guess_dS_sacc[guess_dS_sacc > 0])
    est_fix_mu = np.mean(guess_logdS_fix)
    est_fix_sigma = np.std(guess_logdS_fix)
    est_sacc_mu = np.mean(guess_logdS_sacc)
    est_sacc_sigma = np.std(guess_logdS_sacc)
    Nfix = len(binary_vector) - np.sum(binary_vector)
    Nsacc = np.sum(binary_vector)
    Ntransi = int(np.sum(np.abs(binary_vector[1:] - binary_vector[:-1])) / 2)
    est_lambda_B = -np.log(1 - Ntransi / Nsacc)
    est_lambda_D = -np.log(1 - Ntransi / Nfix)

    prbd = np.exp(-est_lambda_B)
    prbb = 1 - prbd
    prdb = np.exp(-est_lambda_D)
    prdd = 1 - prdb
    s1 = State(LogNormalDistribution(est_fix_mu, est_fix_sigma))  # fixations
    s2 = State(LogNormalDistribution(est_sacc_mu, est_sacc_sigma))  # saccades
    model = HiddenMarkovModel()
    model.add_states(s1, s2)
    model.add_transition(model.start, s1, 0.5)
    model.add_transition(model.start, s2, 0.5)
    model.add_transition(s1, s1, prdd)
    model.add_transition(s1, s2, prdb)
    model.add_transition(s2, s1, prbd)
    model.add_transition(s2, s2, prbb)
    model.add_transition(s2, model.end, 0.5 * len(dS[dS > 0]))
    model.add_transition(s1, model.end, 0.5 * len(dS[dS > 0]))
    model.bake()
    model.fit([dS[dS > 0]])
    temp_vel = dS[dS > 0]

    s1_mu = model.states[0].distribution.parameters[0]
    s2_mu = model.states[1].distribution.parameters[0]

    if s2_mu > s1_mu:
        HMM_est_fix_mu = model.states[0].distribution.parameters[0]
        HMM_est_fix_sigma = model.states[0].distribution.parameters[1]
        HMM_est_sacc_mu = model.states[1].distribution.parameters[0]
        HMM_est_sacc_sigma = model.states[1].distribution.parameters[1]
        HMM_bin_vec = np.swapaxes(model.viterbi(dS[dS > 0])[1], 0, 1)[0][1:-1]




    else:
        HMM_est_fix_mu = model.states[1].distribution.parameters[0]
        HMM_est_fix_sigma = model.states[1].distribution.parameters[1]
        HMM_est_sacc_mu = model.states[0].distribution.parameters[0]
        HMM_est_sacc_sigma = model.states[0].distribution.parameters[1]
        HMM_bin_vec = np.swapaxes(model.viterbi(dS[dS > 0])[1], 0, 1)[0][1:-1]
        HMM_bin_vec = np.abs(np.array(HMM_bin_vec) - np.ones(len(HMM_bin_vec)))

    Nfix = len(HMM_bin_vec) - np.sum(HMM_bin_vec)
    Nsacc = np.sum(HMM_bin_vec)
    Ntransi = int(np.sum(np.abs(HMM_bin_vec[1:] - HMM_bin_vec[:-1])) / 2)
    HMM_est_lambda_B = -np.log(1 - Ntransi / Nsacc)
    HMM_est_lambda_D = -np.log(1 - Ntransi / Nfix)
    HMM_logdS_fix = temp_vel[HMM_bin_vec == 0]
    HMM_logdS_sacc_list = temp_vel[HMM_bin_vec == 1]

    return (HMM_est_fix_mu, HMM_est_fix_sigma, HMM_est_sacc_mu, HMM_est_sacc_sigma, HMM_est_lambda_B, HMM_est_lambda_D)











loc_params = np.swapaxes(np.loadtxt('intermittent_est_params.txt'),0,1)
mean_params = np.mean(np.log(loc_params),axis=1)
std_params = np.std(np.log(loc_params),axis=1)
normed_loc_params = (np.swapaxes(np.log(loc_params),0,1)-mean_params)/std_params
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(normed_loc_params)
new_data = kde.sample()

#og_params = list(np.loadtxt('og_params_int.txt'))
#est_params = list(np.loadtxt('int_process_int_fit_params.txt'))
og_params = []
est_params = []
est_params2 = []
list_est_params = []
list_est_params2 = []
diff_params = []
est_diff_params =[]       
diff_diff_params=[]
diff_est_params=[]
r_square_list=[]




    



integration_factor = 1 ## check this integration factor. 
g_tau = 1
#make this random
#g_v0 = 5
#g_D = 1
#g_lambda_B = 0.05
#g_lambda_D = 0.005
loc_params = np.swapaxes(np.loadtxt('intermittent_est_params.txt'),0,1)
mean_params = np.mean(np.log(loc_params),axis=1)
std_params = np.std(np.log(loc_params),axis=1)
normed_loc_params = (np.swapaxes(np.log(loc_params),0,1)-mean_params)/std_params
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(normed_loc_params)

N = 100000
N_iter = 1000
re_estimations = 5



opt_iter = 20


redim_max = 5
redim_min = 1.0/redim_max
#adj_r_square_int_lev = list(np.loadtxt('r_square_int_process_lev_fit.txt'))
#adj_r_square_int_int = list(np.loadtxt('r_square_int_process_int_fit.txt'))
#lev_params_int = list(np.loadtxt('int_process_int_lev_params.txt'))
#int_params_int = list(np.loadtxt('int_process_int_fit_params.txt'))

adj_r_square_int_lev = []
adj_r_square_int_int = []
lev_params_int = []
int_params_int = []

tau_list = np.arange(1,30)
#tau_list = np.power(1.05,np.arange(1,101)).astype(int)

#create grid with "tighter" parameters
#create 200 initial points at ranfrom from the "tighter" parameter grid


for itera in range(0,N_iter):
#for itera in range(840,841):
   # factor1 =   1.5*random.random()  + 0.5 
   # factor2 =   1.5*random.random()  + 0.5 
   # factor3 =   1.5*random.random()  + 0.5 
   # factor4 =   1.5*random.random()  + 0.5 
    
    factor1 =   1
    factor2 =   1
    factor3 =   1
    factor4 =   1
    new_data = kde.sample()
    [[g_v0,g_D,g_lambda_B,g_lambda_D]] = np.exp(new_data*std_params + mean_params)
    og_g_v0,og_g_D,og_g_lambda_B,og_g_lambda_D = g_v0,g_D,g_lambda_B,g_lambda_D

    #x_loc = np.repeat(np.loadtxt('x_intermittent'+str(itera)+'.txt'),integration_factor)
    #y_loc = np.repeat(np.loadtxt('y_intermittent'+str(itera)+'.txt'),integration_factor)

    x_loc,y_loc = intermittent2(N*integration_factor,g_tau/integration_factor,g_v0*factor1,g_D*factor2,g_lambda_B*factor3,g_lambda_D*factor4)
    
    
    x_list = []
    y_list = []
    dx_list = []
    dy_list = []
    dx4 = []
    dy4 = []
    dx2 = []
    dy2 = []

    N_max = 100

    for i in tau_list:
        x_list.append(x_loc[::i*integration_factor])
        y_list.append(y_loc[::i*integration_factor])
        dx_list.append(np.diff(x_loc[::i*integration_factor]))
        dy_list.append(np.diff(y_loc[::i*integration_factor]))

        dx4.append(np.mean(np.array(dx_list[-1])**4))
        dy4.append(np.mean(np.array(dy_list[-1])**4))
        dx2.append(np.mean(np.array(dx_list[-1])**2))
        dy2.append(np.mean(np.array(dy_list[-1])**2))

    dx4 = np.array(dx4)
    dy4 = np.array(dy4)
    dx4_log = np.log(dx4)
    dy4_log = np.log(dy4)

    dx2 = np.array(dx2)
    dy2 = np.array(dy2)
    dx2_log = np.log(dx2)
    dy2_log = np.log(dy2)

    list_est_params2 = []
    list_est_params = []
    for rrr in range(re_estimations):

        g_emp_points_x = np.array(dx4_log)
        g_emp_points_y = np.array(dy4_log)

        #start initial parameter search grid

        print(g_v0,g_D,g_lambda_B,g_lambda_D)

        ########## Here the HMM estimations ##########

        dS= np.sqrt(np.array(dx_list[0])**2 + np.array(dy_list[0])**2)
        lest_fix_mu,lest_fix_sigma,lest_lambda_B,lest_lambda_D = HMM_first_guess(dS)
        g_v0 = 5*lest_fix_mu
        g_D = lest_fix_sigma
        g_lambda_B = lest_lambda_B
        g_lambda_D = lest_lambda_D
        print(g_v0,g_D,g_lambda_B,g_lambda_D)
        
        ##############################################

        
        grid_nodes = 20
        redim_power = redim_max**(2/grid_nodes)
        redim_vec = np.power(redim_power,np.arange(-grid_nodes*0.5,grid_nodes*0.5))

        gl_v_vec = g_v0 * redim_vec
        gl_d_vec = g_D * redim_vec
        gl_lambdaB_vec = g_lambda_B * redim_vec
        gl_lambdaD_vec = g_lambda_D * redim_vec

        param_grid = np.array(np.meshgrid(gl_v_vec,gl_d_vec,gl_lambdaB_vec,gl_lambdaD_vec)).T.reshape(-1,4)
        temp_adjus_r_int_sq_x = []
        temp_adjus_r_int_sq_x_diff = []
        temp_adjus_r_int_sq_y = []
        del_max_r_s = 0
        del_max_r_s_diff = 0
        print(g_v0,g_D,g_lambda_B,g_lambda_D)
        print(itera)







        




        #new_data = data[0][['Eyetracker timestamp','Gaze point X','Gaze point Y','Eye movement type']]


        
        g_emp_points_x = np.array(dx4_log)
        global_logdx4 = dx4_log
        global_tau_list = tau_list
        temp_adjus_r_int_sq_x = []
        reg2 = LinearRegression().fit(np.log(np.array(tau_list)).reshape(-1, 1), dx4_log)
        coef = reg2.coef_[0]
        intercept = reg2.intercept_
        g_lev_fit_x  = coef*np.log(np.array(tau_list)) + intercept
        r_levy = adjusted_r_square(dx4_log,g_lev_fit_x,2)

    ######### optimizaiton ##############
        global_logdx4 = dx4_log
        global_logdx2 = dx2_log
        global_tau_list = tau_list

        gredim = 10**2
        ggridlen = 3
        rranges = [(g_v0/gredim, g_v0*gredim), (g_D/gredim, g_D*gredim),(g_lambda_B/gredim, g_lambda_B*gredim),(g_lambda_D/gredim, g_lambda_D*gredim)]

        ######## brute force
        #best_popt_x = optimize.brute(to_optimize_mom4_serg_log, rranges, full_output=False,finish=optimize.fmin,Ns = ggridlen)

        ######## shgo
        #gaga = scipy.optimize.shgo(to_optimize_mom4_serg_log,[(g_v0/gredim, g_v0*gredim), (g_D/gredim, g_D*gredim),(g_lambda_B/gredim, g_lambda_B*gredim),(g_lambda_D/gredim, g_lambda_D*gredim)])
        #best_popt_x = gaga['x']

        ######## double annealing
        list_fun = []
        list_poptx = []
        ####### create grid ###########
        grid_nodes = ggridlen
        redim_power = redim_max**(0.5/grid_nodes)
        redim_vec = np.power(redim_power,np.arange(-grid_nodes*0.5,grid_nodes*0.5))
        gl_v_vec = g_v0 * redim_vec
        gl_d_vec = g_D * redim_vec
        gl_lambdaB_vec = g_lambda_B * redim_vec
        gl_lambdaD_vec = g_lambda_D * redim_vec
        print('first calc')

    
        ####### First calculation - fourth moment ###########
        param_grid = np.array(np.meshgrid(gl_v_vec,gl_d_vec,gl_lambdaB_vec,gl_lambdaD_vec)).T.reshape(-1,4)
        #for ki in range(len(param_grid)):
            #print(i)
        #    while True:
        #        try:
        #            with np.errstate(all='raise'):
        #                 la = scipy.optimize.dual_annealing(to_optimize_mom4_serg_log,rranges,x0=param_grid[ki])
        #            break
        #        except Exception:
        #            print("Warning detected")
        #            continue
        la = None
        while la is None:   
            try:
                la = scipy.optimize.dual_annealing(to_optimize_mom4_serg_log,rranges)
            except RuntimeWarning:
                1
                
                
        list_fun.append(la['fun'])
        list_poptx.append(la['x'])
        lbb = la
        best_popt_x = list_poptx[np.argmin(list_fun)]
        print(list_fun)
        print(np.argmin(list_fun))
        print(best_popt_x)
        
        
        g_git_fit = mom4_serg_log(np.array(tau_list),best_popt_x[0],best_popt_x[1],best_popt_x[2],best_popt_x[3])
        current_r_square_fourth = adjusted_r_square(dx4_log,g_git_fit,4) 
        g_git_fit = mom2_serg_log(np.array(tau_list),best_popt_x[0],best_popt_x[1],best_popt_x[2],best_popt_x[3])
        current_r_square_second = adjusted_r_square(dx2_log,g_git_fit,4) 

        if np.sum(np.isnan(g_git_fit))>0:
            print('NaN in fit')         

        param_grid3 = np.array(np.meshgrid(gl_v_vec,gl_d_vec,gl_lambdaB_vec)).T.reshape(-1,3)
        param_grid2 = np.array(np.meshgrid(gl_d_vec,gl_lambdaD_vec)).T.reshape(-1,2)

        last_r = 0.5*current_r_square_fourth + 0.5*current_r_square_second
        
        tos_lambdaD  = best_popt_x[3]
        ################# Optimization 1 #######################
        for klj in range(opt_iter):
            ############# fourth #################
            list_fun = []
            list_poptx = []
            print('first fourth '+str(klj))
            for ki in range(len(param_grid3)):
                la = None
                while la is None:   
                    try:
                        la = scipy.optimize.dual_annealing(to_optimize_mom4_serg_log_vdl,rranges[:3],x0=param_grid3[ki])
                        list_fun.append(la['fun'])
                        list_poptx.append(la['x'])
                    except RuntimeWarning:
                        1


            temp_popt_x= list_poptx[np.argmin(list_fun)]
            g_git_fit = mom4_serg_log(np.array(tau_list),temp_popt_x[0],temp_popt_x[1],temp_popt_x[2],tos_lambdaD)
            new_r_square = adjusted_r_square(dx4_log,g_git_fit,4)            
            if np.sum(np.isnan(g_git_fit))==0:
                if new_r_square > current_r_square_fourth-0.02:
                    current_r_square_fourth =new_r_square
                    best_popt_x[0] = list_poptx[np.argmin(list_fun)][0]
                    best_popt_x[1] = list_poptx[np.argmin(list_fun)][1]
                    best_popt_x[2] = list_poptx[np.argmin(list_fun)][2]
                    best_popt_x[3] =tos_lambdaD
            else:
                print('NaN in fit')   
                
            print(best_popt_x)

            ########### second ################
            print('first second '+str(klj))
            tos_v = best_popt_x[0]
            tos_lambdaB = best_popt_x[2]    
            list_fun = []
            list_poptx = []

            for kii in range(len(param_grid2)):
                la = None
                while la is None:   
                    try:
                        la = scipy.optimize.dual_annealing(to_optimize_second_ld,[(rranges[1][0],rranges[1][1]),(rranges[3][0],rranges[3][1])],x0=param_grid2[kii])  
                        list_fun.append(la['fun'])
                        list_poptx.append(la['x'])
                    except RuntimeWarning:
                        1
                    
            g_git_fit = mom2_serg_log(np.array(tau_list),best_popt_x[0],list_poptx[np.argmin(list_fun)][0],best_popt_x[2],list_poptx[np.argmin(list_fun)][1])
            new_r_square = adjusted_r_square(dx2_log,g_git_fit,4)

            if np.sum(np.isnan(g_git_fit))==0:
                if new_r_square > current_r_square_second-0.02:
                    current_r_square_second =new_r_square
                    
                    tos_lambdaD = list_poptx[np.argmin(list_fun)][1]
                    best_popt_x[1] = list_poptx[np.argmin(list_fun)][0]
                    best_popt_x[3] = tos_lambdaD
            else:
                print('NaN in fit')   
                
            print(best_popt_x)

        param_grid3 = np.array(np.meshgrid(gl_v_vec,gl_lambdaB_vec)).T.reshape(-1,2)
        #param_grid3 = np.array(np.meshgrid(gl_v_vec,gl_d_vec,gl_lambdaB_vec)).T.reshape(-1,3)
        param_grid2 = np.array(np.meshgrid(gl_lambdaB_vec,gl_lambdaD_vec)).T.reshape(-1,2)
        tos_v = best_popt_x[0]
        tos_D = best_popt_x[1]
        tos_lambdaB = best_popt_x[2]
        tos_lambdaD = best_popt_x[3]
        est_params.append([best_popt_x[0],best_popt_x[1],best_popt_x[2],best_popt_x[3]])

        g_int_fit_x4 = mom4_serg_log(np.array(tau_list),est_params[-1][0],est_params[-1][1],est_params[-1][2],est_params[-1][3])
        g_int_fit_x2 = mom2_serg_log(np.array(tau_list),est_params[-1][0],est_params[-1][1],est_params[-1][2],est_params[-1][3])
        r_int4 = adjusted_r_square(dx4_log,g_int_fit_x4,4)
        r_int2 = adjusted_r_square(dx2_log,g_int_fit_x2,4)
        
        if 0.5*(r_int4 +r_int2)<last_r-0.05:
            est_params[-1] = est_params[-2]
            tos_v,tos_D,tos_lambdaB,tos_lambdaD = est_params[-2]
        else:
            last_r = 0.5*(r_int4 +r_int2)
        print('curret_r = ' + str(  r_int4 +r_int2))


        current_r_square_fourth
        current_r_square_second
        ################# Optimization 2 #######################
        for klj in range(opt_iter):
            print('second fourth '+str(klj))
            ############# fourth #################
            list_fun = []
            list_poptx = []
            for ki in range(len(param_grid3)):
                try:
                    la = scipy.optimize.dual_annealing(to_optimize_mom4_serg_log_vl,[rranges[0],rranges[1]],x0=param_grid3[ki])
                    list_fun.append(la['fun'])
                    list_poptx.append(la['x'])
                except RuntimeWarning:
                    1

            temp_popt_x= list_poptx[np.argmin(list_fun)]
            g_git_fit = mom4_serg_log(np.array(tau_list),temp_popt_x[0],tos_D,temp_popt_x[1],tos_lambdaD)
            new_r_square = adjusted_r_square(dx4_log,g_git_fit,4)


            temp_popt_x= list_poptx[np.argmin(list_fun)]
            g_git_fit = mom4_serg_log(np.array(tau_list),list_poptx[np.argmin(list_fun)][0] ,tos_D,list_poptx[np.argmin(list_fun)][1] ,tos_lambdaD)
            new_r_square = adjusted_r_square(dx4_log,g_git_fit,4)            
            
                
            if np.sum(np.isnan(g_git_fit))==0:
                if new_r_square > current_r_square_fourth - 0.02:
                    current_r_square_fourth =new_r_square
                    tos_v =  list_poptx[np.argmin(list_fun)][0] 
                    tos_lambdaB = list_poptx[np.argmin(list_fun)][1] 


                    best_popt_x[0] = tos_v
                    best_popt_x[1] = tos_D
                    best_popt_x[2] = tos_lambdaB
                    best_popt_x[3] = tos_lambdaD
            else:
                print('NaN in fit')                

            ########### second ################
            
            print(best_popt_x)
            list_fun = []
            list_poptx = []
            print('second second '+str(klj))
            for kii in range(len(param_grid2)):
                try:
                    la = scipy.optimize.dual_annealing(to_optimize_second_ll,[(rranges[2][0],rranges[2][1]),(rranges[3][0],rranges[3][1])])  
                    list_fun.append(la['fun'])
                    list_poptx.append(la['x'])
                except:
                    1

            g_git_fit = mom2_serg_log(np.array(tau_list),best_popt_x[0],best_popt_x[1],list_poptx[np.argmin(list_fun)][0],list_poptx[np.argmin(list_fun)][1])
            new_r_square = adjusted_r_square(dx2_log,g_git_fit,4)

            #print('second_iteration. New rsquare:'+ str(new_r_square) +'. Current rsquare:'+ str(current_r_square))
            if np.sum(np.isnan(g_git_fit))==0:
                if new_r_square > current_r_square_second-0.02:
                    current_r_square_second =new_r_square
                    tos_lambdaB =  list_poptx[np.argmin(list_fun)][0] 
                    tos_lambdaD = list_poptx[np.argmin(list_fun)][1]
                    best_popt_x[0] = tos_v
                    best_popt_x[1] = tos_D
                    best_popt_x[2] = tos_lambdaB
                    best_popt_x[3] = tos_lambdaD
            print(best_popt_x)
        ######################################################################     


            #power = 0.5
            #v_mean_power = np.mean((np.array(dx_list[0])**2+ np.array(dy_list[0])**2)**(0.5*power))
            #D_next_est = (np.abs((v_mean_power - ((tos_v*tau_list[0])**power)*(tos_lambdaD / (tos_lambdaB + tos_lambdaD))) / (tos_lambdaB / (tos_lambdaB + tos_lambdaD)))**(1/power))/np.sqrt(tau_list[0])
            power = 0.5


            v_mean_power = np.mean( (np.diff(x_loc)**2 +np.diff(y_loc)**2  )**(0.5*power))


            adj_lbB = 1/ (1/tos_lambdaB - 0.7)
            adj_lbD = 1/ (1/tos_lambdaD + 0.7)
            lbB = adj_lbB
            lbD = adj_lbD
            tmeanB = lbD / (lbB + lbD)
            tmeanD = lbB / (lbB + lbD)


            D_next_est = 0.25*(np.abs( (v_mean_power*1.1 - (tos_v**power)*tmeanB)/ tmeanD)**(2/power))

            emp_v_mean_power = 0.91*( (tos_v**0.5) *tmeanB + ((4*D_next_est)**0.25)*tmeanD)

            #print(D_next_est)
            
            g_int_fit_x4 = mom4_serg_log(np.array(tau_list),best_popt_x[0],D_next_est,best_popt_x[2],best_popt_x[2])
            g_int_fit_x2 = mom2_serg_log(np.array(tau_list),best_popt_x[0],D_next_est,best_popt_x[2],best_popt_x[2])
            r_int4 = adjusted_r_square(dx4_log,g_int_fit_x4,4)
            r_int2 = adjusted_r_square(dx2_log,g_int_fit_x2,4)


            
            
            
            if np.sum(np.isnan(g_git_fit))==0:
                if 0.5*(r_int4 +r_int2)<last_r-0.03:
                    last_r = 0.5*(r_int4 +r_int2)
                    
                    tos_D = 0.7*D_next_est + 0.4*best_popt_x[1]
                    best_popt_x[0] = tos_v
                    best_popt_x[1] = tos_D
                    best_popt_x[2] = tos_lambdaB
                    best_popt_x[3] = tos_lambdaD
            print(best_popt_x)

        est_params2.append([best_popt_x[0],best_popt_x[1],best_popt_x[2],tos_lambdaD])
        

            
        print('curret_r = ' + str(  r_int4 +r_int2))

        list_est_params2.append(est_params2[-1])
        list_est_params.append(est_params[-1])

    try:
        parsed_est_params2 = parse_trials(list_est_params2,10)
    except:
        parsed_est_params2 = [-1,-1,-1,-1]
    try:
        parsed_est_params = parse_trials(list_est_params,10)
    except:
        parsed_est_params = [-1,-1,-1,-1]


    ##########  Saving   #################
    print(i)
    og_params.append([og_g_v0,og_g_D,og_g_lambda_B,og_g_lambda_D])
    lev_params_int.append([coef,intercept])

    print(og_g_v0,og_g_D,og_g_lambda_B,og_g_lambda_D)

    print(parsed_est_params2)
    print(parsed_est_params)
    best_popt_x = parsed_est_params2




    g_int_fit_x = mom4_serg_log(np.array(tau_list),parsed_est_params[0],parsed_est_params[1],parsed_est_params[2],parsed_est_params[3])
    g_lev_fit_x  = coef*np.log(np.array(tau_list)) + intercept
    r_levy = adjusted_r_square(dx4_log,g_lev_fit_x,2)
    r_int = adjusted_r_square(dx4_log,g_int_fit_x,4)
    

    #print(popt_y)
    #print(popt_y_diff)


    reg2 = LinearRegression().fit(np.log(np.array(tau_list)).reshape(-1, 1), dx4_log)
    coef = reg2.coef_[0]
    intercept = reg2.intercept_
    lev_params_int.append([coef,intercept])
    g_lev_fit_x  = coef*np.log(np.array(tau_list)) + intercept
    g_int_fit_x = mom4_serg_log(np.array(tau_list),parsed_est_params[0],parsed_est_params[1],parsed_est_params[2],parsed_est_params[3])
    adj_r_square_int_lev.append(adjusted_r_square(dx4_log,g_lev_fit_x,2))
    adj_r_square_int_int.append(adjusted_r_square(dx4_log,g_int_fit_x,4))
    
    #np.savetxt('final_try2_x_intermittent'+str(itera)+'.txt',x_loc[::integration_factor])
    #np.savetxt('final_try2_y_intermittent'+str(itera)+'.txt',y_loc[::integration_factor])
    #np.savetxt('final_try2_int_process_int_fit_params'+str(itera)+'.txt',parsed_est_params2)
    #np.savetxt('final_try2_int_process_int_fit_params_first_itera'+str(itera)+'.txt',parsed_est_params)
    #np.savetxt('final_try2_og_params_int'+str(itera)+'.txt',og_params)
    #np.savetxt('r_square_int_process_lev_fit.txt',adj_r_square_int_lev)
    #np.savetxt('r_square_int_process_int_fit.txt',adj_r_square_int_int)
    


xtemp = np.loadtxt('x_intermittent2.txt')
    #np.savetxt('y_intermittent'+str(itera)+'.txt',y_loc[::integration_factor])

plt.hist(np.array(adj_r_square_int_int)-np.array(adj_r_square_int_lev),10,label='intermittent_process')
plt.title(r'Adjusted-$R^2$')
plt.legend()
plt.ylabel('frequency')
plt.xlabel(r'Adjusted-$R^2$')
plt.savefig('synth_intm_rsquare.png')

plt.hist(np.array(adj_r_square_int_int) - np.array(adj_r_square_int_lev) ,40,label='intermittent-fit',alpha=0.5)
#plt.hist(adj_r_square_int_lev,40,label='levy-fit',alpha=0.5)
plt.title(r'Adjusted-$R^2$')
plt.legend()
plt.ylabel('frequency')
plt.xlabel(r'Adjusted-$R^2$')


tau_list = np.power(2,np.arange(10))
o_alpha = 2
o_t_mins = 0.007
adj_r_square_lev_lev = []
adj_r_square_lev_int = []
lev_params_lev = []
int_params_lev = []
u_fourth_levy_list = []
u_square_theor_levy_list = []
dy_lev = []
dx_lev = []
int_predict_lev = []
lev_predict_lev = []
u_square_lev_list = []
u_fourth_lev_list = []
len_vector= 10


for factor1 in np.arange(0,1,1.0/325):
    l_alpha = o_alpha + factor1
    for factor2 in np.arange(0,0.008,1):
        g_t_mins = 1.5*o_t_mins - 0.5*factor1*o_t_mins
        print(factor1,factor2)

        for i in range(22):
            lx_lev,ly_lev,lt_lev = levy_flight_2D_2(int(1500000*l_alpha*0.01/g_t_mins),150000,l_alpha,g_t_mins,1)
            #levy_flight_2D_2(n_redirections,n_max,lalpha,tmin,measuring_dt

            ldy_lev = np.diff(ly_lev)
            ldx_lev = np.diff(lx_lev)
            dy_lev.append(ldy_lev)
            dx_lev.append(ldx_lev)
        y_lev = np.cumsum(np.hstack(dy_lev))
        x_lev = np.cumsum(np.hstack(dx_lev))


        u_square_lev = []
        u_fourth_lev = []
        taus_levy = []
        #N = 3000
        #taumax = int(np.log2(len(y_lev)/N))
        tau_max = 10
        for iii in np.arange(tau_max):
            ttau = int(2**(iii))
            num = np.mean(np.diff(y_lev[::ttau])**2 + np.diff(x_lev[::ttau])**2)
            u_square_lev.append(num)
            u_fourth_lev.append(np.mean( (np.diff(y_lev[::ttau])**2 + np.diff(x_lev[::ttau])**2)**2 ) )
            #print(iii)

            

###########################
            


        initial_conditions = []
        for k in range(len_vector-3):
            blabla = first_estimate_ple_averages_inc(u_square_lev[k:k+3],tau_list[k:k+3])
            if blabla[1]:
                initial_conditions.append(blabla[0])


        if len(initial_conditions)!=0:

            best_params = optimize(u_square_lev,tau_list,initial_conditions[0],500)
            curr_min = np.mean( (np.log2(u_square_lev) - log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]))**2)
            best_params_index = 0
            opt_meu = 1
            for k in range(len(initial_conditions)):
                popt3 = optimize(u_square_lev,tau_list,initial_conditions[k],500)
                new_min = np.mean((np.log2(u_square_lev) - log2_moment_scaling(tau_list,popt3[0],popt3[1] ,popt3[2]))**2 )
                if new_min != new_min:
                    new_min = curr_min + 1000
                if new_min < curr_min:
                    best_params = popt3
                    curr_min = new_min
                    best_params_index = k
                    opt_meu = 1

                try:
                    popt3, pcov = scipy.optimize.curve_fit(log2_moment_scaling, tau_list, np.log2(u_square_lev),p0=initial_conditions[k], maxfev=500000)
                    new_min = np.mean((np.log2(u_square_lev) - log2_moment_scaling(tau_list,popt3[0],popt3[1] ,popt3[2]))**2 )
                    if new_min != new_min:
                        new_min = curr_min + 1000
                    if new_min < curr_min:
                        best_params = popt3
                        curr_min = new_min
                        best_params_index = k
                        opt_meu = 0
                except RuntimeError:
                    'a'
            print(best_params_index,opt_meu)




        if len(initial_conditions)==0:
            try:

                popt3, pcov = scipy.optimize.curve_fit(log2_moment_scaling, tau_list, np.log2(u_square_lev), maxfev=50000000)
            except RuntimeError:
                print('minimum could not be found')


        reg2 = LinearRegression().fit(np.log2(np.array(tau_list)).reshape(-1, 1), np.log2(u_square_lev))
        coef = reg2.coef_[0]
        intercept = reg2.intercept_



        adj_r_square_lev_lev.append(adjusted_r_square(np.log2(u_square_lev),np.arange(len(tau_list))*coef + intercept,2))
        adj_r_square_lev_int.append(adjusted_r_square(np.log2(u_square_lev),np.arange(len(tau_list))*coef + intercept,2))
        lev_params_lev.append([coef,intercept])
        lev_predict_lev.append(np.arange(len(tau_list))*coef + intercept)

        u_square_lev_list.append(u_square_lev)
        u_fourth_lev_list.append(u_fourth_lev)



        u_square_theor_levy_list.append(l_alpha)
        int_predict_lev.append(log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]))
        int_params_lev.append(best_params)





        



gamma_lev = np.log10(np.array(adj_r_square_lev_lev)/ np.array(adj_r_square_lev_int))
plt.hist(gamma_int,40)


#np.savetxt('adj_r_square_lev_lev.txt',adj_r_square_lev_lev)
#np.savetxt('adj_r_square_lev_int.txt',adj_r_square_lev_int)
#np.savetxt("gamma_lev.txt",gamma_lev)
#np.savetxt('u_square_lev_list.txt',u_square_lev_list)
#np.savetxt('u_fourth_lev_list.txt',u_fourth_lev_list)

#np.savetxt('int_params_lev.txt',int_params_lev)
#np.savetxt('lev_params_lev.txt',lev_params_lev)
#np.savetxt('int_predict_lev.txt',int_predict_int)
#np.savetxt('lev_predict_lev.txt',lev_predict_int)

















len_vector= 10
lambda1 = 5*0.1**3
lambda2 = 0.1**2
v = 10
D = 1
g_beta = lambda1 + lambda2
g_alpha = lambda2/(g_beta)


integration_time=0.25
skip = int(1/integration_time) #integration time = Delta t in text
tau_list = np.power(2,np.arange(10))

u_square_interm_list = []
u_square_theor_interm_list = []


u_square_theor_interm = []
vector_ranges2 = [0.0625,0.125,0.25,0.5,1,2,4,8,16]
vector_ranges1 = [1,4]
adj_r_square_int_lev = []
adj_r_square_int_int = []
lev_params_int = []
int_params_int = []
u_fourth_interm_list = []
u_square_theor_interm_list = []
int_parameters_theor = []
int_predict_int = []
lev_predict_int = []
for factor1 in vector_ranges1:
    for factor2 in vector_ranges1:
        for factor3 in vector_ranges2:
            for factor4 in vector_ranges2:
                u_square_interm = []
                u_fourth_interm = []
                u_square_theor_interm = []
                synthetic_test2 = intermittent2(15000000,integration_time,v*factor1,D*factor2,lambda1*factor3,lambda2*factor4)
                #intermittent2(nt,dt,mean_bal_sac,diffusion,rate12,rate21)
                print(factor1,factor2,factor3,factor4)
                
                u_square_interm.append( np.mean(np.diff(synthetic_test2[0][::tau*skip])**2 +np.diff(synthetic_test2[1][::tau*skip])**2))
                u_fourth_interm.append( ( np.mean(np.diff(synthetic_test2[0][::tau*skip])**4)+ np.mean(np.diff(synthetic_test2[1][::tau*skip])**4)/2 ))
                dx4_log = np.log10(np.array(u_fourth_interm))
                dx2_log = np.log10(np.array(u_square_interm))
                popt_x, pcov_x = scipy.optimize.curve_fit(mom4_serg_log, np.array(tau_list), np.array(dx4_log), p0 = [1,1,1,1],bounds=(0.0001, [np.inf, np.inf, np.inf,np.inf]),maxfev=500000)
                popt_x_diff, pcov_x = scipy.optimize.curve_fit(mom22_4_diff_serg_log, np.array(tau_list), np.array(dx4_log) - 2*np.array(dx2_log),p0=(popt_x[0],popt_x[1],popt_x[2],popt_x[3]),bounds=(0.0001, [np.inf, np.inf, np.inf,np.inf]),maxfev=500000)

                
                g_emp_points = np.array(dx4_log) - 2*np.array(dx2_log)
                g_interm_fit = mom22_4_diff_serg_log(np.array(tau_list),popt_x_diff[0],popt_x_diff[1],popt_x_diff[2],popt_x_diff[3])
                g_lev_fit = np.ones(len(np.array(g_emp_points))) * np.mean(g_emp_points)
                
                
                #reg2 = LinearRegression().fit(np.log2(np.array(tau_list)).reshape(-1, 1), np.log2(u_square_interm))
                #coef = reg2.coef_[0]
                #intercept = reg2.intercept_
                #lev_params_int.append([coef,intercept])
                adj_r_square_int_lev.append(adjusted_r_square(g_emp_points,g_lev_fit,1))
                adj_r_square_int_int.append(adjusted_r_square(g_emp_points,g_int_fit,4))
                                    
                
#                lev_predict_int.append(np.arange(len(tau_list))*coef + intercept)
#
#                u_square_interm_list.append(u_square_interm)
#                u_fourth_interm_list.append(u_fourth_interm)
#                u_square_theor_interm_list.append(u_square_theor_interm)
#                int_parameters_theor.append([C1_new2(g_alpha,g_beta,0.25*(D*factor2)**2,v*factor1),C2_new2(g_alpha,g_beta,0.25*(D*factor2)**2,v*factor1),lambda2*factor4])
#                adj_r_square_int_int.append(adjusted_r_square(np.log2(u_square_interm),log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]),3))
#                int_predict_int.append(log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]))
#                int_params_int.append(best_params)

                    
                    
                
gamma_int = np.log10(np.array(adj_r_square_int_lev)/ np.array(adj_r_square_int_int))
plt.hist(gamma_int,40)
#np.savetxt('adj_r_square_int_lev.txt',adj_r_square_int_lev)
#np.savetxt('adj_r_square_int_int.txt',adj_r_square_int_int)
#np.savetxt("gamma_int.txt",gamma_int)
#np.savetxt('u_square_interm_list.txt',u_square_interm_list)
#np.savetxt('u_fourth_interm_list.txt',u_fourth_interm_list)
#np.savetxt('u_square_theor_interm_list.txt',u_square_theor_interm_list)
#np.savetxt('int_params_int.txt',int_params_int)
#np.savetxt('lev_params_int.txt',lev_params_int)
#np.savetxt('int_parameters_theor.txt',int_parameters_theor)
#np.savetxt('int_predict_int.txt',int_predict_int)
#np.savetxt('lev_predict_int.txt',lev_predict_int)


len_vector= 10
lambda1 = 5*0.1**3
lambda2 = 0.1**2
v = 10
D = 1
g_beta = lambda1 + lambda2
g_alpha = lambda2/(g_beta)


integration_time=0.25
skip = int(1/integration_time) #integration time = Delta t in text
tau_list = np.power(2,np.arange(10))

u_square_interm_list = []
u_square_theor_interm_list = []


u_square_theor_interm = []
vector_ranges2 = [0.0625,0.125,0.25,0.5,1,2,4,8,16]
vector_ranges1 = [1,4]
adj_r_square_int_lev = []
adj_r_square_int_int = []
lev_params_int = []
int_params_int = []
u_fourth_interm_list = []
u_square_theor_interm_list = []
int_parameters_theor = []
int_predict_int = []
lev_predict_int = []
for factor1 in vector_ranges1:
    for factor2 in vector_ranges1:
        for factor3 in vector_ranges2:
            for factor4 in vector_ranges2:
                u_square_interm = []
                u_fourth_interm = []
                u_square_theor_interm = []
                synthetic_test2 = intermittent2(15000000,integration_time,v*factor1,D*factor2,lambda1*factor3,lambda2*factor4)
                #intermittent2(nt,dt,mean_bal_sac,diffusion,rate12,rate21)
                print(factor1,factor2,factor3,factor4)
                
                u_square_interm.append( np.mean(np.diff(synthetic_test2[0][::tau*skip])**2 +np.diff(synthetic_test2[1][::tau*skip])**2))
                u_fourth_interm.append( ( np.mean(np.diff(synthetic_test2[0][::tau*skip])**4)+ np.mean(np.diff(synthetic_test2[1][::tau*skip])**4)/2 ))
                dx4_log = np.log10(np.array(u_fourth_interm))
                dx2_log = np.log10(np.array(u_square_interm))
                popt_x, pcov_x = scipy.optimize.curve_fit(mom4_serg_log, np.array(tau_list), np.array(dx4_log), p0 = [1,1,1,1],bounds=(0.0001, [np.inf, np.inf, np.inf,np.inf]),maxfev=500000)
                popt_x_diff, pcov_x = scipy.optimize.curve_fit(mom22_4_diff_serg_log, np.array(tau_list), np.array(dx4_log) - 2*np.array(dx2_log),p0=(popt_x[0],popt_x[1],popt_x[2],popt_x[3]),bounds=(0.0001, [np.inf, np.inf, np.inf,np.inf]),maxfev=500000)

                
                g_emp_points = np.array(dx4_log) - 2*np.array(dx2_log)
                g_interm_fit = mom22_4_diff_serg_log(np.array(tau_list),popt_x_diff[0],popt_x_diff[1],popt_x_diff[2],popt_x_diff[3])
                g_lev_fit = np.ones(len(np.array(g_emp_points))) * np.mean(g_emp_points)
                
                
                #reg2 = LinearRegression().fit(np.log2(np.array(tau_list)).reshape(-1, 1), np.log2(u_square_interm))
                #coef = reg2.coef_[0]
                #intercept = reg2.intercept_
                #lev_params_int.append([coef,intercept])
                adj_r_square_int_lev.append(adjusted_r_square(g_emp_points,g_lev_fit,1))
                adj_r_square_int_int.append(adjusted_r_square(g_emp_points,g_int_fit,4))
                                    
                
#                lev_predict_int.append(np.arange(len(tau_list))*coef + intercept)
#
#                u_square_interm_list.append(u_square_interm)
#                u_fourth_interm_list.append(u_fourth_interm)
#                u_square_theor_interm_list.append(u_square_theor_interm)
#                int_parameters_theor.append([C1_new2(g_alpha,g_beta,0.25*(D*factor2)**2,v*factor1),C2_new2(g_alpha,g_beta,0.25*(D*factor2)**2,v*factor1),lambda2*factor4])
#                adj_r_square_int_int.append(adjusted_r_square(np.log2(u_square_interm),log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]),3))
#                int_predict_int.append(log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]))
#                int_params_int.append(best_params)

                    
                    
                
gamma_int = np.log10(np.array(adj_r_square_int_lev)/ np.array(adj_r_square_int_int))
plt.hist(gamma_int,40)
#np.savetxt('adj_r_square_int_lev.txt',adj_r_square_int_lev)
#np.savetxt('adj_r_square_int_int.txt',adj_r_square_int_int)
#np.savetxt("gamma_int.txt",gamma_int)
#np.savetxt('u_square_interm_list.txt',u_square_interm_list)
#np.savetxt('u_fourth_interm_list.txt',u_fourth_interm_list)
#np.savetxt('u_square_theor_interm_list.txt',u_square_theor_interm_list)
#np.savetxt('int_params_int.txt',int_params_int)
#np.savetxt('lev_params_int.txt',lev_params_int)
#np.savetxt('int_parameters_theor.txt',int_parameters_theor)
#np.savetxt('int_predict_int.txt',int_predict_int)
#np.savetxt('lev_predict_int.txt',lev_predict_int)


plt.hist(gamma_int,40)
plt.title('gamma function intermittent')
plt.xlabel(r'$\Gamma$')
plt.ylabel('frequency')
plt.title('gamma function intermittent')
plt.savefig('gamma intermittent.png',dpi=300)



# Levy function definitions

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

#og_params = list(np.loadtxt('og_params_int.txt'))
#est_params = list(np.loadtxt('int_process_int_fit_params.txt'))
og_params = []
est_params = []
est_params2 = []
list_est_params = []
list_est_params2 = []
diff_params = []
est_diff_params =[]       
diff_diff_params=[]
diff_est_params=[]
r_square_list=[]




    



integration_factor = 1 ## check this integration factor. 
g_tau = 1
#make this random
#g_v0 = 5
#g_D = 1
#g_lambda_B = 0.05
#g_lambda_D = 0.005
loc_params = np.swapaxes(np.loadtxt('intermittent_est_params.txt'),0,1)
mean_params = np.mean(np.log(loc_params),axis=1)
std_params = np.std(np.log(loc_params),axis=1)
normed_loc_params = (np.swapaxes(np.log(loc_params),0,1)-mean_params)/std_params
kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(normed_loc_params)

N = 100000
N_iter = 1000
re_estimations = 5



opt_iter = 20


redim_max = 5
redim_min = 1.0/redim_max
#adj_r_square_int_lev = list(np.loadtxt('r_square_int_process_lev_fit.txt'))
#adj_r_square_int_int = list(np.loadtxt('r_square_int_process_int_fit.txt'))
#lev_params_int = list(np.loadtxt('int_process_int_lev_params.txt'))
#int_params_int = list(np.loadtxt('int_process_int_fit_params.txt'))

adj_r_square_int_lev = []
adj_r_square_int_int = []
lev_params_int = []
int_params_int = []

tau_list = np.arange(1,30)
#tau_list = np.power(1.05,np.arange(1,101)).astype(int)

#create grid with "tighter" parameters
#create 200 initial points at ranfrom from the "tighter" parameter grid


for itera in range(0,N_iter):
#for itera in range(840,841):
   # factor1 =   1.5*random.random()  + 0.5 
   # factor2 =   1.5*random.random()  + 0.5 
   # factor3 =   1.5*random.random()  + 0.5 
   # factor4 =   1.5*random.random()  + 0.5 
    
    factor1 =   1
    factor2 =   1
    factor3 =   1
    factor4 =   1
    new_data = kde.sample()
    [[g_v0,g_D,g_lambda_B,g_lambda_D]] = np.exp(new_data*std_params + mean_params)
    og_g_v0,og_g_D,og_g_lambda_B,og_g_lambda_D = g_v0,g_D,g_lambda_B,g_lambda_D

    #x_loc = np.repeat(np.loadtxt('x_intermittent'+str(itera)+'.txt'),integration_factor)
    #y_loc = np.repeat(np.loadtxt('y_intermittent'+str(itera)+'.txt'),integration_factor)

    x_loc,y_loc = intermittent2(N*integration_factor,g_tau/integration_factor,g_v0*factor1,g_D*factor2,g_lambda_B*factor3,g_lambda_D*factor4)
    
    
    x_list = []
    y_list = []
    dx_list = []
    dy_list = []
    dx4 = []
    dy4 = []
    dx2 = []
    dy2 = []

    N_max = 100

    for i in tau_list:
        x_list.append(x_loc[::i*integration_factor])
        y_list.append(y_loc[::i*integration_factor])
        dx_list.append(np.diff(x_loc[::i*integration_factor]))
        dy_list.append(np.diff(y_loc[::i*integration_factor]))

        dx4.append(np.mean(np.array(dx_list[-1])**4))
        dy4.append(np.mean(np.array(dy_list[-1])**4))
        dx2.append(np.mean(np.array(dx_list[-1])**2))
        dy2.append(np.mean(np.array(dy_list[-1])**2))

    dx4 = np.array(dx4)
    dy4 = np.array(dy4)
    dx4_log = np.log(dx4)
    dy4_log = np.log(dy4)

    dx2 = np.array(dx2)
    dy2 = np.array(dy2)
    dx2_log = np.log(dx2)
    dy2_log = np.log(dy2)

    list_est_params2 = []
    list_est_params = []
    for rrr in range(re_estimations):

        g_emp_points_x = np.array(dx4_log)
        g_emp_points_y = np.array(dy4_log)

        #start initial parameter search grid

        print(g_v0,g_D,g_lambda_B,g_lambda_D)

        ########## Here the HMM estimations ##########

        dS= np.sqrt(np.array(dx_list[0])**2 + np.array(dy_list[0])**2)
        lest_fix_mu,lest_fix_sigma,lest_lambda_B,lest_lambda_D = HMM_first_guess(dS)
        g_v0 = 5*lest_fix_mu
        g_D = lest_fix_sigma
        g_lambda_B = lest_lambda_B
        g_lambda_D = lest_lambda_D
        print(g_v0,g_D,g_lambda_B,g_lambda_D)
        
        ##############################################

        
        grid_nodes = 20
        redim_power = redim_max**(2/grid_nodes)
        redim_vec = np.power(redim_power,np.arange(-grid_nodes*0.5,grid_nodes*0.5))

        gl_v_vec = g_v0 * redim_vec
        gl_d_vec = g_D * redim_vec
        gl_lambdaB_vec = g_lambda_B * redim_vec
        gl_lambdaD_vec = g_lambda_D * redim_vec

        param_grid = np.array(np.meshgrid(gl_v_vec,gl_d_vec,gl_lambdaB_vec,gl_lambdaD_vec)).T.reshape(-1,4)
        temp_adjus_r_int_sq_x = []
        temp_adjus_r_int_sq_x_diff = []
        temp_adjus_r_int_sq_y = []
        del_max_r_s = 0
        del_max_r_s_diff = 0
        print(g_v0,g_D,g_lambda_B,g_lambda_D)
        print(itera)







        




        #new_data = data[0][['Eyetracker timestamp','Gaze point X','Gaze point Y','Eye movement type']]


        
        g_emp_points_x = np.array(dx4_log)
        global_logdx4 = dx4_log
        global_tau_list = tau_list
        temp_adjus_r_int_sq_x = []
        reg2 = LinearRegression().fit(np.log(np.array(tau_list)).reshape(-1, 1), dx4_log)
        coef = reg2.coef_[0]
        intercept = reg2.intercept_
        g_lev_fit_x  = coef*np.log(np.array(tau_list)) + intercept
        r_levy = adjusted_r_square(dx4_log,g_lev_fit_x,2)

    ######### optimizaiton ##############
        global_logdx4 = dx4_log
        global_logdx2 = dx2_log
        global_tau_list = tau_list

        gredim = 10**2
        ggridlen = 3
        rranges = [(g_v0/gredim, g_v0*gredim), (g_D/gredim, g_D*gredim),(g_lambda_B/gredim, g_lambda_B*gredim),(g_lambda_D/gredim, g_lambda_D*gredim)]

        ######## brute force
        #best_popt_x = optimize.brute(to_optimize_mom4_serg_log, rranges, full_output=False,finish=optimize.fmin,Ns = ggridlen)

        ######## shgo
        #gaga = scipy.optimize.shgo(to_optimize_mom4_serg_log,[(g_v0/gredim, g_v0*gredim), (g_D/gredim, g_D*gredim),(g_lambda_B/gredim, g_lambda_B*gredim),(g_lambda_D/gredim, g_lambda_D*gredim)])
        #best_popt_x = gaga['x']

        ######## double annealing
        list_fun = []
        list_poptx = []
        ####### create grid ###########
        grid_nodes = ggridlen
        redim_power = redim_max**(0.5/grid_nodes)
        redim_vec = np.power(redim_power,np.arange(-grid_nodes*0.5,grid_nodes*0.5))
        gl_v_vec = g_v0 * redim_vec
        gl_d_vec = g_D * redim_vec
        gl_lambdaB_vec = g_lambda_B * redim_vec
        gl_lambdaD_vec = g_lambda_D * redim_vec
        print('first calc')

    
        ####### First calculation - fourth moment ###########
        param_grid = np.array(np.meshgrid(gl_v_vec,gl_d_vec,gl_lambdaB_vec,gl_lambdaD_vec)).T.reshape(-1,4)
        #for ki in range(len(param_grid)):
            #print(i)
        #    while True:
        #        try:
        #            with np.errstate(all='raise'):
        #                 la = scipy.optimize.dual_annealing(to_optimize_mom4_serg_log,rranges,x0=param_grid[ki])
        #            break
        #        except Exception:
        #            print("Warning detected")
        #            continue
        la = None
        while la is None:   
            try:
                la = scipy.optimize.dual_annealing(to_optimize_mom4_serg_log,rranges)
            except RuntimeWarning:
                1
                
                
        list_fun.append(la['fun'])
        list_poptx.append(la['x'])
        lbb = la
        best_popt_x = list_poptx[np.argmin(list_fun)]
        print(list_fun)
        print(np.argmin(list_fun))
        print(best_popt_x)
        
        
        g_git_fit = mom4_serg_log(np.array(tau_list),best_popt_x[0],best_popt_x[1],best_popt_x[2],best_popt_x[3])
        current_r_square_fourth = adjusted_r_square(dx4_log,g_git_fit,4) 
        g_git_fit = mom2_serg_log(np.array(tau_list),best_popt_x[0],best_popt_x[1],best_popt_x[2],best_popt_x[3])
        current_r_square_second = adjusted_r_square(dx2_log,g_git_fit,4) 

        if np.sum(np.isnan(g_git_fit))>0:
            print('NaN in fit')         

        param_grid3 = np.array(np.meshgrid(gl_v_vec,gl_d_vec,gl_lambdaB_vec)).T.reshape(-1,3)
        param_grid2 = np.array(np.meshgrid(gl_d_vec,gl_lambdaD_vec)).T.reshape(-1,2)

        last_r = 0.5*current_r_square_fourth + 0.5*current_r_square_second
        
        tos_lambdaD  = best_popt_x[3]
        ################# Optimization 1 #######################
        for klj in range(opt_iter):
            ############# fourth #################
            list_fun = []
            list_poptx = []
            print('first fourth '+str(klj))
            for ki in range(len(param_grid3)):
                la = None
                while la is None:   
                    try:
                        la = scipy.optimize.dual_annealing(to_optimize_mom4_serg_log_vdl,rranges[:3],x0=param_grid3[ki])
                        list_fun.append(la['fun'])
                        list_poptx.append(la['x'])
                    except RuntimeWarning:
                        1


            temp_popt_x= list_poptx[np.argmin(list_fun)]
            g_git_fit = mom4_serg_log(np.array(tau_list),temp_popt_x[0],temp_popt_x[1],temp_popt_x[2],tos_lambdaD)
            new_r_square = adjusted_r_square(dx4_log,g_git_fit,4)            
            if np.sum(np.isnan(g_git_fit))==0:
                if new_r_square > current_r_square_fourth-0.02:
                    current_r_square_fourth =new_r_square
                    best_popt_x[0] = list_poptx[np.argmin(list_fun)][0]
                    best_popt_x[1] = list_poptx[np.argmin(list_fun)][1]
                    best_popt_x[2] = list_poptx[np.argmin(list_fun)][2]
                    best_popt_x[3] =tos_lambdaD
            else:
                print('NaN in fit')   
                
            print(best_popt_x)

            ########### second ################
            print('first second '+str(klj))
            tos_v = best_popt_x[0]
            tos_lambdaB = best_popt_x[2]    
            list_fun = []
            list_poptx = []

            for kii in range(len(param_grid2)):
                la = None
                while la is None:   
                    try:
                        la = scipy.optimize.dual_annealing(to_optimize_second_ld,[(rranges[1][0],rranges[1][1]),(rranges[3][0],rranges[3][1])],x0=param_grid2[kii])  
                        list_fun.append(la['fun'])
                        list_poptx.append(la['x'])
                    except RuntimeWarning:
                        1
                    
            g_git_fit = mom2_serg_log(np.array(tau_list),best_popt_x[0],list_poptx[np.argmin(list_fun)][0],best_popt_x[2],list_poptx[np.argmin(list_fun)][1])
            new_r_square = adjusted_r_square(dx2_log,g_git_fit,4)

            if np.sum(np.isnan(g_git_fit))==0:
                if new_r_square > current_r_square_second-0.02:
                    current_r_square_second =new_r_square
                    
                    tos_lambdaD = list_poptx[np.argmin(list_fun)][1]
                    best_popt_x[1] = list_poptx[np.argmin(list_fun)][0]
                    best_popt_x[3] = tos_lambdaD
            else:
                print('NaN in fit')   
                
            print(best_popt_x)

        param_grid3 = np.array(np.meshgrid(gl_v_vec,gl_lambdaB_vec)).T.reshape(-1,2)
        #param_grid3 = np.array(np.meshgrid(gl_v_vec,gl_d_vec,gl_lambdaB_vec)).T.reshape(-1,3)
        param_grid2 = np.array(np.meshgrid(gl_lambdaB_vec,gl_lambdaD_vec)).T.reshape(-1,2)
        tos_v = best_popt_x[0]
        tos_D = best_popt_x[1]
        tos_lambdaB = best_popt_x[2]
        tos_lambdaD = best_popt_x[3]
        est_params.append([best_popt_x[0],best_popt_x[1],best_popt_x[2],best_popt_x[3]])

        g_int_fit_x4 = mom4_serg_log(np.array(tau_list),est_params[-1][0],est_params[-1][1],est_params[-1][2],est_params[-1][3])
        g_int_fit_x2 = mom2_serg_log(np.array(tau_list),est_params[-1][0],est_params[-1][1],est_params[-1][2],est_params[-1][3])
        r_int4 = adjusted_r_square(dx4_log,g_int_fit_x4,4)
        r_int2 = adjusted_r_square(dx2_log,g_int_fit_x2,4)
        
        if 0.5*(r_int4 +r_int2)<last_r-0.05:
            est_params[-1] = est_params[-2]
            tos_v,tos_D,tos_lambdaB,tos_lambdaD = est_params[-2]
        else:
            last_r = 0.5*(r_int4 +r_int2)
        print('curret_r = ' + str(  r_int4 +r_int2))


        current_r_square_fourth
        current_r_square_second
        ################# Optimization 2 #######################
        for klj in range(opt_iter):
            print('second fourth '+str(klj))
            ############# fourth #################
            list_fun = []
            list_poptx = []
            for ki in range(len(param_grid3)):
                try:
                    la = scipy.optimize.dual_annealing(to_optimize_mom4_serg_log_vl,[rranges[0],rranges[1]],x0=param_grid3[ki])
                    list_fun.append(la['fun'])
                    list_poptx.append(la['x'])
                except RuntimeWarning:
                    1

            temp_popt_x= list_poptx[np.argmin(list_fun)]
            g_git_fit = mom4_serg_log(np.array(tau_list),temp_popt_x[0],tos_D,temp_popt_x[1],tos_lambdaD)
            new_r_square = adjusted_r_square(dx4_log,g_git_fit,4)


            temp_popt_x= list_poptx[np.argmin(list_fun)]
            g_git_fit = mom4_serg_log(np.array(tau_list),list_poptx[np.argmin(list_fun)][0] ,tos_D,list_poptx[np.argmin(list_fun)][1] ,tos_lambdaD)
            new_r_square = adjusted_r_square(dx4_log,g_git_fit,4)            
            
                
            if np.sum(np.isnan(g_git_fit))==0:
                if new_r_square > current_r_square_fourth - 0.02:
                    current_r_square_fourth =new_r_square
                    tos_v =  list_poptx[np.argmin(list_fun)][0] 
                    tos_lambdaB = list_poptx[np.argmin(list_fun)][1] 


                    best_popt_x[0] = tos_v
                    best_popt_x[1] = tos_D
                    best_popt_x[2] = tos_lambdaB
                    best_popt_x[3] = tos_lambdaD
            else:
                print('NaN in fit')                

            ########### second ################
            
            print(best_popt_x)
            list_fun = []
            list_poptx = []
            print('second second '+str(klj))
            for kii in range(len(param_grid2)):
                try:
                    la = scipy.optimize.dual_annealing(to_optimize_second_ll,[(rranges[2][0],rranges[2][1]),(rranges[3][0],rranges[3][1])])  
                    list_fun.append(la['fun'])
                    list_poptx.append(la['x'])
                except:
                    1

            g_git_fit = mom2_serg_log(np.array(tau_list),best_popt_x[0],best_popt_x[1],list_poptx[np.argmin(list_fun)][0],list_poptx[np.argmin(list_fun)][1])
            new_r_square = adjusted_r_square(dx2_log,g_git_fit,4)

            #print('second_iteration. New rsquare:'+ str(new_r_square) +'. Current rsquare:'+ str(current_r_square))
            if np.sum(np.isnan(g_git_fit))==0:
                if new_r_square > current_r_square_second-0.02:
                    current_r_square_second =new_r_square
                    tos_lambdaB =  list_poptx[np.argmin(list_fun)][0] 
                    tos_lambdaD = list_poptx[np.argmin(list_fun)][1]
                    best_popt_x[0] = tos_v
                    best_popt_x[1] = tos_D
                    best_popt_x[2] = tos_lambdaB
                    best_popt_x[3] = tos_lambdaD
            print(best_popt_x)
        ######################################################################     


            #power = 0.5
            #v_mean_power = np.mean((np.array(dx_list[0])**2+ np.array(dy_list[0])**2)**(0.5*power))
            #D_next_est = (np.abs((v_mean_power - ((tos_v*tau_list[0])**power)*(tos_lambdaD / (tos_lambdaB + tos_lambdaD))) / (tos_lambdaB / (tos_lambdaB + tos_lambdaD)))**(1/power))/np.sqrt(tau_list[0])
            power = 0.5


            v_mean_power = np.mean( (np.diff(x_loc)**2 +np.diff(y_loc)**2  )**(0.5*power))


            adj_lbB = 1/ (1/tos_lambdaB - 0.7)
            adj_lbD = 1/ (1/tos_lambdaD + 0.7)
            lbB = adj_lbB
            lbD = adj_lbD
            tmeanB = lbD / (lbB + lbD)
            tmeanD = lbB / (lbB + lbD)


            D_next_est = 0.25*(np.abs( (v_mean_power*1.1 - (tos_v**power)*tmeanB)/ tmeanD)**(2/power))

            emp_v_mean_power = 0.91*( (tos_v**0.5) *tmeanB + ((4*D_next_est)**0.25)*tmeanD)

            #print(D_next_est)
            
            g_int_fit_x4 = mom4_serg_log(np.array(tau_list),best_popt_x[0],D_next_est,best_popt_x[2],best_popt_x[2])
            g_int_fit_x2 = mom2_serg_log(np.array(tau_list),best_popt_x[0],D_next_est,best_popt_x[2],best_popt_x[2])
            r_int4 = adjusted_r_square(dx4_log,g_int_fit_x4,4)
            r_int2 = adjusted_r_square(dx2_log,g_int_fit_x2,4)


            
            
            
            if np.sum(np.isnan(g_git_fit))==0:
                if 0.5*(r_int4 +r_int2)<last_r-0.03:
                    last_r = 0.5*(r_int4 +r_int2)
                    
                    tos_D = 0.7*D_next_est + 0.4*best_popt_x[1]
                    best_popt_x[0] = tos_v
                    best_popt_x[1] = tos_D
                    best_popt_x[2] = tos_lambdaB
                    best_popt_x[3] = tos_lambdaD
            print(best_popt_x)

        est_params2.append([best_popt_x[0],best_popt_x[1],best_popt_x[2],tos_lambdaD])
        

            
        print('curret_r = ' + str(  r_int4 +r_int2))

        list_est_params2.append(est_params2[-1])
        list_est_params.append(est_params[-1])

    try:
        parsed_est_params2 = parse_trials(list_est_params2,10)
    except:
        parsed_est_params2 = [-1,-1,-1,-1]
    try:
        parsed_est_params = parse_trials(list_est_params,10)
    except:
        parsed_est_params = [-1,-1,-1,-1]


    ##########  Saving   #################
    print(i)
    og_params.append([og_g_v0,og_g_D,og_g_lambda_B,og_g_lambda_D])
    lev_params_int.append([coef,intercept])

    print(og_g_v0,og_g_D,og_g_lambda_B,og_g_lambda_D)

    print(parsed_est_params2)
    print(parsed_est_params)
    best_popt_x = parsed_est_params2




    g_int_fit_x = mom4_serg_log(np.array(tau_list),parsed_est_params[0],parsed_est_params[1],parsed_est_params[2],parsed_est_params[3])
    g_lev_fit_x  = coef*np.log(np.array(tau_list)) + intercept
    r_levy = adjusted_r_square(dx4_log,g_lev_fit_x,2)
    r_int = adjusted_r_square(dx4_log,g_int_fit_x,4)
    

    #print(popt_y)
    #print(popt_y_diff)


    reg2 = LinearRegression().fit(np.log(np.array(tau_list)).reshape(-1, 1), dx4_log)
    coef = reg2.coef_[0]
    intercept = reg2.intercept_
    lev_params_int.append([coef,intercept])
    g_lev_fit_x  = coef*np.log(np.array(tau_list)) + intercept
    g_int_fit_x = mom4_serg_log(np.array(tau_list),parsed_est_params[0],parsed_est_params[1],parsed_est_params[2],parsed_est_params[3])
    adj_r_square_int_lev.append(adjusted_r_square(dx4_log,g_lev_fit_x,2))
    adj_r_square_int_int.append(adjusted_r_square(dx4_log,g_int_fit_x,4))
    
    #np.savetxt('final_try2_x_intermittent'+str(itera)+'.txt',x_loc[::integration_factor])
    #np.savetxt('final_try2_y_intermittent'+str(itera)+'.txt',y_loc[::integration_factor])
    #np.savetxt('final_try2_int_process_int_fit_params'+str(itera)+'.txt',parsed_est_params2)
    #np.savetxt('final_try2_int_process_int_fit_params_first_itera'+str(itera)+'.txt',parsed_est_params)
    #np.savetxt('final_try2_og_params_int'+str(itera)+'.txt',og_params)
    #np.savetxt('r_square_int_process_lev_fit.txt',adj_r_square_int_lev)
    #np.savetxt('r_square_int_process_int_fit.txt',adj_r_square_int_int)
    


plt.hist(np.array(adj_r_square_int_int) - np.array(adj_r_square_int_lev) ,40,label='intermittent-fit',alpha=0.5)
#plt.hist(adj_r_square_int_lev,40,label='levy-fit',alpha=0.5)
plt.title(r'Adjusted-$R^2$')
plt.legend()
plt.ylabel('frequency')
plt.xlabel(r'Adjusted-$R^2$')


tau_list = np.power(2,np.arange(10))
o_alpha = 2
o_t_mins = 0.007
adj_r_square_lev_lev = []
adj_r_square_lev_int = []
lev_params_lev = []
int_params_lev = []
u_fourth_levy_list = []
u_square_theor_levy_list = []
dy_lev = []
dx_lev = []
int_predict_lev = []
lev_predict_lev = []
u_square_lev_list = []
u_fourth_lev_list = []
len_vector= 10


for factor1 in np.arange(0,1,1.0/325):
    l_alpha = o_alpha + factor1
    for factor2 in np.arange(0,0.008,1):
        g_t_mins = 1.5*o_t_mins - 0.5*factor1*o_t_mins
        print(factor1,factor2)

        for i in range(22):
            lx_lev,ly_lev,lt_lev = levy_flight_2D_2(int(1500000*l_alpha*0.01/g_t_mins),150000,l_alpha,g_t_mins,1)
            #levy_flight_2D_2(n_redirections,n_max,lalpha,tmin,measuring_dt

            ldy_lev = np.diff(ly_lev)
            ldx_lev = np.diff(lx_lev)
            dy_lev.append(ldy_lev)
            dx_lev.append(ldx_lev)
        y_lev = np.cumsum(np.hstack(dy_lev))
        x_lev = np.cumsum(np.hstack(dx_lev))


        u_square_lev = []
        u_fourth_lev = []
        taus_levy = []
        #N = 3000
        #taumax = int(np.log2(len(y_lev)/N))
        tau_max = 10
        for iii in np.arange(tau_max):
            ttau = int(2**(iii))
            num = np.mean(np.diff(y_lev[::ttau])**2 + np.diff(x_lev[::ttau])**2)
            u_square_lev.append(num)
            u_fourth_lev.append(np.mean( (np.diff(y_lev[::ttau])**2 + np.diff(x_lev[::ttau])**2)**2 ) )
            #print(iii)

            

###########################
            


        initial_conditions = []
        for k in range(len_vector-3):
            blabla = first_estimate_ple_averages_inc(u_square_lev[k:k+3],tau_list[k:k+3])
            if blabla[1]:
                initial_conditions.append(blabla[0])


        if len(initial_conditions)!=0:

            best_params = optimize(u_square_lev,tau_list,initial_conditions[0],500)
            curr_min = np.mean( (np.log2(u_square_lev) - log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]))**2)
            best_params_index = 0
            opt_meu = 1
            for k in range(len(initial_conditions)):
                popt3 = optimize(u_square_lev,tau_list,initial_conditions[k],500)
                new_min = np.mean((np.log2(u_square_lev) - log2_moment_scaling(tau_list,popt3[0],popt3[1] ,popt3[2]))**2 )
                if new_min != new_min:
                    new_min = curr_min + 1000
                if new_min < curr_min:
                    best_params = popt3
                    curr_min = new_min
                    best_params_index = k
                    opt_meu = 1

                try:
                    popt3, pcov = scipy.optimize.curve_fit(log2_moment_scaling, tau_list, np.log2(u_square_lev),p0=initial_conditions[k], maxfev=500000)
                    new_min = np.mean((np.log2(u_square_lev) - log2_moment_scaling(tau_list,popt3[0],popt3[1] ,popt3[2]))**2 )
                    if new_min != new_min:
                        new_min = curr_min + 1000
                    if new_min < curr_min:
                        best_params = popt3
                        curr_min = new_min
                        best_params_index = k
                        opt_meu = 0
                except RuntimeError:
                    'a'
            print(best_params_index,opt_meu)




        if len(initial_conditions)==0:
            try:

                popt3, pcov = scipy.optimize.curve_fit(log2_moment_scaling, tau_list, np.log2(u_square_lev), maxfev=50000000)
            except RuntimeError:
                print('minimum could not be found')


        reg2 = LinearRegression().fit(np.log2(np.array(tau_list)).reshape(-1, 1), np.log2(u_square_lev))
        coef = reg2.coef_[0]
        intercept = reg2.intercept_



        adj_r_square_lev_lev.append(adjusted_r_square(np.log2(u_square_lev),np.arange(len(tau_list))*coef + intercept,2))
        adj_r_square_lev_int.append(adjusted_r_square(np.log2(u_square_lev),np.arange(len(tau_list))*coef + intercept,2))
        lev_params_lev.append([coef,intercept])
        lev_predict_lev.append(np.arange(len(tau_list))*coef + intercept)

        u_square_lev_list.append(u_square_lev)
        u_fourth_lev_list.append(u_fourth_lev)



        u_square_theor_levy_list.append(l_alpha)
        int_predict_lev.append(log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]))
        int_params_lev.append(best_params)





        



gamma_lev = np.log10(np.array(adj_r_square_lev_lev)/ np.array(adj_r_square_lev_int))
plt.hist(gamma_int,40)


#np.savetxt('adj_r_square_lev_lev.txt',adj_r_square_lev_lev)
#np.savetxt('adj_r_square_lev_int.txt',adj_r_square_lev_int)
#np.savetxt("gamma_lev.txt",gamma_lev)
#np.savetxt('u_square_lev_list.txt',u_square_lev_list)
#np.savetxt('u_fourth_lev_list.txt',u_fourth_lev_list)

#np.savetxt('int_params_lev.txt',int_params_lev)
#np.savetxt('lev_params_lev.txt',lev_params_lev)
#np.savetxt('int_predict_lev.txt',int_predict_int)
#np.savetxt('lev_predict_lev.txt',lev_predict_int)

















len_vector= 10
lambda1 = 5*0.1**3
lambda2 = 0.1**2
v = 10
D = 1
g_beta = lambda1 + lambda2
g_alpha = lambda2/(g_beta)


integration_time=0.25
skip = int(1/integration_time) #integration time = Delta t in text
tau_list = np.power(2,np.arange(10))

u_square_interm_list = []
u_square_theor_interm_list = []


u_square_theor_interm = []
vector_ranges2 = [0.0625,0.125,0.25,0.5,1,2,4,8,16]
vector_ranges1 = [1,4]
adj_r_square_int_lev = []
adj_r_square_int_int = []
lev_params_int = []
int_params_int = []
u_fourth_interm_list = []
u_square_theor_interm_list = []
int_parameters_theor = []
int_predict_int = []
lev_predict_int = []
for factor1 in vector_ranges1:
    for factor2 in vector_ranges1:
        for factor3 in vector_ranges2:
            for factor4 in vector_ranges2:
                u_square_interm = []
                u_fourth_interm = []
                u_square_theor_interm = []
                synthetic_test2 = intermittent2(15000000,integration_time,v*factor1,D*factor2,lambda1*factor3,lambda2*factor4)
                #intermittent2(nt,dt,mean_bal_sac,diffusion,rate12,rate21)
                print(factor1,factor2,factor3,factor4)
                
                u_square_interm.append( np.mean(np.diff(synthetic_test2[0][::tau*skip])**2 +np.diff(synthetic_test2[1][::tau*skip])**2))
                u_fourth_interm.append( ( np.mean(np.diff(synthetic_test2[0][::tau*skip])**4)+ np.mean(np.diff(synthetic_test2[1][::tau*skip])**4)/2 ))
                dx4_log = np.log10(np.array(u_fourth_interm))
                dx2_log = np.log10(np.array(u_square_interm))
                popt_x, pcov_x = scipy.optimize.curve_fit(mom4_serg_log, np.array(tau_list), np.array(dx4_log), p0 = [1,1,1,1],bounds=(0.0001, [np.inf, np.inf, np.inf,np.inf]),maxfev=500000)
                popt_x_diff, pcov_x = scipy.optimize.curve_fit(mom22_4_diff_serg_log, np.array(tau_list), np.array(dx4_log) - 2*np.array(dx2_log),p0=(popt_x[0],popt_x[1],popt_x[2],popt_x[3]),bounds=(0.0001, [np.inf, np.inf, np.inf,np.inf]),maxfev=500000)

                
                g_emp_points = np.array(dx4_log) - 2*np.array(dx2_log)
                g_interm_fit = mom22_4_diff_serg_log(np.array(tau_list),popt_x_diff[0],popt_x_diff[1],popt_x_diff[2],popt_x_diff[3])
                g_lev_fit = np.ones(len(np.array(g_emp_points))) * np.mean(g_emp_points)
                
                
                #reg2 = LinearRegression().fit(np.log2(np.array(tau_list)).reshape(-1, 1), np.log2(u_square_interm))
                #coef = reg2.coef_[0]
                #intercept = reg2.intercept_
                #lev_params_int.append([coef,intercept])
                adj_r_square_int_lev.append(adjusted_r_square(g_emp_points,g_lev_fit,1))
                adj_r_square_int_int.append(adjusted_r_square(g_emp_points,g_int_fit,4))
                                    
                
#                lev_predict_int.append(np.arange(len(tau_list))*coef + intercept)
#
#                u_square_interm_list.append(u_square_interm)
#                u_fourth_interm_list.append(u_fourth_interm)
#                u_square_theor_interm_list.append(u_square_theor_interm)
#                int_parameters_theor.append([C1_new2(g_alpha,g_beta,0.25*(D*factor2)**2,v*factor1),C2_new2(g_alpha,g_beta,0.25*(D*factor2)**2,v*factor1),lambda2*factor4])
#                adj_r_square_int_int.append(adjusted_r_square(np.log2(u_square_interm),log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]),3))
#                int_predict_int.append(log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]))
#                int_params_int.append(best_params)

                    
                    
                
gamma_int = np.log10(np.array(adj_r_square_int_lev)/ np.array(adj_r_square_int_int))
plt.hist(gamma_int,40)
#np.savetxt('adj_r_square_int_lev.txt',adj_r_square_int_lev)
#np.savetxt('adj_r_square_int_int.txt',adj_r_square_int_int)
#np.savetxt("gamma_int.txt",gamma_int)
#np.savetxt('u_square_interm_list.txt',u_square_interm_list)
#np.savetxt('u_fourth_interm_list.txt',u_fourth_interm_list)
#np.savetxt('u_square_theor_interm_list.txt',u_square_theor_interm_list)
#np.savetxt('int_params_int.txt',int_params_int)
#np.savetxt('lev_params_int.txt',lev_params_int)
#np.savetxt('int_parameters_theor.txt',int_parameters_theor)
#np.savetxt('int_predict_int.txt',int_predict_int)
#np.savetxt('lev_predict_int.txt',lev_predict_int)


tau_list = np.power(2,np.arange(10))
o_alpha = 2
o_t_mins = 0.007
adj_r_square_lev_lev = []
adj_r_square_lev_int = []
lev_params_lev = []
int_params_lev = []
u_fourth_levy_list = []
u_square_theor_levy_list = []
dy_lev = []
dx_lev = []
int_predict_lev = []
lev_predict_lev = []
u_square_lev_list = []
u_fourth_lev_list = []
len_vector= 10


for factor1 in np.arange(0,1,1.0/325):
    l_alpha = o_alpha + factor1
    for factor2 in np.arange(0,0.008,1):
        g_t_mins = 1.5*o_t_mins - 0.5*factor1*o_t_mins
        print(factor1,factor2)

        for i in range(22):
            lx_lev,ly_lev,lt_lev = levy_flight_2D_2(int(1500000*l_alpha*0.01/g_t_mins),150000,l_alpha,g_t_mins,1)
            #levy_flight_2D_2(n_redirections,n_max,lalpha,tmin,measuring_dt

            ldy_lev = np.diff(ly_lev)
            ldx_lev = np.diff(lx_lev)
            dy_lev.append(ldy_lev)
            dx_lev.append(ldx_lev)
        y_lev = np.cumsum(np.hstack(dy_lev))
        x_lev = np.cumsum(np.hstack(dx_lev))


        u_square_lev = []
        u_fourth_lev = []
        taus_levy = []
        #N = 3000
        #taumax = int(np.log2(len(y_lev)/N))
        tau_max = 10
        for iii in np.arange(tau_max):
            ttau = int(2**(iii))
            num = np.mean(np.diff(y_lev[::ttau])**2 + np.diff(x_lev[::ttau])**2)
            u_square_lev.append(num)
            u_fourth_lev.append(np.mean( (np.diff(y_lev[::ttau])**2 + np.diff(x_lev[::ttau])**2)**2 ) )
            #print(iii)

            

###########################
            


        initial_conditions = []
        for k in range(len_vector-3):
            blabla = first_estimate_ple_averages_inc(u_square_lev[k:k+3],tau_list[k:k+3])
            if blabla[1]:
                initial_conditions.append(blabla[0])


        if len(initial_conditions)!=0:

            best_params = optimize(u_square_lev,tau_list,initial_conditions[0],500)
            curr_min = np.mean( (np.log2(u_square_lev) - log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]))**2)
            best_params_index = 0
            opt_meu = 1
            for k in range(len(initial_conditions)):
                popt3 = optimize(u_square_lev,tau_list,initial_conditions[k],500)
                new_min = np.mean((np.log2(u_square_lev) - log2_moment_scaling(tau_list,popt3[0],popt3[1] ,popt3[2]))**2 )
                if new_min != new_min:
                    new_min = curr_min + 1000
                if new_min < curr_min:
                    best_params = popt3
                    curr_min = new_min
                    best_params_index = k
                    opt_meu = 1

                try:
                    popt3, pcov = scipy.optimize.curve_fit(log2_moment_scaling, tau_list, np.log2(u_square_lev),p0=initial_conditions[k], maxfev=500000)
                    new_min = np.mean((np.log2(u_square_lev) - log2_moment_scaling(tau_list,popt3[0],popt3[1] ,popt3[2]))**2 )
                    if new_min != new_min:
                        new_min = curr_min + 1000
                    if new_min < curr_min:
                        best_params = popt3
                        curr_min = new_min
                        best_params_index = k
                        opt_meu = 0
                except RuntimeError:
                    'a'
            print(best_params_index,opt_meu)




        if len(initial_conditions)==0:
            try:

                popt3, pcov = scipy.optimize.curve_fit(log2_moment_scaling, tau_list, np.log2(u_square_lev), maxfev=50000000)
            except RuntimeError:
                print('minimum could not be found')


        reg2 = LinearRegression().fit(np.log2(np.array(tau_list)).reshape(-1, 1), np.log2(u_square_lev))
        coef = reg2.coef_[0]
        intercept = reg2.intercept_



        adj_r_square_lev_lev.append(adjusted_r_square(np.log2(u_square_lev),np.arange(len(tau_list))*coef + intercept,2))
        adj_r_square_lev_int.append(adjusted_r_square(np.log2(u_square_lev),np.arange(len(tau_list))*coef + intercept,2))
        lev_params_lev.append([coef,intercept])
        lev_predict_lev.append(np.arange(len(tau_list))*coef + intercept)

        u_square_lev_list.append(u_square_lev)
        u_fourth_lev_list.append(u_fourth_lev)



        u_square_theor_levy_list.append(l_alpha)
        int_predict_lev.append(log2_moment_scaling(tau_list,best_params[0],best_params[1] ,best_params[2]))
        int_params_lev.append(best_params)





        



gamma_lev = np.log10(np.array(adj_r_square_lev_lev)/ np.array(adj_r_square_lev_int))
plt.hist(gamma_int,40)


np.savetxt('adj_r_square_lev_lev.txt',adj_r_square_lev_lev)
np.savetxt('adj_r_square_lev_int.txt',adj_r_square_lev_int)
np.savetxt("gamma_lev.txt",gamma_lev)
np.savetxt('u_square_lev_list.txt',u_square_lev_list)
np.savetxt('u_fourth_lev_list.txt',u_fourth_lev_list)

np.savetxt('int_params_lev.txt',int_params_lev)
np.savetxt('lev_params_lev.txt',lev_params_lev)
np.savetxt('int_predict_lev.txt',int_predict_int)
np.savetxt('lev_predict_lev.txt',lev_predict_int)




# You can add additional functions or code here if needed

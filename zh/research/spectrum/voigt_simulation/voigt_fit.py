import numpy as np
import matplotlib.pyplot as plt
import lmfit
import voigt_function as voigt


def iteration(model,x,y,param_name,param_min,param_max,start,end,step,delta): # iteration of initial guess
    output = []
    for param in range(start,end,step):
        results = converge(model,x,y,param_name,param,param_min,param_max,delta)
        param = results.best_values[param_name]
        param_stderr = results.params[param_name].stderr
        aic = results.aic
        bic = results.bic
        output.append((param,param_stderr,aic,bic))
    output = np.array(output)
    print output
    argmin = np.argmin(output[:,2])
    param_final = output[argmin,0]
    return param_final

def converge(model,x,y,param_name,param,param_min,param_max,delta):
    param_old = param+delta*2
    while abs(param_old-param) > delta:
        print(param_name,param)
        model.set_param_hint(param_name, value = param, min=param_min, max=param_max)
        results = fitting(model,x,y,False,False)
        param_old = param
        param = results.best_values[param_name]
    else:
        print 'converged !'
    return results

def fitting(model,x,y,ifprint,ifplot): # this is the lmfit usage. # 1) first need to set the model function, then set init values
    params = model.make_params()
    print 'fitting'
    #weights = np.hstack((np.ones(500)*1E-30,np.ones(1106-500)*1E-28))
    result = model.fit(y,params,x=x)
    if ifprint:
        print result.fit_report(show_correl=False)
    if ifplot:
        result.plot_fit()
        result.plot_residuals()
        plt.plot(x, result.best_fit, 'r-',label='fit')
        dely = result.eval_uncertainty(sigma=3)
        plt.fill_between(x, result.best_fit-dely, result.best_fit+dely, color="#ABABAB")
    return result

def main():
    T = [
        550.181818182+273,
        653.545454545+273,
        758.818181818+273,
        863.272727273+273,
        966.363636364+273,
        1082.90909091+273,
        1200.63636364+273,
        1284.00000000+273,
        ]

    coef = np.load('coef_CO2_N2.npy')
    print coef.shape

    fit_results = np.zeros((8,3)) # prepare to save results; 3 columns: [0,:] is direct fit; [1,:] is stderr; [2,:] is corrected

    # This is how to use lmfit: 
    model = lmfit.Model(voigt.f_coef) # 1) set the model
    for i in range(8):
        print i
        x = coef[0,:]
        y = coef[i+1,:]
        plt.plot(x,y,'.',label=str(int(T[i]))+' K') #/max(coef10[1,700:])  # plot expt data
        #T0 = iteration(model,x,y,'Temp',500,2300,800,2200,300,15) # iteration to find init T
        T0 = T[i]# here we do not use iteration
        model.set_param_hint('Temp',value = T0,min=300,max=2000) # 2) set initial value
        results = converge(model,x,y,'Temp',T0,300,2300,15)                        # 3) use lmfit get results
        fit_results[i,0:2] = results.best_values['Temp'],results.params['Temp'].stderr # 4) read out params
        f = open('lmfit_fit_results.txt','a')                 # 5) save results
        f.write('\n')
        f.write('T,no-bkg' + '\n')
        f.write('T0=%+4f\n'%(T0))
        f.write('T:    %+4f   ( %+4f) \n'%(fit_results[i,0],fit_results[i,1]))
        f.close()

    correction_params=np.polyfit(fit_results[:,0],T[0:8],1) # 6) calculate correction parameters
    print correction_params
    fit_results[:,2] = np.polyval(correction_params,fit_results[:,0])

    print fit_results
    print 'initial T',T
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()



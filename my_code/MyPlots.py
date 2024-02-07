import matplotlib.pyplot as plt
from brian2 import *

def plot_poisson_dist(ready_monitors,dist_to_plot):
    mean_PE = mean(ready_monitors['stm_p_e'][1])
    std_PE = std(ready_monitors['stm_p_e'][1])
    mean_BE = mean(ready_monitors['stm_b_e'][1])
    std_BE = std(ready_monitors['stm_b_e'][1])
    if dist_to_plot['P']: plt.hist(ready_monitors['stm_p_e'][1], bins=50, edgecolor='black',alpha=0.5,label=f'PE μ:{mean_PE:0.0f} σ:{std_PE:0.0f}')
    if dist_to_plot['B']:plt.hist(ready_monitors['stm_b_e'][1], bins=50, edgecolor='black',alpha=0.5,label =f'BE μ:{mean_BE:0.0f} σ:{std_BE:0.0f}')
    plt.title('Histogram of the Poisson current')
    plt.xlabel('current [pA]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def plot_potential_distriution(built_network,x_axis_limit):
    '''
    plots the distribution of population potential at three slices
    '''
    time1 = x_axis_limit[0]
    time2 = mean(x_axis_limit)
    time3 = x_axis_limit[1]

    timestep = [int(time1/0.0001),int(time2/0.0001),int(time3/0.0001)]

    slice1 = built_network['stm_p_v'].v[:,timestep[0]]/mV
    slice2 = built_network['stm_p_v'].v[:,timestep[1]]/mV
    slice3 = built_network['stm_p_v'].v[:,timestep[2]-1]/mV

    stat1 = (mean(slice1),std(slice1))
    stat2 = (mean(slice2),std(slice2))
    stat3 = (mean(slice3),std(slice3))

    plt.hist(slice1, bins=50, edgecolor='black',alpha=0.3,label=f't:{time1} μ:{stat1[0]:0.0f} σ:{stat1[1]:0.1f}')
    plt.hist(slice2, bins=50, edgecolor='black',alpha=0.3,label=f't:{time2} μ:{stat2[0]:0.0f} σ:{stat2[1]:0.1f}')
    plt.hist(slice3, bins=50, edgecolor='black',alpha=0.3,label=f't:{time3} μ:{stat3[0]:0.0f} σ:{stat3[1]:0.1f}')
    plt.title(f'Histogram of the of potential')
    plt.xlabel('current [pA]')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

def get_hist_data(built_network):
    time_array = built_network['stm_p_v'].t/second
    voltage_array = built_network['stm_p_v'].v/volt*1000

    x_len = len(time_array)
    y_len = len(voltage_array)

    x_array = np.tile(time_array,(y_len,1)).flatten()
    y_array = voltage_array.flatten()
    heights, x_edges, y_edges = np.histogram2d(x_array, y_array, bins = [x_len, 50])
    return heights, x_edges, y_edges, y_len

def plot_potential_dist(hist_data, sim_time, x_axis_limit,):
    '''
    plots the potential distribution of population P over time
    '''
    heights, x_edges, y_edges, y_len = hist_data
    
    fig = figure(figsize=(12,4))
    plt.rcParams['font.size'] = '18'
    plt.pcolormesh(x_edges, y_edges, heights.T, cmap='Blues', rasterized=True, vmax=int(y_len)*0.10)
    plt.title('Histogram of potential over time')
    plt.xlabel('time [s]')
    plt.ylabel('potential [mV]')
    plt.xticks(np.arange(0,sim_time+3,1))
    plt.xlim(x_axis_limit)
    fig.tight_layout()
    plt.show()

def plot_currents_P_neurons(built_network, currents_to_plot, x_axis_limit,y_axis_limit=None):
    fig = plt.figure(figsize=(12,5))
    if currents_to_plot['curr_adp']: plt.plot(built_network['stm_p_adp'].t,-built_network['stm_p_adp'].curr_adapt[0]/pA,color='magenta', label ='adp')
    if currents_to_plot['curr_p']: plt.plot(built_network['stm_pp'].t,built_network['stm_pp'].curr_p[0]/pA,color='red', label ='P')
    if currents_to_plot['curr_b']: plt.plot(built_network['stm_pb'].t,built_network['stm_pb'].curr_b[0]/pA,color='blue', label ='B')
    if currents_to_plot['curr_bg']: plt.plot(built_network['stm_p_bg'].t,built_network['stm_p_bg'].curr_bg[0]/pA,color='gray', label = 'bg')
    if currents_to_plot['curr_l']: plt.plot(built_network['stm_p_l'].t,built_network['stm_p_l'].curr_l[0]/pA,color='brown', label = 'l')
    if currents_to_plot['curr_net']: plt.plot(built_network['stm_p_net'].t,built_network['stm_p_net'].curr_net[0]/pA,linestyle='-.',color='purple', label ='net')
    if currents_to_plot['curr_e']: plt.plot(built_network['stm_p_e'].t,built_network['stm_p_e'].curr_e[0]/pA,color='black', label ='E',alpha= 0.5)

    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Current [pA]')
    plt.title('Currents to a P neurons')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    plt.legend()
    fig.tight_layout()
    plt.show()

def plot_currents_B_neurons(built_network, currents_to_plot, x_axis_limit,y_axis_limit=None):
    fig = plt.figure(figsize=(12,5))
    if currents_to_plot['curr_adp']: plt.plot(built_network['stm_b_adp'].t,-built_network['stm_b_adp'].curr_adapt[0]/pA,color='magenta', label ='adp')
    if currents_to_plot['curr_p']: plt.plot(built_network['stm_bp'].t,built_network['stm_bp'].curr_p[0]/pA,color='red', label ='P')
    if currents_to_plot['curr_b']: plt.plot(built_network['stm_bb'].t,built_network['stm_bb'].curr_b[0]/pA,color='blue', label ='B')
    if currents_to_plot['curr_bg']: plt.plot(built_network['stm_b_bg'].t,built_network['stm_b_bg'].curr_bg[0]/pA,color='gray', label = 'bg')
    if currents_to_plot['curr_l']: plt.plot(built_network['stm_b_l'].t,built_network['stm_b_l'].curr_l[0]/pA,color='brown', label = 'l')
    if currents_to_plot['curr_net']: plt.plot(built_network['stm_b_net'].t,built_network['stm_b_net'].curr_net[0]/pA,linestyle='-.',color='purple', label ='net')
    if currents_to_plot['curr_e']: plt.plot(built_network['stm_b_e'].t,built_network['stm_b_e'].curr_e[0]/pA,color='black', label ='E',alpha= 0.5)


    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Current [pA]')
    plt.title('Currents to a B neurons')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    plt.legend()
    fig.tight_layout()
    plt.show()

def plot_current_P_pop(ready_monitors, currents_to_plot, x_axis_limit, y_axis_limit=None):
    fig = plt.figure(figsize=(12,5))
    if currents_to_plot['curr_adp']: plt.plot(ready_monitors['stm_p_adp'][0],-ready_monitors['stm_p_adp'][1],color='magenta', label ='adp')
    if currents_to_plot['curr_p']: plt.plot(ready_monitors['stm_pp'][0],ready_monitors['stm_pp'][1],color='red', label ='P')
    if currents_to_plot['curr_b']: plt.plot(ready_monitors['stm_pb'][0],ready_monitors['stm_pb'][1],color='blue', label ='B')
    if currents_to_plot['curr_bg']: plt.plot(ready_monitors['stm_p_bg'][0],ready_monitors['stm_p_bg'][1],color='gray', label = 'bg')
    if currents_to_plot['curr_l']: plt.plot(ready_monitors['stm_p_l'][0],ready_monitors['stm_p_l'][1],color='brown', label = 'l')
    if currents_to_plot['curr_net']: plt.plot(ready_monitors['stm_p_net'][0],ready_monitors['stm_p_net'][1],linestyle='-.', color='purple', label ='net')
    if currents_to_plot['curr_e']: plt.plot(ready_monitors['stm_p_e'][0],ready_monitors['stm_p_e'][1], color='black', label ='E',alpha= 0.5)

    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Current [pA]')
    plt.title('mean Currents to population P')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    plt.legend()
    fig.tight_layout()
    plt.show()

def plot_current_B_pop(ready_monitors, currents_to_plot, x_axis_limit, y_axis_limit=None):
    fig = plt.figure(figsize=(12,5))
    if currents_to_plot['curr_adp']: plt.plot(ready_monitors['stm_b_adp'][0],-ready_monitors['stm_b_adp'][1],color='magenta', label ='adp')
    if currents_to_plot['curr_p']: plt.plot(ready_monitors['stm_bp'][0],ready_monitors['stm_bp'][1],color='red', label ='P')
    if currents_to_plot['curr_b']: plt.plot(ready_monitors['stm_bb'][0],ready_monitors['stm_bb'][1],color='blue', label ='B')
    if currents_to_plot['curr_bg']: plt.plot(ready_monitors['stm_b_bg'][0],ready_monitors['stm_b_bg'][1],color='gray', label = 'bg')
    if currents_to_plot['curr_l']: plt.plot(ready_monitors['stm_b_l'][0],ready_monitors['stm_b_l'][1],color='brown', label = 'l')
    if currents_to_plot['curr_net']: plt.plot(ready_monitors['stm_b_net'][0],ready_monitors['stm_b_net'][1],linestyle='-.', color='purple', label ='net')
    if currents_to_plot['curr_e']: plt.plot(ready_monitors['stm_b_e'][0],ready_monitors['stm_b_e'][1], color='black', label ='E',alpha= 0.5)

    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Current [pA]')
    plt.title('mean Currents to population B')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    plt.legend()
    fig.tight_layout()
    plt.show()

def plot_population_fr(ready_monitors,x_axis_limit,y_axis_limit=None):
    fig = plt.figure(figsize=(12,8))

    plt.rcParams['font.size'] = '18'
    plt.subplot(211)
    plt.plot(ready_monitors['rtm_p'][0],ready_monitors['rtm_p'][1],color='red')
    plt.xlabel('Time [s]')
    plt.ylabel('Firing Rate [spikes/s]')
    plt.title('Population P')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)

    plt.subplot(212) 
    plt.plot(ready_monitors['rtm_b'][0],ready_monitors['rtm_b'][1],color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel('Firing Rate [spikes/s]')
    plt.title('Population B')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)

    fig.tight_layout()
    plt.show()

def plot_rasterplot(tested_network,x_axis_limit,y_axis_limit=None):
    fig = plt.figure(figsize=(12,8))
    plt.rcParams['font.size'] = '18'
    plt.subplot(211)
    plt.scatter(tested_network['spm_p'].t,tested_network['spm_p'].i,s=0.2,color='red')
    plt.xlabel('index [s]')
    plt.ylabel('neuron index i [au]')
    plt.title('Population P raster plot')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    plt.subplot(212)
    plt.scatter(tested_network['spm_b'].t,tested_network['spm_b'].i,s=0.2,color='blue')
    plt.xlabel('index [s]')
    plt.ylabel('neuron index i [au]')
    plt.title('Population B raster plot')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    fig.tight_layout()

def plot_mpt_neuron(built_network,x_axis_limit, y_axis_limit=None):
    fig = figure(figsize=(12,8))
    plt.rcParams['font.size'] = '18'

    plt.subplot(211)
    plt.plot(built_network['stm_p_mempo'].t,built_network['stm_p_mempo'].v[0]/mV,color='red')
    plt.plot(built_network['stm_p_mempo'].t,built_network['stm_p_mempo'].v[25]/mV,color='blue')
    plt.xlabel('Time [s]')
    plt.ylabel('Membrane potential [mV]')
    plt.title('two P neurons')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)

    plt.subplot(212)
    plt.plot(built_network['stm_b_mempo'].t,built_network['stm_b_mempo'].v[0]/mV,color='blue')
    plt.plot(built_network['stm_b_mempo'].t,built_network['stm_b_mempo'].v[25]/mV,color='red')
    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Membrane potential [mV]')
    plt.title('two B neurons')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)

    fig.tight_layout()
    plt.show()

def plot_mpt_pop(ready_monitors,x_axis_limit, y_axis_limit=None):
    fig = plt.figure(figsize=(12,8))

    plt.subplot(211)
    plt.plot(ready_monitors['stm_p_mempo'][0],ready_monitors['stm_p_mempo'][1],color='red')
    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Membrane potential [mV]')
    plt.title('Population P')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)

    plt.subplot(212)
    plt.plot(ready_monitors['stm_b_mempo'][0],ready_monitors['stm_b_mempo'][1],color='blue')
    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Membrane potential [mV]')
    plt.title('population B')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)

    fig.tight_layout()
    plt.show()

def plot_lowpass_LFP(ready_monitors,x_axis_limit,y_axis_limit=None):
    fig = plt.figure(figsize=(12,5))
    plt.plot(ready_monitors['lowpass_lfp'][0],ready_monitors['lowpass_lfp'][1])
    
    plt.xlabel('Time [s]')
    plt.ylabel('current [pA]')
    plt.title('lowpass LFP')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)


    fig.tight_layout()
    plt.show()

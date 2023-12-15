import matplotlib.pyplot as plt
from brian2 import *

def plot_currents_P_neurons(built_network, x_axis_limit,y_axis_limit=None):
    fig = plt.figure(figsize=(12,5))
    plt.plot(built_network['stm_p_adp'].t,-built_network['stm_p_adp'].curr_adapt[0]/pA,color='magenta', label ='adp')
    plt.plot(built_network['stm_pp'].t,built_network['stm_pp'].curr_p[0]/pA,color='red', label ='P')
    plt.plot(built_network['stm_pb'].t,built_network['stm_pb'].curr_b[0]/pA,color='blue', label ='B')
    plt.plot(built_network['stm_p_bg'].t,built_network['stm_p_bg'].curr_bg[0]/pA,color='gray', label = 'bg')
    plt.plot(built_network['stm_p_l'].t,built_network['stm_p_l'].curr_l[0]/pA,color='brown', label = 'l')
    plt.plot(built_network['stm_p_net'].t,built_network['stm_p_net'].curr_net[0]/pA,linestyle='-.',color='purple', label ='net')

    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Current [pA]')
    plt.title('Currents to a P neurons')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    plt.legend()
    fig.tight_layout()
    plt.show()

def plot_currents_B_neurons(built_network, x_axis_limit,y_axis_limit=None):
    fig = plt.figure(figsize=(12,5))
    plt.plot(built_network['stm_b_adp'].t,-built_network['stm_b_adp'].curr_adapt[0]/pA,color='magenta', label ='adp')
    plt.plot(built_network['stm_bp'].t,built_network['stm_bp'].curr_p[0]/pA,color='red', label ='P')
    plt.plot(built_network['stm_bb'].t,built_network['stm_bb'].curr_b[0]/pA,color='blue', label ='B')
    plt.plot(built_network['stm_b_bg'].t,built_network['stm_b_bg'].curr_bg[0]/pA,color='gray', label = 'bg')
    plt.plot(built_network['stm_b_l'].t,built_network['stm_b_l'].curr_l[0]/pA,color='brown', label = 'l')
    plt.plot(built_network['stm_b_net'].t,built_network['stm_b_net'].curr_net[0]/pA,linestyle='-.',color='purple', label ='net')

    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Current [pA]')
    plt.title('Currents to a B neurons')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    plt.legend()
    fig.tight_layout()
    plt.show()

def plot_current_P_pop(ready_monitors, x_axis_limit, y_axis_limit=None):
    fig = plt.figure(figsize=(12,5))
    plt.plot(ready_monitors['stm_p_adp'][0],-ready_monitors['stm_p_adp'][1],color='magenta', label ='adp')
    plt.plot(ready_monitors['stm_pp'][0],ready_monitors['stm_pp'][1],color='red', label ='P')
    plt.plot(ready_monitors['stm_pb'][0],ready_monitors['stm_pb'][1],color='blue', label ='B')
    plt.plot(ready_monitors['stm_p_bg'][0],ready_monitors['stm_p_bg'][1],color='gray', label = 'bg')
    plt.plot(ready_monitors['stm_p_l'][0],ready_monitors['stm_p_l'][1],color='brown', label = 'l')
    plt.plot(ready_monitors['stm_p_net'][0],ready_monitors['stm_p_net'][1],linestyle='-.', color='purple', label ='net')

    plt.rcParams['font.size'] = '18'
    plt.xlabel('Time [s]')
    plt.ylabel('Current [pA]')
    plt.title('mean Currents to population P')
    plt.xlim(x_axis_limit)
    plt.ylim(y_axis_limit)
    plt.legend()
    fig.tight_layout()
    plt.show()

def plot_current_B_pop(ready_monitors, x_axis_limit, y_axis_limit=None):
    fig = plt.figure(figsize=(12,5))
    plt.plot(ready_monitors['stm_b_adp'][0],-ready_monitors['stm_b_adp'][1],color='magenta', label ='adp')
    plt.plot(ready_monitors['stm_bp'][0],ready_monitors['stm_bp'][1],color='red', label ='P')
    plt.plot(ready_monitors['stm_bb'][0],ready_monitors['stm_bb'][1],color='blue', label ='B')
    plt.plot(ready_monitors['stm_b_bg'][0],ready_monitors['stm_b_bg'][1],color='gray', label = 'bg')
    plt.plot(ready_monitors['stm_b_l'][0],ready_monitors['stm_b_l'][1],color='brown', label = 'l')
    plt.plot(ready_monitors['stm_b_net'][0],ready_monitors['stm_b_net'][1],linestyle='-.', color='purple', label ='net')

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
    plt.title('Population A raster plot')
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
    plt.title('Population A')
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

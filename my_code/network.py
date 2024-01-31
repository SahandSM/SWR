from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from my_code.aux_functions import Connectivity
from my_code.parameters import *



def build_network(net_params, initial_condition):
    
    print('BP Model with Adaptation')
    print('Neuron type: adaptive leaky integrate-and-fire')
    
    net_seed = int(net_params['random_seed'].get_param())
    seed(net_seed)
    data_to_save = {}
        
    neuron_eqs = '''
        dv/dt = ( curr_l + curr_syn + curr_bg - curr_adapt + curr_e)/mem_cap: volt (unless refractory)
        curr_syn = curr_p + curr_b : amp
        curr_net = curr_l + curr_syn + curr_bg - curr_adapt + curr_e : amp
        curr_bg : amp
        mem_cap : farad
        g_leak : siemens
        e_rever : volt
        v_reset : volt
        v_stop : volt
        tau_refr : second
        tau_adapt : second
        J_spi : amp
        g_ip : siemens
        g_ib : siemens
        g_ie : siemens
    '''
    curr_l_eqs = '''
        curr_l = g_leak*(v_reset - v) : amp
        '''
    
    curr_adapt_eqs = '''
        dcurr_adapt/dt = -curr_adapt/tau_adapt : amp
        '''

    curr_p_eqs = '''
        curr_p = g_p*(e_p - v) : amp
        dg_p/dt = -g_p/ tau_d_p : siemens
        e_p : volt
        tau_d_p : second
    '''
    
    curr_b_eqs = '''
        curr_b = g_b * (e_b - v): amp
        dg_b / dt = -g_b / tau_d_b: siemens
        e_b : volt
        tau_d_b: second
    '''
    curr_e_eqs = '''
        curr_e = g_e * (e_e - v): amp
        dg_e / dt = -g_e / tau_d_e: siemens
        e_e : volt
        tau_d_e: second
    '''

    
    all_eqs = neuron_eqs + curr_l_eqs + curr_adapt_eqs + curr_p_eqs + curr_b_eqs + curr_e_eqs

    """ CREATE CELL POPULATIONS """

    n_p = int(net_params['n_p'].get_param())
    n_b = int(net_params['n_b'].get_param())
    n_e_p = int(net_params['n_e_p'].get_param())
    n_e_b = int(net_params['n_e_b'].get_param())
    poisson_rate_p = net_params['poisson_rate_p'].get_param()
    poisson_rate_b = net_params['poisson_rate_b'].get_param()
    all_neurons = []
            
    pop_p = NeuronGroup(n_p, model=all_eqs, threshold='v > v_stop', reset='''v = v_reset
                        curr_adapt += J_spi
                        ''', refractory='tau_refr', method='euler', name='pop_p')
    all_neurons.append(pop_p)
    
    pop_b = NeuronGroup(n_b, model=all_eqs, threshold='v > v_stop', reset='''v = v_reset
                        curr_adapt += J_spi
                        ''', refractory='tau_refr', method='euler', name='pop_b')
    all_neurons.append(pop_b)

    pop_e_p = PoissonGroup(n_e_p, rates = poisson_rate_p, name='pop_e_p')
    pop_e_b = PoissonGroup(n_e_b, rates = poisson_rate_b, name='pop_e_b')

    for pop in all_neurons:
        if pop.name == 'pop_p':
            pop.mem_cap = net_params['mem_cap_p'].get_param()
            pop.g_leak = net_params['g_leak_p'].get_param()
            pop.v_reset = net_params['v_reset_p'].get_param()
            pop.tau_adapt = net_params['tau_adapt_p'].get_param()
            pop.curr_bg = net_params['curr_bg_p'].get_param()
            pop.J_spi = net_params['J_spi_p'].get_param()
            pop.g_ip = net_params['g_pp'].get_param()
            pop.g_ib = net_params['g_pb'].get_param()
            pop.g_ie = net_params['g_pe'].get_param()
            pop.v_stop = net_params['v_stop_p'].get_param()
            pop.e_rever = pop.v_reset
            pop.tau_d_e = net_params['tau_d_e_p'].get_param()
            
        elif pop.name == 'pop_b':
            pop.mem_cap = net_params['mem_cap_b'].get_param()
            pop.g_leak = net_params['g_leak_b'].get_param()
            pop.v_reset = net_params['v_reset_b'].get_param()
            pop.tau_adapt = net_params['tau_adapt_b'].get_param()
            pop.curr_bg = net_params['curr_bg_b'].get_param()
            pop.J_spi = net_params['J_spi_b'].get_param()
            pop.g_ip = net_params['g_bp'].get_param()
            pop.g_ib = net_params['g_bb'].get_param()
            pop.g_ie = net_params['g_be'].get_param()
            pop.v_stop = net_params['v_stop_b'].get_param()
            pop.e_rever = pop.v_reset
            pop.tau_d_e = net_params['tau_d_e_b'].get_param()
        
        pop.tau_refr = net_params['tau_refr'].get_param()
        pop.tau_d_p = net_params['tau_d_p'].get_param()
        pop.tau_d_b = net_params['tau_d_b'].get_param()
        pop.e_p = net_params['e_p'].get_param()
        pop.e_b = net_params['e_b'].get_param()
        pop.e_e = net_params['e_e'].get_param()

    # parameters related to current background with noise
    curr_bg_base_p = net_params['curr_bg_base_p'].get_param()
    curr_bg_base_b = net_params['curr_bg_base_b'].get_param()
    curr_bg_noise_amp_p = net_params['curr_bg_noise_amp_p'].get_param()
    curr_bg_noise_amp_b = net_params['curr_bg_noise_amp_b'].get_param()
    noise_dt = net_params['curr_bg_noise_dt'].get_param()
    pop_p_noise_dim = 1 if net_params['curr_bg_equal_to_neurons'].get_param() else pop_p.N
    pop_b_noise_dim = 1 if net_params['curr_bg_equal_to_neurons'].get_param() else pop_b.N
    curr_bg_equal_mean_to_pop = net_params['curr_bg_equal_to_pop'].get_param()
    if net_params['curr_bg_nosie'].get_param():
        @network_operation(dt=noise_dt)
        def change_curr_bg():        
            noise_p = uniform(-1,1,pop_p_noise_dim)
            noise_b = noise_p[:pop_b_noise_dim] if curr_bg_equal_mean_to_pop else uniform(-1,1,pop_b_noise_dim)
            pop_p.curr_bg = curr_bg_base_p - curr_bg_noise_amp_p*noise_p
            pop_b.curr_bg = curr_bg_base_b - curr_bg_noise_amp_b*noise_b

    built_network = Network(collect())
    
    print('Total number of synapses')
    prob_pp = net_params['prob_pp'].get_param()
    conn_pp = Connectivity(prob_pp, n_p, n_p, 'conn_pp')
    data_to_save['conn_pp'] = conn_pp

    prob_pb = net_params['prob_pb'].get_param()
    conn_pb = Connectivity(prob_pb, n_b, n_p, 'conn_pb')
    data_to_save['conn_pb'] = conn_pb

    prob_bp = net_params['prob_bp'].get_param()
    conn_bp = Connectivity(prob_bp, n_p, n_b, 'conn_bp')
    data_to_save['conn_bp'] = conn_bp

    prob_bb = net_params['prob_bb'].get_param()
    conn_bb = Connectivity(prob_bb, n_b, n_b, 'conn_bb')
    data_to_save['conn_bb'] = conn_bb

    prob_pe = net_params['prob_pe'].get_param()
    conn_pe = Connectivity(prob_pe, n_e_p, n_p, 'conn_pe')
    data_to_save['conn_pe'] = conn_pe

    prob_be = net_params['prob_be'].get_param()
    conn_be = Connectivity(prob_be, n_e_b, n_b, 'conn_be')
    data_to_save['conn_be'] = conn_be

    tau_l = net_params['tau_l'].get_param()
    
    syn_pp = Synapses(pop_p, pop_p, on_pre='g_p += g_ip',
                      delay=tau_l, name='syn_pp')
    syn_pp.connect(i=conn_pp.pre_index, j=conn_pp.post_index)
    built_network.add(syn_pp)
    print('P->P: %s' % f'{syn_pp.N[:]:,}')
    
    syn_pb = Synapses(pop_b, pop_p, on_pre='g_b += g_ib',
                      delay=tau_l, name='syn_pb')
    syn_pb.connect(i=conn_pb.pre_index, j=conn_pb.post_index)
    built_network.add(syn_pb)
    print('B->P: %s' % f'{syn_pb.N[:]:,}')

    syn_bp = Synapses(pop_p, pop_b, on_pre='g_p += g_ip',
                      delay=tau_l, name='syn_bp')
    syn_bp.connect(i=conn_bp.pre_index, j=conn_bp.post_index)
    built_network.add(syn_bp)
    print('P->B: %s' % f'{syn_bp.N[:]:,}')
    
    syn_bb = Synapses(pop_b, pop_b, on_pre='g_b += g_ib',
                      delay=tau_l, name='syn_bb')
    syn_bb.connect(i=conn_bb.pre_index, j=conn_bb.post_index)
    built_network.add(syn_bb)
    print('B->B: %s' % f'{syn_bb.N[:]:,}')

    syn_pe = Synapses(pop_e_p, pop_p, on_pre='g_e += g_ie',
                      delay=tau_l, name='syn_pe')
    syn_pe.connect(i=conn_pe.pre_index, j=conn_pe.post_index)
    built_network.add(syn_pe)
    print('E->P: %s' % f'{syn_pe.N[:]:,}')

    syn_be = Synapses(pop_e_b, pop_b, on_pre='g_e += g_ie',
                      delay=tau_l, name='syn_be')
    syn_be.connect(i=conn_be.pre_index, j=conn_be.post_index)
    built_network.add(syn_be)
    print('E->B: %s' % f'{syn_be.N[:]:,}')
        
    if initial_condition == 'none':
        pop_p.v = pop_p.e_rever -10*rand(n_p)*mV
        pop_b.v = pop_b.e_rever -10*rand(n_b)*mV
        pop_p.curr_adapt = 100*rand(n_p)*pA
        pop_b.curr_adapt = 0
    
    used_params = get_used_params(net_params)

    return built_network, used_params




def record_network(built_network, used_net_params, test_params):

    test_seed = int(test_params['random_seed'].get_param())
    rec_adapt_num = int(test_params['rec_adapt_num'].get_param())
    rec_mempo_num = int(test_params['rec_mempo_num'].get_param())
    rec_current_num = int(test_params['rec_current_num'].get_param())
    
    pop_p = built_network['pop_p']
    spm_p = SpikeMonitor(pop_p, name='spm_p')
    built_network.add(spm_p)
    
    rtm_p = PopulationRateMonitor(pop_p, name='rtm_p')
    built_network.add(rtm_p)
    
    stm_p_adp = StateMonitor(pop_p, 'curr_adapt', record=np.random.default_rng(test_seed).choice(pop_p.N,
                            size=rec_adapt_num, replace=False),name='stm_p_adp')
    built_network.add(stm_p_adp)
    
    stm_p_mempo = StateMonitor(pop_p, 'v', record=np.random.default_rng(test_seed).choice(pop_p.N,
                            size=rec_mempo_num, replace=False),name='stm_p_mempo')
    built_network.add(stm_p_mempo)
    
    stm_pb = StateMonitor(pop_p, 'curr_b', record=np.random.default_rng(test_seed).choice(pop_p.N,
                                size=rec_current_num, replace=False), name='stm_pb')
    built_network.add(stm_pb)
    
    pop_b = built_network['pop_b']
    spm_b = SpikeMonitor(pop_b, name='spm_b')
    built_network.add(spm_b)
    
    rtm_b = PopulationRateMonitor(pop_b, name='rtm_b')
    built_network.add(rtm_b)
    
    stm_b_adp = StateMonitor(pop_b, 'curr_adapt', record=np.random.default_rng(test_seed).choice(pop_b.N,
                            size=rec_adapt_num, replace=False),name='stm_b_adp')
    built_network.add(stm_b_adp)
    
    stm_b_mempo = StateMonitor(pop_b, 'v', record=np.random.default_rng(test_seed).choice(pop_b.N,
                            size=rec_mempo_num, replace=False),name='stm_b_mempo')
    built_network.add(stm_b_mempo)
        
    return built_network, test_params

def record_p_currents(built_network, used_net_params, test_params, currents_to_record):

    test_seed = int(test_params['random_seed'].get_param())
    rec_adapt_num = int(test_params['rec_adapt_num'].get_param())

    pop_p = built_network['pop_p']
    neurons_to_record = np.random.default_rng(test_seed).choice(pop_p.N, size=rec_adapt_num, replace=False)

    # monitor P current to neurons in population P
    if currents_to_record['curr_p']:
        stm_pp = StateMonitor(pop_p, 'curr_p', record=neurons_to_record, name='stm_pp')
        built_network.add(stm_pp)


    # monitor background current to neurons in population P
    if currents_to_record['curr_bg']:
        stm_p_bg = StateMonitor(pop_p, 'curr_bg', record=neurons_to_record, name='stm_p_bg')
        built_network.add(stm_p_bg)

    # monitor leak current to neurons in population P
    if currents_to_record['curr_l']:
        stm_p_l = StateMonitor(pop_p, 'curr_l', record=neurons_to_record, name='stm_p_l')
        built_network.add(stm_p_l)

    # monitor net current to neurons in population P
    if currents_to_record['curr_net']:
        stm_p_net = StateMonitor(pop_p, 'curr_net', record=neurons_to_record, name='stm_p_net')
        built_network.add(stm_p_net)

    # monitor net current to neurons in population P
    if currents_to_record['curr_e']:
        stm_p_e = StateMonitor(pop_p, 'curr_e', record=neurons_to_record, name='stm_p_e')
        built_network.add(stm_p_e)

    # a monitor for b current to neurons in population P is already defined.

    return built_network, test_params

def record_b_currents(built_network, used_net_params, test_params,currents_to_record):

    test_seed = int(test_params['random_seed'].get_param())
    rec_adapt_num = int(test_params['rec_adapt_num'].get_param())
    pop_b = built_network['pop_b']
    neurons_to_record = np.random.default_rng(test_seed).choice(pop_b.N, size=rec_adapt_num, replace=False)


    # monitor P current to neurons in population B
    if currents_to_record['curr_p']:
        stm_bp = StateMonitor(pop_b, 'curr_p', record=neurons_to_record, name='stm_bp')
        built_network.add(stm_bp)

    # monitor b current to neurons in population B.
    if currents_to_record['curr_b']:
        stm_bb = StateMonitor(pop_b, 'curr_b', record=neurons_to_record, name='stm_bb')
        built_network.add(stm_bb)

    # monitor background current to neurons in population B
    if currents_to_record['curr_bg']:
        stm_b_bg = StateMonitor(pop_b, 'curr_bg', record=neurons_to_record, name='stm_b_bg')
        built_network.add(stm_b_bg)

    # monitor leak current to neurons in population B
    if currents_to_record['curr_l']:
        stm_b_l = StateMonitor(pop_b, 'curr_l', record=neurons_to_record, name='stm_b_l')
        built_network.add(stm_b_l)

    # monitor net current to neurons in population B
    if currents_to_record['curr_net']:
        stm_b_net = StateMonitor(pop_b, 'curr_net', record=neurons_to_record, name='stm_b_net')
        built_network.add(stm_b_net)

    # monitor net current to neurons in population P
    if currents_to_record['curr_e']:
        stm_b_e = StateMonitor(pop_b, 'curr_e', record=neurons_to_record, name='stm_b_e')
        built_network.add(stm_b_e)


    return built_network, test_params

def record_p_v(built_network, used_net_params, test_params, record_p_v):

    test_seed = int(test_params['random_seed'].get_param())
    rec_num = 500

    pop_p = built_network['pop_p']
    neurons_to_record = np.random.default_rng(test_seed).choice(pop_p.N, size=rec_num, replace=False)

    # monitor P potential for rec_num neurons
    if record_p_v:
        stm_p_v = StateMonitor(pop_p, 'v', record=True, name='stm_p_v')
        built_network.add(stm_p_v)

    return built_network, test_params



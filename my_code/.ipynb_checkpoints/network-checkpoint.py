"""
    network.py: functions for brian network simulations

    network 0: B cells + poisson spikes
    network 1: P and B cells
    network 2: P and A cells
    network 3: P, B, and A cells (without A->J connections)
    network 4: P, B, and A cells (all to all connections)
    network 5: P, B, and A cells + B->A depression
"""

from brian2 import *
import builtins
import os.path
import matplotlib.pyplot as plt
import numpy as np
from my_code.plot_functions import create_empty_figure, draw_in_figure
from my_code.aux_functions import Connectivity
from my_code.parameters import get_used_params, compare_used_params, compare_param
from my_code.tests import run_test, prepare_test, print_test_results, save_test_results


def build_network(build_step, net_params, build_anyway=False):
    """
    build brian network

    Args:
        build_step: specify which network (0-5) is built
        net_params: network parameters
        build_anyway: builds even if parameters for current step differ from previous steps

    Returns:
        'rebuild': instruction to rebuild network from step 0
        or
        built_network: brian network object
        all_used_parameters: dictionary with used network parameters

    """

    if build_step == 0:
        print('======== Network 0: B + poisson spikes ========')
    elif build_step == 1:
        print('=============== Network 1: P-B ================')
    elif build_step == 2:
        print('=============== Network 2: P-A ================')
    elif build_step == 3:
        print('====== Network 3: P-B + P->A, A->A, B->A ======')
    elif build_step == 4:
        print('============== Network 4: P-B-A ===============')
    elif build_step == 5:
        print('===== Network 5: P-B-A + B->A depression ======')
    else:
        raise ValueError('invalid step')

    # initialise pseudo-random seed:
    net_seed = int(net_params['random_seed'].get_param())
    seed(net_seed)

    """ LOAD DATA FROM PREVIOUS BUILD STEPS """
    prev_used_params = {}
    prev_conn = {}
    if build_step > 0:
        for i in range(0, build_step):
            # load data from step i:
            file_name = 'network' + str(i) + '.npz'
            if os.path.isfile(file_name):
                step_data = np.load(file_name, allow_pickle=True)
            else:
                answer = ''
                while answer != 'n':
                    answer = builtins.input('ERROR: previous step data network%d.npz not found!\n'
                                            'Build the previous step with the new parameters [y/n]? ' % i)
                    if answer == 'y':
                        return 'rebuild'
                    if answer == 'n':
                        exit()

            for dic_key in step_data.keys():
                if 'used_params' == dic_key:
                    # check if current step parameters differ from previous step:
                    step_used_params = step_data[dic_key].item()
                    check_params = compare_used_params(i, step_used_params, net_params)

                    # check if user wants to rebuild if a parameter is different:
                    if (not check_params) and (not build_anyway):
                        answer = ''
                        while answer != 'n':
                            answer = builtins.input('Do you wish to rebuild previous '
                                                    'networks with the new parameters [y/n]? ')
                            if answer == 'y':
                                return 'rebuild'

                    # store unique previously used params:
                    for used_param in step_used_params:
                        if used_param not in prev_used_params:
                            prev_used_params[used_param] = step_used_params[used_param]
                # store previously used synaptic connections:
                elif 'conn' in dic_key:
                    prev_conn[dic_key] = step_data[dic_key]
                else:
                    print('Warning: unknown loaded data %s in step %d' % (dic_key, i))

    # initialise dictionary where current step data will be saved:
    data_to_save = {}

    """ NEURON MODEL """

    neuron_eqs = '''
        dv/dt = (curr_leak + curr_syn + curr_bg + curr_stim)/mem_cap : volt (unless refractory)
        curr_leak = g_leak*(e_rest - v) : amp
        curr_syn = curr_p + curr_b + curr_a : amp
        mem_cap : farad
        g_leak : siemens
        e_rest : volt
        v_thres : volt
        v_reset : volt
        tau_refr : second
        curr_bg : amp
        curr_stim : amp
    '''
    curr_p_eqs = '''
        curr_p = g_p*(e_p - v) : amp
        e_p : volt
        dg_p/dt = -g_p/tau_d_p : siemens
        tau_d_p : second
    '''
    curr_b_eqs = '''
        curr_b = g_b * (e_b - v): amp
        e_b : volt
        dg_b / dt = -g_b / tau_d_b: siemens
        tau_d_b: second
    '''
    curr_a_eqs = '''
        curr_a = g_a * (e_a - v): amp
        e_a : volt
        dg_a / dt = -g_a / tau_d_a: siemens
        tau_d_a: second
    '''

    """ CREATE CELL POPULATIONS """

    n_p = int(net_params['n_p'].get_param())
    n_asb = int(net_params['n_asb'].get_param())

    all_neurons = []
    # create P cell population:
    if build_step > 0:
        p_neuron_eqs = neuron_eqs + curr_p_eqs
        if build_step in [2, 4, 5]:
            p_neuron_eqs += curr_a_eqs
        else:
            p_neuron_eqs += '''
                curr_a = 0 * amp : amp
            '''

        if build_step != 2:
            p_neuron_eqs += curr_b_eqs
        else:
            p_neuron_eqs += '''
                curr_b = 0 * amp : amp
            '''

        # noinspection PyTypeChecker
        pop_p = NeuronGroup(n_p, model=p_neuron_eqs,
                            threshold='v > v_thres', reset='v = v_reset',
                            refractory='tau_refr', method='euler',
                            name='pop_p')
        all_neurons.append(pop_p)
        poisson_p = None

    else:
        pop_p = None
        p_rate = net_params['p_rate'].get_param()
        poisson_p = PoissonGroup(n_p, rates=p_rate, name='poisson_p')

    # create B cell population:
    if build_step != 2:
        b_neuron_eqs = neuron_eqs + curr_p_eqs + curr_b_eqs
        if build_step in [4, 5]:
            b_neuron_eqs += curr_a_eqs
        else:
            b_neuron_eqs += '''
                curr_a = 0 * amp : amp
            '''

        n_b = int(net_params['n_b'].get_param())

        # noinspection PyTypeChecker
        pop_b = NeuronGroup(n_b, model=b_neuron_eqs,
                            threshold='v > v_thres', reset='v = v_reset',
                            refractory='tau_refr', method='euler',
                            name='pop_b')
        all_neurons.append(pop_b)
    else:
        pop_b = None
        n_b = None

    # create A cell population:
    if build_step > 1:
        a_neuron_eqs = neuron_eqs + curr_p_eqs + curr_a_eqs
        if build_step in [3, 4, 5]:
            a_neuron_eqs += curr_b_eqs
        else:
            a_neuron_eqs += '''
                curr_b = 0 * amp : amp
            '''

        n_a = int(net_params['n_a'].get_param())
        # noinspection PyTypeChecker
        pop_a = NeuronGroup(n_a, model=a_neuron_eqs,
                            threshold='v > v_thres', reset='v = v_reset',
                            refractory='tau_refr', method='euler',
                            name='pop_a')
        all_neurons.append(pop_a)
    else:
        pop_a = None
        n_a = None

    # assign parameters to each cell type:
    for pop in all_neurons:
        if pop.name == 'pop_p':
            pop.curr_bg = net_params['curr_bg_p'].get_param()
        elif pop.name == 'pop_b':
            pop.curr_bg = net_params['curr_bg_b'].get_param()
        elif pop.name == 'pop_a':
            pop.curr_bg = net_params['curr_bg_a'].get_param()

        pop.mem_cap = net_params['mem_cap'].get_param()
        pop.g_leak = net_params['g_leak'].get_param()
        pop.e_rest = net_params['e_rest'].get_param()
        pop.v_thres = net_params['v_thres'].get_param()
        pop.v_reset = net_params['v_reset'].get_param()
        pop.tau_refr = net_params['tau_refr'].get_param()
        if hasattr(pop, 'e_p'):
            pop.e_p = net_params['e_p'].get_param()
        if hasattr(pop, 'e_b'):
            pop.e_b = net_params['e_b'].get_param()
        if hasattr(pop, 'e_a'):
            pop.e_a = net_params['e_a'].get_param()

    # store all Brian objects created so far:
    built_network = Network(collect())

    """ CONNECT NETWORK """

    print('Total number of synapses')

    if build_step in [1, 2, 3, 4, 5]:
        # P -> P:
        g_pp_str = net_params['g_pp'].get_str()
        tau_l_pp = net_params['tau_l_pp'].get_param()
        pop_p.tau_d_p = net_params['tau_d_pp'].get_param()

        # background:
        if build_step == 1:
            p_pp = net_params['p_pp'].get_param()
            conn_pp_bg = Connectivity(p_pp, n_p, n_p, 'conn_pp')
            data_to_save['conn_pp_bg'] = conn_pp_bg
        else:
            conn_pp_bg = prev_conn['conn_pp_bg'].item()
        syn_pp_bg = Synapses(pop_p, pop_p,
                             on_pre='g_p += %s' % g_pp_str,
                             delay=tau_l_pp, name='syn_pp_bg')
        syn_pp_bg.connect(i=conn_pp_bg.pre_index, j=conn_pp_bg.post_index)
        built_network.add(syn_pp_bg)
        print('P->P: %s (background)' % f'{syn_pp_bg.N[:]:,}')

        # assemblies:
        if n_asb > 0:
            n_p_asb = int(net_params['n_p_asb'].get_param())
            p_pp_asb = net_params['p_pp_asb'].get_param()
            p_pp_ffw = net_params['p_pp_ffw'].get_param()
            asb_name = [None] * n_asb
            conn_pp_asb = [None] * n_asb
            syn_pp_asb = [None] * n_asb
            ffw_circle = net_params['ffw_circle'].get_param()
            if ffw_circle:
                n_ffw = n_asb
            else:
                n_ffw = n_asb - 1
            ffw_name = [None] * n_ffw
            conn_pp_ffw = [None] * n_ffw
            syn_pp_ffw = [None] * n_ffw
            for i in range(n_asb):
                asb_name[i] = 'conn_pp_asb_' + str(i+1)
                if build_step == 1:
                    conn_pp_asb[i] = Connectivity(p_pp_asb, n_p_asb, n_p_asb, asb_name[i])
                    data_to_save[asb_name[i]] = conn_pp_asb[i]
                else:
                    conn_pp_asb[i] = prev_conn[asb_name[i]].item()
                syn_pp_asb[i] = Synapses(pop_p[i*n_p_asb:(i+1)*n_p_asb], pop_p[i*n_p_asb:(i+1)*n_p_asb],
                                         on_pre='g_p += %s' % g_pp_str,
                                         delay=tau_l_pp, name='syn_pp_asb_' + str(i+1))
                syn_pp_asb[i].connect(i=conn_pp_asb[i].pre_index, j=conn_pp_asb[i].post_index)
                built_network.add(syn_pp_asb[i])
                print('P->P: %s (assembly %d)' % (f'{syn_pp_asb[i].N[:]:,}', i + 1))

                # feedforward connections:
                if (i > 0) and (p_pp_ffw > 0):
                    ffw_name[i-1] = 'conn_pp_ffw_' + str(i)
                    if build_step == 1:
                        conn_pp_ffw[i-1] = Connectivity(p_pp_ffw, n_p_asb, n_p_asb, ffw_name[i-1])
                        data_to_save[ffw_name[i-1]] = conn_pp_ffw[i-1]
                    else:
                        conn_pp_ffw[i-1] = prev_conn[ffw_name[i-1]].item()
                    syn_pp_ffw[i-1] = Synapses(pop_p[(i-1) * n_p_asb:i * n_p_asb],
                                               pop_p[i * n_p_asb:(i+1) * n_p_asb],
                                               on_pre='g_p += %s' % g_pp_str,
                                               delay=tau_l_pp, name='syn_pp_ffw_' + str(i))
                    syn_pp_ffw[i-1].connect(i=conn_pp_ffw[i-1].pre_index, j=conn_pp_ffw[i-1].post_index)
                    built_network.add(syn_pp_ffw[i-1])
                    print('P (asb %d) -> P (asb %d): %s' % (i, i+1, f'{syn_pp_ffw[i-1].N[:]:,}'))

                    # "close the circle" connection:
                    if (i == n_asb - 1) and ffw_circle:
                        ffw_name[i] = 'conn_pp_ffw_' + str(i+1)
                        if build_step == 1:
                            conn_pp_ffw[i] = Connectivity(p_pp_ffw, n_p_asb, n_p_asb, ffw_name[i])
                            data_to_save[ffw_name[i]] = conn_pp_ffw[i]
                        else:
                            conn_pp_ffw[i] = prev_conn[ffw_name[i]].item()
                        syn_pp_ffw[i] = Synapses(pop_p[i * n_p_asb:(i + 1) * n_p_asb],
                                                 pop_p[0 * n_p_asb:1 * n_p_asb],
                                                 on_pre='g_p += %s' % g_pp_str,
                                                 delay=tau_l_pp, name='syn_pp_ffw_' + str(i+1))
                        syn_pp_ffw[i].connect(i=conn_pp_ffw[i].pre_index, j=conn_pp_ffw[i].post_index)
                        built_network.add(syn_pp_ffw[i])
                        print('P (asb %d) -> P (asb 1): %s' % (i + 1, f'{syn_pp_ffw[i].N[:]:,}'))

    if build_step in [0, 1, 3, 4, 5]:
        # P -> B:
        g_bp_str = net_params['g_bp'].get_str()
        tau_l_bp = net_params['tau_l_bp'].get_param()
        pop_b.tau_d_p = net_params['tau_d_bp'].get_param()
        if build_step in [0, 1]:
            p_bp = net_params['p_bp'].get_param()
            conn_bp = Connectivity(p_bp, n_p, n_b, 'conn_bp')
            data_to_save['conn_bp'] = conn_bp
        else:
            conn_bp = prev_conn['conn_bp'].item()
        if build_step != 0:
            pop_p_temp = pop_p
        else:
            pop_p_temp = poisson_p
        syn_bp = Synapses(pop_p_temp, pop_b,
                          on_pre='g_p += %s' % g_bp_str,
                          delay=tau_l_bp, name='syn_bp')
        syn_bp.connect(i=conn_bp.pre_index, j=conn_bp.post_index)
        built_network.add(syn_bp)
        print('P->B: %s' % f'{syn_bp.N[:]:,}')

    if build_step in [1, 3, 4, 5]:
        # B -> P:
        g_pb_str = net_params['g_pb'].get_str()
        tau_l_pb = net_params['tau_l_pb'].get_param()
        pop_p.tau_d_b = net_params['tau_d_pb'].get_param()
        if build_step == 1:
            p_pb = net_params['p_pb'].get_param()
            conn_pb = Connectivity(p_pb, n_b, n_p, 'conn_pb')
            data_to_save['conn_pb'] = conn_pb
        else:
            conn_pb = prev_conn['conn_pb'].item()
        syn_pb = Synapses(pop_b, pop_p,
                          on_pre='g_b += %s' % g_pb_str,
                          delay=tau_l_pb, name='syn_pb')
        syn_pb.connect(i=conn_pb.pre_index, j=conn_pb.post_index)
        built_network.add(syn_pb)
        print('B->P: %s' % f'{syn_pb.N[:]:,}')

    if build_step in [0, 1, 3, 4, 5]:
        # B -> B:
        g_bb_str = net_params['g_bb'].get_str()
        tau_l_bb = net_params['tau_l_bb'].get_param()
        pop_b.tau_d_b = net_params['tau_d_bb'].get_param()
        if build_step == 0:
            p_bb = net_params['p_bb'].get_param()
            conn_bb = Connectivity(p_bb, n_b, n_b, 'conn_bb')
            data_to_save['conn_bb'] = conn_bb
        else:
            conn_bb = prev_conn['conn_bb'].item()
        syn_bb = Synapses(pop_b, pop_b,
                          on_pre='g_b += %s' % g_bb_str,
                          delay=tau_l_bb, name='syn_bb')
        syn_bb.connect(i=conn_bb.pre_index, j=conn_bb.post_index)
        built_network.add(syn_bb)
        print('B->B: %s' % f'{syn_bb.N[:]:,}')

    if build_step in [2, 3, 4, 5]:
        # P -> A:
        g_ap_str = net_params['g_ap'].get_str()
        tau_l_ap = net_params['tau_l_ap'].get_param()
        pop_a.tau_d_p = net_params['tau_d_ap'].get_param()
        if build_step == 2:
            p_ap = net_params['p_ap'].get_param()
            conn_ap = Connectivity(p_ap, n_p, n_a, 'conn_ap')
            data_to_save['conn_ap'] = conn_ap
        else:
            conn_ap = prev_conn['conn_ap'].item()
        syn_ap = Synapses(pop_p, pop_a,
                          on_pre='g_p += %s' % g_ap_str,
                          delay=tau_l_ap, name='syn_ap')
        syn_ap.connect(i=conn_ap.pre_index, j=conn_ap.post_index)
        built_network.add(syn_ap)
        print('P->A: %s' % f'{syn_ap.N[:]:,}')

        if build_step != 3:
            # A -> P:
            g_pa_str = net_params['g_pa'].get_str()
            tau_l_pa = net_params['tau_l_pa'].get_param()
            pop_p.tau_d_a = net_params['tau_d_pa'].get_param()
            if build_step == 2:
                p_pa = net_params['p_pa'].get_param()
                conn_pa = Connectivity(p_pa, n_a, n_p, 'conn_pa')
                data_to_save['conn_pa'] = conn_pa
            else:
                conn_pa = prev_conn['conn_pa'].item()
            syn_pa = Synapses(pop_a, pop_p,
                              on_pre='g_a += %s' % g_pa_str,
                              delay=tau_l_pa, name='syn_pa')
            syn_pa.connect(i=conn_pa.pre_index, j=conn_pa.post_index)
            built_network.add(syn_pa)
            print('A->P: %s' % f'{syn_pa.N[:]:,}')

        # A -> A:
        g_aa_str = net_params['g_aa'].get_str()
        tau_l_aa = net_params['tau_l_aa'].get_param()
        pop_a.tau_d_a = net_params['tau_d_aa'].get_param()
        if build_step == 2:
            p_aa = net_params['p_aa'].get_param()
            conn_aa = Connectivity(p_aa, n_a, n_a, 'conn_aa')
            data_to_save['conn_aa'] = conn_aa
        else:
            conn_aa = prev_conn['conn_aa'].item()
        syn_aa = Synapses(pop_a, pop_a,
                          on_pre='g_a += %s' % g_aa_str,
                          delay=tau_l_aa, name='syn_aa')
        syn_aa.connect(i=conn_aa.pre_index, j=conn_aa.post_index)
        built_network.add(syn_aa)
        print('A->A: %s' % f'{syn_aa.N[:]:,}')

    if build_step >= 3:
        # B -> A:
        g_ab_str = net_params['g_ab'].get_str()
        tau_l_ab = net_params['tau_l_ab'].get_param()
        pop_a.tau_d_b = net_params['tau_d_ab'].get_param()
        if build_step == 3:
            p_ab = net_params['p_ab'].get_param()
            conn_ab = Connectivity(p_ab, n_b, n_a, 'conn_ab')
            data_to_save['conn_ab'] = conn_ab
        else:
            conn_ab = prev_conn['conn_ab'].item()
        if build_step < 5:
            syn_ab = Synapses(pop_b, pop_a,
                              on_pre='g_b += 0.5 * %s' % g_ab_str,
                              delay=tau_l_ab, name='syn_ab')
        else:
            # plastic B->A:
            tau_d_str = net_params['tau_d'].get_str()
            eta_d_str = net_params['eta_d'].get_str()
            syn_ab = Synapses(pop_b, pop_a,
                              model='de_ab / dt = (1 - e_ab) / (%s) : 1 (clock-driven)' % tau_d_str,
                              on_pre='''g_b_post += e_ab * %s
                                        e_ab = clip(e_ab*(1 - (%s)), 0, 1)
                                     ''' % (g_ab_str, eta_d_str),
                              delay=tau_l_ab, method='euler',
                              name='syn_ab')
        syn_ab.connect(i=conn_ab.pre_index, j=conn_ab.post_index)
        built_network.add(syn_ab)
        if build_step == 5:
            syn_ab.e_ab = 0.5
        print('B->A: %s' % f'{syn_ab.N[:]:,}')

    if build_step >= 4:
        # A -> B:
        g_ba_str = net_params['g_ba'].get_str()
        tau_l_ba = net_params['tau_l_ba'].get_param()
        pop_b.tau_d_a = net_params['tau_d_ba'].get_param()
        if build_step == 4:
            p_ba = net_params['p_ba'].get_param()
            conn_ba = Connectivity(p_ba, n_a, n_b, 'conn_ba')
            data_to_save['conn_ba'] = conn_ba
        else:
            conn_ba = prev_conn['conn_ba'].item()
        syn_ba = Synapses(pop_a, pop_b,
                          on_pre='g_a += %s' % g_ba_str,
                          delay=tau_l_ba, name='syn_ba')
        syn_ba.connect(i=conn_ba.pre_index, j=conn_ba.post_index)
        built_network.add(syn_ba)
        print('A->B: %s' % f'{syn_ba.N[:]:,}')

    """ SORT AND SAVE USED PARAMETERS """

    # get parameters used in current step:
    curr_used_params = get_used_params(net_params)

    # merge previous and current used parameters:
    if build_step > 0:
        all_used_params = prev_used_params
        for param in curr_used_params:
            if param not in all_used_params:
                all_used_params[param] = curr_used_params[param]
            else:
                # if there is a discrepancy, save the current param:
                match = compare_param(prev_used_params[param], curr_used_params[param])
                if match is False:
                    all_used_params[param] = curr_used_params[param]

    else:
        all_used_params = curr_used_params

    if build_step < 5:
        # save parameters used to build network (for comparison with next build step):
        data_to_save['used_params'] = curr_used_params
        np.savez_compressed('network' + str(build_step) + '.npz', **data_to_save)

    print('================ Network Built ================')

    return built_network, all_used_params


def init_network(built_network, used_net_params, test_params):
    """
    initialise network in one of two states: SWR / non-SWR

    Args:
        built_network: built brian network
        used_net_params: parameters used to build network
        test_params: test parameters

    Returns:
        built_network: initialised network
        test_params: test parameters (for tracking used params)

    """

    init_state = test_params['init_state'].get_param()
    if (init_state != 'swr') and (init_state != 'non-swr'):
        raise ValueError('invalid init state')

    test_seed = int(test_params['random_seed'].get_param())
    seed(test_seed)

    v_reset = used_net_params['v_reset'].get_param()
    v_thres = used_net_params['v_thres'].get_param()

    n_asb = used_net_params['n_asb'].get_param()

    # initialise populations:
    if 'pop_p' in built_network:
        pop_p = built_network['pop_p']
    else:
        pop_p = None

    if 'pop_b' in built_network:
        pop_b = built_network['pop_b']
    else:
        pop_b = None

    if 'pop_a' in built_network:
        pop_a = built_network['pop_a']
    else:
        pop_a = None

    # non-SWR state:
    if init_state == 'non-swr':
        if 'pop_p' in built_network:
            pop_p.v = v_reset

        if 'pop_b' in built_network:
            pop_b.v = v_reset

        if 'pop_a' in built_network:
            pop_a.v = v_reset + (v_thres - v_reset) * np.random.rand(pop_a.N)

    # SWR state:
    else:
        if 'pop_p' in built_network:
            if n_asb > 0:
                init_asb = int(test_params['init_asb'].get_param())
                n_p_asb = int(used_net_params['n_p_asb'].get_param())
                pop_p.v = v_reset
                pop_p_asb = pop_p[n_p_asb*(init_asb - 1):n_p_asb*init_asb]
                pop_p_asb.v += (v_thres - v_reset) * np.random.rand(pop_p_asb.N)
            else:
                pop_p.v = v_reset + (v_thres - v_reset) * np.random.rand(pop_p.N)

        if 'pop_b' in built_network:
            pop_b.v = v_reset + (v_thres - v_reset) * np.random.rand(pop_b.N)

        if 'pop_a' in built_network:
            pop_a.v = v_reset

    # initialise synaptic conductances:
    if 'pop_p' in built_network:
        pop_p.g_p = 0.1 * nS * np.random.rand(pop_p.N)
        if hasattr(pop_p, 'g_b'):
            pop_p.g_b = 0.1 * nS * np.random.rand(pop_p.N)
        if hasattr(pop_p, 'g_a'):
            pop_p.g_a = 0.1 * nS * np.random.rand(pop_p.N)

    if 'pop_b' in built_network:
        if hasattr(pop_b, 'g_p'):
            pop_b.g_p = 0.1 * nS * np.random.rand(pop_b.N)
        if hasattr(pop_b, 'g_b'):
            pop_b.g_b = 0.1 * nS * np.random.rand(pop_b.N)
        if hasattr(pop_b, 'g_a'):
            pop_b.g_a = 0.1 * nS * np.random.rand(pop_b.N)

    if 'pop_a' in built_network:
        if hasattr(pop_a, 'g_p'):
            pop_a.g_p = 0.1 * nS * np.random.rand(pop_a.N)
        if hasattr(pop_a, 'g_b'):
            pop_a.g_b = 0.1 * nS * np.random.rand(pop_a.N)
        if hasattr(pop_a, 'g_a'):
            pop_a.g_a = 0.1 * nS * np.random.rand(pop_a.N)

    return built_network, test_params


def record_network(built_network, used_net_params, test_params):
    """
    create monitors to record from network

    Args:
        built_network: built brian network
        used_net_params: parameters used to build network
        test_params: test parameters

    Returns:
        built_network: network with added monitors
        test_params: test parameters (for tracking used params)

    """
    test_seed = int(test_params['random_seed'].get_param())

    """ MONITOR P CELLS """

    if 'pop_p' in built_network:
        pop_p = built_network['pop_p']

        spm_p = SpikeMonitor(pop_p, name='spm_p')
        built_network.add(spm_p)

        rtm_p = PopulationRateMonitor(pop_p, name='rtm_p')
        built_network.add(rtm_p)

        # assembly monitors:
        n_asb = int(used_net_params['n_asb'].get_param())
        if n_asb > 0:
            n_p_asb = int(used_net_params['n_p_asb'].get_param())
            spm_p_asb = [None] * n_asb
            rtm_p_asb = [None] * n_asb
            for i in range(n_asb):
                spm_p_asb[i] = SpikeMonitor(pop_p[i*n_p_asb:(i+1)*n_p_asb], name='spm_p_asb_'+str(i+1))
                built_network.add(spm_p_asb[i])

                rtm_p_asb[i] = PopulationRateMonitor(pop_p[i*n_p_asb:(i+1)*n_p_asb], name='rtm_p_asb_'+str(i+1))
                built_network.add(rtm_p_asb[i])

            # neurons outside assemblies:
            spm_p_out = SpikeMonitor(pop_p[n_asb*n_p_asb:], name='spm_p_out')
            built_network.add(spm_p_out)

            rtm_p_out = PopulationRateMonitor(pop_p[n_asb*n_p_asb:], name='rtm_p_out')
            built_network.add(rtm_p_out)

    # monitor poisson process mimicking P cells:
    elif 'poisson_p' in built_network:
        poisson_p = built_network['poisson_p']

        spm_p = SpikeMonitor(poisson_p, name='spm_p')
        built_network.add(spm_p)

        rtm_p = PopulationRateMonitor(poisson_p, name='rtm_p')
        built_network.add(rtm_p)

    """ MONITOR B CELLS """

    if 'pop_b' in built_network:
        pop_b = built_network['pop_b']

        spm_b = SpikeMonitor(pop_b, name='spm_b')
        built_network.add(spm_b)

        rtm_b = PopulationRateMonitor(pop_b, name='rtm_b')
        built_network.add(rtm_b)

        # record B->P current from n random P cells:
        if 'pop_p' in built_network:
            pop_p = built_network['pop_p']
            rec_pb_num = int(test_params['rec_pb_num'].get_param())

            stm_pb = StateMonitor(pop_p, 'curr_b',
                                  record=np.random.default_rng(test_seed).choice(pop_p.N,
                                                                                 size=rec_pb_num,
                                                                                 replace=False),
                                  name='stm_pb')
            built_network.add(stm_pb)

    """ MONITOR A CELLS """

    if 'pop_a' in built_network:
        pop_a = built_network['pop_a']

        spm_a = SpikeMonitor(pop_a, name='spm_a')
        built_network.add(spm_a)

        rtm_a = PopulationRateMonitor(pop_a, name='rtm_a')
        built_network.add(rtm_a)

    # monitor short-term depression from n random B->A synapses:
    if 'syn_ab' in built_network:
        syn_ab = built_network['syn_ab']
        if hasattr(syn_ab, 'e_ab'):
            rec_e_num = int(test_params['rec_e_num'].get_param())

            stm_e = StateMonitor(syn_ab, 'e_ab',
                                 record=np.random.default_rng(test_seed).choice(syn_ab.N[:],
                                                                                size=rec_e_num,
                                                                                replace=False),
                                 name='stm_e')
            built_network.add(stm_e)

    return built_network, test_params


def build_and_test_network(net_params, test_params, plot_params, live_plotting=True, save_test=False):
    """
    build brian network and run network test

    Args:
        net_params: network parameters to build network
        test_params: test parameters to run test
        plot_params: plot parameters to plot test
        live_plotting: show live plots while test runs
        save_test: if true, save test outputs as .npz and .png files

    """

    """ BUILD BRIAN NETWORK """

    sim_dt = test_params['sim_dt'].get_param()
    defaultclock.dt = sim_dt

    build_step = int(test_params['build_step'].get_param())
    build_anyway = test_params['build_anyway'].get_param()

    build_return = build_network(build_step, net_params, build_anyway)
    if build_return == 'rebuild':
        # rebuild from step 0 with new parameters
        for i in range(build_step):
            build_network(i, net_params)
        built_network, used_net_params = build_network(build_step, net_params)
    else:
        built_network, used_net_params = build_return

    # add brian monitors for recording
    built_network, test_params_u1 = record_network(built_network, used_net_params, test_params)

    # initialise network with the specified initial conditions
    ready_network, test_params_u2 = init_network(built_network, used_net_params, test_params_u1)

    """ PREPARE LIVE PLOTTING """

    if live_plotting:
        # create figure where test will be shown live:
        fig_to_save, subplot_groups = create_empty_figure(ready_network, test_params_u2, used_net_params, plot_params)
        plt.ion()
        plt.show()

        live_plot_dt = test_params['live_plot_dt'].get_param()

        # brian will call live_operations with periodicity live_plot_dt:
        @network_operation(dt=live_plot_dt, name='net_operation')
        def live_operations():
            if len(built_network['rtm_p'].t) > 0:  # check for positive time
                # prepare brian monitors for live plotting and test calculations:
                live_monitors, _, _ = prepare_test(ready_network, test_params, plot_params, test_over=False)

                # print current SWR IEI to command line:
                if 'swr_iei' in live_monitors:
                    iei = live_monitors['swr_iei']
                    print('IEI = (%.2f +/- %.2f) sec' % (float(np.mean(iei)), float(np.std(iei))))

                # refresh figure with live test data:
                draw_in_figure(test_params, plot_params, subplot_groups, live_monitors)

        ready_network.add(Network(collect())['net_operation'])

    """ RUN TEST """

    test_id = test_params['test_id'].get_param()
    time_str = test_params['time_str'].get_param()
    print('\n====== Starting Test %s | %s ======' % (test_id, time_str))

    # run actual brian simulation:
    tested_network, test_params_u3 = run_test(ready_network, test_params_u2, used_net_params)

    # close live plots:
    if live_plotting:
        plt.ioff()
        plt.close()

    # prepare brian monitors for plotting and test calculations:
    ready_monitors, test_data, test_params_u4 = prepare_test(tested_network, test_params_u3, plot_params)

    # print test results:
    test_params_u5 = print_test_results(ready_monitors, test_data, test_params_u4)

    # save test outputs and figure:
    if save_test:
        save_test_results(test_data, ready_monitors, test_params_u5, used_net_params)

        fig_to_save, subplot_groups = create_empty_figure(tested_network, test_params, used_net_params, plot_params)
        draw_in_figure(test_params, plot_params, subplot_groups, ready_monitors)
        fig_to_save.savefig('outputs/test' + test_id + '_' + time_str + '.png')
        plt.close(fig_to_save)

    print('\n================ Finished Test ================\n')

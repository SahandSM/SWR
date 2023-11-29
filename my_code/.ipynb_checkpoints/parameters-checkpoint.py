from brian2 import *
import os
import tkinter as tk
from tkinter import filedialog


class Parameter:

    def __init__(self, val, pref_unit=1):
        """
        initialise parameter object, attributing it a quantity and preferred unit
        the parameter is marked as not used

        Args:
            val: parameter value
            pref_unit: preferred parameter unit
        """

        if type(val) is Quantity:
            print('ERROR: Parameter value should not have units')
        if type(val) is bool:
            self.quantity = val
        else:
            self.quantity = val * pref_unit
        self.pref_unit = pref_unit
        self.used = False

    def get_param(self):
        """
        output parameter quantity and mark parameter as used

        Returns:
            self.quantity: parameter quantity
        """

        self.used = True
        return self.quantity

    def get_str(self):
        """
        output parameter quantity as string in preferred unit and mark parameter as used

        Returns:
            output_string: string of parameter quantity in preferred unit
        """

        self.used = True

        pref_unit_str = str(self.pref_unit)
        if '1. ' in pref_unit_str:
            pref_unit_str = pref_unit_str[3:]

        output_string = str(self.quantity / self.pref_unit) + ' * ' + pref_unit_str
        return output_string


def get_used_params(param_group):
    """
    for a given group of parameters, outputs only the parameters which have been used

    Args:
        param_group: parameter group

    Returns:
        used_param_group: group with parameters that have been used
    """

    used_param_group = {}
    for param_name in param_group:
        if param_group[param_name].used:
            used_pref_unit = param_group[param_name].pref_unit
            used_val = param_group[param_name].quantity
            if type(param_group[param_name].quantity) is not str:
                used_val = used_val / used_pref_unit

            used_param_group[param_name] = Parameter(used_val, used_pref_unit)

    return used_param_group


def compare_param(param1, param2):
    """
    compares two parameters

    Args:
        param1: first parameter
        param2: second parameter

    Returns:
        match: [true/false] if parameters match
    """

    param1_val = param1.get_param() / param1.pref_unit
    param2_val = param2.get_param() / param2.pref_unit

    if (param2_val == 0) and (param1_val == 0):
        ratio = 1.0
    else:
        ratio = param1.get_param() / param2.get_param()

    match = False
    if type(ratio) is not Quantity:
        if ratio == 1.0:
            match = True

    return match


def compare_used_params(build_step, used_params, curr_params):
    """
    compares current network parameters with parameters used in a previous build step

    Args:
        build_step: build step where used_params were used
        used_params: used parameters in previous build_step
        curr_params: current network parameters being used

    Returns:
        check: [true/false] if all parameters match
    """

    check = True
    for param in used_params:
        curr = curr_params[param]
        used = used_params[param]

        match = compare_param(curr, used)
        if match is False:
            check = False
            print('WARNING: Trying to use param %s with value %s, while in step %d it was %s' %
                  (param, curr.quantity, build_step, used.quantity))
            break

    return check


def get_dft_net_params():
    """
    dictionary with default network parameters

    Returns:
        dft_net_params: default network parameters
    """

    dft_net_params = {'random_seed': Parameter(123)}

    # LIF neuron:
    dft_net_params = {**dft_net_params,
                      **{'mem_cap': Parameter(200, pF), 'g_leak': Parameter(10, nS),
                         'e_rest': Parameter(-60, mV), 'v_thres': Parameter(-50, mV),
                         'v_reset': Parameter(-60, mV), 'tau_refr': Parameter(1, ms)}
                      }

    # neuron populations:
    dft_net_params = {**dft_net_params,
                      **{'n_p': Parameter(8200), 'e_p': Parameter(0, mV), 'curr_bg_p': Parameter(200, pA),
                         'n_b': Parameter(135), 'e_b': Parameter(-70, mV), 'curr_bg_b': Parameter(200, pA),
                         'n_a': Parameter(50), 'e_a': Parameter(-70, mV), 'curr_bg_a': Parameter(200, pA)}
                      }

    # poisson process rate (for build step 0):
    dft_net_params = {**dft_net_params,
                      **{'p_rate': Parameter(25, Hz)}
                     }

    # connectivities:
    dft_net_params = {**dft_net_params,
                      **{'p_pp': Parameter(0.01), 'p_bp': Parameter(0.20), 'p_ap': Parameter(0.01),
                         'p_pb': Parameter(0.50), 'p_bb': Parameter(0.20), 'p_ab': Parameter(0.20),
                         'p_pa': Parameter(0.60), 'p_aa': Parameter(0.60), 'p_ba': Parameter(0.60)}
                      }

    # conductances:
    dft_net_params = {**dft_net_params,
                      **{'g_pp': Parameter(0.20, nS), 'g_bp': Parameter(0.05, nS), 'g_ap': Parameter(0.20, nS),
                         'g_pb': Parameter(0.70, nS), 'g_bb': Parameter(5.00, nS), 'g_ab': Parameter(8.00, nS),
                         'g_pa': Parameter(6.00, nS), 'g_ba': Parameter(7.00, nS), 'g_aa': Parameter(4.00, nS)}
                      }

    # synaptic decay:
    dft_net_params = {**dft_net_params,
                      **{'tau_d_pp': Parameter(2.0, ms), 'tau_d_bp': Parameter(2.0, ms), 'tau_d_ap': Parameter(2.0, ms),
                         'tau_d_pb': Parameter(1.5, ms), 'tau_d_bb': Parameter(1.5, ms), 'tau_d_ab': Parameter(1.5, ms),
                         'tau_d_pa': Parameter(4.0, ms), 'tau_d_ba': Parameter(4.0, ms), 'tau_d_aa': Parameter(4.0, ms)}
                      }

    # synaptic latency:
    dft_net_params = {**dft_net_params,
                      **{'tau_l_pp': Parameter(1, ms), 'tau_l_bp': Parameter(1, ms), 'tau_l_ap': Parameter(1, ms),
                         'tau_l_pb': Parameter(1, ms), 'tau_l_bb': Parameter(1, ms), 'tau_l_ab': Parameter(1, ms),
                         'tau_l_pa': Parameter(1, ms), 'tau_l_ba': Parameter(1, ms), 'tau_l_aa': Parameter(1, ms)}
                      }

    # B->A synaptic depression:
    dft_net_params = {**dft_net_params,
                      **{'eta_d': Parameter(0.18), 'tau_d': Parameter(250, ms)}
                      }

    # P assembly parameters:
    dft_net_params = {**dft_net_params,
                      **{'n_asb': Parameter(0), 'n_p_asb': Parameter(1000),
                         'p_pp_asb': Parameter(0.15), 'p_pp_ffw': Parameter(0.08),
                         'ffw_circle': Parameter(False)}
                      }

    return dft_net_params


def get_dft_test_params(test_id):
    """
    dictionary with default test parameters for a given test ID

    Args:
        test_id: test ID

    Returns:
        dft_test_params: default test parameters
    """

    dft_test_params = {'test_id': Parameter(test_id),
                       'time_str': Parameter(time.strftime("%Y%m%d_%H%M%S")),
                       'build_step': Parameter(int(test_id[0])),
                       'random_seed': Parameter(100)}

    # simulation parameters:
    dft_test_params = {**dft_test_params,
                       **{'sim_dt': Parameter(0.10, ms),
                          'prep_time': Parameter(2, second),
                          'sim_time': Parameter(7, second),
                          'live_plot_dt': Parameter(1, second),
                          'build_anyway': Parameter(False)}
                       }
    if test_id == '42':
        dft_test_params['stim_time'] = Parameter(50, ms)
        dft_test_params['wait_time'] = Parameter(1, second)
        dft_test_params['stim_strength'] = Parameter(500, pA)
    if test_id[0] == '5':
        dft_test_params['sim_time'] = Parameter(60, second)

    # limits and thresholds for tests:
    dft_test_params = {**dft_test_params,
                       **{'isi_thres_irr': Parameter(0.5),
                          'isi_thres_reg': Parameter(0.3),
                          'q_factor_thres': Parameter(0.1),
                          'max_diff': Parameter(3, hertz),
                          'max_a_swr': Parameter(2, hertz),
                          'max_b_nswr': Parameter(2, hertz),
                          'max_net_freq': Parameter(350, Hz),
                          'min_peak_height': Parameter(30, pA),
                          'min_peak_dist': Parameter(100, ms)}
                       }

    # test calculations:
    dft_test_params = {**dft_test_params,
                       **{'lfp_cutoff': Parameter(5, Hz),
                          'gauss_window': Parameter(3, ms),
                          'track_psd': Parameter(False)}
                       }
    if test_id in ['01', '11', '21', '31', '41']:
        dft_test_params['track_unit_rates'] = Parameter(True)
        dft_test_params['track_isi'] = Parameter(True)
        dft_test_params['track_psd'] = Parameter(True)
    else:
        dft_test_params['track_unit_rates'] = Parameter(False)
        dft_test_params['track_isi'] = Parameter(False)
        dft_test_params['track_psd'] = Parameter(False)
    if test_id[0] == '5':
        dft_test_params['track_events'] = Parameter(True)
    else:
        dft_test_params['track_events'] = Parameter(False)

    if test_id == '42':
        dft_test_params['track_states'] = Parameter(True)
    else:
        dft_test_params['track_states'] = Parameter(False)

    # initial conditions:
        if test_id in ['01', '11', '31']:
            dft_test_params['init_state'] = Parameter('swr')
        else:
            dft_test_params['init_state'] = Parameter('non-swr')
        dft_test_params['init_asb'] = Parameter(1)

    # recording:
    dft_test_params = {**dft_test_params,
                       **{'rec_pb_num': Parameter(50),
                          'rec_e_num': Parameter(50)}}

    return dft_test_params


def get_dft_plot_params(test_id):
    """
    dictionary with default plot parameters for a given test ID

    Args:
        test_id: test ID

    Returns:
        dft_plot_params: default plot parameters
    """

    dft_plot_params = {'plot_width': Parameter(20, cmeter),
                       'params_width': Parameter(10, cmeter),
                       'fig_height': Parameter(20, cmeter),
                       'show_params': Parameter(True),
                       'p_color': Parameter('#ef3b53'),
                       'b_color': Parameter('dodgerblue'),
                       'a_color': Parameter('#0a9045'),
                       'e_color': Parameter('#e67e22'),
                       'pb_color': Parameter('darkblue'),
                       'text_font': Parameter(9),
                       'spine_width': Parameter(1.0),
                       'plot_range': Parameter(2, second),
                       'save_plot_start': Parameter(0, second),
                       'save_plot_stop': Parameter(0, second),
                       'num_time_ticks': Parameter(4),
                       'num_psd_ticks': Parameter(4),
                       'auto_y': Parameter(True),
                       'rtm_p_min': Parameter(0),
                       'rtm_p_max': Parameter(90),
                       'rtm_b_min': Parameter(0),
                       'rtm_b_max': Parameter(150),
                       'rtm_a_min': Parameter(0),
                       'rtm_a_max': Parameter(30),
                       'stm_pb_min': Parameter(0),
                       'stm_pb_max': Parameter(100)
                       }

    if test_id in ['01', '11']:
        dft_plot_params['plot_range'] = Parameter(0.5, second)

    return dft_plot_params


def load_parameters():
    """
    loads network and test parameters
    from a *.npz output file of a previously run test

    Returns:
        net_params: loaded network parameters
        test_params: loaded test parameters
    """

    # select test file to import
    root = tk.Tk()
    root.withdraw()
    init_dir = os.getcwd() + '/outputs'
    filename = filedialog.askopenfilename(initialdir=init_dir, title="Select file",
                                          filetypes=[("npz Files", "*.npz")])
    root.destroy()

    test_file = np.load(filename, encoding='latin1', allow_pickle=True)

    net_params = test_file['used_net_params'].item()
    test_params = test_file['used_test_params'].item()

    return net_params, test_params


def get_param_diff(used_dict, dft_dict, param_name, dec=0, print_unit=1, show_unit=False):
    """
    for a given parameter being used, checks whether it is different from default
    if the parameter is different from default, output string is bold

    Args:
        used_dict: used parameter dictionary
        dft_dict: default parameter dictionary
        param_name: name of parameter being checked
        dec: number of decimal places for print
        print_unit: unit to print parameter
        show_unit: [true/false] show parameter unit

    Returns:
        print_output: string with parameter value
    """

    default_param = dft_dict[param_name].get_param()
    param_val = used_dict[param_name].get_param()

    print_param = '{:.{prec}f}'.format(param_val / print_unit, prec=dec)
    if show_unit:
        print_param += ' ' + str(print_unit)

    if param_val / print_unit == 0:
        if default_param / print_unit == 0:
            print_output = print_param
        else:
            print_output = r'\textbf{%s}' % print_param
    else:
        if round(default_param / param_val, 9) == 1.0:
            print_output = print_param
        else:
            print_output = r'\textbf{%s}' % print_param

    return print_output


def print_param_list(used_net_params, used_test_params):
    """
    print a list of used network and test parameters
    parameters different from default are printed in bold

    Args:
        used_net_params: used network parameters
        used_test_params: used test parameters

    Returns:
        text_str: full string with parameter list
    """

    test_id = used_test_params['test_id'].get_param()
    build_step = used_test_params['build_step'].get_param()

    dft_net_params = get_dft_net_params()
    dft_test_params = get_dft_test_params(test_id)

    tab_str = r'\begin{tabular}{ c c c c c } ' + \
              r'$J \to I$ & $p_{IJ}$ & $g_{IJ}$ (nS) & $\tau_d^{IJ}$ (ms) & $\tau_l^{IJ}$ (ms) \\\hline'

    if build_step > 0:
        tab_str += r'$P \to P$ & %s & %s & %s & %s \\' % \
                   (get_param_diff(used_net_params, dft_net_params, 'p_pp', 2),
                    get_param_diff(used_net_params, dft_net_params, 'g_pp', 2, nS),
                    get_param_diff(used_net_params, dft_net_params, 'tau_d_pp', 1, ms),
                    get_param_diff(used_net_params, dft_net_params, 'tau_l_pp', 1, ms))

    if build_step != 2:
        tab_str += r'$P \to B$ & %s & %s & %s & %s \\' % \
                   (get_param_diff(used_net_params, dft_net_params, 'p_bp', 2),
                    get_param_diff(used_net_params, dft_net_params, 'g_bp', 2, nS),
                    get_param_diff(used_net_params, dft_net_params, 'tau_d_bp', 1, ms),
                    get_param_diff(used_net_params, dft_net_params, 'tau_l_bp', 1, ms))

        if build_step != 0:
            tab_str += r'$B \to P$ & %s & %s & %s & %s \\' % \
                       (get_param_diff(used_net_params, dft_net_params, 'p_pb', 2),
                        get_param_diff(used_net_params, dft_net_params, 'g_pb', 2, nS),
                        get_param_diff(used_net_params, dft_net_params, 'tau_d_pb', 1, ms),
                        get_param_diff(used_net_params, dft_net_params, 'tau_l_pb', 1, ms))

        tab_str += r'$B \to B$ & %s & %s & %s & %s \\' % \
                   (get_param_diff(used_net_params, dft_net_params, 'p_bb', 2),
                    get_param_diff(used_net_params, dft_net_params, 'g_bb', 2, nS),
                    get_param_diff(used_net_params, dft_net_params, 'tau_d_bb', 1, ms),
                    get_param_diff(used_net_params, dft_net_params, 'tau_l_bb', 1, ms))

    if build_step > 1:
        tab_str += r'$P \to A$ & %s & %s & %s & %s \\' % \
                   (get_param_diff(used_net_params, dft_net_params, 'p_ap', 2),
                    get_param_diff(used_net_params, dft_net_params, 'g_ap', 2, nS),
                    get_param_diff(used_net_params, dft_net_params, 'tau_d_ap', 1, ms),
                    get_param_diff(used_net_params, dft_net_params, 'tau_l_ap', 1, ms))

    if (build_step > 1) and (build_step != 3):
        tab_str += r'$A \to P$ & %s & %s & %s & %s \\' % \
                   (get_param_diff(used_net_params, dft_net_params, 'p_pa', 2),
                    get_param_diff(used_net_params, dft_net_params, 'g_pa', 2, nS),
                    get_param_diff(used_net_params, dft_net_params, 'tau_d_pa', 1, ms),
                    get_param_diff(used_net_params, dft_net_params, 'tau_l_pa', 1, ms))

    if build_step > 1:
        tab_str += r'$A \to A$ & %s & %s & %s & %s \\' % \
                   (get_param_diff(used_net_params, dft_net_params, 'p_aa', 2),
                    get_param_diff(used_net_params, dft_net_params, 'g_aa', 2, nS),
                    get_param_diff(used_net_params, dft_net_params, 'tau_d_aa', 1, ms),
                    get_param_diff(used_net_params, dft_net_params, 'tau_l_aa', 1, ms))

    if build_step >= 3:
        tab_str += r'$B \to A$ & %s & %s & %s & %s \\' % \
                   (get_param_diff(used_net_params, dft_net_params, 'p_ab', 2),
                    get_param_diff(used_net_params, dft_net_params, 'g_ab', 2, nS),
                    get_param_diff(used_net_params, dft_net_params, 'tau_d_ab', 1, ms),
                    get_param_diff(used_net_params, dft_net_params, 'tau_l_ab', 1, ms))
    if build_step >= 4:
        tab_str += r'$A \to B$ & %s & %s & %s & %s \\' % \
                   (get_param_diff(used_net_params, dft_net_params, 'p_ba', 2),
                    get_param_diff(used_net_params, dft_net_params, 'g_ba', 2, nS),
                    get_param_diff(used_net_params, dft_net_params, 'tau_d_ba', 1, ms),
                    get_param_diff(used_net_params, dft_net_params, 'tau_l_ba', 1, ms))
    tab_str += r'\end{tabular}\\ '

    text_str = r'\setlength{\parindent}{0em}Test %s Parameters:\\ ' % test_id + \
               r'$\cdot$ seed = %s, dt = %s\\' % \
               (get_param_diff(used_test_params, dft_test_params, 'random_seed', 0),
                get_param_diff(used_test_params, dft_test_params, 'sim_dt', 2, ms, True)) + \
               r'$\cdot$ $\sigma_w$ = %s' % \
               get_param_diff(used_test_params, dft_test_params, 'gauss_window', 1, ms, True)
    if build_step == 5:
        text_str += r', $\overline{B \to P}$ $ < $ %s' % \
                    get_param_diff(used_test_params, dft_test_params, 'lfp_cutoff', 0, Hz, True)
    text_str += r'\\'

    text_str += r'\\ ' + \
                r'Network Model:\\ ' + \
                r'$\cdot$ seed = %s\\' % \
                get_param_diff(used_net_params, dft_net_params, 'random_seed', 0) + \
                r'$\cdot$ $C$ = %s, $g_L$ = %s\\ ' % \
                (get_param_diff(used_net_params, dft_net_params, 'mem_cap', 0, pF, True),
                 get_param_diff(used_net_params, dft_net_params, 'g_leak', 0, nS, True)) + \
                r'$\cdot$ $V_{thres}$ = %s, $E_{rest}$ = %s\\ ' % \
                (get_param_diff(used_net_params, dft_net_params, 'v_thres', 0, mV, True),
                 get_param_diff(used_net_params, dft_net_params, 'e_rest', 0, mV, True)) + \
                r'$\cdot$ $V_{reset}$ = %s, $\tau_{refr}$ = %s\\ ' % \
                (get_param_diff(used_net_params, dft_net_params, 'v_reset', 0, mV, True),
                 get_param_diff(used_net_params, dft_net_params, 'tau_refr', 0, ms, True)) + \
                r'$\cdot$ '
    if build_step != 0:
        text_str += '$I^P_{BG}$ = %s ' % get_param_diff(used_net_params, dft_net_params, 'curr_bg_p', 0, pA, True)
    if build_step != 2:
        text_str += '$I^B_{BG}$ = %s ' % get_param_diff(used_net_params, dft_net_params, 'curr_bg_b', 0, pA, True)
    if build_step >= 2:
        text_str += '$I^A_{BG}$ = %s' % get_param_diff(used_net_params, dft_net_params, 'curr_bg_a', 0, pA, True)
    text_str += r'\\'

    text_str += r'$\cdot$ $N_P$ = %s' % get_param_diff(used_net_params, dft_net_params, 'n_p')

    if build_step != 2:
        text_str += r', $N_B$ = %s' % get_param_diff(used_net_params, dft_net_params, 'n_b')
    if build_step > 1:
        text_str += r', $N_A$ = %s' % get_param_diff(used_net_params, dft_net_params, 'n_a')

    n_asb = used_net_params['n_asb'].get_param()
    if n_asb > 0:
        text_str += r' \\ $\cdot$ $N_{\textrm{asb}}$ = %s, $N_{\textrm{asb}}^P$ = %s, $p_{\textrm{asb}}^{pp}$ = %s' %\
                    (get_param_diff(used_net_params, dft_net_params, 'n_asb'),
                     get_param_diff(used_net_params, dft_net_params, 'n_p_asb'),
                     get_param_diff(used_net_params, dft_net_params, 'p_pp_asb', 2))

    text_str += r' \\ $\cdot$ $E_P$ = %s' % get_param_diff(used_net_params, dft_net_params, 'e_p', 0, mV, True)
    if build_step != 2:
        text_str += r', $E_B$ = %s' % get_param_diff(used_net_params, dft_net_params, 'e_b', 0, mV, True)
    if build_step > 1:
        text_str += r', $E_A$ = %s' % get_param_diff(used_net_params, dft_net_params, 'e_a', 0, mV, True)

    if build_step == 0:
        text_str += r'\\ $\cdot$ $P_{rate}$ = %s' % \
                    get_param_diff(used_net_params, dft_net_params, 'p_rate', 0, Hz, True)

    text_str += r'\\ \\ ' + tab_str

    if build_step == 5:
        text_str += r' \\ $\cdot$ $\eta_d$ = %s, $\tau_d$ = %s' % \
                    (get_param_diff(used_net_params, dft_net_params, 'eta_d', 2),
                     get_param_diff(used_net_params, dft_net_params, 'tau_d', 0, ms, True))

    return text_str

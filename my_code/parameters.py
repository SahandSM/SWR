from brian2 import *

class Parameter:
    
    def __init__(self, value, unit=1):
        
        if type(value) is Quantity:
            print('ERROR: Give value without unit')
            
        if type(value) is bool:
            self.quantity = value
            
        else:
            self.quantity = value * unit
            
        self.unit = unit
        self.value = value
        self.used = False
        
        
        
    def get_param(self):
        
        self.used = True
        return self.quantity
    
    
    
    def set_param(self, x):
        
        self.quantity = x * self.unit
        
        
    
    def get_str(self):
        
        self.used = True
        unit_str = str(self.unit)
        
        if '1. ' in unit_str:
            unit_str = unit_str[3:]

        output_string = str(self.quantity / self.unit) + ' * ' + unit_str
        return output_string

    
    
def get_used_params(param_group):

    used_param_group = {}

    for param_name in param_group:

        if param_group[param_name].used:
            used_unit = param_group[param_name].unit
            used_val = param_group[param_name].quantity

            if type(param_group[param_name].quantity) is not str:
                used_val = used_val / used_unit

            used_param_group[param_name] = Parameter(used_val, used_unit)

    return used_param_group


def get_default_net_params():

    default_params = {'random_seed': Parameter(123)}
    
    # AdEx neuron, shared parameters
    default_params = {**default_params,
                      **{'tau_refr': Parameter(1, ms)}
                      }
    
    # AdEx neuron, population-specific parameters
    default_params = {**default_params,
                      **{'g_leak_p': Parameter(10, nS), 'v_stop_p': Parameter(-50, mV), 'v_reset_p': Parameter(-60, mV),
                         'g_leak_b': Parameter(10, nS), 'v_stop_b': Parameter(-50, mV), 'v_reset_b': Parameter(-60, mV),                         
                         
                         'mem_cap_p': Parameter(200, pF),
                         'mem_cap_b': Parameter(200, pF)}
                      }
    
    # Adaptation:
    default_params = {**default_params,
                      **{'J_spi_p': Parameter(50, pA), 'tau_adapt_p': Parameter(250, ms),
                         'J_spi_b': Parameter(0, pA), 'tau_adapt_b': Parameter(250, ms)}
                      }
    
    # neuron populations:
    default_params = {**default_params,
                      **{'n_p': Parameter(8200), 'curr_bg_p': Parameter(200, pA),'curr_bg_base_p':Parameter(110,pA), 'curr_bg_noise_amp_p': Parameter(20,pA),
                         'n_b': Parameter(135), 'curr_bg_b': Parameter(200, pA), 'curr_bg_base_b':Parameter(110,pA), 'curr_bg_noise_amp_b': Parameter(20,pA), 
                         'curr_bg_noise_dt': Parameter(10,ms),'curr_bg_equal_to_neurons': Parameter(False), 'curr_bg_equal_to_pop': Parameter(False),
                         'curr_bg_nosie': Parameter(False)}
                      }

    # connectivity and other dimensionless parameters:
    default_params = {**default_params,
                      **{'prob_pp': Parameter(0.01), 'prob_bp': Parameter(0.2),
                         'prob_pb': Parameter(0.5), 'prob_bb': Parameter(0.2)}
                      }

    # weights (conductances*(V-Erever) already):
    default_params = {**default_params,
                      **{'g_pp': Parameter(0.2, nS), 'g_bp': Parameter(0.05, nS),
                         'g_pb': Parameter(0.7, nS), 'g_bb': Parameter(5, nS)}
                      }
    
    # synapses:
    default_params = {**default_params,
                      **{'tau_d_p': Parameter(2, ms), 'tau_d_b': Parameter(1.5, ms),
                         'e_p': Parameter(0, mV), 'e_b': Parameter(-70, mV),
                         'tau_l': Parameter(1, ms)}
                      }

    # parameters related to Poisson population
    default_params = {**default_params,
                      **{'n_e_p': Parameter(500), 'tau_d_e_p': Parameter(2, ms),
                         'poisson_rate_p': Parameter(50, Hz),
                         'n_e_b': Parameter(500), 'tau_d_e_b': Parameter(2, ms),
                         'poisson_rate_b': Parameter(50, Hz),
                         'e_e': Parameter(0, mV),
                         'g_pe': Parameter(0, nS), 'g_be': Parameter(0, nS),
                         'prob_pe': Parameter(0.1), 'prob_be': Parameter(0.1)}
                     }

    return default_params



def get_dft_test_params():

    dft_test_params = {'random_seed': Parameter(100)}

    dft_test_params = {**dft_test_params,
                       **{'sim_dt': Parameter(0.10, ms),
                          'prep_time': Parameter(2, second),
                          'sim_time': Parameter(10, second)
                          }
                       }

    dft_test_params = {**dft_test_params,
                       **{'max_net_freq': Parameter(350, Hz),
                          'min_peak_height': Parameter(30, pA),
                          'min_peak_dist': Parameter(0.2, second),
                          'lfp_cutoff': Parameter(5, Hz),
                          'gauss_window': Parameter(3, ms)}
                       }
    
    dft_test_params['track_unit_rates'] = Parameter(True)
    dft_test_params['track_isi'] = Parameter(True)
    dft_test_params['track_events'] = Parameter(True)
    dft_test_params['track_states'] = Parameter(True)
    dft_test_params['stim_time'] = Parameter(100, ms)
    dft_test_params['wait_time'] = Parameter(2, second)
    dft_test_params['stim_strength'] = Parameter(100, pA)
    dft_test_params['n_stim'] = Parameter(100)


    dft_test_params = {**dft_test_params,
                       **{'rec_adapt_num': Parameter(50),
                          'rec_mempo_num': Parameter(50),
                          'rec_current_num': Parameter(50)}}

    return dft_test_params

from brian2 import *
from my_code.parameters import *
from my_code.aux_functions import *
from my_code.detect_peaks import detect_peaks


def run_test(network, test_params, used_net_params, stimulation):

    # test_seed = int(test_params['random_seed'].get_param())
    prep_time = test_params['prep_time'].get_param()
    sim_time = test_params['sim_time'].get_param()
    stim_time = test_params['stim_time'].get_param()
    n_stim = test_params['n_stim'].get_param()
    # network.run(stim_time, report='text')

    curr_bg_length = int((prep_time+sim_time)/defaultclock.dt)
    I_random = TimedArray((110-20*rand(curr_bg_length))*pA,dt=100*ms)

    network.run(prep_time, report = 'text')

    if stimulation == 'none':
        network.run(sim_time, report='text')

    if stimulation == 'p':
        pop_p = network['pop_p']
        stim_strength = test_params['stim_strength'].get_param()
        lucky_ones=np.random.choice(pop_p.N,size=n_stim)
        pop_p.curr_bg[lucky_ones] += stim_strength
        network.run(stim_time)
        pop_p.curr_bg[lucky_ones] -= stim_strength
        network.run(sim_time-stim_time)

    if stimulation == 'b':
        pop_b = network['pop_b']
        stim_strength = test_params['stim_strength'].get_param()
        lucky_ones=np.random.choice(pop_b.N,size=n_stim)
        pop_p.curr_bg[lucky_ones] += stim_strength
        network.run(stim_time)
        pop_b.curr_bg[lucky_ones] -= stim_strength
        network.run(sim_time-stim_time)
        
    return network, test_params



def prepare_test(network, test_params, used_net_params):

    sim_dt = test_params['sim_dt'].get_param()
    sim_time = test_params['sim_time'].get_param()
    prep_time = test_params['prep_time'].get_param()
    track_events = test_params['track_events'].get_param()
    track_states = test_params['track_states'].get_param()
    track_unit_rates = test_params['track_unit_rates'].get_param()
    track_isi = test_params['track_isi'].get_param()
    max_freq = test_params['max_net_freq'].get_param() / Hz
    ready_monitors = {}
    test_data = {}

    end_time = np.max(network['rtm_p'].t / second) * second
    start_time = 0 * second
    measure_time = prep_time
    spm_array = ['spm_p', 'spm_b']
                 
    for spm in spm_array:
        if check_brian_monitor(network, spm, 'i'):
            ready_monitors[spm] = trim_brian_monitor(network[spm], network[spm].i, 1, start_time, end_time)
            pop_name = spm[4:]

            if track_unit_rates:
                mean_rate, std_rate = calc_unit_rates(network[spm], measure_time, end_time)
                test_data['mean_rate_' + pop_name] = mean_rate

            if track_isi:
                mean_cv, std_cv, check_cv = calc_isi_cv(network[spm], measure_time, end_time)
                test_data['mean_cv_' + pop_name] = mean_cv
                test_data['std_cv_' + pop_name] = std_cv
                test_data['check_cv_' + pop_name] = check_cv
    
                 
    if track_events:
        lfp_time, lfp_trace = trim_brian_monitor(network['stm_pb'],np.mean(-network['stm_pb'].curr_b,0), pA, start_time, end_time)

        lowpass_lfp = calc_low_pass_filter(lfp_trace, test_params['lfp_cutoff'].get_param() / Hz, sim_dt / second)
        ready_monitors['lowpass_lfp'] = lfp_time, lowpass_lfp
        min_peak_dist = test_params['min_peak_dist'].get_param()
        mpd = int(min_peak_dist / sim_dt)
        min_peak_height = test_params['min_peak_height'].get_param() / pA
        peak_idx = detect_peaks(lowpass_lfp, mph=min_peak_height, mpd=int(min_peak_dist / sim_dt))
        peak_idx = peak_idx[lfp_time[peak_idx] >= (measure_time / second)]
        peak_idx = peak_idx[:-1]
        n_events = len(peak_idx)
        test_data['n_events'] = n_events
                 
    if track_states and n_events >= 1:

        # calculate LFP baseline as the mean of the [-200,-100] ms preceding peaks:
        baseline_start = int(-200 * ms / sim_dt)
        baseline_stop = int(-100 * ms / sim_dt)
        baseline_window = np.array([range(peak + baseline_start, peak + baseline_stop + 1) for peak in peak_idx], dtype=int)
        baseline = np.mean(lowpass_lfp[baseline_window])

        # get amplitude of SWRs:
        swr_amp = lowpass_lfp[peak_idx]
        test_data['swr_amp'] = swr_amp

        # calculate times at half-maximum to get event start and end times:
        swr_halfmax = (swr_amp - baseline) / 2.0 + baseline
        event_pre_index = np.zeros(n_events, dtype=int)
        event_post_index = np.zeros(n_events, dtype=int)
        # look within +- 200 ms from peak center
        window = int(200 * ms / sim_dt)

        for i in range(n_events):
            # find event start:
            aux_idx = (np.abs(lowpass_lfp[peak_idx[i] - window:peak_idx[i]] - swr_halfmax[i])).argmin()
            event_pre_index[i] = int(aux_idx + peak_idx[i] - window)

            # find event end:
            aux_idx = (np.abs(lowpass_lfp[peak_idx[i]: peak_idx[i] + window] - swr_halfmax[i])).argmin()
            event_post_index[i] = int(aux_idx + peak_idx[i])

        # get FWHM (duration) of SWRs:
        swr_fwhm = lfp_time[event_post_index] - lfp_time[event_pre_index]
        test_data['event_durations'] = swr_fwhm * 1e3  # in ms

        # get Inter-Event-Interval of SWRs:
        swr_iei = lfp_time[event_pre_index[1:]] - lfp_time[event_post_index[:-1]]
        test_data['event_intervals'] = swr_iei

        test_data['event_pre_index'] = event_pre_index
        test_data['event_peak_index'] = peak_idx
        test_data['event_post_index'] = event_post_index
           
    
    gauss_window = test_params['gauss_window'].get_param()
    rtm_array = ['rtm_p', 'rtm_b']
    adapt_array = ['stm_p_adp', 'stm_b_adp']
    mempo_array = ['stm_p_mempo', 'stm_b_mempo']
                              
    for rtm in rtm_array: 
        ready_monitors[rtm] = trim_brian_monitor(network[rtm],network[rtm].smooth_rate('gaussian', width=gauss_window),Hz,start_time, end_time)

        if n_events > 0:
            event_mean_firing = np.zeros(n_events)
            nswr_mean_firing = np.zeros(n_events-1)
            event_argmax_firing = np.zeros(n_events)
            event_max_firing = np.zeros(n_events)

            for i in range(n_events):
                event_mean_firing[i] = np.mean(ready_monitors[rtm][1][event_pre_index[i]:event_post_index[i]+1])
                event_argmax_firing[i] = np.argmax(ready_monitors[rtm][1][event_pre_index[i]:event_post_index[i]+1])
                event_max_firing[i] = np.max(ready_monitors[rtm][1][event_pre_index[i]:event_post_index[i]+1])
                
            for i in range(n_events-1):
                nswr_mean_firing[i] = np.mean(ready_monitors[rtm][1][event_post_index[i]:event_pre_index[i+1]])

            test_data[rtm + '_event'] = event_mean_firing
            test_data[rtm + '_nswr'] = nswr_mean_firing
            test_data[rtm + '_event_argmax'] = event_argmax_firing
            test_data[rtm + '_event_max'] = event_max_firing
        
    for adp in adapt_array: 
        ready_monitors[adp] = trim_brian_monitor(network[adp],np.mean(network[adp].curr_adapt, axis=0),pA, start_time, end_time)

        if n_events > 0:
            event_mean_adapt = np.zeros(n_events)
            nswr_mean_adapt = np.zeros(n_events-1)
            event_argmax_adapt = np.zeros(n_events)

            for i in range(n_events):
                event_mean_adapt[i] = np.mean(ready_monitors[adp][1][event_pre_index[i]:event_post_index[i]+1])
                event_argmax_adapt[i] = np.argmax(ready_monitors[adp][1][event_pre_index[i]:event_post_index[i]+1])
                
            for i in range(n_events-1):
                nswr_mean_adapt[i] = np.mean(ready_monitors[adp][1][event_post_index[i]:event_pre_index[i+1]])
                
            test_data[adp + '_event'] = event_mean_adapt
            test_data[adp + '_nswr'] = nswr_mean_adapt
            test_data[adp + '_event_argmax'] = event_argmax_adapt
    
    for pot in mempo_array: 
        ready_monitors[pot] = trim_brian_monitor(network[pot],np.mean(network[pot].v, axis=0),mV, start_time, end_time)

        if n_events > 0:
            event_mean_mempo = np.zeros(n_events)
            nswr_mean_mempo = np.zeros(n_events-1)
            event_argmax_mempo = np.zeros(n_events)

            for i in range(n_events):
                event_mean_mempo[i] = np.mean(ready_monitors[pot][1][event_pre_index[i]:event_post_index[i]+1])
                event_argmax_mempo[i] = np.argmax(ready_monitors[pot][1][event_pre_index[i]:event_post_index[i]+1])
                
            for i in range(n_events-1):
                nswr_mean_mempo[i] = np.mean(ready_monitors[pot][1][event_post_index[i]:event_pre_index[i+1]])

            test_data[pot + '_event'] = event_mean_mempo
            test_data[pot + '_nswr'] = nswr_mean_mempo
            test_data[pot + '_event_argmax'] = event_argmax_mempo
            
    return ready_monitors, test_data, test_params

def average_p_currents(network, ready_monitors, currents_to_record):
    end_time = np.max(network['rtm_p'].t / second) * second
    start_time = 0 * second
    current_array = ['stm_p_adp','stm_pp','stm_pb', 'stm_p_bg', 'stm_p_l', 'stm_p_net']

    # ready_monitors['stm_p_adp'] = trim_brian_monitor(network['stm_p_adp'],np.mean(network['stm_p_adp'].curr_adapt, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_p']: ready_monitors['stm_pp'] = trim_brian_monitor(network['stm_pp'],np.mean(network['stm_pp'].curr_p, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_b']: ready_monitors['stm_pb'] = trim_brian_monitor(network['stm_pb'],np.mean(network['stm_pb'].curr_b, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_bg']: ready_monitors['stm_p_bg'] = trim_brian_monitor(network['stm_p_bg'],np.mean(network['stm_p_bg'].curr_bg, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_l']: ready_monitors['stm_p_l'] = trim_brian_monitor(network['stm_p_l'],np.mean(network['stm_p_l'].curr_l, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_net']: ready_monitors['stm_p_net'] = trim_brian_monitor(network['stm_p_net'],np.mean(network['stm_p_net'].curr_net, axis=0),pA, start_time, end_time)

    return ready_monitors

def average_b_currents(network, ready_monitors, currents_to_record):
    end_time = np.max(network['rtm_b'].t / second) * second
    start_time = 0 * second
    current_array = ['stm_b_adp','stm_bb','stm_bp', 'stm_p_bg', 'stm_p_l', 'stm_b_net']

    # ready_monitors['stm_b_adp'] = trim_brian_monitor(network['stm_p_adp'],np.mean(network['stm_p_adp'].curr_adapt, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_b']: ready_monitors['stm_bb'] = trim_brian_monitor(network['stm_bb'],np.mean(network['stm_bb'].curr_b, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_p']: ready_monitors['stm_bp'] = trim_brian_monitor(network['stm_bp'],np.mean(network['stm_bp'].curr_p, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_bg']: ready_monitors['stm_b_bg'] = trim_brian_monitor(network['stm_b_bg'],np.mean(network['stm_b_bg'].curr_bg, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_l']: ready_monitors['stm_b_l'] = trim_brian_monitor(network['stm_b_l'],np.mean(network['stm_b_l'].curr_l, axis=0),pA, start_time, end_time)
    if currents_to_record['curr_net']: ready_monitors['stm_b_net'] = trim_brian_monitor(network['stm_b_net'],np.mean(network['stm_b_net'].curr_net, axis=0),pA, start_time, end_time)

    return ready_monitors
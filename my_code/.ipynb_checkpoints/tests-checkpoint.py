"""
    tests.py: functions for network tests

    list of standard tests:
        network 0: B cells + poisson spikes
            test 01: tests network 0

        network 1: P and B cells
            test 11: tests if network 1 is in SWR state

        network 2: P and A cells
            test 21: tests if network 2 is in non-SWR state

        network 3: P, B, and A cells (without A->J connections)
            test 31: tests if network 3 is in SWR state

        network 4: P, B, and A cells (all to all connections)
            test 41: tests if network 4 remains in non-SWR state
            test 42: tests if network 4 can transition between states through current injection

        network 5: P, B, and A cells + B->A depression
            test 51: tests properties of spontaneous SWR events
"""

from brian2 import *
from imported_code.detect_peaks import detect_peaks
from my_code.parameters import get_used_params
from my_code.aux_functions import check_network_state, check_brian_monitor,\
    calc_network_frequency, trim_brian_monitor, calc_low_pass_filter, calc_unit_rates, \
    calc_isi_cv, get_q_factor, get_newest_file


def prepare_test(network, test_params, plot_params, test_over=True):
    """
    prepare brian monitors for test plots and calculations

    Args:
        network: brian network
        test_params: test parameters
        plot_params: plot parameters
        test_over: [true/false] if test is not over, calculations are
                                only performed for a fixed time_range

    Returns:
        ready_monitors: monitor data ready for plotting
        test_data: calculated test data
        test_params: test parameters (for tracking used params)

    """

    sim_dt = test_params['sim_dt'].get_param()  # simulation step time
    prep_time = test_params['prep_time'].get_param()  # only data for t > prep_time is used for calculations

    track_events = test_params['track_events'].get_param()  # track SWR events
    track_states = test_params['track_states'].get_param()  # track network state (for test 42)

    track_unit_rates = test_params['track_unit_rates'].get_param()  # track mean and std unit firing rates
    track_isi = test_params['track_isi'].get_param()  # track inter spike intervals
    track_psd = test_params['track_psd'].get_param()  # track power spectral densities
    max_freq = test_params['max_net_freq'].get_param() / Hz  # maximum frequency for power spectral densities

    """ SELECT TIME LIMITS FOR PLOTTING AND CALCULATIONS """

    # get current simulation time:
    end_time = np.max(network['rtm_p'].t / second) * second
    plot_range = plot_params['plot_range'].get_param()

    plot_start = 0 * second
    plot_stop = end_time
    if end_time > plot_range:
        plot_start = end_time - plot_range

    # if test is over, get time plot range if specified:
    if test_over:
        save_plot_start = plot_params['save_plot_start'].get_param()
        save_plot_stop = plot_params['save_plot_stop'].get_param()
        if (save_plot_start / second != 0) or (save_plot_stop / second != 0):
            plot_start = save_plot_start
            plot_stop = save_plot_stop

        # calculations will be performed for t > prep_time:
        if end_time > prep_time:
            calc_start = prep_time
        else:
            calc_start = 0 * second

    # if test not over, calculations are only performed for current plot range
    else:
        calc_start = plot_start

    ready_monitors = {'plot_start': plot_start,
                      'plot_stop': plot_stop}

    test_data = {}

    """ SPIKE MONITOR OPERATIONS """

    # create array of all possible spike monitors:
    spm_array = ['spm_p', 'spm_b', 'spm_a', 'spm_p_out']
    asb_idx = 1
    while asb_idx < 1000:
        if check_brian_monitor(network, 'spm_p_asb_' + str(asb_idx), 'i'):
            spm_array.append('spm_p_asb_' + str(asb_idx))
            asb_idx += 1
        else:
            break

    # trim all existing spike monitors:
    for spm in spm_array:
        if check_brian_monitor(network, spm, 'i'):
            # prepare spike monitor for plotting:
            ready_monitors[spm] = trim_brian_monitor(network[spm], network[spm].i, 1, plot_start, plot_stop)

            pop_name = spm[4:]

            # calculate unit firing rates:
            if track_unit_rates:
                mean_rate, std_rate = calc_unit_rates(network[spm], calc_start, end_time)
                test_data['mean_rate_' + pop_name] = mean_rate
                test_data['std_rate_' + pop_name] = std_rate

            # calculate inter-spike-interval coefficient of variation:
            if track_isi:
                mean_cv, std_cv, check_cv = calc_isi_cv(network[spm], calc_start, end_time)
                test_data['mean_cv_' + pop_name] = mean_cv
                test_data['std_cv_' + pop_name] = std_cv
                test_data['check_cv_' + pop_name] = check_cv

    """ RATE MONITOR OPERATIONS """

    # smoothing gaussian window for population rate calculation:
    gauss_window = test_params['gauss_window'].get_param()

    # create array of all possible rate monitors:
    rtm_array = ['rtm_p', 'rtm_b', 'rtm_a', 'rtm_p_out']
    asb_idx = 1
    while asb_idx < 1000:
        if check_brian_monitor(network, 'rtm_p_asb_' + str(asb_idx), 'rate'):
            rtm_array.append('rtm_p_asb_' + str(asb_idx))
            asb_idx += 1
        else:
            break

    for rtm in rtm_array:
        if check_brian_monitor(network, rtm, 'rate'):

            # prepare rate monitor for plotting with a smooth gaussian window:
            ready_monitors[rtm] = trim_brian_monitor(network[rtm],
                                                     network[rtm].smooth_rate('gaussian', width=gauss_window), Hz,
                                                     plot_start, plot_stop)

            # calculate power spectral density of population rate:
            if track_psd:
                pop_name = rtm[4:]

                # trim raw population rate monitor:
                _, raw_rtm = trim_brian_monitor(network[rtm], network[rtm].rate, Hz,
                                                calc_start, end_time)

                # get power spectrum of population rate in the [0, max_freq] range and fit it to a lorentzian:
                pop_freq, pop_psd, lorentz_fit = calc_network_frequency(raw_rtm, end_time - calc_start,
                                                                        sim_dt / second, max_freq, fit=True)
                ready_monitors['psd_' + pop_name] = pop_freq, pop_psd

                # calculate q-factor of lorentzian fit:
                fit_worked = False
                if len(lorentz_fit) > 0:
                    lorentz_a, lorentz_mu, lorentz_sigma = lorentz_fit
                    if lorentz_mu >= 0:
                        fit_worked = True

                if fit_worked:
                    q_factor = get_q_factor(lorentz_a, lorentz_sigma)
                else:
                    q_factor = 0
                    lorentz_mu = 0
                test_data['q_factor_' + pop_name] = q_factor
                test_data['net_freq_' + pop_name] = lorentz_mu

    """ STATE MONITOR OPERATIONS """

    # prepare synaptic efficacy e_ab plot:
    if check_brian_monitor(network, 'stm_e', 'e_ab'):
        ready_monitors['stm_e'] = trim_brian_monitor(network['stm_e'], np.mean(network['stm_e'].e_ab, axis=0), 1,
                                                     plot_start, plot_stop)

    # calculate LFP and SWR properties:
    if check_brian_monitor(network, 'stm_pb', 'curr_b'):

        if track_events or track_states:
            pb_start = 0 * second
            pb_stop = end_time
        elif track_psd:
            pb_start = calc_start
            pb_stop = end_time
        else:
            pb_start = plot_start
            pb_stop = plot_stop

        # our LFP is the negative mean B->P current:
        lfp_time, lfp_trace = trim_brian_monitor(network['stm_pb'],
                                                 -np.mean(network['stm_pb'].curr_b, axis=0), pA,
                                                 pb_start, pb_stop)

        lowpass_lfp = calc_low_pass_filter(lfp_trace, test_params['lfp_cutoff'].get_param() / Hz, sim_dt / second)
        ready_monitors['lowpass_lfp'] = lfp_time, lowpass_lfp

        # get power spectrum of LFP:
        if track_psd:
            lfp_freq, lfp_psd, _ = calc_network_frequency(lfp_trace, pb_stop - pb_start, sim_dt / second,
                                                          max_freq, fit=False)
            ready_monitors['psd_lfp'] = lfp_freq, lfp_psd

        # get SWR properties:
        if track_events:
            # get lowpass-filtered LFP peaks:
            min_peak_height = test_params['min_peak_height'].get_param() / pA
            min_peak_dist = test_params['min_peak_dist'].get_param()
            peak_idx = detect_peaks(lowpass_lfp, mph=min_peak_height, mpd=int(min_peak_dist / sim_dt))

            # remove peaks for t < prep_time:
            peak_idx = peak_idx[lfp_time[peak_idx] >= (prep_time / second)]

            # remove last event as it might not end:
            peak_idx = peak_idx[:-1]

            num_events = len(peak_idx)
            if num_events > 1:

                # calculate LFP baseline as the mean of the [-200,-100] ms preceding peaks:
                baseline_start = int(-200 * ms / sim_dt)
                baseline_stop = int(-100 * ms / sim_dt)
                baseline_window = np.array(
                    [range(peak + baseline_start, peak + baseline_stop + 1) for peak in peak_idx], dtype=int)
                baseline = np.mean(lowpass_lfp[baseline_window])

                # get amplitude of SWRs:
                swr_amp = lowpass_lfp[peak_idx]
                ready_monitors['swr_amp'] = swr_amp

                # calculate times at half-maximum to get event start and end times:
                swr_halfmax = (swr_amp - baseline) / 2.0 + baseline
                swr_start_idx = np.zeros(num_events, dtype=int)
                swr_end_idx = np.zeros(num_events, dtype=int)
                # look within +- 100 ms from peak center
                window = int(100 * ms / sim_dt)
                for i in range(num_events):
                    # find event start:
                    aux_idx = (np.abs(lowpass_lfp[peak_idx[i] - window:peak_idx[i]] - swr_halfmax[i])).argmin()
                    swr_start_idx[i] = int(aux_idx + peak_idx[i] - window)

                    # find event end:
                    aux_idx = (np.abs(lowpass_lfp[peak_idx[i]: peak_idx[i] + window] - swr_halfmax[i])).argmin()
                    swr_end_idx[i] = int(aux_idx + peak_idx[i])

                # get FWHM (duration) of SWRs:
                swr_fwhm = lfp_time[swr_end_idx] - lfp_time[swr_start_idx]
                ready_monitors['swr_fwhm'] = swr_fwhm * 1e3  # in ms

                # get Inter-Event-Interval of SWRs:
                swr_iei = lfp_time[swr_start_idx[1:]] - lfp_time[swr_end_idx[:-1]]
                ready_monitors['swr_iei'] = swr_iei

    return ready_monitors, test_data, test_params


def run_test(network, test_params, used_net_params):
    """
    run network test

    Args:
        network: brian network object where test will be
        test_params: test parameters
        used_net_params: network parameters used to build network

    Returns:
        network: brian network object where test was run
        test_params: test parameters (for tracking used params)

    """

    # run network for prep_time:
    prep_time = test_params['prep_time'].get_param()
    network.run(prep_time)

    # run network for the remaining test time:
    sim_time = test_params['sim_time'].get_param()
    test_id = test_params['test_id'].get_param()

    if test_id not in ['42']:
        network.run(sim_time - prep_time, report='text')

    elif test_id == '42':

        stim_time = test_params['stim_time'].get_param()
        wait_time = test_params['wait_time'].get_param()
        stim_strength = test_params['stim_strength'].get_param()

        # stim P cells:
        pop_p = network['pop_p']
        n_asb = used_net_params['n_asb'].get_param()
        if n_asb > 0:
            init_asb = int(test_params['init_asb'].get_param())
            n_p_asb = int(used_net_params['n_p_asb'].get_param())
            pop_p[(init_asb-1)*n_p_asb:init_asb*n_p_asb].curr_stim = stim_strength * np.random.rand(n_p_asb)
        else:
            pop_p.curr_stim = stim_strength * np.random.rand(pop_p.N)
        network.run(stim_time)
        pop_p.curr_stim = 0 * pA
        network.run(wait_time)

        pop_p.curr_stim = -stim_strength * np.random.rand(pop_p.N)
        network.run(stim_time)
        pop_p.curr_stim = 0 * pA
        network.run(wait_time)

        # stim B cells:
        pop_b = network['pop_b']
        pop_b.curr_stim = stim_strength * np.random.rand(pop_b.N)
        network.run(stim_time)
        pop_b.curr_stim = 0 * pA
        network.run(wait_time)

        pop_b.curr_stim = -stim_strength * np.random.rand(pop_b.N)
        network.run(stim_time)
        pop_b.curr_stim = 0 * pA
        network.run(wait_time)

        # stim A cells:
        pop_a = network['pop_a']
        pop_a.curr_stim = -stim_strength * np.random.rand(pop_a.N)
        network.run(stim_time)
        pop_a.curr_stim = 0 * pA
        network.run(wait_time)

        pop_a.curr_stim = +stim_strength * np.random.rand(pop_a.N)
        network.run(stim_time)
        pop_a.curr_stim = 0 * pA
        network.run(wait_time)

    return network, test_params


def print_test_results(monitors, test_data, test_params):
    """
    print test results

    Args:
        monitors: prepared monitors with simulation recordings
        test_data: calculated test data
        test_params: test parameters

    Returns:
        test parameters (for tracking used params)

    """
    test_id = test_params['test_id'].get_param()
    prep_time = test_params['prep_time'].get_param()

    if test_id in ['11', '21', '31', '41']:
        # load results from previous Test for comparison:
        prev_test_data = {}
        if test_id in ['31', '41']:
            if test_id == '31':
                prev_test_id = '11'
            else:
                prev_test_id = '21'

            prev_test_path = get_newest_file('outputs/test' + prev_test_id + '*.npz')
            if prev_test_path != '':
                prev_test_data = np.load(prev_test_path, encoding='latin1', allow_pickle=True)
                prev_test_data = dict(zip(("{}".format(i) for i in prev_test_data),
                                          (prev_test_data[j] for j in prev_test_data)))
            else:
                prev_test_data = None

        max_diff = test_params['max_diff'].get_param() / Hz
        max_a_swr = test_params['max_a_swr'].get_param() / Hz
        max_b_nswr = test_params['max_b_nswr'].get_param() / Hz

        print('\n============== UNIT FIRING RATES ===============')

        # create array of all possible populations for which mean unit rate was calculated:
        mean_rate_array = ['mean_rate_p', 'mean_rate_b', 'mean_rate_a', 'mean_rate_p_out']
        asb_idx = 1
        while asb_idx < 1000:
            if 'mean_rate_p_asb_' + str(asb_idx) in test_data:
                mean_rate_array.append('mean_rate_p_asb_' + str(asb_idx))
                asb_idx += 1
            else:
                break

        # print unit firing rates:
        for mean_rate_str in mean_rate_array:
            if mean_rate_str in test_data:
                pop_name = mean_rate_str[10:]
                mean_rate = test_data[mean_rate_str]
                std_rate = test_data['std_rate_' + pop_name]
                print('%s firing rate (%.2f +/- %.2f) Hz' % (pop_name.upper(), mean_rate, std_rate))

                if test_id == '31':
                    if pop_name == 'a':
                        if mean_rate < max_a_swr:
                            print('\t A activity is lower than %.2f PASSED' % max_a_swr)
                        else:
                            print('\t WARNING: A activity is higher than %.2f! X' % max_a_swr)

                if test_id == '41':
                    if pop_name == 'b':
                        if mean_rate < max_b_nswr:
                            print('\t B activity is lower than %.2f PASSED' % max_b_nswr)
                        else:
                            print('\t WARNING: B activity is higher than %.2f! X' % max_b_nswr)

        # compare with previous test:
        if test_id in ['31', '41'] and (prev_test_data is not None):
            print('compare data from latest Test %s:' % prev_test_id)
            for mean_rate_str in mean_rate_array:
                if mean_rate_str in test_data:
                    pop_name = mean_rate_str[10:]
                    if mean_rate_str in prev_test_data:
                        mean_rate = test_data[mean_rate_str]
                        prev_mean_rate = prev_test_data[mean_rate_str]
                        if np.abs(mean_rate - prev_mean_rate) - max_diff <= 0:
                            print('\t %s firing rate %.2f Hz (within %.2f Hz threshold PASSED)' %
                                  (pop_name.upper(), prev_mean_rate, max_diff))
                        else:
                            print('\t %s firing rate %.2f Hz (outside %.2f Hz threshold! FAILED)' %
                                  (pop_name.upper(), prev_mean_rate, max_diff))
                    else:
                        print('\t %s firing rate: no data to compare with.' % pop_name.upper())

        print('\n========== REGULARITY OF UNIT FIRING ===========')

        # create array of all possible populations for which isi cv was calculated:
        mean_cv_array = ['mean_cv_p', 'mean_cv_b', 'mean_cv_a', 'mean_cv_p_out']
        asb_idx = 1
        while asb_idx < 1000:
            if 'mean_cv_p_asb_' + str(asb_idx) in test_data:
                mean_cv_array.append('mean_cv_p_asb_' + str(asb_idx))
                asb_idx += 1
            else:
                break

        # print ISI CVs:
        for mean_cv_str in mean_cv_array:
            if mean_cv_str in test_data:
                pop_name = mean_cv_str[8:]
                mean_cv = test_data[mean_cv_str]
                std_cv = test_data['std_cv_' + pop_name]
                check_cv = test_data['check_cv_' + pop_name]
                if check_cv:
                    print('%s ISI CV = %.2f +/- %.2f' % (pop_name.upper(), mean_cv, std_cv))
                else:
                    print('%s ISI CV could not be calculated!' % pop_name.upper())

        # compare with previous test:
        if test_id in ['31', '41'] and (prev_test_data is not None):
            print('compare data from latest Test %s:' % prev_test_id)
            for mean_cv_str in mean_cv_array:
                if mean_cv_str in test_data:
                    pop_name = mean_cv_str[8:]
                    if mean_cv_str in prev_test_data:
                        if prev_test_data['check_cv_' + pop_name]:
                            print('\t %s ISI CV %.2f +/- %.2f' % (pop_name.upper(),
                                                                  prev_test_data[mean_cv_str],
                                                                  prev_test_data['std_cv_' + pop_name]))
                        else:
                            print('\t %s ISI CV could not be calculated!' % pop_name.upper())
                    else:
                        print('\t %s ISI CV: no data to compare with.' % pop_name.upper())

        print('\n================ SYNCHRONICITY =================')

        # create array of all possible populations for which Q-factor was calculated:
        q_factor_array = ['q_factor_p', 'q_factor_b', 'q_factor_a', 'q_factor_p_out']
        asb_idx = 1
        while asb_idx < 1000:
            if 'q_factor_p_asb_' + str(asb_idx) in test_data:
                q_factor_array.append('q_factor_p_asb_' + str(asb_idx))
                asb_idx += 1
            else:
                break

        # print synchronicity and Q-factor:
        q_factor_thres = test_params['q_factor_thres'].get_param()
        for q_factor_str in q_factor_array:
            if q_factor_str in test_data:
                pop_name = q_factor_str[9:]
                q_factor = test_data[q_factor_str]
                net_freq = test_data['net_freq_' + pop_name]
                if q_factor == 0:
                    print('%s is Asynchronous: Q-factor could not be calculated' % pop_name.upper())
                elif q_factor <= q_factor_thres:
                    print('%s is Asynchronous: Q-factor = %s (<= %.2f)' %
                          (pop_name.upper(), '{:.2e}'.format(q_factor), q_factor_thres))
                elif q_factor > q_factor_thres:
                    print('%s network freq. %.2f Hz; Q-factor = %.2f (> %.2f)' %
                          (pop_name.upper(), net_freq, q_factor, q_factor_thres))

        # compare with previous test:
        if test_id in ['31', '41'] and (prev_test_data is not None):
            print('compare data from latest Test %s:' % prev_test_id)
            for q_factor_str in q_factor_array:
                if q_factor_str in test_data:
                    pop_name = q_factor_str[9:]
                    if q_factor_str in prev_test_data:
                        prev_q_factor = prev_test_data[q_factor_str]
                        prev_net_freq = prev_test_data['net_freq_' + pop_name]
                        if prev_q_factor == 0:
                            print('\t %s is Asynchronous: Q-factor could not be calculated' % pop_name.upper())
                        elif prev_q_factor <= q_factor_thres:
                            print('\t %s is Asynchronous: Q-factor = %s (<= %.2f)' %
                                  (pop_name.upper(), '{:.2e}'.format(prev_q_factor), q_factor_thres))
                        elif prev_q_factor > q_factor_thres:
                            print('\t %s network freq. %.2f Hz; Q-factor = %.2f (> %.2f)' %
                                  (pop_name.upper(), prev_net_freq, prev_q_factor, q_factor_thres))
                    else:
                        print('\t %s: no data to compare with.' % pop_name.upper())

    if test_id == '42':
        stim_time = test_params['stim_time'].get_param()
        wait_time = test_params['wait_time'].get_param()
        total_time = prep_time + 6 * (stim_time + wait_time)
        lfp_cutoff = test_params['lfp_cutoff'].get_param()

        lfp_time, lowpass_lfp = monitors['lowpass_lfp']

        thres = test_params['min_peak_height'].get_param() / pA

        check1 = check_network_state('nswr', thres, lfp_time, lowpass_lfp, 0,
                                     (prep_time - 1 / lfp_cutoff) / second)

        if check1:
            print('PASSED: stayed in non-SWR state during prep time.')
        else:
            print('WARNING: Prep time failed to stay in non-SWR state!')

        # P stim:
        check2 = check_network_state('swr', thres, lfp_time, lowpass_lfp,
                                     (prep_time + stim_time) / second,
                                     (prep_time + stim_time + wait_time) / second)
        if check2:
            print('PASSED: stayed in SWR state after P activation.')
        else:
            print('WARNING: Failed to stay in SWR state after P activation!')

        check3 = check_network_state('nswr', thres, lfp_time, lowpass_lfp,
                                     (prep_time + 2 * stim_time + wait_time) / second,
                                     (prep_time + 2 * stim_time + 2 * wait_time - 1 / lfp_cutoff) / second)
        if check3:
            print('PASSED: stayed in non-SWR state after P inactivation.')
        else:
            print('WARNING: Failed to stay in non-SWR state after P inactivation!')

        # B stim:
        check4 = check_network_state('swr', thres, lfp_time, lowpass_lfp,
                                     (prep_time + 3 * stim_time + 2 * wait_time) / second,
                                     (prep_time + 3 * stim_time + 3 * wait_time) / second)
        if check4:
            print('PASSED: stayed in SWR state after B activation.')
        else:
            print('WARNING: Failed to stay in SWR state after B activation!')

        check5 = check_network_state('nswr', thres, lfp_time, lowpass_lfp,
                                     (prep_time + 4 * stim_time + 3 * wait_time) / second,
                                     (prep_time + 4 * stim_time + 4 * wait_time - 1 / lfp_cutoff) / second)
        if check5:
            print('PASSED: stayed in non-SWR state after B inactivation.')
        else:
            print('WARNING: Failed to stay in non-SWR state after B inactivation!')

        # A stim:
        check6 = check_network_state('swr', thres, lfp_time, lowpass_lfp,
                                     (prep_time + 5 * stim_time + 4 * wait_time) / second,
                                     (prep_time + 5 * stim_time + 5 * wait_time) / second)
        if check6:
            print('PASSED: stayed in SWR state after A inactivation.')
        else:
            print('WARNING: Failed to stay in SWR state after A inactivation!')

        check7 = check_network_state('nswr', thres, lfp_time, lowpass_lfp,
                                     (prep_time + 6 * stim_time + 5 * wait_time) / second,
                                     total_time / second)
        if check7:
            print('PASSED: stayed in non-SWR state after A activation.')
        else:
            print('WARNING: Failed to stay in non-SWR state after A activation!')

    if 'swr_iei' in monitors:
        swr_iei = monitors['swr_iei']
        swr_amp = monitors['swr_amp']
        swr_fwhm = monitors['swr_fwhm']

        print('IEI = (%.2f +/- %.2f) s' % (float(np.mean(swr_iei)), float(np.std(swr_iei))))
        print('Amp = (%.2f +/- %.2f) pA' % (float(np.mean(swr_amp)), float(np.std(swr_amp))))
        print('FWHM = (%.2f +/- %.2f) ms' % (float(np.mean(swr_fwhm)), float(np.std(swr_fwhm))))

    return test_params


def save_test_results(test_data, monitors, used_test_params, used_net_params):
    """
    save test results to *.npz file
    Args:
        test_data: calculated test data
        monitors: prepared monitors with simulation recordings
        used_test_params: used test parameters
        used_net_params: used network parameters

    """
    data_to_save = test_data

    if 'swr_iei' in monitors:
        data_to_save['swr_iei'] = monitors['swr_iei']
        data_to_save['swr_amp'] = monitors['swr_amp']
        data_to_save['swr_fwhm'] = monitors['swr_fwhm']

    used_test_params = get_used_params(used_test_params)

    data_to_save['used_test_params'] = used_test_params
    data_to_save['used_net_params'] = used_net_params

    test_id = used_test_params['test_id'].get_param()
    time_str = used_test_params['time_str'].get_param()
    np.savez_compressed('outputs/test' + test_id + '_' + time_str + '.npz', **data_to_save)

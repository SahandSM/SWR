from brian2 import *
import os.path
from my_code import parameters as param
from matplotlib import colors
from matplotlib import gridspec
from matplotlib import rc
rc('text', usetex=True)


def color_gradient(color_i, color_f, n):
    """
    calculates array of intermediate colors between two given colors

    Args:
        color_i: initial color
        color_f: final color
        n: number of colors

    Returns:
        color_array: array of n colors
    """

    rgb_color_i = np.array(colors.to_rgb(color_i))
    rgb_color_f = np.array(colors.to_rgb(color_f))
    color_array = [None] * n
    for i in range(n):
        color_array[i] = colors.to_hex(rgb_color_i*(1-i/(n-1)) + rgb_color_f*i/(n-1))

    return color_array


def shade_stimulation(ax, t_start, t_dur, shade_color, shade_prop='1/1'):
    """
    shade portion of subplot to highlight where a stimulation happened

    Args:
        ax: subplot axis
        t_start: shade starting time
        t_dur: shade duration
        shade_color: shade color
        shade_prop: vertical proportion of subplot to shade
    """

    # calculate top and bottom of shading area:
    y_bottom, y_top = ax.get_ylim()
    y_height = y_top - y_bottom
    prop_numerator = int(shade_prop[0])
    prop_denominator = int(shade_prop[-1])
    shade_bottom = y_bottom + y_height*(prop_numerator - 1)/prop_denominator
    shade_top = shade_bottom + y_height*1/prop_denominator

    # create shading rectangle:
    vertices = [(t_start, shade_bottom),
                (t_start + t_dur, shade_bottom),
                (t_start + t_dur, shade_top),
                (t_start, shade_top)]
    rect = Polygon(vertices, facecolor=shade_color, edgecolor=shade_color)
    ax.add_patch(rect)


class MySubplot:
    def __init__(self, subplot_name, monitor_names, plot_type, plot_colors, plot_params, auto_y, x_label):
        """
        initialise subplot object

        Args:
            subplot_name: subplot name
            monitor_names: array with name of monitors to be plotted on this subplot
            plot_type: type of plot
                        'raster': raster plot
                        'trace': "normal" 2d plot
                        'hist': histogram
                        ...
            plot_colors: array of colors for each monitor to be plotted on this subplot
            plot_params: plot parameters
            auto_y: [true/false] if true, y axis will automatically adjust limits and ticks
            x_label: x axis label
        """

        self.subplot_name = subplot_name
        self.monitor_names = monitor_names
        self.num_traces = len(self.monitor_names)
        self.plot_type = plot_type
        self.plot_colors = plot_colors
        self.auto_y = auto_y
        self.x_label = x_label
        self.ax = None
        self.lines = [None] * self.num_traces
        self.font_size = plot_params['text_font'].get_param()
        self.spine_width = plot_params['spine_width'].get_param()

        # if fixed y limits have been specified on plot parameters, store them:
        if ((self.subplot_name + '_min') in list(plot_params.keys())) and \
           ((self.subplot_name + '_max') in list(plot_params.keys())):
            y_min = plot_params[self.subplot_name + '_min'].get_param()
            y_max = plot_params[self.subplot_name + '_max'].get_param()
            if y_max > 20:
                y_ticks = [y_min, int(((y_max - y_min) / 2) / 10) * 10]
            else:
                y_ticks = [y_min, int(((y_max - y_min) / 2) / 2) * 2]
            self.fixed_y_ticks = y_ticks
            self.fixed_y_lims = [y_min, y_max]
        else:
            self.fixed_y_ticks = None
            self.fixed_y_lims = None

        self.fixed_x_ticks = None
        self.fixed_x_lims = None
        self.y_log = False  # TODO implement log scale

    def attr_ax(self, subplot_axes):
        self.ax = subplot_axes

    def set_title(self, title_text):
        self.ax.set_title(title_text, loc='left', x=0., y=1.02, fontsize=self.font_size)

    def hide_bottom(self):
        self.ax.spines['bottom'].set_visible(False)
        self.ax.set_xticks([])

    def hide_left(self):
        self.ax.spines['left'].set_visible(False)
        self.ax.set_yticks([])

    def show_time_labels(self):
        self.ax.set_xlabel('time (s)')

    def hide_time_labels(self):
        self.ax.set_xticklabels(labels=[])
        self.ax.set_xlabel('')

    def fix_y_axis(self, y_lims, y_ticks):
        self.fixed_y_ticks = y_ticks
        self.fixed_y_lims = y_lims

    def fix_x_axis(self, x_lims, x_ticks):
        self.fixed_x_ticks = x_ticks
        self.fixed_x_lims = x_lims

    def set_x_ticks(self, x_ticks):
        self.ax.set_xticks(x_ticks)
        self.ax.set_xticklabels(labels=x_ticks, fontsize=self.font_size)

    def set_y_ticks(self, y_ticks):
        self.ax.set_yticks(y_ticks)
        self.ax.set_yticklabels(labels=y_ticks, fontsize=self.font_size)

    def set_y_label(self, label):
        self.ax.set_ylabel(label, fontsize=self.font_size)

    def pretty_ticks(self):
        y_ticks = self.ax.get_yticks()
        self.set_y_ticks(y_ticks)
        x_ticks = self.ax.get_xticks()
        self.set_x_ticks(x_ticks)

    def set_time_ticks(self, plot_start, plot_stop, n_ticks):
        time_ticks = [round(plot_start + (plot_stop - plot_start) * i / n_ticks, 2) for i in
                      range(n_ticks + 1)]
        self.set_x_ticks(time_ticks)

    def add_lines(self):
        for i in range(self.num_traces):
            if self.plot_type == 'raster':
                self.lines[i], = self.ax.plot([], [], '.', ms=1.0, color=self.plot_colors[i])

            elif self.plot_type == 'trace':
                self.lines[i], = self.ax.plot([], [], lw=1, color=self.plot_colors[i])

    def general_format(self):

        self.ax.tick_params(top=False, which='both', labelsize=self.font_size, direction='in', width=self.spine_width)
        self.ax.spines['top'].set_visible(False)
        self.ax.spines['right'].set_visible(False)

        # all raster plots include only 50 neurons:
        if self.plot_type == 'raster':
            self.ax.set_ylim([-1, 51])
            self.hide_left()

            # if more than one monitor in one raster plot (for assemblies), split the 50 neurons equally:
            if self.num_traces > 1:
                asb_height = int(50 / self.num_traces)
                h_lines = np.arange(0, asb_height*(self.num_traces + 1), asb_height)
                for h in h_lines:
                    self.ax.axhline(h, lw=0.5, color='lightgray')

        # if y axis is not automatic, used fixed y ticks and limits:
        if self.auto_y is False:
            if self.fixed_y_ticks is not None:
                self.set_y_ticks(self.fixed_y_ticks)
            if self.fixed_y_lims is not None:
                self.ax.set_ylim(self.fixed_y_lims)

        # if fixed x ticks and limits have been set, use those@
        if self.fixed_x_ticks is not None:
            self.set_x_ticks(self.fixed_x_ticks)
        if self.fixed_x_lims is not None:
            self.ax.set_xlim(self.fixed_x_lims)

        # if any x label has been set, use it:
        if self.x_label != '':
            self.ax.set_xlabel(self.x_label, fontsize=self.font_size)

        self.pretty_ticks()

    def set_lines(self, monitors):
        """
        attribute the correct monitor to each subplot line

        Args:
            monitors: dictionary containing all prepared brian monitors from test run
        """

        # iterate through each line to be plotted
        for i in range(self.num_traces):
            prev_height = 0

            # if required monitor has been calculated:
            if self.monitor_names[i] in list(monitors.keys()):

                # multiple raster plots in one subplot are split equally:
                if (self.plot_type == 'raster') and (self.num_traces > 1):
                    num = int(50 / self.num_traces)
                    times, neurons = monitors[self.monitor_names[i]]
                    idx = (neurons >= num*i) & (neurons < num*(i+1))
                    monitor = times[idx], neurons[idx]
                else:
                    monitor = monitors[self.monitor_names[i]]

                # get previous y axis data to auto update y axis for trace plots:
                if (self.plot_type == 'trace') and self.auto_y:
                    _, prev_height = self.ax.get_ylim()
                    if prev_height < 1:
                        y_ticks = [0, 1]
                        self.set_y_ticks(y_ticks)
                        self.ax.set_ylim([0, 1.05])

                # set monitor x and y arrays to subplot line
                if self.plot_type in ['raster', 'trace']:
                    self.lines[i].set_xdata(monitor[0])
                    self.lines[i].set_ydata(monitor[1])

                # auto update y axis for trace plots:
                if (self.plot_type == 'trace') and self.auto_y:
                    if (len(monitor[1]) > 0) and (np.max(monitor[1]) > prev_height):
                        height = np.max(monitor[1])
                        if height > 10:
                            y_ticks = [0, int(math.ceil(height / 20) * 10)]
                        elif height > 5:
                            y_ticks = [0, int(math.ceil(height / 10) * 5)]
                        elif height > 1:
                            y_ticks = [0, int(math.ceil(height / 4) * 2)]
                        else:
                            y_ticks = [0, 1]
                            height = 1
                        self.ax.set_ylim([0, height * 1.05])
                        self.set_y_ticks(y_ticks)

                # create histogram plots:
                if self.plot_type == 'hist':
                    # axes have to be erased and re-written every time:
                    store_xlabel = self.ax.get_xlabel()
                    store_ylabel = self.ax.get_ylabel()
                    self.ax.clear()
                    self.ax.set_xlabel(store_xlabel, fontsize=self.font_size)
                    self.ax.set_ylabel(store_ylabel, fontsize=self.font_size)

                    # create histogram
                    self.ax.hist(monitor, color=self.plot_colors[i])

                    # set x and y ticks:
                    if len(monitor) == 0:
                        x_ticks = []
                        y_ticks = []
                    else:
                        n_x_ticks = 3
                        x_max = np.max(monitor) * 1.5
                        x_ticks = np.unique(
                            [round((x_max * i / n_x_ticks) * 2) / 2 for i in range(n_x_ticks + 1)]
                        )
                        n_y_ticks = 2
                        _, top = self.ax.get_ylim()
                        y_ticks = np.unique([round((top * i / n_y_ticks)) for i in range(n_y_ticks + 1)])
                    self.set_x_ticks(x_ticks)
                    self.set_y_ticks(y_ticks)


class MySubplotGroup:
    def __init__(self, group_name, group_title, group_type, y_label=''):
        """
        create group of subplot

        Args:
            group_name: subplot group name
            group_title: subplot group title
            group_type: subplot group type
                        'time': subplot group where time is on the x axis
                        ...: other
            y_label: subplot group y axis label
        """

        self.group_name = group_name
        self.group_title = group_title
        self.group_type = group_type
        self.time_labels = False
        self.y_label = y_label
        self.subplots = []

    def add_subplot(self, subplot_name, monitor_names, plot_type, plot_colors, plot_params, auto_y=True, x_label=''):
        """
        add a subplot to this subplot group.
        each subplot within the group can have more than one monitor.

        Args:
            subplot_name: name of subplot
            monitor_names: array with name of monitors to be plotted
            plot_type: type of subplot
                        'raster': raster plot
                        'trace': "normal" 2d plot
                        'hist': histogram
                        ...
            plot_colors: array with colors for each monitor to be plotted
            plot_params: plot parameters
            auto_y: [true/false] if true, y axis ticks and limits are automatically updated
            x_label: label for subplot x axis
        """

        self.subplots.append(MySubplot(subplot_name, monitor_names,
                                       plot_type, plot_colors, plot_params,
                                       auto_y, x_label))

    def get_num_vert(self):
        """
        get total number of rows of subplots in this group
        """
        if self.group_type == 'time':
            return len(self.subplots)
        else:
            return 1

    def get_num_horiz(self):
        """
        get total number of columns of subplots in this group
        ('time' group can only have one column)
        """

        if self.group_type != 'time':
            return len(self.subplots)
        else:
            return 1

    def reveal_time_labels(self):
        """
        show/hide time labels if it is/isn't a 'time' group
        """

        if self.group_type == 'time':
            if self.time_labels:
                self.subplots[-1].show_time_labels()
            else:
                self.subplots[-1].hide_time_labels()

    def init_group_format(self, time_labels=False):
        self.time_labels = time_labels
        self.subplots[0].set_title(self.group_title)

        # for every subplot in group:
        for i in range(len(self.subplots)):
            self.subplots[i].general_format()

            # hide bottom of all subplots in 'time' group except for the last:
            if self.group_type == 'time':
                if i < len(self.subplots) - 1:
                    self.subplots[i].hide_bottom()

        # if a y label has given, set it on the middle subplot of the group:
        if self.y_label != '':
            if self.group_type == 'time':
                label_idx = int(len(self.subplots) / 2)
            else:
                label_idx = 0
            self.subplots[label_idx].set_y_label(self.y_label)

        self.reveal_time_labels()

    def set_time_axes(self, plot_start, plot_stop, n_ticks):
        if self.group_type == 'time':
            for i in range(len(self.subplots)):
                self.subplots[i].ax.set_xlim([plot_start, plot_stop])

            self.subplots[-1].set_time_ticks(plot_start, plot_stop, n_ticks)
            self.reveal_time_labels()


def create_empty_figure(network, test_params, net_params, plot_params):
    """
    create figure where test results will be drawn.
    all subplots and monitors are specified here, but no data is attributed to them.

    Args:
        network: brian network where test is run
        test_params: test parameters
        net_params: network parameters
        plot_params: plot parameters

    Returns:
        fig: matplotlib figure object
        subplot_groups: dictionary with all subplot groups
    """

    text_font = plot_params['text_font'].get_param()
    p_color = plot_params['p_color'].get_param()
    b_color = plot_params['b_color'].get_param()
    a_color = plot_params['a_color'].get_param()
    e_color = plot_params['e_color'].get_param()
    pb_color = plot_params['pb_color'].get_param()

    test_id = test_params['test_id'].get_param()

    n_asb = net_params['n_asb'].get_param()

    # initialise dictionary where all subplot groups will be stored:
    subplot_groups = {}

    """ TIME PLOTS """

    # create group of raster plots:
    group_idx = 0
    subplot_groups[group_idx] = MySubplotGroup('spm', r'\textbf{Spikes}', 'time')
    if 'pop_p' in network:
        n_asb = int(net_params['n_asb'].get_param())
        if n_asb > 0:
            monitors_spm_p = []
            colors_spm_p = color_gradient(p_color, 'darkred', n_asb)
            colors_spm_p += ['gray']
            for i in range(n_asb):
                monitors_spm_p += ['spm_p_asb_' + str(i + 1)]
            monitors_spm_p += ['spm_p_out']
            subplot_groups[group_idx].add_subplot('spm_p', monitors_spm_p, 'raster', colors_spm_p,
                                                  plot_params)
        else:
            subplot_groups[group_idx].add_subplot('spm_p', ['spm_p'], 'raster', [p_color],
                                                  plot_params)

    if 'pop_b' in network:
        subplot_groups[group_idx].add_subplot('spm_b', ['spm_b'], 'raster', [b_color],
                                              plot_params)

    if 'pop_a' in network:
        subplot_groups[group_idx].add_subplot('spm_a', ['spm_a'], 'raster', [a_color],
                                              plot_params)

    auto_y = plot_params['auto_y'].get_param()

    # create group of population rate plots:
    group_idx += 1
    subplot_groups[group_idx] = MySubplotGroup('rtm', r'\textbf{Population Rates}', 'time',
                                               y_label=r'Population Rate (Hz)')
    if 'pop_p' in network:
        if n_asb > 0:
            monitors_rtm_p = ['rtm_p']
            colors_rtm_p = ['black'] + color_gradient(p_color, 'darkred', n_asb)
            for i in range(n_asb):
                monitors_rtm_p += ['rtm_p_asb_'+str(i+1)]
            subplot_groups[group_idx].add_subplot('rtm_p', monitors_rtm_p, 'trace', colors_rtm_p,
                                                  plot_params, auto_y)
        else:
            subplot_groups[group_idx].add_subplot('rtm_p', ['rtm_p'], 'trace', [p_color],
                                                  plot_params, auto_y)

    if 'pop_b' in network:
        subplot_groups[group_idx].add_subplot('rtm_b', ['rtm_b'], 'trace', [b_color],
                                              plot_params, auto_y)

    if 'pop_a' in network:
        subplot_groups[group_idx].add_subplot('rtm_a', ['rtm_a'], 'trace', [a_color],
                                              plot_params, auto_y)

    # create B->A synaptic efficacy plot:
    build_step = int(test_params['build_step'].get_param())
    if build_step == 5:
        group_idx += 1
        subplot_groups[group_idx] = MySubplotGroup('e_ab', r'\textbf{Synaptic Efficacy}', 'time',
                                                   y_label=r'e')
        subplot_groups[group_idx].add_subplot('stm_e', ['stm_e'], 'trace', [e_color],
                                              plot_params, auto_y=False)
        subplot_groups[group_idx].subplots[0].fix_y_axis([0, 1], [0.0, 0.5, 1.0])

    # create lowpass-filtered LFP plot:
    track_events = test_params['track_events'].get_param()
    track_states = test_params['track_states'].get_param()
    if track_events or track_states:
        group_idx += 1
        subplot_groups[group_idx] = MySubplotGroup('lowpass_lfp', r'\textbf{Low-pass-filtered LFP}', 'time',
                                                   y_label=r'$\overline{B \to P}$ (pA)')
        subplot_groups[group_idx].add_subplot('lowpass_lfp', ['lowpass_lfp'], 'trace', [pb_color],
                                              plot_params, auto_y)

    """ OTHER PLOTS """

    # create group of Power Spectral Density plots:
    track_psd = test_params['track_psd'].get_param()
    if track_psd:
        group_idx += 1
        subplot_groups[group_idx] = MySubplotGroup('psd',
                                                   r'\textbf{Network Oscillations}',
                                                   'horizontal',
                                                   y_label=r'PSD (Hz)')

        if 'pop_p' in network:
            subplot_groups[group_idx].add_subplot('psd_p', ['psd_p'], 'trace', [p_color],
                                                  plot_params, auto_y, x_label=r'P Frequency (Hz)')

        if 'pop_b' in network:
            subplot_groups[group_idx].add_subplot('psd_b', ['psd_b'], 'trace', [b_color],
                                                  plot_params, auto_y, x_label=r'B Frequency (Hz)')

        if ('pop_a' in network) and ('pop_b' not in network):
            subplot_groups[group_idx].add_subplot('psd_a', ['psd_a'], 'trace', [a_color],
                                                  plot_params, auto_y, x_label=r'A Frequency (Hz)')

        if ('pop_p' in network) and ('pop_b' in network):
            subplot_groups[group_idx].add_subplot('psd_lfp', ['psd_lfp'], 'trace', [pb_color],
                                                  plot_params, auto_y, x_label=r'LFP Frequency (Hz)')

        max_net_freq = test_params['max_net_freq'].get_param() / Hz
        n_psd_ticks = plot_params['num_psd_ticks'].get_param()
        psd_x_lims = [0, max_net_freq]
        psd_x_ticks = np.unique([round((max_net_freq * i / n_psd_ticks) * 2) / 2 for i in range(n_psd_ticks + 1)])
        for psd_subplot in subplot_groups[group_idx].subplots:
            psd_subplot.fix_x_axis(psd_x_lims, psd_x_ticks)
            psd_subplot.y_log = True

    # create group to plot SWR properties:
    track_events = test_params['track_events'].get_param()
    if track_events:
        group_idx += 1
        subplot_groups[group_idx] = MySubplotGroup('events',
                                                   r'\textbf{Event Properties}',
                                                   'horizontal',
                                                   y_label=r'\# Events')
        subplot_groups[group_idx].add_subplot('swr_iei', ['swr_iei'], 'hist', ['gray'],
                                              plot_params, x_label='IEI (s)')
        subplot_groups[group_idx].add_subplot('swr_amp', ['swr_amp'], 'hist', ['gray'],
                                              plot_params, x_label='Amplitude (pA)')
        subplot_groups[group_idx].add_subplot('swr_fwhm', ['swr_fwhm'], 'hist', ['gray'],
                                              plot_params, x_label='FWHM (ms)')

        # TODO add previous IEI vs amplitude plot

    """ CREATE FIGURE """

    # get fig dimensions in inches
    plot_width = plot_params['plot_width'].get_param() / (cmeter * 2.54)
    fig_height = plot_params['fig_height'].get_param() / (cmeter * 2.54)

    # plot a list of parameters on the side:
    show_params = plot_params['show_params'].get_param()
    params_width = plot_params['params_width'].get_param() / (cmeter * 2.54)
    if show_params:
        fig_width = plot_width + params_width
    else:
        fig_width = plot_width

    # create figure object:
    fig = plt.figure(figsize=(fig_width, fig_height))
    fig.canvas.set_window_title('Test ' + test_id)

    # get total number of columns and rows of subplots
    num_groups = len(subplot_groups)
    num_vert = 0
    num_horiz = 0
    last_vert = 0
    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]
        if group.group_type == 'time':
            last_vert = group_idx
        num_vert += group.get_num_vert()
        if group.get_num_horiz() > num_horiz:
            num_horiz = group.get_num_horiz()

    # calculate array of width ratios for all subplots:
    plot_prop = plot_width / fig_width
    if num_horiz > 1:
        width_ratios = []
        size_horiz = plot_prop/(1.25*num_horiz)
        for i in range(num_horiz):
            width_ratios += [size_horiz]
            if i < num_horiz - 1:
                width_ratios += [0.25*size_horiz]
        width_ratios = width_ratios
    else:
        width_ratios = [plot_prop]
    if show_params:
        params_prop = 1 - plot_prop
        width_ratios += [0.1*params_prop, 0.9*params_prop]

    # calculate array of height ratios for all subplots:
    height_ratios = []
    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]
        for i in range(group.get_num_vert()):
            if group.group_type == 'time':
                height_ratios += [1]
            else:
                height_ratios += [1.9]
        if group_idx < len(subplot_groups) - 1:
            if group_idx == last_vert:
                height_ratios += [1.1]
            else:
                height_ratios += [0.5]

    # create grid where all subplots will be added:
    num_vert_plots = num_vert + num_groups - 1
    num_horiz_plots = num_horiz * 2 - 1
    if show_params:
        num_horiz_plots += 2
    gs = gridspec.GridSpec(num_vert_plots, num_horiz_plots,
                           width_ratios=width_ratios,
                           height_ratios=height_ratios)

    # add all subplots to grid:
    idx_vert = 0
    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]
        if group.group_type == 'time':
            for i in range(len(group.subplots)):
                group.subplots[i].attr_ax(fig.add_subplot(gs[idx_vert, 0:(num_horiz*2-1)]))
                idx_vert += 1
        else:
            idx_horiz = 0
            for i in range(len(group.subplots)):
                group.subplots[i].attr_ax(fig.add_subplot(gs[idx_vert, idx_horiz]))
                idx_horiz += 2
            idx_vert += 1
        idx_vert += 1

    # initialise groups and attribute line objects to each subplot:
    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]
        if group_idx == last_vert:
            time_labels = True
        else:
            time_labels = False
        group.init_group_format(time_labels)
        for i in range(len(group.subplots)):
            group.subplots[i].add_lines()

    # create side plot with parameter list and network sketch:
    if show_params:
        text_str = param.print_param_list(net_params, test_params)
        ax_params = fig.add_subplot(gs[0:3, -1])

        # Make invisible:
        for spine in ['top', 'right', 'bottom', 'left']:
            ax_params.spines[spine].set_visible(False)
        ax_params.set_xticks([])
        ax_params.set_xticklabels(labels=[])
        ax_params.set_yticks([])
        ax_params.set_yticklabels(labels=[])

        ax_params.text(-1500, 0, text_str, fontsize=text_font, clip_on=False,
                       horizontalalignment='left', verticalalignment='top', linespacing=1.5)

        sketch_path = 'network_sketches/step' + str(build_step) + '.png'
        if os.path.isfile(sketch_path):
            diagram = plt.imread('network_sketches/step' + str(build_step) + '.png')
            ax_params.imshow(diagram, interpolation='nearest')
        else:
            print('\nERROR: network sketch .png not available!\n\tRun network_sketches/svg_to_png.sh')

        ax_params.set_ylim([1500, 0])
        ax_params.set_xlim([-1300, 2000])

    plt.subplots_adjust(wspace=0., hspace=0.15)

    return fig, subplot_groups


def draw_in_figure(test_params, plot_params, subplot_groups, monitors):
    """
    draws test results by attributing the prepared test monitors
    to each of the subplot objects previously created

    Args:
        test_params: test parameters
        plot_params: plot parameters
        subplot_groups: dictionary with all subplot groups previously created
        monitors: dictionary with all brian monitors prepared for test
    """

    plot_start = monitors['plot_start']
    plot_stop = monitors['plot_stop']

    n_time_ticks = plot_params['num_time_ticks'].get_param()
    min_peak_height = test_params['min_peak_height'].get_param() / pA

    test_id = test_params['test_id'].get_param()

    if test_id == '42':
        prep_time = test_params['prep_time'].get_param() / second
        stim_time = test_params['stim_time'].get_param() / second
        wait_time = test_params['wait_time'].get_param() / second
        init_asb = int(test_params['init_asb'].get_param())
    else:
        prep_time = []
        stim_time = []
        wait_time = []
        init_asb = []

    for group_idx in subplot_groups:
        group = subplot_groups[group_idx]

        if group.group_type == 'time':
            group.set_time_axes(plot_start / second, plot_stop / second, n_time_ticks)

        for sub_plot in group.subplots:
            sub_plot.set_lines(monitors)

            # draw SWR detection threshold:
            if sub_plot.subplot_name == 'lowpass_lfp':
                subplot_groups[group_idx].subplots[0].ax.axhline(y=min_peak_height, lw=0.7, ls='--', color='lightgray')

            # draw shaded rectangles in test '42' to indicate where stimulations were applied:
            if test_id == '42':
                if sub_plot.subplot_name == 'spm_p':
                    n_asb = int(sub_plot.num_traces - 1)
                    if n_asb > 0:
                        frac_stim = str(init_asb) + '/' + str(n_asb + 1)
                    else:
                        frac_stim = '1/1'

                    # P positive stim:
                    shade_stimulation(sub_plot.ax, prep_time, stim_time, 'lightgray', frac_stim)
                    # P negative stim:
                    shade_stimulation(sub_plot.ax, prep_time + stim_time + wait_time, stim_time, 'lightpink', frac_stim)

                elif sub_plot.subplot_name == 'spm_b':
                    # B positive stim:
                    shade_stimulation(sub_plot.ax, prep_time + 2*stim_time + 2*wait_time, stim_time, 'lightgray')
                    # B negative stim:
                    shade_stimulation(sub_plot.ax, prep_time + 3*stim_time + 3*wait_time, stim_time, 'lightpink')

                elif sub_plot.subplot_name == 'spm_a':
                    # A negative stim:
                    shade_stimulation(sub_plot.ax, prep_time + 4*stim_time + 4*wait_time, stim_time, 'lightpink')
                    # A positive stim:
                    shade_stimulation(sub_plot.ax, prep_time + 5*stim_time + 5*wait_time, stim_time, 'lightgray')

    # short pause necessary for live plotting:
    plt.pause(0.05)

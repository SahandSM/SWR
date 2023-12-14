# ad2296be49f19d6ae177063854ae693de1e0ba7b: created a branch for the first phase of project. making shure the model generates SWR that oscilat.
# 812ca2edcfc442de2f8c036f4d5d09e0f70ba93d: created branch dedicated to get population P to oscilate detached from pop. B.
# 7d8990b558b137cefe6c1d2f764a5f77a75f0533: oscilation in small time scale.
# fa6246bcc3fb7a124c921e79e8c113f0c7714c7f: to increase the time sclae of the oscilation, increase adaptation current and decrease background current

# f890c385db8d9bff303399c7fb6fc3d92698da22: summary:
#       threre is a barckground thereshold for the the isolated population P to jumpt to activity. (100pA)
#       the threshold doesn't change with the addition of P current. but the jup increases dramaticlly.
#       when we are close enough to the threshold and above it, adding enough adaptation leads to oscilation.
#       in presense of P current and in oscilation state, the adaptation curret oscilates between 0 and adaptation increment J_spi_p
#       the value of the adaptation increment affect the oscilation time scale.

# c1c4cc6b4327f4e01f2f03f1c6bdac5d6f424c22: population B also has the same background current threshold as P (100pA). but it is oscilating (droping to zero) without the adaptaiton current. why?

# d3fca3b18848811e8348edacffa3898a365d0471: I set the initial potentail of two neuron in both population equal to -65mV. But the behaviour is still very different.
#       to do this, I had to note which index neurons are being recorded by the monitor and set them equal.

# 51bd0527abdd57a52a8464c307d7d267fc3745c0: the reason is that neurons in population P receive random initial adaption currents. This initial adaptation current 
#       goes to zero in time. but the neurons are out of sync by this then.

# 6d10f1f92f1269e43a53701a0bbbe37465cf5d54: summary:
#       withough initial adaptation, all neurons fire in syncrony.
#       the 100pA constant background current pushes the resting potential to above threshold.
#       by adding adaption (J_spi_p), we are adding a time dependent current which decays in time and increments each time the population fires.
#       therefore, the resting potential becomes time dependent. also, it casues the potential to increases slower since the net current is now
#       smaller or equal to 100pA. Therefore, the firing rate also decreases.


# just a comment to prepare for merging

# 628b49729ab646f53224e101eb6f940ecb416f3b: added a branch to work on adding current monitors
# c077e9688a95e55f53cd8608ded3d7ac43e2f526: at this point, all current monitors and ready monitors are added.
#       also decided to put all initial potentials at -60mV to begin with.

# 100pA is the current threshold. At back ground currents below 100pA, the resting potential is below -50mV and the cell doesn't fire.
#   at background currents above 100pA, the resting potential is above -50mV and the cell fries.

# 4bf29629d4f8b5ccbef589bcfd6235ce7016624b: to decrease the firing rate (increase the interspike interval) we should reduce the background current.
#       reducing the background current, reduces the resting potential and neuron doesn't fire. so we need to reduce the leak conductance to reduce the resting potential as well.

# 2f47508449e8e034660dbebeff38a816691487b5: I tried to get population B to oscilate in isolation and monitor b current (LFP) to get peaks. the problem is that b current is constantly above min_peak.
#       to make it oscilate I need an oscilatory current to B population. so I add adaptation to po P to make p current oscilatory. and then connected population P to population B.
#       the result is both populations are oscillating. LFP is also oscilating. but both firing rates and LFP amplitude are verly small.
#       it gets a bit complicated to increase the firing rats and the LFP amplitude.

# c0c720eac7a09fd9f8b5d372ef3b8729976d4efa: I could increase the LFP amplitude to above 20 by increasing the g_pb. I had to increase background current to P as well to prevent total inhibition.
#       I also had to increase adaptation increments.
#       problem is population P firing rate is still as low as 10 spike/s during event!

# 82b05ff0397c213ef4284ce833ba40761e63abee: instead of getting the oscilatory current to pop B from population P wit adaptation, I added addaptation to population B itself.
#       the results: b to p current peaks are above 30.
#                    both population firing rates are 0 during non-states. 
#                    both P and B firing rates are still relatively low during even (10 and 22 spikes/s respectively)
#       note: initial values are not random yet.
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
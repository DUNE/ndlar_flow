################################################################################
##                                                                            ##
##    CONTAINS: Definitions of particle information from PDG (e.g. particle   ##
##              PDG IDs, masses, etc.)                                        ##
##                                                                            ##
################################################################################

####-------------------------- PDG ID DEFINITIONS --------------------------####

neutral_pdg=[111] #, 22] #, 2112] # add K0, rho0, eta0?
meson_pdg={111,211,-211,130,310,311,321,-321,221,331,421,-421,411,-411, 431,-431}
nu_mu_pdg=14


####------------------ PDG ID/PARTICLE LABEL DICTIONARIES ------------------####

hadron_pdg_dict ={2112:'n',
                 -2112:r'$\bar{n}$',
                  2212:'p',
                 -2212:r'$\bar{p}$', 
                  3112:r'$\Sigma^-$',
                  3122:r'$\Lambda^0$',
                 -3122:r'$\bar{\Lambda}^0$',
                  3212:r'$\Sigma^0$',
                  3222:r'$\Sigma^+$', 
                  4212:r'$\Sigma_c^+$',
                  4222:r'$\Sigma_c^{++}$',
                  4112:r'$\Sigma_c^0$', 
                  4122:r'$\Lambda_c^+$'} 

neutral_hadron_pdg_dict ={2112:'n',
                         -2112:r'$\bar{n}$',
                          3122:r'$\Lambda^0$',
                         -3122:r'$\bar{\Lambda}^0$',
                          3212:r'$\Sigma^0$',
                          4112:r'$\Sigma_c^0$'} 

selection_pdg_dict ={2212:'Protons',
                       13:'Muons'}
                     


####---------------- PDG ID/PARTICLE PROPERTY DICTIONARIES -----------------####

rest_mass_dict ={2212: 938.27208816,
                   13: 105.6583755  }  # Masses in MeV (from PDG)
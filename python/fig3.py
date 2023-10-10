# Copyright (c) 2021-2023 Ivan Iakoupov
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import os.path

import matplotlib.pyplot as p
import numpy as np
from matplotlib_util import convert_to_matplotlib_format,\
                            extract_params_from_file_name,\
                            get_column_dic
import matplotlib.ticker as ticker
from common_plot_setup import do_common_setup

do_common_setup()

prefix = 'plot_data'

decay1_data_file = 'decay_with_jqf_startEigenstate_kappa_1_gamma2_50_gamma1I_0_gamma2I_0_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_omega2_3997_kxr_0_kx2_1.csv'
decay2_data_file = 'decay_with_jqf_startEigenstate_kappa_1_gamma2_0_gamma1I_0_gamma2I_0_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_omega2_3997_kxr_0_kx2_1.csv'

fig = p.figure(figsize=(3.3,2.0))
param_dict_decay1 = extract_params_from_file_name(decay1_data_file)
full_path_decay1 = os.path.join(prefix, decay1_data_file)
if not os.path.exists(full_path_decay1):
    print('Path {} doesn\'t exist'.format(full_path_decay1))
data_decay1 = np.loadtxt(full_path_decay1, dtype=np.float64, delimiter=';',
                         unpack=True, skiprows=1)
column_dic_decay1 = get_column_dic(full_path_decay1)

param_dict_decay2 = extract_params_from_file_name(decay2_data_file)
full_path_decay2 = os.path.join(prefix, decay2_data_file)
if not os.path.exists(full_path_decay2):
    print('Path {} doesn\'t exist'.format(full_path_decay2))
data_decay2 = np.loadtxt(full_path_decay2, dtype=np.float64, delimiter=';',
                         unpack=True, skiprows=1)
column_dic_decay2 = get_column_dic(full_path_decay2)

kappaMHz = 2
kappatFactor = 1.0/(2*np.pi*kappaMHz*1e6)*1e9
t_ns_decay1 = data_decay1[column_dic_decay1['t']]*kappatFactor
t_ns_decay2 = data_decay2[column_dic_decay2['t']]*kappatFactor

omega_1 = param_dict_decay1['omega1']
omega_r = param_dict_decay1['omegar']
g = param_dict_decay1['g']

eigenvalue0 = 0.5*(omega_1+omega_r)+np.sqrt((0.5*(omega_r-omega_1))**2+g**2)
eigenvalue1 = 0.5*(omega_1+omega_r)-np.sqrt((0.5*(omega_r-omega_1))**2+g**2)
theta = 0.5*np.angle(0.5*(omega_r-omega_1)+1j*g);
print(eigenvalue0)
print(eigenvalue1)
print(eigenvalue1/omega_r)

ax1 = p.subplot(111)
print((g/(omega_r-omega_1))**2)
print(np.sin(theta)**2)
purcell_decay_rate = np.sin(theta)**2*eigenvalue1/omega_r
print('Purcell decay rate kappa_Purcell = {}*kappa'.format(purcell_decay_rate))
print('Purcell decay rate kappa_Purcell/(2*pi) = {} Hz'.format(purcell_decay_rate*kappaMHz*1e6))
population_2 = max(data_decay1[column_dic_decay1['population_2']])
print('max JQF population = {}'.format(population_2))
data_decay = np.exp(-purcell_decay_rate*data_decay2[column_dic_decay2['t']])
F_ss = (param_dict_decay1['gamma2']/(purcell_decay_rate+param_dict_decay1['gamma2']))**2
data_jqf_steady = np.full_like(t_ns_decay1, F_ss)
p.plot(t_ns_decay2, 1-data_decay2[column_dic_decay2['fidelity']], label='no JQF')
p.plot(t_ns_decay1, 1-data_decay1[column_dic_decay1['fidelity']], '--', label='JQF')
p.plot(t_ns_decay2, 1-data_decay, '-.' , label='no JQF (analytical)')
p.plot(t_ns_decay2, 1-data_jqf_steady, 'k:' , label='JQF (analytical)')
print(F_ss)
#p.axhline(F_ss, ls=':', color='k')
p.ylim(5e-5, 1e-1)
p.legend(loc='center right', handlelength=2.2)
#p.title(r'(a)')
p.ylabel(r'$1-F$')
p.xlabel(r'$t\,[{\rm ns}]$')
p.yscale('log')


p.tight_layout(pad=0.2)
#p.show()
p.savefig('fig3.eps')

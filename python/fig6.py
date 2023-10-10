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

Rabi_data_file = 'control_with_jqf_kappa_1_gamma2_50_g_54.772_Omega_gauss_omegad_3997_omegar_5000_omega1_4000_omega2_3997_kxr_0_kx2_1_t_0.628_sigmaF_0.02_OmegaMax_100_Nc_3.csv'
Rabi_data_file2 = 'control_with_jqf_kappa_1_gamma2_50_g_54.772_Omega_adrk4iter13_omegad_3997_omegar_5000_omega1_4000_omega2_3997_kxr_0_kx2_1_t_0.628_Nc_100.csv'

p.figure(figsize=(3.3,4.5))
param_dict_Rabi = extract_params_from_file_name(Rabi_data_file)
param_dict_Rabi2 = extract_params_from_file_name(Rabi_data_file2)
#g1d = param_dict['g1d']
full_path_Rabi = os.path.join(prefix, Rabi_data_file)
if not os.path.exists(full_path_Rabi):
    print('Path {} doesn\'t exist'.format(full_path_Rabi))
data_Rabi = np.loadtxt(full_path_Rabi, dtype=np.float64, delimiter=';',
                       unpack=True, skiprows=1)
column_dic_Rabi = get_column_dic(full_path_Rabi)

full_path_Rabi2 = os.path.join(prefix, Rabi_data_file2)
if not os.path.exists(full_path_Rabi2):
    print('Path {} doesn\'t exist'.format(full_path_Rabi2))
data_Rabi2 = np.loadtxt(full_path_Rabi2, dtype=np.float64, delimiter=';',
                       unpack=True, skiprows=1)
column_dic_Rabi2 = get_column_dic(full_path_Rabi2)

max_val_arg = data_Rabi[column_dic_Rabi['tilde_F']].argmax()
val_array_max = data_Rabi[column_dic_Rabi['tilde_F']][max_val_arg]
t_for_val_array_max = data_Rabi[column_dic_Rabi['t']][max_val_arg]

last_val = data_Rabi[column_dic_Rabi['tilde_F']][len(data_Rabi[column_dic_Rabi['t']])-1]
print('last DRAG tilde_F = {}'.format(last_val))

last_val2 = data_Rabi2[column_dic_Rabi2['tilde_F']][len(data_Rabi2[column_dic_Rabi2['t']])-1]
print('last ADRK4 tilde_F = {}'.format(last_val2))

kappaMHz = 2
kappatFactor = 1.0/(2*np.pi*kappaMHz*1e6)*1e9
t_ns_Rabi = data_Rabi[column_dic_Rabi['t']]*kappatFactor
t_ns_Rabi2 = data_Rabi2[column_dic_Rabi2['t']]*kappatFactor

ax2 = p.subplot(311)
ax2.xaxis.set_major_locator(ticker.MultipleLocator(10))
#p.plot(t_ns_Rabi, data_Rabi[column_dic_Rabi['population_1']], label=r'AA population')
p.plot(t_ns_Rabi, data_Rabi[column_dic_Rabi['population_2']], label=r'$\langle b_2^\dagger b_2\rangle$')
p.plot(t_ns_Rabi, 1-data_Rabi[column_dic_Rabi['tilde_F']], '--', label=r'$1-\tilde{F}$')
p.plot(t_ns_Rabi, data_Rabi[column_dic_Rabi['res_population']], ':', label=r'$\langle a^\dagger a\rangle$')
#p.plot(t_ns_Rabi, data_Rabi[column_dic_Rabi['s_1_1']], linestyle='-.', label=r'$\langle \sigma_{2,11}\rangle$')
p.ylim(4e-5, 1e1)
p.title(r'(a)')
p.legend(loc='upper right', handlelength=2.2)
p.xlabel(r'$t\,[{\rm ns}]$')
p.yscale('log')
ax2.yaxis.set_major_locator(ticker.FixedLocator([1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1]))

ax3 = p.subplot(312)
ax3.xaxis.set_major_locator(ticker.MultipleLocator(10))
line_styles_list = ['-', '--', ':', '-.']
for n in range(3):
    p.plot(t_ns_Rabi, data_Rabi[column_dic_Rabi['s_1_{}'.format(n)]], linestyle=line_styles_list[n], label=r'$j={}$'.format(n))
n=10
p.plot(t_ns_Rabi, data_Rabi[column_dic_Rabi['s_1_{}'.format(n)]], linestyle=line_styles_list[3], label=r'$j={}$'.format(n))
p.ylim(1e-10, 1e1)
p.title(r'(b)')
#p.legend(loc='upper left', handlelength=2.2)
p.xlabel(r'$t\,[{\rm ns}]$')
#In the simulation code the indices are zero-based, while
#in the paper they are one-based. Hence s_1_j corresponds
#to \sigma_{2,jj}.
p.ylabel(r'$\langle \sigma_{2,jj}\rangle$')
p.yscale('log')
ax3.annotate(r'$j=0$',
            xy=(45, 7e-2), xycoords='data', size=6
)
ax3.annotate(r'$j=1$',
            xy=(45, 3e-6), xycoords='data', size=6
)
ax3.annotate(r'$j=2$',
            xy=(36, 5e-10), xycoords='data', size=6
)
ax3.annotate(r'$j=10$',
            xy=(21, 5e-10), xycoords='data', size=6
)

ax4 = p.subplot(313)
ax4.xaxis.set_major_locator(ticker.MultipleLocator(10))
ax4.yaxis.set_major_locator(ticker.MultipleLocator(100))
p.plot(t_ns_Rabi, kappaMHz*data_Rabi[column_dic_Rabi['Omega_Re']], label=r'${\rm Re}[\Omega]/(2\pi)$')
p.plot(t_ns_Rabi, kappaMHz*data_Rabi[column_dic_Rabi['Omega_Im']], '--', label=r'${\rm Im}[\Omega]/(2\pi)$ DRAG')
p.plot(t_ns_Rabi2, kappaMHz*data_Rabi2[column_dic_Rabi2['Omega_Re']], ':', label=r'${\rm Re}[\Omega]/(2\pi)$ ${\rm iter}=13$')
p.plot(t_ns_Rabi2, kappaMHz*data_Rabi2[column_dic_Rabi2['Omega_Im']], '-.', label=r'${\rm Im}[\Omega]/(2\pi)$ ${\rm iter}=13$')
p.ylabel(r'$[{\rm MHz}]$')
p.ylim(-20, 220)
p.title(r'(c)')
p.legend(loc='upper right', handlelength=2.2)
p.xlabel(r'$t\,[{\rm ns}]$')

p.tight_layout(pad=0.3, h_pad=0.4, w_pad=0.4)
#p.show()
p.savefig('fig6.eps')

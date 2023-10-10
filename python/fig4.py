# Copyright (c) 2022-2023 Ivan Iakoupov
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

decay1_data_file = 'decay_with_jqf_omega2_kappa_1_gamma2_50_gamma1I_0_gamma2I_0_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_kxr_0_kx2_1.csv'
decay2_data_file = 'decay_with_jqf_omega2_kappa_1_gamma2_0_gamma1I_0_gamma2I_0_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_kxr_0_kx2_1.csv'
decay3_data_file = 'decay_with_jqf_omega2_kappa_1_gamma2_50_gamma1I_0_gamma2I_1.5_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_kxr_0_kx2_1.csv'
decay5_data_file = 'decay_with_jqf_k0x2_kappa_1_gamma2_50_gamma1I_0_gamma2I_0_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_omega2_3997_kxr_0.csv'
decay6_data_file = 'decay_with_jqf_k0x2_kappa_1_gamma2_0_gamma1I_0_gamma2I_0_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_omega2_3997_kxr_0.csv'
decay7_data_file = 'decay_with_jqf_k0x2_kappa_1_gamma2_50_gamma1I_0_gamma2I_1.5_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_omega2_3997_kxr_0.csv'
decay9_data_file = 'decay_with_jqf_gamma1I_kappa_1_gamma2_50_gamma2I_0_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_omega2_3997_kxr_0_kx2_1.csv'
decay10_data_file = 'decay_with_jqf_gamma1I_kappa_1_gamma2_0_gamma2I_0_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_omega2_3997_kxr_0_kx2_1.csv'
decay11_data_file = 'decay_with_jqf_gamma1I_kappa_1_gamma2_50_gamma2I_1.5_g_54.772_Omega_0_omegad_4000_omegar_5000_omega1_4000_omega2_3997_kxr_0_kx2_1.csv'

fig = p.figure(figsize=(3.3,4.5))
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

param_dict_decay3 = extract_params_from_file_name(decay3_data_file)
full_path_decay3 = os.path.join(prefix, decay3_data_file)
if not os.path.exists(full_path_decay3):
    print('Path {} doesn\'t exist'.format(full_path_decay3))
data_decay3 = np.loadtxt(full_path_decay3, dtype=np.float64, delimiter=';',
                         unpack=True, skiprows=1)
column_dic_decay3 = get_column_dic(full_path_decay3)

param_dict_decay5 = extract_params_from_file_name(decay5_data_file)
full_path_decay5 = os.path.join(prefix, decay5_data_file)
if not os.path.exists(full_path_decay5):
    print('Path {} doesn\'t exist'.format(full_path_decay5))
data_decay5 = np.loadtxt(full_path_decay5, dtype=np.float64, delimiter=';',
                         unpack=True, skiprows=1)
column_dic_decay5 = get_column_dic(full_path_decay5)

param_dict_decay6 = extract_params_from_file_name(decay6_data_file)
full_path_decay6 = os.path.join(prefix, decay6_data_file)
if not os.path.exists(full_path_decay6):
    print('Path {} doesn\'t exist'.format(full_path_decay6))
data_decay6 = np.loadtxt(full_path_decay6, dtype=np.float64, delimiter=';',
                         unpack=True, skiprows=1)
column_dic_decay6 = get_column_dic(full_path_decay6)

param_dict_decay7 = extract_params_from_file_name(decay7_data_file)
full_path_decay7 = os.path.join(prefix, decay7_data_file)
if not os.path.exists(full_path_decay7):
    print('Path {} doesn\'t exist'.format(full_path_decay7))
data_decay7 = np.loadtxt(full_path_decay7, dtype=np.float64, delimiter=';',
                         unpack=True, skiprows=1)
column_dic_decay7 = get_column_dic(full_path_decay7)

param_dict_decay9 = extract_params_from_file_name(decay9_data_file)
full_path_decay9 = os.path.join(prefix, decay9_data_file)
if not os.path.exists(full_path_decay9):
    print('Path {} doesn\'t exist'.format(full_path_decay9))
data_decay9 = np.loadtxt(full_path_decay9, dtype=np.float64, delimiter=';',
                         unpack=True, skiprows=1)
column_dic_decay9 = get_column_dic(full_path_decay9)

param_dict_decay10 = extract_params_from_file_name(decay10_data_file)
full_path_decay10 = os.path.join(prefix, decay10_data_file)
if not os.path.exists(full_path_decay10):
    print('Path {} doesn\'t exist'.format(full_path_decay10))
data_decay10 = np.loadtxt(full_path_decay10, dtype=np.float64, delimiter=';',
                         unpack=True, skiprows=1)
column_dic_decay10 = get_column_dic(full_path_decay10)

param_dict_decay11 = extract_params_from_file_name(decay11_data_file)
full_path_decay11 = os.path.join(prefix, decay11_data_file)
if not os.path.exists(full_path_decay11):
    print('Path {} doesn\'t exist'.format(full_path_decay11))
data_decay11 = np.loadtxt(full_path_decay11, dtype=np.float64, delimiter=';',
                         unpack=True, skiprows=1)
column_dic_decay11 = get_column_dic(full_path_decay11)

kappaMHz = 2
kappatFactor = 1.0/(2*np.pi*kappaMHz*1e6)*1e9

omega_1 = param_dict_decay1['omega1']
omega_r = param_dict_decay1['omegar']
g = param_dict_decay1['g']

eigenvalue0 = 0.5*(omega_1+omega_r)+np.sqrt((0.5*(omega_r-omega_1))**2+g**2)
eigenvalue1 = 0.5*(omega_1+omega_r)-np.sqrt((0.5*(omega_r-omega_1))**2+g**2)
theta = 0.5*np.angle(0.5*(omega_r-omega_1)+1j*g);
print(eigenvalue0)
print(eigenvalue1)
print(eigenvalue1/omega_r)

omega = kappaMHz*(data_decay1[column_dic_decay1['omega_2']]-eigenvalue1)

ax1 = p.subplot(311)
print((g/(omega_r-omega_1))**2)
print(np.sin(theta)**2)
purcell_decay_rate = np.sin(theta)**2*eigenvalue1/omega_r
print(purcell_decay_rate)
p.plot(omega, 1-data_decay2[column_dic_decay2['last_fidelity']], '-', color='C0', label='no JQF')
p.plot(omega, 1-data_decay1[column_dic_decay1['last_fidelity']], '--', color='C1', label='JQF')
p.plot(omega, 1-data_decay3[column_dic_decay3['last_fidelity']], '-.', color='C2', label=r'JQF, $\gamma_{2,{\rm int}}/(2\pi)=3\,{\rm MHz}$')
p.ylim(5e-5, 1e-1)
p.legend(loc='upper right', handlelength=2.2)
p.title(r'(a)')
p.ylabel(r'$1-F$')
p.xlabel(r'$(\omega_{{\rm t},2}-\omega_{1,10})/(2\pi)\,[{\rm MHz}]$')
p.yscale('log')
ax1.yaxis.set_major_locator(ticker.FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))

ax2 = p.subplot(312)
max_last_fidelity_index = data_decay5[column_dic_decay5['last_fidelity']].argmax()
print('k0x for max F = {}'.format(data_decay6[column_dic_decay6['k0x_2']][max_last_fidelity_index]))
p.plot(data_decay6[column_dic_decay6['k0x_2']], 1-data_decay6[column_dic_decay6['last_fidelity']], '-', color='C0', label='no JQF')
p.plot(data_decay5[column_dic_decay5['k0x_2']], 1-data_decay5[column_dic_decay5['last_fidelity']], '--', color='C1', label='JQF')
p.plot(data_decay7[column_dic_decay7['k0x_2']], 1-data_decay7[column_dic_decay7['last_fidelity']], '-.', color='C2', label=r'JQF, $\gamma_{2,{\rm int}}/(2\pi)=3\,{\rm MHz}$')
p.ylim(5e-5, 1e-1)
p.legend(loc='upper right', handlelength=2.2)
p.title(r'(b)')
p.ylabel(r'$1-F$')
p.xlabel(r'$k_{\omega_{1,10}}x_2/\pi$')
p.yscale('log')
ax2.yaxis.set_major_locator(ticker.FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))

ax3 = p.subplot(313)
p.plot(1.0/(2*np.pi*data_decay10[column_dic_decay10['gamma1Internal']]*kappaMHz*1e6)*1e6, 1-data_decay10[column_dic_decay10['last_fidelity']], '-', color='C0', label='no JQF')
p.plot(1.0/(2*np.pi*data_decay9[column_dic_decay9['gamma1Internal']]*kappaMHz*1e6)*1e6, 1-data_decay9[column_dic_decay9['last_fidelity']], '--', color='C1', label='JQF')
p.plot(1.0/(2*np.pi*data_decay11[column_dic_decay11['gamma1Internal']]*kappaMHz*1e6)*1e6, 1-data_decay11[column_dic_decay11['last_fidelity']], '-.', color='C2', label=r'JQF, $\gamma_{2,{\rm int}}/(2\pi)=3\,{\rm MHz}$')
p.ylim(5e-5, 1e-1)
p.legend(loc='lower left', handlelength=2.2)
p.title(r'(c)')
p.ylabel(r'$1-F$')
p.xlabel(r'$T_{1,{\rm q,int}}\,[{\rm \mu s}]$')
p.yscale('log')
ax3.yaxis.set_major_locator(ticker.FixedLocator([1e-4, 1e-3, 1e-2, 1e-1]))

p.tight_layout(pad=0.2)
#p.show()
p.savefig('fig4.eps')

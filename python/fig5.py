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

r1_data_file = 'measure_with_jqf_omegad_kappa_1_kappaI_0_gamma2_50_gamma2I_0_gamma2D_0_g_54.772_Omega_2_omegar_5000_omega1_4000_omega2_3997_kxr_0_kx2_1_phi_0.csv'
r2_data_file = 'measure_with_jqf_omegad_kappa_1_kappaI_0_gamma2_0_gamma2I_0_gamma2D_0_g_54.772_Omega_2_omegar_5000_omega1_4000_omega2_3997_kxr_0_kx2_1_phi_0.csv'

r3_data_file = 'measure_with_jqf_Omega_kappa_1_kappaI_0_gamma2_0_gamma2I_0_gamma2D_0_g_54.772_omegar_5000_omega1_4000_omega2_3997_omegad_5002.5_kxr_0_kx2_1_phi_0.csv'
r4_data_file = 'measure_with_jqf_Omega_kappa_1_kappaI_0_gamma2_50_gamma2I_0_gamma2D_0_g_54.772_omegar_5000_omega1_4000_omega2_3997_omegad_5002.5_kxr_0_kx2_1_phi_0.csv'

r5_data_file = 'measure_with_jqf_g_kappa_1_kappaI_0_gamma2_0_gamma2I_0_gamma2D_0_Omega_0.1_omegar_5000_omega1_4000_omega2_3997_omegad_opt_kxr_0_kx2_1_phi_0.csv'
r6_data_file = 'measure_with_jqf_g_kappa_1_kappaI_0_gamma2_50_gamma2I_0_gamma2D_0_Omega_0.1_omegar_5000_omega1_4000_omega2_3997_omegad_opt_kxr_0_kx2_1_phi_0.csv'
r7_data_file = 'measure_with_jqf_g_kappa_1_kappaI_0_gamma2_0_gamma2I_0_gamma2D_0_Omega_2_omegar_5000_omega1_4000_omega2_3997_omegad_opt_kxr_0_kx2_1_phi_0.csv'
r8_data_file = 'measure_with_jqf_g_kappa_1_kappaI_0_gamma2_50_gamma2I_0_gamma2D_0_Omega_2_omegar_5000_omega1_4000_omega2_3997_omegad_opt_kxr_0_kx2_1_phi_0.csv'

def format_function(x, pos):
    val = x/np.pi
    if val == 0:
        return '0'
    elif val == 1:
        return r'$\pi$'
    elif val == -1:
        return r'$-\pi$'
    return str(int(x/np.pi)) + r'$\pi$'

def find_angle(data,column_dic):
    angle = np.zeros_like(data[column_dic['r_re_0']])
    for n in range(len(data[column_dic['r_re_0']])):
        prod = data[column_dic['r_re_0']][n]*data[column_dic['r_re_1']][n]+data[column_dic['r_im_0']][n]*data[column_dic['r_im_1']][n]
        mag0 = np.sqrt(data[column_dic['r_re_0']][n]**2+data[column_dic['r_im_0']][n]**2)
        mag1 = np.sqrt(data[column_dic['r_re_1']][n]**2+data[column_dic['r_im_1']][n]**2)
        cosTheta = prod/(mag0*mag1)
        angle[n] = np.arccos(cosTheta)
    return angle

def find_chi(data,column_dic,param_dict):
    alpha = -200 # anharmonicity
    return 0.5*data[column_dic['g']]**2/(param_dict['omegar']-param_dict['omega1'])*(1-(param_dict['omegar']-param_dict['omega1']+alpha)/(param_dict['omegar']-param_dict['omega1']-alpha))

fig = p.figure(figsize=(3.3,4.5))

param_dict_r1 = extract_params_from_file_name(r1_data_file)
full_path_r1 = os.path.join(prefix, r1_data_file)
if not os.path.exists(full_path_r1):
    print('Path {} doesn\'t exist'.format(full_path_r1))
data_r1 = np.loadtxt(full_path_r1, dtype=np.float64, delimiter=';',
                    unpack=True, skiprows=1)
column_dic_r1 = get_column_dic(full_path_r1)

param_dict_r2 = extract_params_from_file_name(r2_data_file)
full_path_r2 = os.path.join(prefix, r2_data_file)
if not os.path.exists(full_path_r2):
    print('Path {} doesn\'t exist'.format(full_path_r2))
data_r2 = np.loadtxt(full_path_r2, dtype=np.float64, delimiter=';',
                    unpack=True, skiprows=1)
column_dic_r2 = get_column_dic(full_path_r2)

param_dict_r3 = extract_params_from_file_name(r3_data_file)
full_path_r3 = os.path.join(prefix, r3_data_file)
if not os.path.exists(full_path_r3):
    print('Path {} doesn\'t exist'.format(full_path_r3))
data_r3 = np.loadtxt(full_path_r3, dtype=np.float64, delimiter=';',
                    unpack=True, skiprows=1)
column_dic_r3 = get_column_dic(full_path_r3)

param_dict_r4 = extract_params_from_file_name(r4_data_file)
full_path_r4 = os.path.join(prefix, r4_data_file)
if not os.path.exists(full_path_r4):
    print('Path {} doesn\'t exist'.format(full_path_r4))
data_r4 = np.loadtxt(full_path_r4, dtype=np.float64, delimiter=';',
                    unpack=True, skiprows=1)
column_dic_r4 = get_column_dic(full_path_r4)

param_dict_r5 = extract_params_from_file_name(r5_data_file)
full_path_r5 = os.path.join(prefix, r5_data_file)
if not os.path.exists(full_path_r5):
    print('Path {} doesn\'t exist'.format(full_path_r5))
data_r5 = np.loadtxt(full_path_r5, dtype=np.float64, delimiter=';',
                    unpack=True, skiprows=1)
column_dic_r5 = get_column_dic(full_path_r5)

param_dict_r6 = extract_params_from_file_name(r6_data_file)
full_path_r6 = os.path.join(prefix, r6_data_file)
if not os.path.exists(full_path_r6):
    print('Path {} doesn\'t exist'.format(full_path_r6))
data_r6 = np.loadtxt(full_path_r6, dtype=np.float64, delimiter=';',
                    unpack=True, skiprows=1)
column_dic_r6 = get_column_dic(full_path_r6)

param_dict_r7 = extract_params_from_file_name(r7_data_file)
full_path_r7 = os.path.join(prefix, r7_data_file)
if not os.path.exists(full_path_r7):
    print('Path {} doesn\'t exist'.format(full_path_r7))
data_r7 = np.loadtxt(full_path_r7, dtype=np.float64, delimiter=';',
                    unpack=True, skiprows=1)
column_dic_r7 = get_column_dic(full_path_r7)

param_dict_r8 = extract_params_from_file_name(r8_data_file)
full_path_r8 = os.path.join(prefix, r8_data_file)
if not os.path.exists(full_path_r8):
    print('Path {} doesn\'t exist'.format(full_path_r8))
data_r8 = np.loadtxt(full_path_r8, dtype=np.float64, delimiter=';',
                    unpack=True, skiprows=1)
column_dic_r8 = get_column_dic(full_path_r8)

kappaMHz = 2
kappatFactor = 1.0/(2*np.pi*kappaMHz*1e6)*1e9
omega = kappaMHz*(data_r1[column_dic_r1['omega_d']]-param_dict_r1['omegar'])

population_r1_0 = max(data_r1[column_dic_r1['last_population_2_0']])
population_r1_1 = max(data_r1[column_dic_r1['last_population_2_1']])
print('max last population_r1_0 = {}'.format(population_r1_0))
print('max last population_r1_1 = {}'.format(population_r1_1))
population_r4_0 = max(data_r4[column_dic_r4['last_population_2_0']])
population_r4_1 = max(data_r4[column_dic_r4['last_population_2_1']])
print('max last population_r4_0 = {}'.format(population_r4_0))
print('max last population_r4_1 = {}'.format(population_r4_1))
population_r8_0 = max(data_r8[column_dic_r8['last_population_2_0']])
population_r8_1 = max(data_r8[column_dic_r8['last_population_2_1']])
print('max last population_r8_0 = {}'.format(population_r8_0))
print('max last population_r8_1 = {}'.format(population_r8_1))

ax1 = p.subplot(311)
r_0_1 = data_r1[column_dic_r1['r_re_0']]+1j*data_r1[column_dic_r1['r_im_0']]
r_1_1 = data_r1[column_dic_r1['r_re_1']]+1j*data_r1[column_dic_r1['r_im_1']]
r_0_2 = data_r2[column_dic_r2['r_re_0']]+1j*data_r2[column_dic_r2['r_im_0']]
r_1_2 = data_r2[column_dic_r2['r_re_1']]+1j*data_r2[column_dic_r2['r_im_1']]
ax1.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(np.pi))
piFormatter = ticker.FuncFormatter(format_function)
ax1.yaxis.set_major_formatter(piFormatter)
p.ylim(-np.pi*1.02, np.pi*1.02)
p.plot(omega, np.angle(r_0_2), label=r'$|0\rangle$ (no JQF)')
p.plot(omega, np.angle(r_0_1), '--', label=r'$|0\rangle$ (JQF)')
p.plot(omega, np.angle(r_1_2), '-.', label=r'$|1\rangle$ (no JQF)')
p.plot(omega, np.angle(r_1_1), ':', label=r'$|1\rangle$ (JQF)')
p.legend(loc='upper right', handlelength=2.2)
p.xlabel(r'$(\omega_{\rm d}-\omega_{\rm r})/(2\pi)\,[{\rm MHz}]$')
p.ylabel(r'$\arg(r)$')
p.title(r'${\rm (a)}$')
p.axvline(kappaMHz*2.5, color='k', linestyle=':')

ax2 = p.subplot(312)
r_0_3 = data_r3[column_dic_r3['r_re_0']]+1j*data_r3[column_dic_r3['r_im_0']]
r_1_3 = data_r3[column_dic_r3['r_re_1']]+1j*data_r3[column_dic_r3['r_im_1']]
r_0_4 = data_r4[column_dic_r4['r_re_0']]+1j*data_r4[column_dic_r4['r_im_0']]
r_1_4 = data_r4[column_dic_r4['r_re_1']]+1j*data_r4[column_dic_r4['r_im_1']]
angle_r3 = find_angle(data_r3, column_dic_r3)
angle_r4 = find_angle(data_r4, column_dic_r4)
angle_ratio_r34 = angle_r4/angle_r3
min_angle_ratio_r34 = min(angle_ratio_r34)
max_angle_ratio_r34 = max(angle_ratio_r34)
print("Subfigure (b) angle ratio: min = {}, max = {}".format(min_angle_ratio_r34, max_angle_ratio_r34))
p.ylim(0.968, 1)
p.plot(kappaMHz*data_r3[column_dic_r3['Omega']], angle_r3/np.pi, label=r'no JQF')
p.plot(kappaMHz*data_r4[column_dic_r4['Omega']], angle_r4/np.pi, '--', label=r'JQF')
#Uncomment the below plot to see that it is indeed constant
#p.plot(kappaMHz*data_r4[column_dic_r4['Omega']], angle_ratio_r34, label=r'JQF/no JQF ratio')
p.legend(loc='upper right', handlelength=2.2)
p.xlabel(r'$\Omega_1/(2\pi)\,[{\rm MHz}]$')
p.ylabel(r'angle $\theta_{\rm d}/\pi$')
p.title(r'${\rm (b)}$')

ax2 = p.subplot(313)
r_0_5 = data_r5[column_dic_r5['r_re_0']]+1j*data_r5[column_dic_r5['r_im_0']]
r_1_5 = data_r5[column_dic_r5['r_re_1']]+1j*data_r5[column_dic_r5['r_im_1']]
r_0_6 = data_r6[column_dic_r6['r_re_0']]+1j*data_r6[column_dic_r6['r_im_0']]
r_1_6 = data_r6[column_dic_r6['r_re_1']]+1j*data_r6[column_dic_r6['r_im_1']]
r_0_7 = data_r7[column_dic_r7['r_re_0']]+1j*data_r7[column_dic_r7['r_im_0']]
r_1_7 = data_r7[column_dic_r7['r_re_1']]+1j*data_r7[column_dic_r7['r_im_1']]
r_0_8 = data_r8[column_dic_r8['r_re_0']]+1j*data_r8[column_dic_r8['r_im_0']]
r_1_8 = data_r8[column_dic_r8['r_re_1']]+1j*data_r8[column_dic_r8['r_im_1']]

angle_r5 = find_angle(data_r5, column_dic_r5)
angle_r6 = find_angle(data_r6, column_dic_r6)
angle_r7 = find_angle(data_r7, column_dic_r7)
angle_r8 = find_angle(data_r8, column_dic_r8)
chi_r5 = find_chi(data_r5,column_dic_r5,param_dict_r5)
chi_r6 = find_chi(data_r6,column_dic_r6,param_dict_r6)
chi_r7 = find_chi(data_r7,column_dic_r7,param_dict_r7)
chi_r8 = find_chi(data_r8,column_dic_r8,param_dict_r8)
p.plot(kappaMHz*chi_r5, angle_r5/np.pi, label=r'${}$ MHz (no JQF)'.format(kappaMHz*param_dict_r5['Omega']))
p.plot(kappaMHz*chi_r6, angle_r6/np.pi, '--', label=r'${}$ MHz (JQF)'.format(kappaMHz*param_dict_r6['Omega']))
p.plot(kappaMHz*chi_r7, angle_r7/np.pi, '-.', label=r'${}$ MHz (no JQF)'.format(kappaMHz*param_dict_r7['Omega']))
p.plot(kappaMHz*chi_r8, angle_r8/np.pi, ':', label=r'${}$ MHz (JQF)'.format(kappaMHz*param_dict_r8['Omega']))

#Below is a complicated way to plot a vertical line at chi/(2\pi) = 1MHz
#This is to check that the chosen value of g_r/\kappa = 54.772
#indeed gives chi/(2\pi) = 1MHz
alpha = -200
chi_one = 0.5*54.772**2/(param_dict_r5['omegar']-param_dict_r5['omega1'])*(1-(param_dict_r5['omegar']-param_dict_r5['omega1']+alpha)/(param_dict_r5['omegar']-param_dict_r5['omega1']-alpha))
p.axvline(kappaMHz*chi_one, color='k', linestyle=':')

p.ylim(0.955, 1.002)
p.legend(loc='lower right', handlelength=2.2)
p.xlabel(r'$\chi/(2\pi)\,[{\rm MHz}]$')
p.ylabel(r'angle $\theta_{\rm d}/\pi$')
p.title(r'${\rm (c)}$')

p.tight_layout(pad=0.2)
#p.show()
p.savefig('fig5.eps')

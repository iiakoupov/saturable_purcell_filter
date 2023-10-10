# Copyright (c) 2017-2021 Ivan Iakoupov
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

import numpy as np
import gzip

def convert_to_matplotlib_format(dataX, dataY):
    '''
    If x takes values 1,2,3 and y takes values 4,5
    then input should be of the form:
    dataX=(1,2,3,1,2,3)
    dataY=(4,4,4,5,5,5)

    The function will return a meshgrid of
    dataY_reduced=(1,2,3)
    dataX_reduced=(4,5)
    '''
    for n, i in enumerate(dataX[1:]):
        if dataX[0] == i:
            break
    dataX_reduced = np.array(dataX[:n+1])
    #First sanity check
    dataX_len = len(dataX)
    reduced_len = len(dataX_reduced)
    if dataX_len % reduced_len != 0:
        print('Error: first argument reduced length doesn\'t divide the total'\
              'length')
    divisor = dataX_len // reduced_len
    #Second sanity check
    for i in range(divisor):
        if not np.array_equal(dataX_reduced,
                              dataX[i*reduced_len:(i+1)*reduced_len]):
            print('Error: chunk number {} of the first argument is not equal'\
                  ' to the very first chunk of the argument'.format(i))
            print(dataX_reduced)
            print(dataX[i*reduced_len:(i+1)*reduced_len])
    #Third sanity check - begin checking second argument
    dataY_reduced = []
    for i in range(divisor):
        for j in dataY[i*reduced_len:(i+1)*reduced_len]:
            if j != dataY[i*reduced_len]:
                print('Error: chunk {} of the second argument doesn\'t consist'\
                      ' of entries that have the same value'.format(i))
        dataY_reduced.append(j)
    dataY_reduced = np.array(dataY_reduced)
    return np.meshgrid(dataX_reduced, dataY_reduced)

def extract_params_from_file_name(file_name):
    '''
    This function assumes that the file name
    is of the form "file_name_header_nameA_valA_nameB_valB.txt",
    where "file_name_header" can be any string (even including
    underscores "_"), "nameA" and "nameB" are names of the parameters
    and "valA" and "valB" are their respective values. The
    values are assumed to be either floating point numbers
    or integers.
    The function then builds a dictionary
    { "nameA" : valA, "nameB" : valB }
    and returns it.
    '''
    file_split = file_name.rpartition('.')[0].split('_')
    file_split_len = len(file_split)
    param_dict = {}
    #Sometimes we encounter strings as the values
    #of the parameters (like "optimal" or "random")
    #The mechanism we introduce is the tentative
    #pair (name,value) where we didn't succed to
    #convert "value" to a number. However if one 
    #of the next (name,value) pairs works out fine then
    #we shall also append the "tentative" ones too.
    #NOTE: this makes it impossible to have such
    #      non-number valued parameters as the
    #      very first ones following the header
    #      in the file name of in the file name, as
    #      we iterate from back to front.
    tentative_pair_list = []
    #print(file_split)
    for n in range(1, file_split_len, 2):
        param_val = file_split[-n]
        param_name = file_split[-(n+1)]
        #print('n = {}, param_name = {}, param_val = {}'.format(n,param_name, param_val))
        try:
            param_val_num = float(param_val)
            if float(int(param_val_num)) == param_val_num:
                param_dict[param_name] = int(param_val_num)
            else:
                param_dict[param_name] = param_val_num
            #If tentative_pair_list is non-empty
            if tentative_pair_list:
                for pair in tentative_pair_list:
                    param_dict[pair[0]] = pair[1]
                #Clear the list
                del tentative_pair_list[:]
        except ValueError:
            tentative_pair_list.append((param_name, param_val))
            continue
    return param_dict

def read_column_names(path):
    if path.endswith('.gz'):
        f = gzip.open(path, 'rt')
    else:
        f = open(path, 'r')
    names_string = f.readline()
    if names_string[-1] == '\n':
        names_string = names_string[:-1]
    if names_string[0] == '#':
        names_string = names_string[1:]
    names_string = names_string.strip()
    name_array = names_string.split(';')
    return name_array

def get_column_dic(path):
    column_names = read_column_names(path)
    column_dic = dict(zip(column_names, range(len(column_names))))
    return column_dic


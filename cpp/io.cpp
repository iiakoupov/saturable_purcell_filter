// Copyright (c) 2015-2020 Ivan Iakoupov
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.

#include "io.h"

#include <fcntl.h>
#include <fstream>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>

std::vector<std::vector<double>> loadtxt(const std::string &path, char delimiter, int skiprows)
{
    std::vector<std::vector<double>> data;

    int fd = open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        return data;
    }
    const int64_t length = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    char *str = (char *)mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);

    // Do a first pass to find number of lines and number of
    // data points in each line
    int64_t num_columns = 0;
    int64_t last_position_row = 0;
    int64_t i = 0;
    int64_t row = 0;
    while (i < length) {
        if (str[i] == '\r' || str[i] == '\n') {
            int64_t num_data_values = 0;
            for (int64_t n = last_position_row; n < i; ++n) {
                if (str[n] == delimiter) {
                    ++num_data_values;
                }
            }
            // The last column doesn't have a delimiter at the end
            ++num_data_values;
            if (num_data_values > num_columns) {
                num_columns = num_data_values;
            }
            int64_t j = std::min(i+1, length-1);
            while (j < length && (str[j] == '\r' || str[j] == '\n')) {
                ++j;
            }
            i = j;
            last_position_row = i;
            ++row;
        } else {
            ++i;
        }
    }
    data.reserve(num_columns);
    const int64_t num_rows = row - skiprows;
    for (int64_t k = 0; k < num_columns; ++k) {
        std::vector<double> data_column(num_rows);
        data.push_back(data_column);
    }
    last_position_row = 0;
    i = 0;
    row = 0;
    while (row < skiprows) {
        if (str[i] == '\r' || str[i] == '\n') {
            int64_t j = std::min(i+1, length-1);
            while (j < length && (str[j] == '\r' || str[j] == '\n')) {
                ++j;
            }
            i = j;
            last_position_row = i;
            ++row;
        } else {
            ++i;
        }
    }
    // Reset the row counter since the data vector
    // has not allocated space for the skipped rows
    row = 0;
    while (i < length) {
        if (str[i] == '\r' || str[i] == '\n') {
            int64_t last_position_column = last_position_row;
            int64_t column = 0;
            for (int64_t n = last_position_row; n < i; ++n) {
                if (str[n] == delimiter) {
                    const int64_t column_width = n-last_position_column;
                    const std::string val_string(&str[last_position_column], column_width);
                    const double val = stod(val_string);
                    data[column][row] = val;
                    ++column;
                    last_position_column = n + 1;
                }
            }
            // The last column doesn't have a delimiter at the end
            if (last_position_column < i) {
                const int64_t column_width = i-last_position_column;
                const std::string val_string(&str[last_position_column], column_width);
                const double val = stod(val_string);
                data[column][row] = val;
            }
            int64_t j = std::min(i+1, length-1);
            while (j < length && (str[j] == '\r' || str[j] == '\n')) {
                ++j;
            }
            i = j;
            last_position_row = i;
            ++row;
        } else {
            ++i;
        }
    }
    munmap(str, length);
    //for (int64_t m = 0; m < num_rows; ++m) {
    //    for (int64_t n = 0; n < num_columns; ++n) {
    //        std::cout << data[n][m] << ",";
    //    }
    //    std::cout << std::endl;
    //}
    return data;
}

void savetxt(const std::string &path, const std::vector<std::vector<double>> &data, char delimiter, const std::string &header)
{
    std::ofstream file(path);
    file.precision(17);
    const int columns = data.size();
    if (columns == 0) {
        return;
    }
    const int rows = data[0].size();
    int old_pos = 0;
    int n = 0;
    const int headerSize = header.size();
    while (n < headerSize) {
        if (header[n] == '\r' || header[n] == '\n') {
            const int substr_size = n-old_pos;
            file << "# " << header.substr(old_pos, substr_size) << '\n';
            old_pos = std::min(n+1,headerSize);
            while (old_pos < headerSize && (header[old_pos] == '\r' || header[old_pos] == '\n')) {
                ++old_pos;
            }
            n = old_pos;
        } else {
            ++n;
        }
    }
    if (old_pos < headerSize) {
        file << "# " << header.substr(old_pos, headerSize) << '\n';
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns-1; ++j) {
            file << data[j][i] << delimiter;
        }
        file << data[columns-1][i] << '\n';
    }
}

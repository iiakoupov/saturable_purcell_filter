This is the code for the publication:

Ivan Iakoupov and Kazuki Koshino
"Saturable Purcell filter for circuit quantum electrodynamics"  
[Phys. Rev. Research 5, 013148 (2023)](https://link.aps.org/doi/10.1103/PhysRevResearch.5.013148)  
Preprint: [arXiv:2202.07229](https://arxiv.org/abs/2202.07229)

There are two subdirectories:
* "cpp"
* "python"

Subdirectory "cpp" contains C++ code for the time-consuming calculations that output data files, and the subdirectory "python" contains Python scripts that use these data files to produce figures used in the above paper.

To build the C++ code, run:

```sh
cd cpp
mkdir build
cd build
cmake ..
make
```

The following executables are produced:
* decay_with_jqf_plots (data for Figs. 3, 4, 9)
* measure_with_jqf_plots (data for Fig. 5)
* control_with_jqf_plots (data for Figs. 6, 7)

The executables just need to be run to produce the data files for their respective figures. The data files can then be used by the scripts in the "python" directory to produce the figures.

The optimal control optimization in "control_with_jqf_plots" is disabled, because it takes too long to run. To enable it, uncomment the last piece of code in the function "control_with_jqf_article_plots()" of the file "control_with_jqf_plots.cpp".

The C++ executables "measure_with_jqf_plots" and "control_with_jqf_plot" could be sped up significantly if one uses the GPU acceleration with AMD ROCm.

# ROCm

To compile with ROCm, the rocsparse library is needed. The code was tested to compile and run on ROCm 5.6.0 on Arch Linux. To compile, use

```sh
CXX='/opt/rocm/bin/hipcc --rocm-path=/opt/rocm' cmake -DUSE_ROCM=ON ..
```
instead of

```sh
cmake ..
```
in the commands above.

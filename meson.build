project('FFT3DFilter',
        'cpp',
        meson_version : '>=0.38.0',
        default_options : ['cpp_std=c++17', 'buildtype=release'],
        license : 'GPL2',
        version : '1')


# Dependencies
vapoursynth = dependency('vapoursynth')
fftw = dependency('fftw3f_threads')


# Sources
src = [
    'info.h',
    'FFT3DFilter.h',
    'FFT3DFilter.cpp',
    'fft3dfilter_c.cpp',
    'Plugin.cpp']


# Libs
library(
    'fft3dfilter',
    src,
    dependencies : [vapoursynth, fftw],
    install_dir : join_paths(get_option('prefix'), get_option('libdir'), 'vapoursynth'),
    install : true
)

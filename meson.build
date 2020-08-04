project('FFT3DFilter', 'cpp',
  default_options : ['buildtype=release', 'b_ndebug=if-release', 'cpp_std=c++1z'],
  license : 'GPL2',
  meson_version : '>=0.38.0',
  version : '2'
)

cc = meson.get_compiler('cpp')

sources = [
  'FFT3DFilter.cpp',
  'FFT3DFilter.h',
  'fft3dfilter_c.cpp',
  'FFT3DFilterTransform.cpp',
  'Plugin.cpp'
]

vapoursynth_dep = dependency('vapoursynth')

fftw3f_dep = dependency('fftw3f')

deps = [vapoursynth_dep, fftw3f_dep]

code = '''
#include <fftw3.h>
int main() {
    fftwf_init_threads();
    return 0;
}
'''
if not cc.links(code, dependencies : fftw3f_dep)
  deps += cc.find_library('fftw3f_threads')
endif

if cc.has_argument('-fvisibility=hidden')
  add_project_arguments('-fvisibility=hidden', language : 'cpp')
endif

if host_machine.cpu_family().startswith('x86')
  add_project_arguments('-mfpmath=sse', '-msse2', language : 'cpp')
endif

shared_module('fft3dfilter', sources,
  dependencies : deps,
  install : true,
  install_dir : join_paths(vapoursynth_dep.get_pkgconfig_variable('libdir'), 'vapoursynth')
)

# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow_lite_micro_cmake_wrapper-src"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow_lite_micro_cmake_wrapper-build"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow_lite_micro_cmake_wrapper-subbuild/tensorflow_lite_micro_cmake_wrapper-populate-prefix"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow_lite_micro_cmake_wrapper-subbuild/tensorflow_lite_micro_cmake_wrapper-populate-prefix/tmp"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow_lite_micro_cmake_wrapper-subbuild/tensorflow_lite_micro_cmake_wrapper-populate-prefix/src/tensorflow_lite_micro_cmake_wrapper-populate-stamp"
  "/home/linzejia/app/mlek_lee/b_yv5s"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow_lite_micro_cmake_wrapper-subbuild/tensorflow_lite_micro_cmake_wrapper-populate-prefix/src/tensorflow_lite_micro_cmake_wrapper-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow_lite_micro_cmake_wrapper-subbuild/tensorflow_lite_micro_cmake_wrapper-populate-prefix/src/tensorflow_lite_micro_cmake_wrapper-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow_lite_micro_cmake_wrapper-subbuild/tensorflow_lite_micro_cmake_wrapper-populate-prefix/src/tensorflow_lite_micro_cmake_wrapper-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

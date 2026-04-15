# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-ruy-src"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-ruy-build"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/tmp"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src/tensorflow-ruy-populate-stamp"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src/tensorflow-ruy-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src/tensorflow-ruy-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-ruy-subbuild/tensorflow-ruy-populate-prefix/src/tensorflow-ruy-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

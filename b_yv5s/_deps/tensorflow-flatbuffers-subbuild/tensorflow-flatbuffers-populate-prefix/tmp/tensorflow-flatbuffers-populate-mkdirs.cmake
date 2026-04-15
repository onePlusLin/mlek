# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.5)

file(MAKE_DIRECTORY
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-flatbuffers-src"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-flatbuffers-build"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/tmp"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src/tensorflow-flatbuffers-populate-stamp"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src"
  "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src/tensorflow-flatbuffers-populate-stamp"
)

set(configSubDirs )
foreach(subDir IN LISTS configSubDirs)
    file(MAKE_DIRECTORY "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src/tensorflow-flatbuffers-populate-stamp/${subDir}")
endforeach()
if(cfgdir)
  file(MAKE_DIRECTORY "/home/linzejia/app/mlek_lee/b_yv5s/_deps/tensorflow-flatbuffers-subbuild/tensorflow-flatbuffers-populate-prefix/src/tensorflow-flatbuffers-populate-stamp${cfgdir}") # cfgdir has leading slash
endif()

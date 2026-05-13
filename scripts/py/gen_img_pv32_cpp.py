#  SPDX-FileCopyrightText:  Copyright 2021-2024 Arm Limited and/or
#  its affiliates <open-source-office@arm.com>
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""
Utility script to convert a set of RGB images in a given location into
corresponding cpp files and a single hpp file referencing the vectors
from the cpp files.
"""
import glob
import math
import os
import typing
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError
from jinja2 import Environment, FileSystemLoader

from gen_utils import GenUtils

# pylint: disable=duplicate-code
parser = ArgumentParser()

parser.add_argument(
    "--image_path",
    type=str,
    help="path to images folder or image file to convert."
)

parser.add_argument(
    "--package_gen_dir",
    type=str,
    help="path to directory to be generated."
)

parser.add_argument(
    "--image_size",
    type=int,
    nargs=2,
    help="Height and Width of the converted images (e.g. --image_size 128 320)."
)

parser.add_argument(
    "--channels_displayed",
    type=int,
    nargs=1,
    help="Number of channels (1=grayscale, 3=RGB)."
)

parser.add_argument(
    "--license_template",
    type=str,
    help="Header template file",
    default="header_template.txt"
)

parsed_args = parser.parse_args()

env = Environment(loader=FileSystemLoader(Path(__file__).parent / 'templates'),
                  trim_blocks=True,
                  lstrip_blocks=True)


# pylint: enable=duplicate-code
@dataclass
class ImagesParams:
    """
    Template params for Images.hpp and Images.cc
    """
    num_images: int
    image_size: typing.Sequence
    image_array_names: typing.List[str]
    image_filenames: typing.List[str]


def write_hpp_file(
        images_params: ImagesParams,
        header_file_path: Path,
        cc_file_path: Path,
        header_template_file: str,
        channels: int = 3,
):
    """
    Write Images.hpp and Images.cc

    @param images_params:           Template params
    @param header_file_path:        Images.hpp path
    @param cc_file_path:            Images.cc path
    @param header_template_file:    Header template file name
    @param channels:                Number of channels (1 or 3)
    """
    print(f"++ Generating {header_file_path}")
    hdr = GenUtils.gen_header(env, header_template_file)

    img_w, img_h = images_params.image_size
    image_size = str(img_w * img_h * channels)

    env \
        .get_template('sample-data/images/images.h.template') \
        .stream(common_template_header=hdr,
                imgs_count=images_params.num_images,
                img_size=image_size,
                var_names=images_params.image_array_names,
                img_width=img_w,
                img_height=img_h) \
        .dump(str(header_file_path))

    env \
        .get_template('sample-data/images/images.c.template') \
        .stream(common_template_header=hdr,
                var_names=images_params.image_array_names,
                img_names=images_params.image_filenames,
                header_filename=os.path.basename(header_file_path)) \
        .dump(str(cc_file_path))

def resize_crop_image(
        original_image: Image.Image,
        image_width: int,
        image_height: int,
        channels_displayed: int,
) -> np.ndarray:
    """
    Preprocess image to match model input (aligns with map_tflite.py pipeline).

    For PV32-LightOA:
      1. BGR→GRAY (via cv2) → here: RGB→L (PIL, equivalent result)
      2. Direct resize to target size (no aspect-ratio, no center crop)
      3. uint8 flat array

    @param original_image:      PIL Image in RGB mode
    @param image_width:         Model input width (e.g. 320)
    @param image_height:        Model input height (e.g. 128)
    @param channels_displayed:  1 for grayscale, 3 for RGB
    @return:                    Flat uint8 numpy array
    """
    if channels_displayed == 1:
        # PIL 'L' mode: L = 0.299*R + 0.587*G + 0.114*B
        # Equivalent to cv2.COLOR_BGR2GRAY applied on BGR input
        img = original_image.convert('L')
        print("Converting to grayscale")
    else:
        img = original_image  # keep RGB
        print("Converting to RGB")

    # Direct resize, no aspect-ratio preservation, no center crop
    img_resized = img.resize((image_width, image_height), Image.Resampling.BILINEAR)
    print(f"Resizing to image_width X image_height: {image_width}x{image_height}")

    return np.array(img_resized, dtype=np.uint8).flatten()


def write_individual_img_cc_file(
        rgb_data: np.ndarray,
        image_filename: str,
        cc_filename: Path,
        header_template_file: str,
        array_name: str
):
    """
    Write image.cc

    @param rgb_data:                Image data
    @param image_filename:          Image file name
    @param cc_filename:             image.cc path
    @param header_template_file:    Header template file name
    @param array_name:              C++ array name
    """
    print(f"++ Converting {image_filename} to {cc_filename.name}")

    hdr = GenUtils.gen_header(env, header_template_file, image_filename)

    hex_line_generator = (', '.join(map(hex, sub_arr))
                          for sub_arr in np.array_split(rgb_data, math.ceil(len(rgb_data) / 20)))
    env \
        .get_template('sample-data/images/image.c.template') \
        .stream(common_template_header=hdr,
                var_name=array_name,
                img_data=hex_line_generator) \
        .dump(str(cc_filename))


def main(args):
    """
    Convert images
    @param args:    Parsed args
    """
    # Keep the count of the images converted
    image_idx = 0
    image_filenames = []
    image_array_names = []

    if Path(args.image_path).is_dir():
        filepaths = sorted(glob.glob(str(Path(args.image_path) / '**/*.*'), recursive=True))
    elif Path(args.image_path).is_file():
        filepaths = [args.image_path]
    else:
        raise OSError("Directory or file does not exist.")

    img_w = args.image_size[1]   # width
    img_h = args.image_size[0]   # height
    channels = args.channels_displayed[0]

    for filepath in filepaths:
        filename = Path(filepath).name

        try:
            original_image = Image.open(filepath).convert("RGB")
        except UnidentifiedImageError:
            print(f"-- Skipping file {filepath} due to unsupported image format.")
            continue

        image_filenames.append(filename)

        # Save the C file
        os.makedirs(args.package_gen_dir, exist_ok=True)

        cc_filename = (Path(args.package_gen_dir) /
                       (Path(filename).stem.replace(" ", "_") + ".c"))
        array_name = "im" + str(image_idx)
        image_array_names.append(array_name)

        rgb_data = resize_crop_image(original_image,
                                     img_w,
                                     img_h,
                                     channels)
        
        write_individual_img_cc_file(rgb_data,
                                     filename,
                                     cc_filename,
                                     args.license_template,
                                     array_name)

        # Increment image index
        image_idx = image_idx + 1

    header_filepath = Path(args.package_gen_dir) / "sample_files.h"
    common_cc_filepath = Path(args.package_gen_dir) / "sample_files.c"

    images_params = ImagesParams(image_idx, (img_w, img_h), image_array_names, image_filenames)

    if len(image_filenames) > 0:
        write_hpp_file(images_params, header_filepath, common_cc_filepath, args.license_template, channels)
    else:
        raise FileNotFoundError("No valid images found.")


if __name__ == '__main__':
    main(parsed_args)

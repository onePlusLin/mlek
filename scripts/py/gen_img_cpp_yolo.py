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
from PIL import Image, UnidentifiedImageError, ImageOps
from jinja2 import Environment, FileSystemLoader
# import cv2

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
    help="Size (width and height) of the converted images."
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
    image_ratio:typing.List[float]
    imgae_dw:typing.List[float]
    imgae_dh:typing.List[float]


def write_hpp_file(
        images_params: ImagesParams,
        header_file_path: Path,
        cc_file_path: Path,
        header_template_file: str,
):
    """
    Write Images.hpp and Images.cc

    @param images_params:           Template params
    @param header_file_path:        Images.hpp path
    @param cc_file_path:            Images.cc path
    @param header_template_file:    Header template file name
    """
    print(f"++ Generating {header_file_path}")
    hdr = GenUtils.gen_header(env, header_template_file)

    img_w, img_h = images_params.image_size
    image_size = str(img_w * img_h * 3)

    env \
        .get_template('sample-data/yolo_images/images.h.template') \
        .stream(common_template_header=hdr,
                imgs_count=images_params.num_images,
                img_size=image_size,
                var_names=images_params.image_array_names,
                img_width=img_w,
                img_height=img_h,
                image_ratios=images_params.image_ratio,
                image_dws=images_params.imgae_dw,
                image_dhs=images_params.imgae_dh) \
        .dump(str(header_file_path))

    env \
        .get_template('sample-data/yolo_images/images.c.template') \
        .stream(common_template_header=hdr,
                var_names=images_params.image_array_names,
                img_names=images_params.image_filenames,
                header_filename=os.path.basename(header_file_path),
                image_ratios=images_params.image_ratio,
                image_dws=images_params.imgae_dw,
                image_dhs=images_params.imgae_dh) \
        .dump(str(cc_file_path))

def resize_pad_rgb_image(
        original_image: Image.Image,
        image_size: typing.Sequence,
) -> np.ndarray:
    """
    Resize and pad input image
    use for yolo image preprocessing

    @param original_image:  Image to resize and pad --RGB
    @param image_size:      New image size          --(H, W)
    @return:                Resized and padded image
    """
    
    
    img_array = np.array(original_image) # NumPy Array: img.shape 返回 (H, W, C) （高, 宽, 通道数）
    shape = img_array.shape[:2]
    r = min(image_size[0] / shape[0], image_size[1] / shape[1])
    new_unpad = round(shape[1] * r), round(shape[0] * r)    # 方便cv2.resize的宽高顺序是W, H
    dw, dh = (image_size[1] - new_unpad[0]) / 2, (image_size[0] - new_unpad[1]) / 2  # wh pad (float类型）

    img = original_image 
    if shape[::-1] != new_unpad: 
        img = original_image.resize(new_unpad, Image.Resampling.BILINEAR)
        # img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR) # resize的时候是宽高顺序
        
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = ImageOps.expand(img, border=(int(left), int(top), int(right), int(bottom)), fill=(114, 114, 114))
    # img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    # 图片流的数据只能是uint8 后续在图片预处理进行处理吧
    # 这里进行resize和pad
    img = np.ascontiguousarray(img)
    img_flat = np.array( img , dtype = np.uint8).flatten()
    print(f"++ new image shape: {img_flat.shape},ratio: {r},pading: dw-- {dw}, dh-- {dh}")
    
    return img_flat,r,dw,dh
    
# 生成C++数组:
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
        .get_template('sample-data/yolo_images/image.c.template') \
        .stream(common_template_header=hdr,
                var_name=array_name,
                img_data=hex_line_generator) \
        .dump(str(cc_filename))


def main(args):
    """
    Convert images
    @param args:    Parsed args
    """
    print(f"###########################################################################")
    print(f"####################### straring processing images ########################")
    # Keep the count of the images converted
    image_idx = 0
    image_filenames = []
    image_array_names = []
    image_ratio = []
    imgae_dw = []
    imgae_dh = []


    if Path(args.image_path).is_dir():
        filepaths = sorted(glob.glob(str(Path(args.image_path) / '**/*.*'), recursive=True))
    elif Path(args.image_path).is_file():
        filepaths = [args.image_path]
    else:
        raise OSError("Directory or file does not exist.")

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
        
        print(f"++ processing original image: {filename} with shape {original_image.size}") # PIL Image: img.size 返回 (W, H) （宽, 高）
        rgb_data,r,dw,dh = resize_pad_rgb_image(original_image, args.image_size)
        image_ratio.append(r)
        imgae_dw.append(dw)
        imgae_dh.append(dh)
        
        write_individual_img_cc_file(rgb_data,
                                     filename,
                                     cc_filename,
                                     args.license_template,
                                     array_name)

        # Increment image index
        image_idx = image_idx + 1

    header_filepath = Path(args.package_gen_dir) / "sample_files.h"
    common_cc_filepath = Path(args.package_gen_dir) / "sample_files.c"

    images_params = ImagesParams(image_idx, args.image_size, image_array_names, image_filenames, image_ratio, imgae_dw, imgae_dh)

    if len(image_filenames) > 0:
        write_hpp_file(images_params, header_filepath, common_cc_filepath, args.license_template)
    else:
        raise FileNotFoundError("No valid images found.")
    print(f"############################ done processing images ########################")


if __name__ == '__main__':
    main(parsed_args)
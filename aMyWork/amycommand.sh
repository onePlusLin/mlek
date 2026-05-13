vela /home/linzejia/app/ethos-u-vela/azejiaTest/yolov5s/yolov5s-int8.tflite \
--output-dir ./output/our \
--accelerator-config ethos-u65-512 \
--optimise Performance \
--system-config Ethos_U65_High_Our \
--memory-mode Dedicated_Sram_2MB \
--config Arm/vela.ini \
--arena-cache-size=2097152 \
--verbose-all 2>&1 | tee yolov5s_ds_2M.log

# python :
--debug-force-legacy-core

cmake ../ \
 -DUSE_CASE_BUILD=yolov8s \
 -DETHOSU_TARGET_NPU_CONFIG=ethos-u65-512 \
 -DETHOS_U_NPU_CONFIG_ID=Y512 \
 -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
 -Dyolov8s_MODEL_TFLITE_PATH=/home/linzejia/app/ethos-u-vela/work/yolo_66/v8s/output/yolov8s_full_integer_quant_vela.tflite \
 -DETHOS_U_NPU_ID=U65 \
 -DETHOS_U_NPU_CACHE_SIZE=2097152

# cpu profile:
add_definitions(-DCPU_PROFILE_ENABLED)

# semihosting:
-DSEMIHOSTING_ENABLED=ON

# 验证测试输出：
-DVERIFY_TEST_OUTPUT=1 


# 设置单个输入，防止ddr溢出？
 -DUSE_SINGLE_INPUT=ON

make -j$(nproc)

FVP_Corstone_SSE-300_Ethos-U65 \
 -C cpu0.CFGDTCMSZ=15 \
 -C cpu0.CFGITCMSZ=15 \
 -C mps3_board.visualisation.disable-visualisation=1 \
 -C mps3_board.uart0.out_file="-" \
 -C mps3_board.uart0.shutdown_tag="EXITTHESIM" \
 -C ethosu.num_macs=512 \
 -C mps3_board.FPGA_SRAM_SIZE=2 \
 -C ethosu.extra_args="--fast" \
 -a ethos-u-yolov8s.axf
 
# fast 
 -C ethosu.extra_args="--fast"

# dump:
 -C cpu0.semihosting-enable=1

# model
# v8s:
/home/linzejia/app/ethos-u-vela/work/yolo_66/v8s/output/new/yolov8s_full_integer_quant_vela.tflite
/home/linzejia/app/ethos-u-vela/work/yolo_66/v8s/output/our/yolov8s_full_integer_quant_vela.tflite
# v8s_46:
/home/linzejia/app/ethos-u-vela/work/yolo46/v8s/output/test/yolov8s_full_integer_quant_vela.tflite


cmake ../ \
 -DUSE_CASE_BUILD=yolov8snpp \
 -DETHOSU_TARGET_NPU_CONFIG=ethos-u65-512 \
 -DETHOS_U_NPU_CONFIG_ID=Y512 \
 -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
 -Dyolov8snpp_MODEL_TFLITE_PATH=/home/linzejia/app/model/vela_model/output/v8s66_npp/yolov8s_modified_full_integer_quant_vela.tflite \
 -DETHOS_U_NPU_ID=U65 \
 -DETHOS_U_NPU_CACHE_SIZE=2097152 \
 -DBUILD_FVP_TESTS=ON \
 -DUSE_SINGLE_INPUT=ON

cmake ../ \
 -DUSE_CASE_BUILD=yolov8ssegnpp \
 -DETHOSU_TARGET_NPU_CONFIG=ethos-u65-512 \
 -DETHOS_U_NPU_CONFIG_ID=Y512 \
 -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
 -Dyolov8ssegnpp_MODEL_TFLITE_PATH=/home/linzejia/app/u01/u01/model/tfliteModel/v8sseg/seg66npp/yolov8s-seg_npp_66_full_integer_quant_vela.tflite \
 -DETHOS_U_NPU_ID=U65 \
 -DETHOS_U_NPU_CACHE_SIZE=2097152 \
 
 -DBUILD_FVP_TESTS=ON \
 -DUSE_SINGLE_INPUT=ON

 

cmake ../ \
 -DUSE_CASE_BUILD=pv32 \
 -DETHOSU_TARGET_NPU_CONFIG=ethos-u65-512 \
 -DETHOS_U_NPU_CONFIG_ID=Y512 \
 -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
 -Dpv32_MODEL_TFLITE_PATH=/home/linzejia/app/mlek_lee/resources_downloaded/pv32/pv32-lightOA-v6.6.9_full_integer_quant_vela.tflite \
 -DETHOS_U_NPU_ID=U65 \
 -DETHOS_U_NPU_CACHE_SIZE=2097152

cmake ../ \
 -DUSE_CASE_BUILD=pv32 \
 -Dpv32_IMAGE_channel=3 \
 -Dpv32_IMAGE_WIDTH=320 \
 -Dpv32_IMAGE_HEIGHT=320 \
 -DETHOSU_TARGET_NPU_CONFIG=ethos-u65-512 \
 -DETHOS_U_NPU_CONFIG_ID=Y512 \
 -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
 -Dpv32_MODEL_TFLITE_PATH=/home/linzejia/app/u01/u01/other/pv32_fur/pv32-furniture-v1.15.35_full_integer_quant_vela.tflite \
 -DETHOS_U_NPU_ID=U65 \
 -DETHOS_U_NPU_CACHE_SIZE=2097152

cmake ../ \
 -DUSE_CASE_BUILD=pv32 \
 -Dpv32_IMAGE_channel=3 \
 -Dpv32_IMAGE_WIDTH=320 \
 -Dpv32_IMAGE_HEIGHT=320 \
 -DETHOSU_TARGET_NPU_CONFIG=ethos-u65-512 \
 -DETHOS_U_NPU_CONFIG_ID=Y512 \
 -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
 -Dpv32_MODEL_TFLITE_PATH=/home/linzejia/app/u01/u01/other/tflite_fu/pv32-furniture_tflite/output/pv32-furniture-v1.15.35_full_integer_quant_vela.tflite \
 -DETHOS_U_NPU_ID=U65 \
 -DETHOS_U_NPU_CACHE_SIZE=2097152
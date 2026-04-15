vela /home/linzejia/app/ethos-u-vela/azejiaTest/yolov5s/yolov5s-int8.tflite \
--output-dir ./output \
--accelerator-config ethos-u65-512 \
--optimise Performance \
--system-config Ethos_U65_High_End \
--memory-mode Dedicated_Sram_512KB \
--config Arm/vela.ini \
--arena-cache-size=2097152 \
--verbose-all 2>&1 | tee yolov5s_ds_2M.log

# python :
--debug-force-legacy-core

cmake ../ \
 -DUSE_CASE_BUILD=yolov5s \
 -DETHOSU_TARGET_NPU_CONFIG=ethos-u65-512 \
 -DETHOS_U_NPU_CONFIG_ID=Y512 \
 -DETHOS_U_NPU_MEMORY_MODE=Dedicated_Sram \
 -Dyolov5s_MODEL_TFLITE_PATH=/home/linzejia/app/ethos-u-vela/work/v5s_silu/output/v5s_our/yolov5s-int8_vela.tflite \
 -DETHOS_U_NPU_ID=U65 \
 -DETHOS_U_NPU_CACHE_SIZE=2097152

# cpu profile:
add_definitions(-DCPU_PROFILE_ENABLED)

# semihosting:
-DSEMIHOSTING_ENABLED=ON

make -j$(nproc)

FVP_Corstone_SSE-300_Ethos-U65 \
 -C cpu0.CFGDTCMSZ=15 \
 -C cpu0.CFGITCMSZ=15 \
 -C mps3_board.visualisation.disable-visualisation=1 \
 -C mps3_board.uart0.out_file="-" \
 -C mps3_board.uart0.shutdown_tag="EXITTHESIM" \
 -C ethosu.num_macs=512 \
 -C mps3_board.FPGA_SRAM_SIZE=2 \
 -a ethos-u-yolov5s.axf
 
# fast 
 -C ethosu.extra_args="--fast"

# dump:
 -C cpu0.semihosting-enable=1
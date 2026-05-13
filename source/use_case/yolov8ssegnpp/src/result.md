count: 15
62 5.0840 166.7535 155.0192 263.0374 0.9336 
56 292.6485 218.1440 351.6886 320.1483 0.8984 
39 549.9644 297.1945 587.7144 400.6746 0.8789 
56 361.5070 218.5002 417.5781 316.0073 0.8594 
0 413.1063 156.8765 465.4615 297.9151 0.8008 
60 317.2915 221.4856 445.4751 319.2342 0.6914 
72 443.3072 168.8514 512.8552 288.4368 0.6875 
56 409.4438 219.8020 442.8842 306.6635 0.6875 
62 558.3724 208.2909 639.6097 287.8507 0.6445 
58 230.4106 177.2009 266.0999 213.0419 0.5391 
75 348.3314 205.6233 362.4703 230.2273 0.3828 
72 490.2218 169.2749 513.2524 285.2953 0.3555 
75 241.9948 196.9908 253.1111 212.6329 0.3125 
74 447.9500 120.6343 461.7136 141.9983 0.3125 
58 333.1939 174.9988 369.8719 220.5909 0.2812 
INFO - post-processing done.
INFO - Final results:
INFO - Total number of inferences: 1
INFO - Using sample image: 000000000285.jpg
INFO - capturedFrameSize 1228800.
INFO - inputTensor->bytes 1228800.
INFO - Extracted image ID: 285 from filename: 000000000285.jpg
INFO - get img_id 285 ratio 1.000000 padding 27.000000 x 0.000000.
ERROR - Invalid image size for given location!
INFO - use input shape 1228800 
INFO - DoPreProcess start in yolov8ssegnpp.
INFO - Input tensor quantization: scale=0.003921568859, zero_point=-128
INFO - pre-processing done.
INFO - 
D: ETHOSU_PMU_Set_EVTYPER(): num=0, type=5, val=35
D: ETHOSU_PMU_Set_EVTYPER(): num=1, type=40, val=130
D: ETHOSU_PMU_Set_EVTYPER(): num=2, type=45, val=135
D: ETHOSU_PMU_Set_EVTYPER(): num=3, type=62, val=386
D: ETHOSU_PMU_Set_CNTR_OVS(): 
D: ETHOSU_PMU_Enable(): Enable PMU
D: ETHOSU_PMU_CNTR_Disable(): mask=0x8000000f
D: ETHOSU_PMU_CYCCNT_Reset(): Reset PMU cycle counter
D: ETHOSU_PMU_EVCNTR_ALL_Reset(): Reset all events
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=0, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=1, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=2, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=3, val=0
D: ETHOSU_PMU_Get_CCNTR(): val=0
I: Acquiring NPU driver handle
D: ethosu_reserve_driver(): NPU driver handle 0x20000068 reserved
D: ethosu_invoke_async(): OPTIMIZER_CONFIG
I: Optimizer release nbr: 0 patch: 1
I: Optimizer config. product=1, cmd_stream_version=0, macs_per_cc=9, shram_size=96, custom_dma=0
I: Optimizer config. arch version: 1.0.6
I: Ethos-U config. product=1, cmd_stream_version=0, macs_per_cc=9, shram_size=96, custom_dma=0
I: Ethos-U. arch version=1.1.0
D: ethosu_invoke_async(): NOP
D: ethosu_invoke_async(): NOP
D: ethosu_invoke_async(): NOP
D: ethosu_invoke_async(): COMMAND_STREAM
I: handle_command_stream: cmd_stream=0x70e73510, cms_length 6014
D: ETHOSU_PMU_CNTR_Disable(): mask=0x8000000f
D: ETHOSU_PMU_CNTR_Enable(): mask=0x8000000f
D: ethosu_dev_run_command_stream(): QBASE=0x0000000070e73510, QSIZE=24056, cmd_stream_ptr=0x70e73510
D: ethosu_dev_run_command_stream(): BASEP0=0x00000000704b0ac0
D: ethosu_dev_run_command_stream(): BASEP1=0x0000000070e797a0
D: ethosu_dev_run_command_stream(): BASEP2=0x0000000031000000
D: ethosu_dev_run_command_stream(): BASEP3=0x00000000710097a0
D: ethosu_dev_run_command_stream(): BASEP4=0x0000000070f417a0
D: ethosu_dev_run_command_stream(): BASEP5=0x0000000070ff6ba0
D: ethosu_dev_run_command_stream(): BASEP6=0x0000000070ffe8a0
D: ethosu_dev_run_command_stream(): BASEP7=0x0000000071001aa0
D: ethosu_dev_run_command_stream(): CMD=0x00000001
D: ETHOSU_PMU_CNTR_Disable(): mask=0x8000000f
D: ethosu_wait(): Inference finished successfully...
D: ethosu_release_driver(): NPU driver handle 0x20000068 released
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=0, val=207665
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=1, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=2, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=3, val=0
D: ETHOSU_PMU_Get_CCNTR(): val=208086
INFO - 
INFO - Profile for Inference:
INFO - NPU ACTIVE: 207665 cycles
INFO - NPU AXI0_RD_DATA_BEAT_RECEIVED: 0 beats
INFO - NPU AXI0_WR_DATA_BEAT_WRITTEN: 0 beats
INFO - NPU AXI1_RD_DATA_BEAT_RECEIVED: 0 beats
INFO - NPU IDLE: 421 cycles
INFO - NPU TOTAL: 208086 cycles
INFO - 
INFO - inference done.
INFO - DoPostProcess start in yolov8ssegnpp.
INFO - Output tensor 0: role=1, channels=64, grid=20x20, stride=32
INFO - Output tensor 1: role=3, channels=1, grid=80x80, stride=8
INFO - Output tensor 2: role=1, channels=64, grid=80x80, stride=8
INFO - Output tensor 3: role=4, channels=32, grid=40x40, stride=16
INFO - Output tensor 4: role=3, channels=1, grid=20x20, stride=32
INFO - Output tensor 5: role=1, channels=64, grid=40x40, stride=16
INFO - Output tensor 6: role=4, channels=32, grid=80x80, stride=8
INFO - Proto tensor 7: channels=32, grid=160x160 (int8, on-the-fly dequant)
INFO - Output tensor 8: role=2, channels=80, grid=80x80, stride=8
INFO - Output tensor 9: role=3, channels=1, grid=40x40, stride=16
INFO - Output tensor 10: role=4, channels=32, grid=20x20, stride=32
INFO - Output tensor 11: role=2, channels=80, grid=20x20, stride=32
INFO - Output tensor 12: role=2, channels=80, grid=40x40, stride=16
count: 1
21 5.9166 73.7708 583.1050 640.0000 0.9609 
INFO - post-processing done.
INFO - Final results:
INFO - Total number of inferences: 1
INFO - Using sample image: 000000000632.jpg
INFO - capturedFrameSize 1228800.
INFO - inputTensor->bytes 1228800.
INFO - Extracted image ID: 632 from filename: 000000000632.jpg
INFO - get img_id 632 ratio 1.000000 padding 0.000000 x 78.500000.
ERROR - Invalid image size for given location!
INFO - use input shape 1228800 
INFO - DoPreProcess start in yolov8ssegnpp.
INFO - Input tensor quantization: scale=0.003921568859, zero_point=-128
INFO - pre-processing done.
INFO - 
D: ETHOSU_PMU_Set_EVTYPER(): num=0, type=5, val=35
D: ETHOSU_PMU_Set_EVTYPER(): num=1, type=40, val=130
D: ETHOSU_PMU_Set_EVTYPER(): num=2, type=45, val=135
D: ETHOSU_PMU_Set_EVTYPER(): num=3, type=62, val=386
D: ETHOSU_PMU_Set_CNTR_OVS(): 
D: ETHOSU_PMU_Enable(): Enable PMU
D: ETHOSU_PMU_CNTR_Disable(): mask=0x8000000f
D: ETHOSU_PMU_CYCCNT_Reset(): Reset PMU cycle counter
D: ETHOSU_PMU_EVCNTR_ALL_Reset(): Reset all events
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=0, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=1, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=2, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=3, val=0
D: ETHOSU_PMU_Get_CCNTR(): val=0
I: Acquiring NPU driver handle
D: ethosu_reserve_driver(): NPU driver handle 0x20000068 reserved
D: ethosu_invoke_async(): OPTIMIZER_CONFIG
I: Optimizer release nbr: 0 patch: 1
I: Optimizer config. product=1, cmd_stream_version=0, macs_per_cc=9, shram_size=96, custom_dma=0
I: Optimizer config. arch version: 1.0.6
I: Ethos-U config. product=1, cmd_stream_version=0, macs_per_cc=9, shram_size=96, custom_dma=0
I: Ethos-U. arch version=1.1.0
D: ethosu_invoke_async(): NOP
D: ethosu_invoke_async(): NOP
D: ethosu_invoke_async(): NOP
D: ethosu_invoke_async(): COMMAND_STREAM
I: handle_command_stream: cmd_stream=0x70e73510, cms_length 6014
D: ETHOSU_PMU_CNTR_Disable(): mask=0x8000000f
D: ETHOSU_PMU_CNTR_Enable(): mask=0x8000000f
D: ethosu_dev_run_command_stream(): QBASE=0x0000000070e73510, QSIZE=24056, cmd_stream_ptr=0x70e73510
D: ethosu_dev_run_command_stream(): BASEP0=0x00000000704b0ac0
D: ethosu_dev_run_command_stream(): BASEP1=0x0000000070e797a0
D: ethosu_dev_run_command_stream(): BASEP2=0x0000000031000000
D: ethosu_dev_run_command_stream(): BASEP3=0x00000000710097a0
D: ethosu_dev_run_command_stream(): BASEP4=0x0000000070f417a0
D: ethosu_dev_run_command_stream(): BASEP5=0x0000000070ff6ba0
D: ethosu_dev_run_command_stream(): BASEP6=0x0000000070ffe8a0
D: ethosu_dev_run_command_stream(): BASEP7=0x0000000071001aa0
D: ethosu_dev_run_command_stream(): CMD=0x00000001
D: ETHOSU_PMU_CNTR_Disable(): mask=0x8000000f
D: ethosu_wait(): Inference finished successfully...
D: ethosu_release_driver(): NPU driver handle 0x20000068 released
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=0, val=207665
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=1, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=2, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=3, val=0
D: ETHOSU_PMU_Get_CCNTR(): val=208086
INFO - 
INFO - Profile for Inference:
INFO - NPU ACTIVE: 207665 cycles
INFO - NPU AXI0_RD_DATA_BEAT_RECEIVED: 0 beats
INFO - NPU AXI0_WR_DATA_BEAT_WRITTEN: 0 beats
INFO - NPU AXI1_RD_DATA_BEAT_RECEIVED: 0 beats
INFO - NPU IDLE: 421 cycles
INFO - NPU TOTAL: 208086 cycles
INFO - 
INFO - inference done.
INFO - DoPostProcess start in yolov8ssegnpp.
INFO - Output tensor 0: role=1, channels=64, grid=20x20, stride=32
INFO - Output tensor 1: role=3, channels=1, grid=80x80, stride=8
INFO - Output tensor 2: role=1, channels=64, grid=80x80, stride=8
INFO - Output tensor 3: role=4, channels=32, grid=40x40, stride=16
INFO - Output tensor 4: role=3, channels=1, grid=20x20, stride=32
INFO - Output tensor 5: role=1, channels=64, grid=40x40, stride=16
INFO - Output tensor 6: role=4, channels=32, grid=80x80, stride=8
INFO - Proto tensor 7: channels=32, grid=160x160 (int8, on-the-fly dequant)
INFO - Output tensor 8: role=2, channels=80, grid=80x80, stride=8
INFO - Output tensor 9: role=3, channels=1, grid=40x40, stride=16
INFO - Output tensor 10: role=4, channels=32, grid=20x20, stride=32
INFO - Output tensor 11: role=2, channels=80, grid=20x20, stride=32
INFO - Output tensor 12: role=2, channels=80, grid=40x40, stride=16
count: 6
59 0.0391 278.2543 398.8136 476.8647 0.9023 
58 342.6320 211.7197 429.7512 349.7274 0.8789 
56 246.6000 230.6992 350.5451 317.2393 0.8320 
58 184.2664 129.4328 241.0419 228.1649 0.8008 
39 96.9658 190.3404 113.9551 231.1448 0.5781 
73 432.2241 185.7390 555.2552 229.4771 0.2656 
INFO - post-processing done.
INFO - Final results:
INFO - Total number of inferences: 1
INFO - Using sample image: bus.jpg
INFO - capturedFrameSize 1228800.
INFO - inputTensor->bytes 1228800.
INFO - Extracted image ID: 0 from filename: bus.jpg
INFO - get img_id 0 ratio 0.592593 padding 80.000000 x 0.000000.
ERROR - Invalid image size for given location!
INFO - use input shape 1228800 
INFO - DoPreProcess start in yolov8ssegnpp.
INFO - Input tensor quantization: scale=0.003921568859, zero_point=-128
INFO - pre-processing done.
INFO - 
D: ETHOSU_PMU_Set_EVTYPER(): num=0, type=5, val=35
D: ETHOSU_PMU_Set_EVTYPER(): num=1, type=40, val=130
D: ETHOSU_PMU_Set_EVTYPER(): num=2, type=45, val=135
D: ETHOSU_PMU_Set_EVTYPER(): num=3, type=62, val=386
D: ETHOSU_PMU_Set_CNTR_OVS(): 
D: ETHOSU_PMU_Enable(): Enable PMU
D: ETHOSU_PMU_CNTR_Disable(): mask=0x8000000f
D: ETHOSU_PMU_CYCCNT_Reset(): Reset PMU cycle counter
D: ETHOSU_PMU_EVCNTR_ALL_Reset(): Reset all events
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=0, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=1, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=2, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=3, val=0
D: ETHOSU_PMU_Get_CCNTR(): val=0
I: Acquiring NPU driver handle
D: ethosu_reserve_driver(): NPU driver handle 0x20000068 reserved
D: ethosu_invoke_async(): OPTIMIZER_CONFIG
I: Optimizer release nbr: 0 patch: 1
I: Optimizer config. product=1, cmd_stream_version=0, macs_per_cc=9, shram_size=96, custom_dma=0
I: Optimizer config. arch version: 1.0.6
I: Ethos-U config. product=1, cmd_stream_version=0, macs_per_cc=9, shram_size=96, custom_dma=0
I: Ethos-U. arch version=1.1.0
D: ethosu_invoke_async(): NOP
D: ethosu_invoke_async(): NOP
D: ethosu_invoke_async(): NOP
D: ethosu_invoke_async(): COMMAND_STREAM
I: handle_command_stream: cmd_stream=0x70e73510, cms_length 6014
D: ETHOSU_PMU_CNTR_Disable(): mask=0x8000000f
D: ETHOSU_PMU_CNTR_Enable(): mask=0x8000000f
D: ethosu_dev_run_command_stream(): QBASE=0x0000000070e73510, QSIZE=24056, cmd_stream_ptr=0x70e73510
D: ethosu_dev_run_command_stream(): BASEP0=0x00000000704b0ac0
D: ethosu_dev_run_command_stream(): BASEP1=0x0000000070e797a0
D: ethosu_dev_run_command_stream(): BASEP2=0x0000000031000000
D: ethosu_dev_run_command_stream(): BASEP3=0x00000000710097a0
D: ethosu_dev_run_command_stream(): BASEP4=0x0000000070f417a0
D: ethosu_dev_run_command_stream(): BASEP5=0x0000000070ff6ba0
D: ethosu_dev_run_command_stream(): BASEP6=0x0000000070ffe8a0
D: ethosu_dev_run_command_stream(): BASEP7=0x0000000071001aa0
D: ethosu_dev_run_command_stream(): CMD=0x00000001
D: ETHOSU_PMU_CNTR_Disable(): mask=0x8000000f
D: ethosu_wait(): Inference finished successfully...
D: ethosu_release_driver(): NPU driver handle 0x20000068 released
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=0, val=207665
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=1, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=2, val=0
D: ETHOSU_PMU_Get_CNTR_OVS(): 
D: ETHOSU_PMU_Get_EVCNTR(): num=3, val=0
D: ETHOSU_PMU_Get_CCNTR(): val=208086
INFO - 
INFO - Profile for Inference:
INFO - NPU ACTIVE: 207665 cycles
INFO - NPU AXI0_RD_DATA_BEAT_RECEIVED: 0 beats
INFO - NPU AXI0_WR_DATA_BEAT_WRITTEN: 0 beats
INFO - NPU AXI1_RD_DATA_BEAT_RECEIVED: 0 beats
INFO - NPU IDLE: 421 cycles
INFO - NPU TOTAL: 208086 cycles
INFO - 
INFO - inference done.
INFO - DoPostProcess start in yolov8ssegnpp.
INFO - Output tensor 0: role=1, channels=64, grid=20x20, stride=32
INFO - Output tensor 1: role=3, channels=1, grid=80x80, stride=8
INFO - Output tensor 2: role=1, channels=64, grid=80x80, stride=8
INFO - Output tensor 3: role=4, channels=32, grid=40x40, stride=16
INFO - Output tensor 4: role=3, channels=1, grid=20x20, stride=32
INFO - Output tensor 5: role=1, channels=64, grid=40x40, stride=16
INFO - Output tensor 6: role=4, channels=32, grid=80x80, stride=8
INFO - Proto tensor 7: channels=32, grid=160x160 (int8, on-the-fly dequant)
INFO - Output tensor 8: role=2, channels=80, grid=80x80, stride=8
INFO - Output tensor 9: role=3, channels=1, grid=40x40, stride=16
INFO - Output tensor 10: role=4, channels=32, grid=20x20, stride=32
INFO - Output tensor 11: role=2, channels=80, grid=20x20, stride=32
INFO - Output tensor 12: role=2, channels=80, grid=40x40, stride=16
count: 6
5 12.6862 231.1845 799.8768 741.2241 0.9180 
0 50.6516 398.5554 246.3696 900.9402 0.9180 
0 670.5076 392.0984 809.7134 878.9827 0.8828 
0 222.5992 407.1767 343.5375 858.4033 0.8828 
0 0.0000 545.7411 78.6211 867.1135 0.5977 
27 285.6256 480.7857 301.9341 524.0493 0.4219 
INFO - post-processing done.
INFO - Final results:
INFO - Total number of inferences: 1
INFO - Main loop terminated successfully.
INFO - program terminating...
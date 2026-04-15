/*
 * SPDX-FileCopyrightText: Copyright 2021-2023 Arm Limited and/or its affiliates <open-source-office@arm.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/****************************************************************************
 * Includes
 ****************************************************************************/

#include "command_stream.hpp"
#include <stdio.h>

//#include <inttypes.h>


using namespace std;

extern struct ethosu_device *get_ethos_drv(void);


namespace EthosU {
namespace CommandStream {

/****************************************************************************
 * DataPointer
 ****************************************************************************/

DataPointer::DataPointer() : data(nullptr), size(0) {}

DataPointer::DataPointer(const char *_data, size_t _size) : data(_data), size(_size) {}

bool DataPointer::operator!=(const DataPointer &other) {
    if (size != other.size) {
        return true;
    }

    for (size_t i = 0; i < size; i++) {
        if (data[i] != other.data[i]) {
            return true;
        }
    }

    return false;
}

/****************************************************************************
 * PmuConfig
 ****************************************************************************/

Pmu::Pmu(ethosu_driver *_drv, const PmuEvents &_config) : drv(_drv), config(_config) {
    // Enable PMU block
	  printf("en pmu\r\n",0);
    ETHOSU_PMU_Enable(drv);
    printf("en cycle\r\n",0);
    // Enable cycle counter
    ETHOSU_PMU_CNTR_Enable(drv, ETHOSU_PMU_CCNT_Msk);

	  printf("en event\r\n",0);
    // Configure event types
    for (size_t i = 0; i < config.size(); i++) {
        ETHOSU_PMU_Set_EVTYPER(drv, i, config[i]);
        ETHOSU_PMU_CNTR_Enable(drv, 1u << i);
    }
		printf("end pmu\r\n",0);
}

void Pmu::clear() {
    ETHOSU_PMU_CYCCNT_Reset(drv);
    ETHOSU_PMU_EVCNTR_ALL_Reset(drv);
}

void Pmu::print() {
//    printf("PMU={cycleCount=%llu, events=[%" PRIu32 ", %" PRIu32 ", %" PRIu32 ", %" PRIu32 "]}\n",
//           ETHOSU_PMU_Get_CCNTR(drv),
//           ETHOSU_PMU_Get_EVCNTR(drv, 0),
//           ETHOSU_PMU_Get_EVCNTR(drv, 1),
//           ETHOSU_PMU_Get_EVCNTR(drv, 2),
//           ETHOSU_PMU_Get_EVCNTR(drv, 3));
}

uint64_t Pmu::getCycleCount() const {
    return ETHOSU_PMU_Get_CCNTR(drv);
}

uint32_t Pmu::getEventCount(size_t index) const {
    return ETHOSU_PMU_Get_EVCNTR(drv, index);
}

/****************************************************************************
 * CommandStream
 ****************************************************************************/
//drv(ethosu_reserve_driver()),
CommandStream::CommandStream(const DataPointer &_commandStream,
                             const BasePointers &_basePointers,
                             const PmuEvents &_pmuEvents) :
    drv(ethosu_reserve_driver()),
    commandStream(_commandStream), basePointers(_basePointers), pmu(drv, _pmuEvents) {}

CommandStream::~CommandStream() {
    ethosu_release_driver(drv);
}

int CommandStream::run(size_t repeat) {
    // Base pointer array
    uint64_t baseAddress[ETHOSU_BASEP_INDEXES];
    size_t baseAddressSize[ETHOSU_BASEP_INDEXES];
	  //printf("in command stream run\r\n");
	
    for (size_t i = 0; i < ETHOSU_BASEP_INDEXES; i++) {
        baseAddress[i]     = reinterpret_cast<uint64_t>(basePointers[i].data);
        baseAddressSize[i] = reinterpret_cast<size_t>(basePointers[i].size);
			  printf("baseAddress %d,\n",baseAddress[i]);
			  printf("baseAddress %d,\n", baseAddressSize[i]);
    }
		
		//printf("in command stream run2\r\n");

    while (repeat-- > 0) {
        int error = ethosu_invoke_v3(
            drv, commandStream.data, commandStream.size, &(baseAddress[0]), &(baseAddressSize[0]), ETHOSU_BASEP_INDEXES, nullptr);

        if (error != 0) {
            printf("Inference failed v3. error=%d\n", error);
            return 1;
        }
				//printf("in command while run\r\n");
    }

    return 0;
}

int CommandStream::run_async() {
    // Base pointer array
    uint64_t baseAddress[ETHOSU_BASEP_INDEXES];
    size_t baseAddressSize[ETHOSU_BASEP_INDEXES];

    for (size_t i = 0; i < ETHOSU_BASEP_INDEXES; i++) {
        baseAddress[i]     = reinterpret_cast<uint64_t>(basePointers[i].data);
        baseAddressSize[i] = reinterpret_cast<size_t>(basePointers[i].size);
				printf("baseAddress %d,\n",baseAddress[i]);
			  printf("baseAddress %d,\n", baseAddressSize[i]);
    }

    int error = ethosu_invoke_async(
        drv, commandStream.data, commandStream.size, &(baseAddress[0]), &(baseAddressSize[0]), ETHOSU_BASEP_INDEXES, nullptr);

    if (error != 0) {
        printf("Inference invoke async failed. error=%d\n", error);
        return 1;
    }

    return 0;
}

int CommandStream::wait_async(bool block) {
    return ethosu_wait(drv, block);
}

DataPointer &CommandStream::getCommandStream() {
    return commandStream;
}

BasePointers &CommandStream::getBasePointers() {
    return basePointers;
}

Pmu &CommandStream::getPmu() {
    return pmu;
}

}; // namespace CommandStream
}; // namespace EthosU

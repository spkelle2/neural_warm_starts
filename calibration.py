# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Timer API used during MIP solving."""

import abc
import time
from typing import Optional


class Timer(abc.ABC):
    """Class describing the API that need to be implemented for the Timer.

    This Timer class is used to time the duration of the MIP solving process, and
    is used in solvers_old.py. The methods that need to be provided are:

    - start_and_wait: This should start the timer, and waits for further calls.
    - terminate_and_wait: This should terminate the timer, and waits for further
        calls.
    - elapsed_real_time: This method should return the elapsed time in seconds
    since the last time
        it was started.
    - elapsed_calibrated_time: This method can be implemented to return version of
        the elapsed time that is calibrated to the machine speed.
    """

    cpu_start = None
    clock_start = None

    def start_and_wait(self):
        self.cpu_start = time.process_time()
        self.clock_start = time.time()

    def terminate_and_wait(self):
        pass

    # do these two need to be properties?
    def elapsed_real_time(self) -> float:
        return time.time() - self.clock_start

    def elapsed_calibrated_time(self) -> Optional[float]:
        return time.process_time() - self.cpu_start

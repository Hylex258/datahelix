# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Client IndexSenderThread"""

import queue
import threading
import time

from distributed_dataloader.client.microbatch import MicroBatchIndexResponseGroup, split_microbatch_indices
from distributed_dataloader.utils.logger import logger


class IndexSenderThread(threading.Thread):
    """Send sampler index to coordinator"""

    def __init__(
        self,
        name,
        client,
        index_sampler,
        cache_queue,
        result_queue,
        stop_event,
        epoch_event,
        inflight_semaphore,
        microbatch_size=None,
        max_microbatch_parts=None,
    ):
        super().__init__(name=name, daemon=True)
        self.client = client
        self.index_sampler = index_sampler
        self.sampler_iterator = iter(self.index_sampler)
        self.cache_queue = cache_queue
        self.result_queue = result_queue
        self.stop_event = stop_event
        self.epoch_event = epoch_event
        self.sent_count = 0
        self.queue_check_time = 0.001
        self.inflight_semaphore = inflight_semaphore
        self._queue_full_log_interval = 5.0
        self._last_queue_full_log_ts = 0.0
        self._queue_full_suppressed_logs = 0
        self._result_queue_was_full = False
        self.microbatch_size = int(microbatch_size) if microbatch_size is not None else None
        self.max_microbatch_parts = int(max_microbatch_parts) if max_microbatch_parts is not None else None

    def run(self):
        while not self.stop_event.is_set():
            if self.result_queue.full():
                now = time.monotonic()
                self._queue_full_suppressed_logs += 1
                if (
                    not self._result_queue_was_full
                    or self._last_queue_full_log_ts == 0.0
                    or now - self._last_queue_full_log_ts >= self._queue_full_log_interval
                ):
                    logger.info(
                        f"Result queue is full (size: {self.result_queue.qsize()}), waiting... "
                        f"check_interval={self.queue_check_time}s "
                        f"suppressed={max(self._queue_full_suppressed_logs - 1, 0)}"
                    )
                    self._last_queue_full_log_ts = now
                    self._queue_full_suppressed_logs = 0
                self._result_queue_was_full = True
                time.sleep(self.queue_check_time)
                continue
            if self._result_queue_was_full:
                self._result_queue_was_full = False
                self._last_queue_full_log_ts = 0.0
                self._queue_full_suppressed_logs = 0
            try:
                # get index
                index = next(self.sampler_iterator)
            except StopIteration:
                logger.info(f"Sampler exhausted, sent {self.sent_count} indices")
                # end mark
                self.cache_queue.put("end")

                logger.info("Waiting for next epoch...")
                self.epoch_event.wait()
                self.epoch_event.clear()

                self.sent_count = 0
                self.sampler_iterator = iter(self.index_sampler)
                logger.info("New epoch started.")
                continue

            self._send_index(index)

        logger.info("Thread stopped.")

    def _send_index(self, index):
        chunks = split_microbatch_indices(index, self.microbatch_size, max_parts=self._max_inflight_parts())
        acquired = 0
        try:
            for _ in chunks:
                self.inflight_semaphore.acquire()
                acquired += 1
            logger.info(f"The remaining space in the send queue is {self.inflight_semaphore._value}.")

            responses = [self._send_one_index(chunk) for chunk in chunks]
            self.sent_count += 1
            if len(responses) == 1:
                self.cache_queue.put(responses[0])
                return

            self.cache_queue.put(
                MicroBatchIndexResponseGroup(
                    responses=tuple(responses),
                    chunk_sizes=tuple(len(chunk) for chunk in chunks),
                )
            )
            logger.info(
                "Sent one logical batch as microbatch requests: "
                f"num_microbatches={len(chunks)} chunk_sizes={[len(chunk) for chunk in chunks]}"
            )
        except Exception:
            for _ in range(acquired):
                self.inflight_semaphore.release()
            raise

    def _send_one_index(self, index):
        current_index_send_times = 0
        while True:
            try:
                index_response = self.client.send_index_to_coordinator(index)
                if current_index_send_times > 0:
                    logger.info(f"Send index successful after retries={current_index_send_times}: index={index}")
                return index_response
            except Exception as error_info:
                if "Exception calling application: 'stub'" in str(error_info):
                    current_index_send_times += 1
                    logger.warning(
                        f'Got error: "Exception calling application: \'stub\'" from coordinator. '
                        f"Will attempt {current_index_send_times} to resend."
                    )
                    continue
                raise RuntimeError(f"Failed to send index {index}: {error_info}") from error_info

    def _max_inflight_parts(self):
        return self.max_microbatch_parts

    def stop(self):
        self.stop_event.set()
        try:
            while True:
                self.cache_queue.get_nowait()
        except queue.Empty:
            pass

import threading
import time


class RepeatedTimer:
    def __init__(self, interval: float, func, *args, **kwargs):
        self.interval = interval

        self._func = func
        self._args = args
        self._kwargs = kwargs

        self._thread = threading.Thread(target=self._call_each_interval)
        self._continue = True

    def _call_each_interval(self):
        while self._continue:
            time.sleep(self.interval)
            if not self._continue:
                return
            self._func(*self._args, **self._kwargs)

    def start(self):
        self._thread.start()

    def stop(self):
        self._continue = False
        self._thread.join()

    def __del__(self):
        self.stop()



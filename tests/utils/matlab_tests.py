import unittest
import time

import qctoolkit.utils.matlab


class RepeatedTimerTests(unittest.TestCase):

    def test_all(self):
        in_args = (1, 2, 'asd')
        in_kwargs = {'asd': 8, 'h': 9}

        interval = 1e-1
        started = False
        stopped = False
        self._counter = 0
        self._last_call = None

        def callback(*args, **kwargs):
            current_time = time.perf_counter()
            if self._last_call:
                self.assertLess(self._last_call + interval, current_time)
            self._last_call = current_time

            self.assertTrue(started)
            self.assertFalse(stopped)

            self.assertEqual(in_args, args)
            self.assertEqual(in_kwargs, kwargs)

            self._counter += 1

        timer = qctoolkit.utils.matlab.RepeatedTimer(interval, callback, *in_args, **in_kwargs)

        started = True
        timer.start()

        time.sleep(1)

        timer.stop()
        stopped = True

        self.assertIn(self._counter, [9, 10, 11])

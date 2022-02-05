import collections
from typing import Union

import numpy as np
from scipy.signal import butter


class ActionFilter(object):
    """Implements a generic lowpass or bandpass action filter."""

    def __init__(self, a, b, order, num_joints, ftype="lowpass"):
        """Initializes filter.
        Either one per joint or same for all joints.
        Args:
          a: filter output history coefficients
          b: filter input coefficients
          order: filter order
          num_joints: robot DOF
          ftype: filter type. 'lowpass' or 'bandpass'
        """
        self.num_joints = num_joints
        if isinstance(a, list):
            self.a = a
            self.b = b
        else:
            self.a = [a]
            self.b = [b]

        # Either a set of parameters per joint must be specified as a list
        # Or one filter is applied to every joint
        if not ((len(self.a) == len(self.b) == num_joints) or (len(self.a) == len(self.b) == 1)):
            raise ValueError("Incorrect number of filter values specified")

        # Normalize by a[0]
        for i in range(len(self.a)):
            self.b[i] /= self.a[i][0]
            self.a[i] /= self.a[i][0]

        # Convert single filter to same format as filter per joint
        if len(self.a) == 1:
            self.a *= num_joints
            self.b *= num_joints
        self.a = np.stack(self.a)
        self.b = np.stack(self.b)

        if ftype == "bandpass":
            assert len(self.b[0]) == len(self.a[0]) == 2 * order + 1
            self.hist_len = 2 * order
        elif ftype == "lowpass":
            assert len(self.b[0]) == len(self.a[0]) == order + 1
            self.hist_len = order
        else:
            raise ValueError("%s filter type not supported" % (ftype))

        self.yhist = collections.deque(maxlen=self.hist_len)
        self.xhist = collections.deque(maxlen=self.hist_len)
        self.reset()

    def reset(self):
        """Resets the history buffers to 0."""
        self.yhist.clear()
        self.xhist.clear()
        for _ in range(self.hist_len):
            self.yhist.appendleft(np.zeros((self.num_joints, 1)))
            self.xhist.appendleft(np.zeros((self.num_joints, 1)))

    def filter(self, x):
        """Returns filtered x."""
        xs = np.concatenate(list(self.xhist), axis=-1)
        ys = np.concatenate(list(self.yhist), axis=-1)
        y = (
            np.multiply(x, self.b[:, 0])
            + np.sum(np.multiply(xs, self.b[:, 1:]), axis=-1)
            - np.sum(np.multiply(ys, self.a[:, 1:]), axis=-1)
        )
        self.xhist.appendleft(x.reshape((self.num_joints, 1)).copy())
        self.yhist.appendleft(y.reshape((self.num_joints, 1)).copy())
        return y

    def init_history(self, x):
        x = np.expand_dims(x, axis=-1)
        for i in range(self.hist_len):
            self.xhist[i] = x
            self.yhist[i] = x


class ActionFilterButter(ActionFilter):
    """Butterworth filter."""

    def __init__(
        self,
        sampling_rate: float,
        num_joints: int,
        order: int = 2,
        lowcut: Union[float, np.ndarray] = 0.0,
        highcut: Union[float, np.ndarray] = 4.0,
    ):
        """Initializes a butterworth filter.
        Either one per joint or same for all joints.
        Args:
          lowcut: Low cutoff frequencies.
            If np.ndarray is provided, must contain same number of elements as highcut
            All 0 for lowpass or all > 0 for bandpass.
          highcut: High cutoff frequencies.
            If np.ndarray is provided, must contain same number of elements as lowcut
            All must be > 0
          sampling_rate: frequency of samples in Hz
          num_joints: int,
          order: filter order
        """
        self.lowcut = lowcut if isinstance(lowcut, np.ndarray) else np.full(num_joints, lowcut)
        self.highcut = highcut if isinstance(highcut, np.ndarray) else np.full(num_joints, highcut)

        if self.lowcut.shape != self.highcut.shape:
            raise ValueError("Number of lowcut and highcut filter values should " "be the same")

        if np.any(self.lowcut):
            if not np.all(self.lowcut):
                raise ValueError("All the filters must be of the same type: " "lowpass or bandpass")
            self.ftype = "bandpass"
        else:
            self.ftype = "lowpass"

        a_coeffs = []
        b_coeffs = []
        for i, (l, h) in enumerate(zip(self.lowcut, self.highcut)):
            if h <= 0.0:
                raise ValueError("Highcut must be > 0")

            b, a = self.butter_filter(l, h, sampling_rate, order)
            b_coeffs.append(b)
            a_coeffs.append(a)

        super(ActionFilterButter, self).__init__(a_coeffs, b_coeffs, order, num_joints, self.ftype)

    def butter_filter(self, lowcut, highcut, fs, order=5):
        """Returns the coefficients of a butterworth filter.
        If lowcut = 0, the function returns the coefficients of a low pass filter.
        Otherwise, the coefficients of a band pass filter are returned.
        Highcut should be > 0
        Args:
          lowcut: low cutoff frequency
          highcut: high cutoff frequency
          fs: sampling rate
          order: filter order
        Return:
          b, a: parameters of a butterworth filter
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        if low:
            b, a = butter(order, [low, high], btype="band")
        else:
            b, a = butter(order, [high], btype="low")
        return b, a

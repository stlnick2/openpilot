import numpy as np
from common.numpy_fast import clip, interp
import matplotlib.pyplot as plt
from selfdrive.config import Conversions as CV


class LatControlModel:
  def __init__(self):
    # Model generated using Konverter: https://github.com/ShaneSmiskol/Konverter
    model_weights_file = f'C:/Git/op/op-smiskol/torque_models/corolla_model_v5_weights.npz'
    self.w, self.b = np.load(model_weights_file, allow_pickle=True)['wb']

    self.use_rates = False
    self.sat_count_rate = 1.0 * 0.01
    self.sat_limit = 5

    self.reset()

  def reset(self):
    self.sat_count = 0.0

  def _check_saturation(self, control, check_saturation, limit):
    saturated = abs(control) == limit

    if saturated and check_saturation:
      self.sat_count += self.sat_count_rate
    else:
      self.sat_count -= self.sat_count_rate

    self.sat_count = clip(self.sat_count, 0.0, 1.0)

    return self.sat_count > self.sat_limit

  def predict(self, x):
    x = np.array(x, dtype=np.float32)
    l0 = np.dot(x, self.w[0]) + self.b[0]
    l0 = np.where(l0 > 0, l0, l0 * 0.3)
    l1 = np.dot(l0, self.w[1]) + self.b[1]
    l1 = np.where(l1 > 0, l1, l1 * 0.3)
    l2 = np.dot(l1, self.w[2]) + self.b[2]
    return l2


model = LatControlModel()

# speeds = np.linspace(0, 70 * CV.MPH_TO_MS, 1000)
angles = np.linspace(0, 90, 1000)
speed = 40 * CV.MPH_TO_MS
torque_40_left = np.array([model.predict([angle, angle-5, 0, 0, speed])[0] for angle in angles])
torque_40_right = np.array([model.predict([-angle, -(angle-5), 0, 0, speed])[0] *  interp(abs(angle), [0, 90.], [1.27, interp(speed, [17.8816, 31.2928], [1., 1.1])]) for angle in angles])

plt.plot(angles, torque_40_left, label='left torque at 40 mph')
plt.plot(angles, -torque_40_right, label='right torque at 40 mph')
plt.legend()

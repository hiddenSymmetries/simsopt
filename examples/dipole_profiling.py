import numpy as np

from simsopt.field.magneticfieldclasses import DipoleField

Ndipoles = 4095
m = np.outer(np.ones(Ndipoles), np.array([0.5, 0.5, 0.5]))
m_loc = np.outer(np.ones(Ndipoles), np.array([0.1, -0.1, 1]))
field_loc = np.outer(np.ones(Ndipoles), np.array([1, 0.2, 0.5]))
print(m.shape, field_loc.shape)
Bfield = DipoleField(m_loc, m)
Bfield.set_points(field_loc)



B_simsopt = Bfield.B()
B_correct = np.outer(np.ones(Ndipoles), Ndipoles * 1e-7 * np.array([0.260891, -0.183328, -0.77562]))
print(B_simsopt.shape, B_correct.shape)
print(B_simsopt, B_correct)
assert np.allclose(B_simsopt, B_correct)

import time
tic = time.time()

for i in range(200):
    Bfield.clear_cached_properties()
    Bfield.B()
toc = time.time()
print(toc-tic)

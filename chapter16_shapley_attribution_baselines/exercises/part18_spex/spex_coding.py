"""Small, visible finite-field lookup plumbing for the SPEX exercises.

These are the binary primitive polynomials used by galois.BCH 0.4.11, not learned
tables or precomputed support decoders. Bit i is the coefficient of x**i.
The method exercises (code construction, syndromes, BM, Chien) live in
spex_methods.py and can be executed in the notebook namespace.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np


BCH_PRIMITIVE_POLYNOMIALS = {
    2: 7, 3: 11, 4: 19, 5: 37, 6: 67, 7: 137, 8: 285, 9: 529, 10: 1033,
}


@dataclass(frozen=True)
class BinaryField:
    m: int
    polynomial: int
    exp: np.ndarray
    log: np.ndarray

    @property
    def order(self):
        return (1 << self.m) - 1

    def multiply(self, a: int, b: int) -> int:
        """Multiply field elements (addition is ordinary bitwise XOR)."""
        if a == 0 or b == 0:
            return 0
        return int(self.exp[int(self.log[a]) + int(self.log[b])])

    def divide(self, a: int, b: int) -> int:
        if b == 0:
            raise ZeroDivisionError("Division by zero in GF(2**m)")
        if a == 0:
            return 0
        return int(self.exp[(int(self.log[a]) - int(self.log[b])) % self.order])

    def multiply_array(self, a, b):
        """Broadcast multiplication, used only to make root search inexpensive."""
        a, b = np.broadcast_arrays(np.asarray(a, dtype=int), np.asarray(b, dtype=int))
        return np.where((a == 0) | (b == 0), 0, self.exp[self.log[a] + self.log[b]])


@lru_cache(maxsize=None)
def binary_field(m: int) -> BinaryField:
    """Generate all powers of primitive element x; never enumerate supports."""
    if m not in BCH_PRIMITIVE_POLYNOMIALS:
        raise ValueError("The supplied field table supports 2 <= m <= 10")
    order = (1 << m) - 1
    polynomial = BCH_PRIMITIVE_POLYNOMIALS[m]
    exp = np.zeros(2 * order, dtype=np.int64)
    log = np.zeros(order + 1, dtype=np.int64)
    value = 1
    for power in range(order):
        exp[power] = value
        log[value] = power
        value <<= 1
        if value & (1 << m):
            value ^= polynomial
    if value != 1 or len(set(exp[:order])) != order:
        raise ValueError("The field polynomial did not generate every nonzero element")
    exp[order:] = exp[:order]
    exp.flags.writeable = log.flags.writeable = False
    return BinaryField(m, polynomial, exp, log)

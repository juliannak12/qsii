"""
polynomial.py

Polynomial arithmetic over the reals, used as entries in the transfer matrix.
Coefficients stored as list: coeffs[i] = coefficient of z^i.
"""


class Polynomial:
    def __init__(self, coeffs=None):
        if coeffs is None:
            self.coeffs = [0.0]
        else:
            self.coeffs = [float(c) for c in coeffs]
        self._trim()

    def _trim(self):
        while len(self.coeffs) > 1 and abs(self.coeffs[-1]) < 1e-12:
            self.coeffs.pop()

    def __add__(self, other):
        n = max(len(self.coeffs), len(other.coeffs))
        result = [0.0] * n
        for i, c in enumerate(self.coeffs):
            result[i] += c
        for i, c in enumerate(other.coeffs):
            result[i] += c
        return Polynomial(result)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Polynomial([c * other for c in self.coeffs])
        result = [0.0] * (len(self.coeffs) + len(other.coeffs) - 1)
        for i, a in enumerate(self.coeffs):
            for j, b in enumerate(other.coeffs):
                result[i + j] += a * b
        return Polynomial(result)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __repr__(self):
        terms = []
        for i, c in enumerate(self.coeffs):
            if abs(c) < 1e-9:
                continue
            if i == 0:
                terms.append(f"{c:.6g}")
            elif i == 1:
                terms.append(f"{c:.6g}z")
            else:
                terms.append(f"{c:.6g}z^{i}")
        return " + ".join(terms) if terms else "0"

    def evaluate(self, z):
        result = 0.0
        for c in reversed(self.coeffs):
            result = result * z + c
        return result

    def truncate(self, max_deg):
        return Polynomial(self.coeffs[:max_deg + 1])

    @staticmethod
    def zero():
        return Polynomial([0.0])

    @staticmethod
    def one():
        return Polynomial([1.0])


class PolynomialMatrix:
    """
    Square matrix of Polynomials.
    data[i][j] is a Polynomial.
    """
    def __init__(self, data):
        self.data = data
        self.dim = len(data)

    @staticmethod
    def zeros(n):
        return PolynomialMatrix([[Polynomial.zero() for _ in range(n)] for _ in range(n)])

    @staticmethod
    def identity(n):
        m = PolynomialMatrix.zeros(n)
        for i in range(n):
            m.data[i][i] = Polynomial.one()
        return m

    def __matmul__(self, other):
        n = self.dim
        result = PolynomialMatrix.zeros(n)
        for i in range(n):
            for k in range(n):
                if all(abs(c) < 1e-12 for c in self.data[i][k].coeffs):
                    continue  # skip zero entries (sparse optimization from MATLAB)
                for j in range(n):
                    if all(abs(c) < 1e-12 for c in other.data[k][j].coeffs):
                        continue
                    result.data[i][j] = result.data[i][j] + (self.data[i][k] * other.data[k][j])
        return result

    def power(self, exp):
        """Repeated squaring: O(log n) matrix multiplications."""
        result = PolynomialMatrix.identity(self.dim)
        base = self
        while exp > 0:
            if exp % 2 == 1:
                result = result @ base
            base = base @ base
            exp //= 2
        return result

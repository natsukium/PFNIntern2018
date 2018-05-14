class array(object):
    def __init__(self, x):
        self.x = x
        self.shape = self._shape()
        self.T = self.transpose

    def __iter__(self):
        return self.x.__iter__()

    def __getitem__(self, *idx):
        if isinstance(idx[0], int):
            pass
        elif len(idx[0]) == 2:
            return self.x[idx[0][0]][idx[0][1]]
        if isinstance(self.x[idx[0]], (int, float)):
            return self.x[idx[0]]
        return array(self.x[idx[0]])

    def __repr__(self):
        prefix = f"array([{self.x[0]}"
        inner = ""
        suffix = "])"
        if len(self.x) > 1:
            for i in range(len(self.x)-1):
                inner += (f",\n       {self.x[i+1]}")
        return prefix + inner + suffix

    def __str__(self):
        prefix = f"[{self.x[0]}"
        inner = ""
        suffix = "]"
        if len(self.x) > 1:
            for i in range(len(self.x)-1):
                inner += f"\n {self.x[i+1]}"
        return prefix + inner + suffix

    def __len__(self):
        return len(self.x)

    def __eq__(self, y):
        return self.x == y

    def __neg__(self):
        if self._is_vector():
            return array([-i for i in self.x])
        else:
            return array([[-i for i in j] for j in self.x])

    def __abs__(self):
        if self._is_vector():
            return array([abs(i) for i in self.x])
        else:
            return array([[abs(i) for i in j] for j in self.x])

    def __round__(self, n):
        if self._is_vector():
            return array([round(i, n) for i in self.x])
        else:
            return array([[round(i, n) for i in j] for j in self.x])

    def __add__(self, y):
        return self.add(y)

    def __sub__(self, y):
        return self.sub(y)

    def __mul__(self, y):
        return self.multiply(y)

    def __matmul__(self, y):
        return self.dot(y)

    def __truediv__(self, y):
        return self.div(y)

    def add(self, y):
        # vector + scalar
        if self._is_vector() and self._is_scalar(y):
            return array([xi + y for xi in self.x])
        if self._shape(self.x) != self._shape(y):
            raise ValueError
        if self._is_vector():
            return array([xi + yi for xi, yi in zip(self.x, y)])
        return array([[xij + yij for xij, yij in zip(xi, yi)]
                      for xi, yi in zip(self.x, y)])

    def sub(self, y):
        # vector - scalar
        if self._is_scalar(y):
            return array([xi - y for xi in self.x])
        # vector - vector
        if self._is_vector(y):
            return array([xi - yi for xi, yi in zip(self.x, y)])
        # marix - matrix
        return array([[xij - yij for xij, yij in zip(xi, yi)]
                      for xi, yi in zip(self.x, y)])

    def multiply(self, y):
        # vector * scalar
        if self._is_vector() and self._is_scalar(y):
            return array([xi * y for xi in self.x])
        # matrix * scalar
        if self._is_scalar(y):
            return array([[xij * y for xij in xi] for xi in self.x])
        # matrix * matrix
        return array([[xij * yij for xij, yij in zip(xi, yi)]
                      for xi, yi in zip(self.x, y)])

    def dot(self, y):
        # vector @ vector
        if self._is_vector():
            if self._shape() != self._shape(y):
                raise ValueError
            return sum([xi * yi for xi, yi in zip(self.x, y)])
        # matrix @ vector
        if self._is_vector(y):
            if self._shape(self.x)[-1] != self._shape(y):
                raise ValueError
            return array(
                [sum([xj*yj for xj, yj in zip(xi, y)]) for xi in self.x])
        if self._shape(self.x)[-1] != self._shape(y)[0]:
            raise ValueError
        # matrix @ matrix
        return array([[sum([xij*yij for xij, yij in zip(xi, yj)])
                       for yj in zip(*y)] for xi in self.x])

    def div(self, y):
        if self._is_vector():
            return array([xi/y for xi in self.x])
        return array([[xij/y for xij in xi] for xi in self.x])

    def transpose(self):
        return array(list(map(list, zip(*self.x))))

    def concat(self, y):
        x = list(self.x)
        y = list(y)

        if self._is_vector(self.x):
            x_new = []
            x_new.append(x)
            x_new.append(y)
            return array(x_new)

        x.append(y)
        return array(x)

    def _shape(self, x=None):
        if x is None:
            x = self.x
        if self._is_vector(x):
            shape = len(x)
        else:
            shape = (len(x), len(x[0]))

        return shape

    def _is_scalar(self, x=None):
        if x is None:
            x = self.x
        return isinstance(x, (int, float))

    def _is_vector(self, x=None):
        if x is None:
            x = self.x
        return isinstance(x[0], (int, float))

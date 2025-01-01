from dataclasses import dataclass
import math
from typing import Union

    
@dataclass
class Matrix2x2:
    def __init__(self, r1:list[float|int], r2:list[float|int]):
        self.a11, self.a12 = r1
        self.a21, self.a22 = r2
        self.determinant = self.determinant()
        self.inverse = self.inverse()
        self.transpose = self.transpose()
    def __add__(self, other: 'Matrix2x2') -> 'Matrix2x2':
        return Matrix2x2([self.a11 + other.a11, self.a12 + other.a12],
                         [self.a21 + other.a21, self.a22 + other.a22])
    def __sub__(self, other: 'Matrix2x2') -> 'Matrix2x2':
        return Matrix2x2([self.a11 - other.a11, self.a12 - other.a12],
                         [self.a21 - other.a21, self.a22 - other.a22])
    def __mul__(self, scalar: float) -> 'Matrix2x2':
        return Matrix2x2([self.a11 * scalar, self.a12 * scalar],
                         [self.a21 * scalar, self.a22 * scalar])
    def __str__(self) -> str:
        return f"Matrix2x2([{self.a11}, {self.a12}], [{self.a21}, {self.a22}])"
    def determinant(self) -> float:
        return self.a11 * self.a22 - self.a12 * self.a21
    def inverse(self) -> 'Matrix2x2':
        determinant = self.determinant
        if determinant == 0:
            raise ValueError("Matrix is not invertible")
        return Matrix2x2([self.a22 / determinant, -self.a12 / determinant],
                         [-self.a21 / determinant, self.a11 / determinant])
    def transpose(self) -> 'Matrix2x2':
        return Matrix2x2([self.a11, self.a21], [self.a12, self.a22])

@dataclass
class Matrix3x3:
    def __init__(self, r1:list[float|int], r2:list[float|int], r3:list[float|int]):
        self.a11, self.a12, self.a13 = r1
        self.a21, self.a22, self.a23 = r2
        self.a31, self.a32, self.a33 = r3

        self.determinant = self.determinant()
        self.inverse = self.inverse()
        self.transpose = self.transpose()
    
    def __add__(self, other: 'Matrix3x3') -> 'Matrix3x3':
        return Matrix3x3([self.a11 + other.a11, self.a12 + other.a12, self.a13 + other.a13],
                         [self.a21 + other.a21, self.a22 + other.a22, self.a23 + other.a23],
                         [self.a31 + other.a31, self.a32 + other.a32, self.a33 + other.a33])
    def __sub__(self, other: 'Matrix3x3') -> 'Matrix3x3':
        return Matrix3x3([self.a11 - other.a11, self.a12 - other.a12, self.a13 - other.a13],
                         [self.a21 - other.a21, self.a22 - other.a22, self.a23 - other.a23],
                         [self.a31 - other.a31, self.a32 - other.a32, self.a33 - other.a33])
    
    def __mul__(self, other: 'Matrix3x3') -> 'Matrix3x3':
        n = 3
        result = [[0] * n for _ in range(n)]
    
        for x in range(n):
            for y in range(n):
                t = 0
                for i in range(0,n-1):
                    t += self[f"a{x}{(y+i)%(n+1)}"] * other[f"a{(x+i)%(n+1)}{y}"]
                result[x][y] = t
        r1, r2, r3 = result
        return Matrix3x3(r1, r2, r3)
    def __str__(self) -> str:
        return f"Matrix3x3([{self.a11}, {self.a12}, {self.a13}], [{self.a21}, {self.a22}, {self.a23}], [{self.a31}, {self.a32}, {self.a33}])"
    
    def determinant(self) -> float:
        '''
        WARNING: This function is turned into a value after INIT is called.
        '''
        return (self.a11 * (self.a22 * self.a33 - self.a23 * self.a32) -
                self.a12 * (self.a21 * self.a33 - self.a23 * self.a31) +
                self.a13 * (self.a21 * self.a32 - self.a22 * self.a31))

    def inverse(self) -> 'Matrix3x3':
        '''
        WARNING: This function is turned into a value after INIT is called.
        '''
        det = self.determinant
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")
        
        inv_det = 1 / det
        return Matrix3x3(
            [(self.a22 * self.a33 - self.a23 * self.a32) * inv_det,
             (self.a13 * self.a32 - self.a12 * self.a33) * inv_det,
             (self.a12 * self.a23 - self.a13 * self.a22) * inv_det],
            [(self.a23 * self.a31 - self.a21 * self.a33) * inv_det,
             (self.a11 * self.a33 - self.a13 * self.a31) * inv_det,
             (self.a13 * self.a21 - self.a11 * self.a23) * inv_det],
            [(self.a21 * self.a32 - self.a22 * self.a31) * inv_det,
             (self.a12 * self.a31 - self.a11 * self.a32) * inv_det,
             (self.a11 * self.a22 - self.a12 * self.a21) * inv_det]
        )

    def transpose(self) -> 'Matrix3x3':
        '''
        WARNING: This function is turned into a value after INIT is called.
        '''
        return Matrix3x3(
            [self.a11, self.a21, self.a31],
            [self.a12, self.a22, self.a32],
            [self.a13, self.a23, self.a33]
        )

class Matrix4x4:
    def __init__(self, r1:list[float|int], r2:list[float|int], r3:list[float|int], r4:list[float|int]):
        self.a11, self.a12, self.a13, self.a14 = r1
        self.a21, self.a22, self.a23, self.a24 = r2
        self.a31, self.a32, self.a33, self.a34 = r3
        self.a41, self.a42, self.a43, self.a44 = r4
        self.determinant = self.determinant()
        self.inverse = self.inverse()
        self.transpose = self.transpose()

    def __add__(self, other: 'Matrix4x4') -> 'Matrix4x4':
        return Matrix4x4([self.a11 + other.a11, self.a12 + other.a12, self.a13 + other.a13, self.a14 + other.a14],
                         [self.a21 + other.a21, self.a22 + other.a22, self.a23 + other.a23, self.a24 + other.a24],
                         [self.a31 + other.a31, self.a32 + other.a32, self.a33 + other.a33, self.a34 + other.a34],
                         [self.a41 + other.a41, self.a42 + other.a42, self.a43 + other.a43, self.a44 + other.a44])
    def __sub__(self, other: 'Matrix4x4') -> 'Matrix4x4':
        return Matrix4x4([self.a11 - other.a11, self.a12 - other.a12, self.a13 - other.a13, self.a14 - other.a14],
                         [self.a21 - other.a21, self.a22 - other.a22, self.a23 - other.a23, self.a24 - other.a24],
                         [self.a31 - other.a31, self.a32 - other.a32, self.a33 - other.a33, self.a34 - other.a34],
                         [self.a41 - other.a41, self.a42 - other.a42, self.a43 - other.a43, self.a44 - other.a44])
    def __mul__(self, other: 'Matrix4x4') -> 'Matrix4x4':
        n = 4
        result = [[0] * n for _ in range(n)]
    
        for x in range(n):
            for y in range(n):
                t = 0
                for i in range(0,n-1):
                    t += self[f"a{x}{(y+i)%(n+1)}"] * other[f"a{(x+i)%(n+1)}{y}"]
                result[x][y] = t
        r1, r2, r3, r4 = result
        return Matrix3x3(r1, r2, r3, r4)
    def __str__(self) -> str:
        return f"Matrix4x4([{self.a11}, {self.a12}, {self.a13}, {self.a14}], [{self.a21}, {self.a22}, {self.a23}, {self.a24}], [{self.a31}, {self.a32}, {self.a33}, {self.a34}], [{self.a41}, {self.a42}, {self.a43}, {self.a44}])"
    def determinant(self) -> float:
        '''
        WARNING: This function is turned into a value after INIT is called.
        '''
        return (self.a11 * (self.a22 * self.a33 * self.a44 + self.a23 * self.a34 * self.a42 + self.a24 * self.a32 * self.a43 - self.a24 * self.a33 * self.a42 - self.a23 * self.a32 * self.a44 - self.a22 * self.a34 * self.a43) -
                self.a12 * (self.a21 * self.a33 * self.a44 + self.a23 * self.a34 * self.a41 + self.a24 * self.a31 * self.a43 - self.a24 * self.a33 * self.a41 - self.a23 * self.a31 * self.a44 - self.a21 * self.a34 * self.a43) +
                self.a13 * (self.a21 * self.a32 * self.a44 + self.a22 * self.a34 * self.a41 + self.a24 * self.a31 * self.a42 - self.a24 * self.a32 * self.a41 - self.a22 * self.a31 * self.a44 - self.a21 * self.a34 * self.a42) -
                self.a14 * (self.a21 * self.a32 * self.a43 + self.a22 * self.a33 * self.a41 + self.a23 * self.a31 * self.a42 - self.a23 * self.a32 * self.a41 - self.a22 * self.a31 * self.a43 - self.a21 * self.a33 * self.a42))
    def inverse(self) -> 'Matrix4x4':
        '''
        WARNING: This function is turned into a value after INIT is called.
        '''
        det = self.determinant
        if det == 0:
            raise ValueError("Matrix is singular and cannot be inverted")

        inv_det = 1 / det
        return Matrix4x4(
            [(self.a22 * self.a33 * self.a44 + self.a23 * self.a34 * self.a42 + self.a24 * self.a32 * self.a43 - self.a24 * self.a33 * self.a42 - self.a23 * self.a32 * self.a44 - self.a22 * self.a34 * self.a43) * inv_det,
             (self.a14 * self.a33 * self.a42 + self.a13 * self.a32 * self.a44 + self.a12 * self.a34 * self.a43 - self.a14 * self.a32 * self.a43 - self.a13 * self.a34 * self.a42 - self.a12 * self.a33 * self.a44) * inv_det,
             (self.a12 * self.a23 * self.a44 + self.a13 * self.a24 * self.a42 + self.a14 * self.a22 * self.a43 - self.a14 * self.a23 * self.a42 - self.a13 * self.a22 * self.a44 - self.a12 * self.a24 * self.a43) * inv_det,
             (self.a14 * self.a22 * self.a33 + self.a12 * self.a23 * self.a34 + self.a13 * self.a24 * self.a32 - self.a13 * self.a22 * self.a34 - self.a12 * self.a24 * self.a33 - self.a14 * self.a23 * self.a32) * inv_det],
            [(self.a24 * self.a33 * self.a41 + self.a21 * self.a34 * self.a43 + self.a23 * self.a31 * self.a44 - self.a23 * self.a34 * self.a41 - self.a21 * self.a33 * self.a44 - self.a24 * self.a31 * self.a43) * inv_det,
             (self.a11 * self.a34 * self.a43 + self.a13 * self.a31 * self.a44 + self.a14 * self.a33 * self.a41 - self.a14 * self.a31 * self.a43 - self.a13 * self.a34 * self.a41 - self.a11 * self.a33 * self.a44) * inv_det,
             (self.a14 * self.a21 * self.a33 + self.a11 * self.a23 * self.a34 + self.a13 * self.a24 * self.a31 - self.a13 * self.a21 * self.a34 - self.a11 * self.a24 * self.a33 - self.a14 * self.a23 * self.a31) * inv_det,
             (self.a11 * self.a22 * self.a34 + self.a12 * self.a24 * self.a31 + self.a14 * self.a21 * self.a32 - self.a14 * self.a22 * self.a31 - self.a12 * self.a21 * self.a34 - self.a11 * self.a24 * self.a32) * inv_det],
            [(self.a21 * self.a32 * self.a44 + self.a22 * self.a34 * self.a41 + self.a24 * self.a31 * self.a42 - self.a24 * self.a32 * self.a41 - self.a22 * self.a31 * self.a44 - self.a21 * self.a34 * self.a42) * inv_det,
             (self.a12 * self.a31 * self.a44 + self.a11 * self.a34 * self.a42 + self.a14 * self.a32 * self.a41 - self.a14 * self.a31 * self.a42 - self.a12 * self.a34 * self.a41 - self.a11 * self.a32 * self.a44) * inv_det,
             (self.a14 * self.a22 * self.a31 + self.a12 * self.a24 * self.a31 + self.a11 * self.a22 * self.a34 - self.a14 * self.a21 * self.a32 - self.a12 * self.a21 * self.a34 - self.a11 * self.a24 * self.a32) * inv_det,
             (self.a11 * self.a23 * self.a32 + self.a12 * self.a21 * self.a33 + self.a13 * self.a22 * self.a31 - self.a13 * self.a21 * self.a32 - self.a12 * self.a23 * self.a31 - self.a11 * self.a22 * self.a33) * inv_det],
            [(self.a22 * self.a31 * self.a43 + self.a23 * self.a32 * self.a41 + self.a21 * self.a33 * self.a42 - self.a21 * self.a32 * self.a43 - self.a23 * self.a31 * self.a42 - self.a22 * self.a33 * self.a41) * inv_det,
             (self.a13 * self.a31 * self.a42 + self.a11 * self.a32 * self.a43 + self.a12 * self.a33 * self.a41 - self.a12 * self.a31 * self.a43 - self.a13 * self.a32 * self.a41 - self.a11 * self.a33 * self.a42) * inv_det,
             (self.a11 * self.a22 * self.a33 + self.a12 * self.a23 * self.a31 + self.a13 * self.a21 * self.a32 - self.a13 * self.a22 * self.a31 - self.a12 * self.a21 * self.a33 - self.a11 * self.a23 * self.a32) * inv_det,
             (self.a12 * self.a21 * self.a34 + self.a14 * self.a22 * self.a31 + self.a11 * self.a24 * self.a32 - self.a11 * self.a22 * self.a34 - self.a14 * self.a21 * self.a32 - self.a12 * self.a24 * self.a31) * inv_det]
        )
    
    def transpose(self) -> 'Matrix4x4':
        '''
        WARNING: This function is turned into a value after INIT is called.
        '''
        return Matrix4x4(
            [self.a11, self.a21, self.a31, self.a41],
            [self.a12, self.a22, self.a32, self.a42],
            [self.a13, self.a23, self.a33, self.a43],
            [self.a14, self.a24, self.a34, self.a44]
        )

@dataclass
class Vector2:
    def __init__(self, x: Union[float|int], y: Union[float|int]):
        self.x: Union[float|int] = x
        self.y: Union[float|int] = y
    
    def __add__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other: 'Vector2') -> 'Vector2':
        return Vector2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar: float) -> 'Vector2':
        return Vector2(self.x * scalar, self.y * scalar)
    
    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y)
    
    def normalize(self) -> 'Vector2':
        length = self.length()
        if length > 0:
            return self * (1.0 / length)
        return Vector2(0, 0)
    
    def __str__(self) -> str:
        return f"Vector2({self.x}, {self.y})"
    
    def __dot__(self, other) -> float:
        if isinstance(other, Vector2):
            return self.x * other.x + self.y * other.y
        elif isinstance(other, Vector3):
            return self.x * other.x + self.y * other.y + other.z
        else:
            raise TypeError("Unsupported operand type for __dot__")
        
    def cross(self, other) -> float:
        return self.x * other.y - self.y * other.x
    def normalize(self) -> 'Vector2':
        length = self.length()
        if length > 0:
            return self * (1.0 / length)
        return Vector2(0, 0)
    def transform(self, matrix: Matrix2x2) -> 'Vector2':
        x = self.x * matrix.a11 + self.y * matrix.a21
        y = self.x * matrix.a21 + self.y * matrix.a22
        return Vector2(x, y)
    def rotate(self, angle) -> 'Vector2':
        matrix = Matrix2x2([math.cos(angle), -math.sin(angle)],
            [math.sin(angle), math.cos(angle)]
        )
        return self.transform(matrix)
    
@dataclass
class Vector3:
    def __init__(self, x: Union[float|int], y: Union[float|int], z: Union[float|int]):
        self.x: Union[float|int] = x
        self.y: Union[float|int] = y
        self.z: Union[float|int] = z

    def __add__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other: 'Vector3') -> 'Vector3':
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, scalar: float) -> 'Vector3':
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def length(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalize(self) -> 'Vector3':
        length = self.length()
        if length > 0:
            return self * (1.0 / length)
        return Vector3(0, 0, 0)

    def __str__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"
    
    def __dot__(self, other) -> float:
        if isinstance(other, Vector2):
            return self.x * other.x + self.y * other.y + self.z
        elif isinstance(other, Vector3):
            return self.x * other.x + self.y * other.y + self.z * other.z
        else:
            raise TypeError("Unsupported operand type for __dot__")
        
    def cross(self, other):
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )
    
    def normalize(self):
        length = math.sqrt(self.dot(self))
        if length == 0:
            return Vector3(0, 0, 0)
        return Vector3(self.x / length, self.y / length, self.z / length)
    
    def transform(self, matrix: Matrix4x4) -> 'Vector3':
        x = self.x * matrix.a11 + self.y * matrix.a21 + self.z * matrix.a31 + matrix.a41
        y = self.x * matrix.a12 + self.y * matrix.a22 + self.z * matrix.a32 + matrix.a42
        z = self.x * matrix.a13 + self.y * matrix.a23 + self.z * matrix.a33 + matrix.a43
        w = self.x * matrix.a14 + self.y * matrix.a24 + self.z * matrix.a34 + matrix.a44
        if w != 0:
            return Vector3(x/w, y/w, z/w)
        return Vector3(x, y, z)
    
    def rotate(self, angle):
        z_matrix = Matrix4x4(
            [math.cos(angle.z), -math.sin(angle.z), 0, 0],
            [math.sin(angle.z), math.cos(angle.z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        )
        y_matrix = Matrix4x4(
            [math.cos(angle.y), 0, math.sin(angle.y), 0],
            [0, 1, 0, 0],
            [-math.sin(angle.y), 0, math.cos(angle.y), 0],
            [0, 0, 0, 1]
        )
        x_matrix = Matrix4x4(
            [1, 0, 0, 0],
            [0, math.cos(angle.x), -math.sin(angle.x), 0],
            [0, math.sin(angle.x), math.cos(angle.x), 0],
            [0, 0, 0, 1]
        )
        return self.transform(z_matrix).transform(y_matrix).transform(x_matrix)
    
def dot(v1: Union[Vector2 | Vector3], v2: Union[Vector2 | Vector3]) -> float:
    return v1.__dot__(v2) 
from entity2_protocol import Protocol
import sympy as sy


if __name__ == "__main__":
    print("asdasd")
    x0 = sy.Matrix([1,1,1,1])
    x1 = sy.Matrix([0,1,2,3])
    y0 = sy.Matrix([1,1,1,1])
    y1 = sy.Matrix([-1,-1,-1,0])
    z0, z1 = Protocol.dot_product_triples(4, x0, x1, y0, y1)
    print("z: ", z0 + z1)

    print("direct inner product: ", (x0 + x1).dot(y0 + y1))
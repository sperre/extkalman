package extkalman

import "code.google.com/p/gomatrix/matrix"

func Kalman(W, R, Q, x0, P0, A, B, H matrix.Matrix) (k *ExtKalmanFilter) {

	f := func(x, u matrix.Matrix) matrix.Matrix {
		// f(x) = Ax + Bu
		return matrix.Sum( matrix.Product(A, x), matrix.Product(B, u))
	}
	dfdx := func(x, u matrix.Matrix) matrix.Matrix {
		// df/dx = A
		return A
	}
	h := func(x matrix.Matrix) matrix.Matrix  {
		// h(x) = H*x
		return matrix.Product(A, x)
	}
	dhdx := func(x matrix.Matrix) matrix.Matrix {
		// dh/dx = H
		return H
	}
	return ExtendedKalman(W, R, Q, x0, P0, f, dfdx, h, dhdx, nil)
}


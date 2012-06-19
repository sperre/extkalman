package extkalman

import (
	"code.google.com/p/gomatrix/matrix"
)

// System equation: v and w are white noize, Q and R are design matrixes, P0
// and x0 are initial states.
//
// x(k+1) = f(x,u) + W*w
//   y(x) = h(x,u) + v
//
// Q(k) = E[w(k) w(j)^t], where j=k else 0
// R(k) = E[v(k) v(j)^t], where j=k else 0
type ExtKalmanFilter struct {
	// States
	x matrix.Matrix   // State vector
	mP matrix.Matrix  // Error covariance matrix

	// Constants
	mW  matrix.Matrix    // Which states are noisy
	mQ matrix.Matrix     // Process noise variance
	mR matrix.Matrix     // Measurement noise variance
	cWQWt matrix.Matrix  // W*Q*W^t

	// Function pointers
	f func(matrix.Matrix, matrix.Matrix) matrix.Matrix     // x = f(x,u)
	dfdx func(matrix.Matrix, matrix.Matrix) matrix.Matrix  // F = df(x,u)/dx
	h func(matrix.Matrix) matrix.Matrix                    // y = h(x)
	dhdx func(matrix.Matrix) matrix.Matrix                 // H = dh(x)/dx

	// Useful to normalize angle error (TODO: remove)
	normalize_ydiff func(matrix.Matrix) matrix.Matrix
}

func ExtendedKalman(
		W, R, Q, x0, P0 matrix.Matrix,
		f, dfdx func(matrix.Matrix, matrix.Matrix) matrix.Matrix,
		h, dhdx, normalize_ydiff func(matrix.Matrix) matrix.Matrix) (k *ExtKalmanFilter) {
	k = new(ExtKalmanFilter)
	k.mW = W
	k.mR = R
	k.mQ = Q

	k.x = x0
	k.mP = P0

	k.f    =  f
	k.dfdx = dfdx
	k.h    =  h
	k.dhdx = dhdx
	k.normalize_ydiff = normalize_ydiff

	// Pre calculated constant
	k.cWQWt = matrix.Product( W, Q, matrix.Transpose(W) )

	return
}

// y: Measurments
// u: Actuator thrust
func (k *ExtKalmanFilter)Step(y, u matrix.Matrix)(x matrix.Matrix) {
	var y_ = k.h(k.x)
	var H  = k.dhdx(k.x)
	var Ht = matrix.Transpose(H)


	// -----------------
	// Estimation

	// Kalman gain matrix
	// K = P * H^t * (H*P*Ht + R)⁻¹
	var inv = matrix.Inverse( matrix.Sum( matrix.Product(H, k.mP, Ht), k.mR) )
	var K = matrix.Product(k.mP, Ht, inv)
	var Kt = matrix.Transpose(K)


	// State estimation update
	// x_ = x + K * (y - h(x))
	var ydiff, _ = y.Minus(y_)
	if k.normalize_ydiff != nil {
		ydiff = k.normalize_ydiff(ydiff)
	}
	var x_ = matrix.Sum(k.x, matrix.Product(K, ydiff))

	// Error covariance update
	// P_ = (I - K*H) * P * (I - K*H)^t + K*R*K^t
	var IKH, _ = matrix.Eye( H.Cols() ).Minus( matrix.Product(K, H) )
	var IKHt = matrix.Transpose(IKH)
	var P_ = matrix.Sum( matrix.Product(IKH, k.mP, IKHt), matrix.Product(K, k.mR, Kt) )


	// -----------------
	// Propagation
	var F = k.dfdx(x_, u)
	var Ft = matrix.Transpose(F)

	// State estimation propagation
	// x(k+1) = A*x_(k) + B*u
	k.x = k.f(x_, u)

	// Error covariance propagation
	// P(k+1) = A*P_*A^t + W*Q*W^t

	k.mP = matrix.Sum( matrix.Product(F, P_, Ft), k.cWQWt )
	return k.x
}

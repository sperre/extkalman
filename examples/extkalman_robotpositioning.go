package main

import (
	"fmt"
	"math"
	"code.google.com/p/gomatrix/matrix"
	"eurobot/extkalman"
)


type position struct {
	X float64
	Y float64
}

func newPos(X, Y float64) *position {
	return &position{X,Y}
}

func main() {
	// Beacon positions
	var beaconA = newPos(   0.0,    0.0)
	var beaconB = newPos(   0.0, 2000.0)
	var beaconC = newPos(3000.0, 1000.0)

	// First state
	var x0 matrix.Matrix = matrix.MakeDenseMatrix([]float64{200, 0, 200, 0}, 4, 1)
	var P0 matrix.Matrix = matrix.Diagonal([]float64{10000, 100, 10000, 100})

	fmt.Printf(">init:\n")
	fmt.Printf("x0:\n%v\n\n", x0)
	fmt.Printf("P0:\n%v\n\n", P0)

	// x(k+1) = Ax + Bu + Ww
	var W matrix.Matrix = matrix.MakeDenseMatrix([]float64{
		0.0, 0.0,
		1.0, 0.0,
		0.0, 0.0,
		0.0, 1.0}, 4, 2)
	fmt.Printf("> noise\n")
	fmt.Printf("W:\n%v\n\n", W)

	// Design
	var d = 0.01*0.01
	var Q matrix.Matrix = matrix.Diagonal([]float64{d, d})
	var R matrix.Matrix = matrix.Diagonal([]float64{d, d, d})

	fmt.Printf(">Design\n")
	fmt.Printf("Q:\n%v\n\n", Q)
	fmt.Printf("R:\n%v\n\n", R)

	// Init filter
	var u matrix.Matrix = matrix.MakeDenseMatrix([]float64{0, 0}, 2, 1)
	var y matrix.Matrix = matrix.MakeDenseMatrix([]float64{1000, 1000, 3000}, 3, 1)

	// df(x,u)/dx
	dfdx := func(x, u matrix.Matrix) matrix.Matrix {
		// The process is independent of possition, df/dx = A
		T := 1.0
		A := matrix.MakeDenseMatrix([]float64{
			1.0,   T, 0.0, 0.0,
			0.0, 1.0, 0.0, 0.0,
			0.0, 0.0, 1.0,   T,
			0.0, 0.0, 0.0, 1.0}, 4, 4)
		return A
	}

	// Estimated movement of robot given state x
	f := func(x, u matrix.Matrix) matrix.Matrix {
		// x(k+1) = Ax + Bu
		var A = dfdx(x, u)

		return matrix.Product(A, x)
		// Commented out: take motor gain in to account:
		/*
		var B matrix.Matrix = matrix.MakeDenseMatrix([]float64{
			0.0, 0.0,
			0.0, 0.0,
			0.0, 0.0,
			0.0, 0.0}, 4, 2)
		return matrix.Sum( matrix.Product(A, x), matrix.Product(B, u))
		*/
	}

	// dh(x,u)/dx
	dhdx := func(x matrix.Matrix) matrix.Matrix {
		// Linearisation of H around x(k)
		robot := newPos(x.Get(0, 0), x.Get(2, 0))

		denA := math.Sqrt( math.Pow(robot.X-beaconA.X, 2.0) + math.Pow(robot.Y-beaconA.Y, 2.0) )
		denB := math.Sqrt( math.Pow(robot.X-beaconB.X, 2.0) + math.Pow(robot.Y-beaconB.Y, 2.0) )
		denC := math.Sqrt( math.Pow(robot.X-beaconC.X, 2.0) + math.Pow(robot.Y-beaconC.Y, 2.0) )

		H := matrix.MakeDenseMatrix([]float64{
			(robot.X-beaconA.X)/denA, 0.0, (robot.Y-beaconA.Y)/denA, 0.0,
			(robot.X-beaconB.X)/denB, 0.0, (robot.Y-beaconB.Y)/denB, 0.0,
			(robot.X-beaconC.X)/denC, 0.0, (robot.Y-beaconC.Y)/denC, 0.0}, 3, 4)

		return H
	}

	// Measure estimate given state x
	h:= func(x matrix.Matrix) matrix.Matrix {
		var robot = newPos(x.Get(0, 0), x.Get(2, 0))

		var y matrix.Matrix = matrix.MakeDenseMatrix([]float64{
			math.Sqrt( math.Pow(robot.X-beaconA.X, 2.0) + math.Pow(robot.Y-beaconA.Y, 2.0) ),
			math.Sqrt( math.Pow(robot.X-beaconB.X, 2.0) + math.Pow(robot.Y-beaconB.Y, 2.0) ),
			math.Sqrt( math.Pow(robot.X-beaconC.X, 2.0) + math.Pow(robot.Y-beaconC.Y, 2.0) )}, 3, 1)
		return y
	}

	// State variable
	var x matrix.Matrix

	// Initalize kalman filter
	var filter = extkalman.ExtendedKalman(W, R, Q, x0, P0, f, dfdx, h, dhdx)

	// Run filter
	fmt.Printf("Running:\n")
	for {
		x = filter.Step(y, u)
		fmt.Printf("x:\n%v\n\n", x)
	}

}



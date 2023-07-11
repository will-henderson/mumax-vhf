// Package setup declares system parameters and tells them to mumax
package data

import (
	en "github.com/mumax/3/engine"
)

var (
	DynamicFactor float64
)

// Setup assigns the variables to global variables, and tells them to mumax.
// This should be called after "defer en.InitAndClose()()"
func Setup(s string) {

	ResetGlobals()
	en.Eval(s)

}

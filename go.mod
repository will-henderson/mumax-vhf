module github.com/will-henderson/mumax-vhf

go 1.19

require (
	github.com/mumax/3 v3.9.3+incompatible
	gonum.org/v1/gonum v0.13.0
)

//replace github.com/mumax/3 => ../mumax/3
replace github.com/mumax/3 => ../mumax-gradients

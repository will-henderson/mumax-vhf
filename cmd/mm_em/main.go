package main

import (
	"fmt"
	"os"

	en "github.com/mumax/3/engine"
	"github.com/will-henderson/mumax-vhf/data"
	"github.com/will-henderson/mumax-vhf/solver"
)

func main() {
	defer en.InitAndClose()

	filename := os.Args[1]
	bytes, err := os.ReadFile(filename)
	if err != nil {
		fmt.Println(err)
	}
	data.Setup(string(bytes))

	en.Relax()
	solver.SetSolver(os.Args[2])
	solver.Modes()

}

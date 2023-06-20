package field

import (
	"fmt"

	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

func SetDemagField(s, dst *data.Slice) {

	//The easiest thing to do is to just temporarily point the magnetisation variable to something else
	m0 := magnetisationBuffer
	fmt.Println("magnetisation Buffer:", magnetisationBuffer)

	magnetisationBuffer = s

	en.SetDemagField(dst)

	magnetisationBuffer = m0

}

package field

import (
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

func SetDemagField(s, dst *data.Slice) {

	magnetisationBuffer := getMagnetisationBuffer()
	m0 := *magnetisationBuffer

	*magnetisationBuffer = s

	en.SetDemagField(dst)

	*magnetisationBuffer = m0

}

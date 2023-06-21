package field

import (
	"github.com/mumax/3/data"
	en "github.com/mumax/3/engine"
)

func AddExchangeField(s, dst *data.Slice) {

	magnetisationBuffer := getMagnetisationBuffer()
	m0 := *magnetisationBuffer
	*magnetisationBuffer = s

	en.AddExchangeField(dst)

	*magnetisationBuffer = m0

}

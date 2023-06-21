package tests

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/will-henderson/mumax-vhf/setup"
)

func Load() []setup.SystemParameters {

	jsonFile, err := os.Open("users.json")
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var tests []setup.SystemParameters
	json.Unmarshal(byteValue, &tests)

	return tests
}

func Name(s setup.SystemParameters) string {
	return fmt.Sprint(s)
}

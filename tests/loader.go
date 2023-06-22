// Package tests contains utilities for tests in the module.
package tests

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/will-henderson/mumax-vhf/setup"
)

// Load returns an array containing the test case system parameters stored in testcases.json
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

// Name returns a string representation of SystemParameters
func Name(s setup.SystemParameters) string {
	return fmt.Sprint(s)
}

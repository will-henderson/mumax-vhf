// Package tests contains utilities for tests in the module.
package tests

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"

	. "github.com/will-henderson/mumax-vhf/data"
)

// Load returns an array containing the test case system parameters stored in testcases.json
func Load() []SystemParameters {

	jsonFile, err := os.Open("../tests/testcases.json") //should use a proper path joining here
	if err != nil {
		fmt.Println(err)
	}
	defer jsonFile.Close()

	byteValue, _ := ioutil.ReadAll(jsonFile)

	var tests []SystemParameters
	json.Unmarshal(byteValue, &tests)

	return tests
}

// Name returns a string representation of SystemParameters
func Name(s SystemParameters) string {
	return fmt.Sprint(s)
}

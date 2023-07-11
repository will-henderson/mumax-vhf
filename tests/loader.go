// Package tests contains utilities for tests in the module.
package tests

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
)

// Load returns an array containing the test case system parameters stored in testcases.json
func Load() []string {

	wd, err := os.Getwd()
	if err != nil {
		fmt.Println(err)
	}
	directory := path.Join(path.Dir(wd), "tests", "testcases")
	files, err := ioutil.ReadDir(directory)
	if err != nil {
		fmt.Println(err)
	}

	var fileStrings []string
	for _, file := range files {
		bytes, err := os.ReadFile(path.Join(directory, file.Name()))
		if err != nil {
			fmt.Println(err)
		}
		fileStrings = append(fileStrings, string(bytes))
	}

	return fileStrings
}

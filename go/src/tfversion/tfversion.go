package main

import (
    tf "github.com/tensorflow/tensorflow/tensorflow/go"
    "fmt"
)

func main() {
    fmt.Printf("TensorFlow version %s\n", tf.Version())
}

package main

// NB! I cannot get this to work. I'm storing it in git for future reference
// State is messy and experimental
// Example. https://github.com/tensorflow/tensorflow/issues/20511

import (
	"flag"
	"fmt"
	_ "image/png"
	"io/ioutil"
	"log"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func main() {
	imagefile := flag.String("image", "", "Path of a JPEG-image to extract label for")
	flag.Parse()
	if *imagefile == "" {
		flag.Usage()
		return
	}

	model, err := tf.LoadSavedModel("cc-predictor-model", []string{"serve"}, nil)
	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		return
	}

	defer model.Session.Close()

	/*
		tensor, terr := dummyInputTensor(128 * 128 * 3) // replace this with your own data
		if terr != nil {
			fmt.Printf("Error creating input tensor: %s\n", terr.Error())
			return
		}
	*/

	tensorr, _ := tf.NewTensor([1][9]float32{})
	if err != nil {
		log.Fatal(err)
	}

	tensor, err := makeTensorFromImage(*imagefile)
	if err != nil {
		log.Fatal(err)
	}

	result, runErr := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("x").Output(0):      tensor,
			model.Graph.Operation("y_true").Output(0): tensorr,
			//model.Graph.Operation("keep_prob").Output(0): keepProb,
		},
		[]tf.Output{
			model.Graph.Operation("infer").Output(0),
		},
		nil,
	)

	if runErr != nil {
		fmt.Printf("Error running the session with input, err: %s\n", runErr.Error())
		return
	}

	fmt.Printf("Most likely number in input is %v \n", result[0].Value())

	//probabilities := result[0].Value().([]int64)[0]
	//fmt.Printf("Probablilities %v \n", probabilities)
}

func dummyInputTensor(size int) (*tf.Tensor, error) {

	imageData := [][]float32{make([]float32, size)}
	return tf.NewTensor(imageData)
}

// Convert the image in filename to a Tensor suitable as input to the cc-classifier model.
func makeTensorFromImage(filename string) (*tf.Tensor, error) {
	bytes, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	// DecodeJpeg uses a scalar String-valued tensor as input.
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}
	// Construct a graph to normalize the image
	graph, input, output, err := constructGraphToNormalizeImage()
	if err != nil {
		return nil, err
	}

	// Execute that graph to normalize this one image
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()

	normalized, err := session.Run(
		map[tf.Output]*tf.Tensor{input: tensor},
		[]tf.Output{output},
		nil)
	if err != nil {
		return nil, err
	}

	return normalized[0], nil

}

func decodeJpegGraph() (graph *tf.Graph, input, output tf.Output, err error) {
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	output = op.ExpandDims(s,
		op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)),
		op.Const(s.SubScope("make_batch"), int32(0)))
	graph, err = s.Finalize()
	return graph, input, output, err
}

// The inception model takes as input the image described by a Tensor in a very
// specific normalized format (a particular image size, shape of the input tensor,
// normalized pixel values etc.).
//
// This function constructs a graph of TensorFlow operations which takes as
// input a JPEG-encoded string and returns a tensor suitable as input to the
// inception model.
func constructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {
	// Some constants specific to the pre-trained model at:
	// https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
	//
	// - The model was trained after with images scaled to 224x224 pixels.
	// - The colors, represented as R, G, B in 1-byte each were converted to
	//   float using (value - Mean)/Scale.
	const (
		H, W  = 128, 128
		Mean  = float32(0.0)
		Scale = float32(255.0)
	)
	// - input is a String-Tensor, where the string the JPEG-encoded image.
	// - The inception model takes a 4D tensor of shape
	//   [BatchSize, Height, Width, Colors=3], where each pixel is
	//   represented as a triplet of floats
	// - Apply normalization on each pixel and use ExpandDims to make
	//   this single image be a "batch" of size 1 for ResizeBilinear.
	s := op.NewScope()
	input = op.Placeholder(s, tf.String)
	decode := op.DecodeJpeg(s, input, op.DecodeJpegChannels(3))

	output = op.Div(s,
		op.Sub(s,
			op.ResizeBilinear(s,
				op.ExpandDims(s,
					op.Cast(s, decode, tf.Float),
					op.Const(s.SubScope("make_batch"), int32(0))),
				op.Const(s.SubScope("size"), []int32{H, W})),
			op.Const(s.SubScope("mean"), Mean)),
		op.Const(s.SubScope("scale"), Scale))
	graph, err = s.Finalize()

	return graph, input, output, err
}

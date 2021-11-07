package predict

import (
	"flag"
	"fmt"
	_ "image/png"
	"io/ioutil"
	"log"
	"os"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func init() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "1")
	os.Setenv("KMP_AFFINITY", "noverbose")
}

func argmax([]float32) {

}

func Predict() {
	imagefile := flag.String("image", "", "Path of a JPEG-image to extract label for")
	flag.Parse()
	if *imagefile == "" {
		flag.Usage()
		os.Exit(66)
	}
	predict(*imagefile)

}

func predict(imagefile string) {

	model, err := tf.LoadSavedModel("/lustre/storeB/users/espenm/cc-classifier/checkpoints/saved_model_145.pb", []string{"serve"}, nil)
	if err != nil {
		fmt.Printf("Error loading saved model: %s\n", err.Error())
		os.Exit(65)
	}

	/*
		ops := model.Graph.Operations()
		for _, op := range ops {
			fmt.Println(op.Name())
		}
	*/

	defer model.Session.Close()

	tensor, err := makeTensorFromImage(imagefile)
	if err != nil {
		log.Fatal(err)
	}

	result, runErr := model.Session.Run(
		map[tf.Output]*tf.Tensor{
			model.Graph.Operation("serving_default_conv2d_input").Output(0): tensor,

			//model.Graph.Operation("keep_prob").Output(0): keepProb,
		},
		[]tf.Output{
			model.Graph.Operation("StatefulPartitionedCall").Output(0),
		},
		nil,
	)

	if runErr != nil {
		fmt.Printf("Error running the session with input, err: %s\n", runErr.Error())
		os.Exit(64)
	}
	fmt.Printf("********\n")
	arr32 := result[0].Value().([][]float32)[0]
	//_ = dill
	argmax(arr32)
	fmt.Printf("Result: %v\n", result[0].Value().([][]float32))
	//fmt.Printf("Result: %v\n", dill)

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

// The inception model takes as input the image described by a Tensor in a very
// specific normalized format (a particular image size, shape of the input tensor,
// normalized pixel values etc.).
//
// This function constructs a graph of TensorFlow operations which takes as
// input a JPEG-encoded string and returns a tensor suitable as input to the
// inception model.
func constructGraphToNormalizeImage() (graph *tf.Graph, input, output tf.Output, err error) {

	// - The model was trained after with images scaled to 128x128 pixels.
	// - The colors, represented as R, G, B in 1-byte each were converted to
	//   float using (value - Mean)/Scale.
	const (
		H, W  = 128, 128
		Mean  = float32(0.0)
		Scale = float32(255)
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

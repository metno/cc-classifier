package main

// Run predictions concurrently.

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strconv"

	"github.com/metno/go-tf-cnn-predict/predict"
)

/*
Use a datadir organized like this
├── test
│   ├── 0
│   ├── 1
│   ├── 2
│   ├── 3
│   ├── 4
│   ├── 5
│   ├── 6
│   ├── 7
│   └── 8

*/

type labelPath struct {
	Path       string
	Label      int
	Prediction int
	Stddev     float32
}

func readLabelsAndPaths(testdir string) ([]labelPath, error) {
	items := []labelPath{}

	entries, err := os.ReadDir(testdir)
	if err != nil {
		log.Fatal(err)
	}

	for _, e := range entries {
		files, err := os.ReadDir(testdir + "/" + e.Name())
		if err != nil {
			log.Fatal(err)
		}
		label, err := strconv.Atoi(e.Name())
		if err != nil {
			log.Printf("%v", err)
			continue
		}
		for _, f := range files {
			path := fmt.Sprintf("%s/%d/%s", testdir, label, f.Name())

			item1 := labelPath{Path: path, Label: label, Prediction: -1}
			items = append(items, item1)
		}
	}
	return items, nil
}

func predictImage(imagepath string, modeldir string, predictor predict.Predictor) (int, float32, error) {

	pred, dev, err := predictor.PredictWithDeviation(imagepath, predictor.Model)
	if err != nil {
		return -1, -1, err
	}
	return pred, dev, err
}

// Here's the worker, of which we'll run several
// concurrent instances. These workers will receive
// work on the `jobs` channel and send the corresponding
// results on `results`.
func worker(id int, jobs <-chan labelPath, modelPath string, results chan<- labelPath, predictor predict.Predictor) {

	for j := range jobs {

		pred, dev, err := predictImage(j.Path, modelPath, predictor)
		if err != nil {
			log.Printf("error: %v", err)
			continue
		}

		j.Prediction = pred
		j.Stddev = dev
		results <- j
	}
}

func main() {

	testDatadir := flag.String("testdatadir", "/lustre/storeB/users/espenm/data/v2.0.3/test", "Path to the test data directory")
	modelPath := flag.String("modelpath", "/lustre/storeB/project/metproduction/static_data/camsatrec/models/v2.0.3-tensorflow2.x+keras/saved_model_057.pb", "Path to the model")
	//epoch := flag.String("epoch", "", "Model snapshot")
	flag.Parse()
	if *testDatadir == "" || *modelPath == "" {
		flag.Usage()
		return
	}

	labelPaths, err := readLabelsAndPaths(*testDatadir)
	if err != nil {
		log.Fatal(err)
	}

	//cpus := runtime.NumCPU()
	cpus := 8
	runtime.GOMAXPROCS(cpus)

	predictor, err := predict.NewPredictor(*modelPath)
	if err != nil {
		log.Printf("NewPredictor: %v", err)
		os.Exit(68)

	}

	model := predictor.Model
	defer model.Session.Close()

	// In order to use our pool of workers we need to send
	// them work and collect their results. We make 2
	// channels for this.
	jobs := make(chan labelPath, len(labelPaths))
	results := make(chan labelPath, len(labelPaths))

	// This starts up cpus workers, initially blocked
	// because there are no jobs yet.
	for w := 0; w < cpus; w++ {
		go worker(w, jobs, *modelPath, results, predictor)
	}

	// Here we send `len(labelPaths)` jobs and then `close` that
	// channel to indicate that's all the work we have.
	for j := 0; j < len(labelPaths); j++ {
		jobs <- labelPaths[j]
	}
	close(jobs)

	// Finally we collect all the results of the work.
	for a := 0; a < len(labelPaths); a++ {
		res := <-results
		fmt.Printf("%s %d %d %0.3f\n", res.Path, res.Label, res.Prediction, res.Stddev)
	}
}

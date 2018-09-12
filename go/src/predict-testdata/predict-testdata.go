package main

import (
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strconv"
	"strings"
)

var imagedir = "/lustre/storeB/project/metproduction/products/webcams"

type labelPath struct {
	Path       string
	Label      int
	Prediction int
}

func parseInt(value string) int {
	if len(value) == 0 {
		return 0
	}
	i, err := strconv.Atoi(value)
	if err != nil {
		return -2
	}
	return i
}

// /home/espenm/space/projects/cc-classifier/predict.py --modeldir /home/espenm/space/projects/models/v24_9999 --epoch 831 --filename /lustre/storeB/project/metproduction/products/webcams/2018/06/14/13/13_20180614T2200Z.jpg  2>/dev/null
func execCommand(command string, arg ...string) (string, string, error) {
	cmd := exec.Command(command, arg...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	outStr, errStr := string(stdout.Bytes()), string(stderr.Bytes())
	return outStr, errStr, err

}

func readLabels(path string) ([]labelPath, error) {
	items := []labelPath{}

	//CAMID_YYYYMMDDThhmmZ.jpg
	r := regexp.MustCompile(`(\d+)_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})Z\.jpg (\d)`)

	file, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		line := scanner.Text()
		matches := r.FindStringSubmatch(line)
		if len(matches) != 8 {
			fmt.Printf("Could not parse input %s\n", line)
			return nil, fmt.Errorf("Could not parse input %s", line)
		}

		camid := parseInt(matches[1])
		year := parseInt(matches[2])
		month := parseInt(matches[3])
		day := parseInt(matches[4])
		hour := parseInt(matches[5])
		minute := parseInt(matches[6])
		cclabel := parseInt(matches[7])

		path := fmt.Sprintf("%s/%0.4d/%0.2d/%0.2d/%d/%d_%0.4d%0.2d%0.2dT%0.2d%0.2dZ.jpg", imagedir, year, month, day,
			camid, camid, year, month, day,
			hour, minute)

		item1 := labelPath{Path: path, Label: cclabel}
		items = append(items, item1)

	}

	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return items, nil
}

// Here's the worker, of which we'll run several
// concurrent instances. These workers will receive
// work on the `jobs` channel and send the corresponding
// results on `results`. We'll sleep a second per job to
// simulate an expensive task.
func worker(id int, jobs <-chan labelPath, prog string, modeldir string, epoch string, results chan<- labelPath) {
	count := 0
	for j := range jobs {

		count++
		//fmt.Println("worker", id, "started  job", j, "Path:", j.Path)
		outStr, errStr, err := execCommand(prog,
			"--modeldir", modeldir,
			"--epoch", epoch,
			"--filename", j.Path)

		if err != nil {
			fmt.Printf("Failed: %v\n", err)
			fmt.Printf("Failed: %s\n", errStr)
		}
		//fmt.Println(parseInt(strings.TrimRight(outStr, "\n")))
		//fmt.Println("worker", id, "finished job", j)
		j.Prediction = parseInt(strings.TrimRight(outStr, "\n"))
		results <- j
	}
}

func main() {
	predictionProg := flag.String("predictscript", "", "Path to the prediction script")
	labelFile := flag.String("labelfile", "", "Path to the labelsfile")
	modelDir := flag.String("modeldir", "", "Path to the modeldir")
	epoch := flag.String("epoch", "", "Model snapshot")
	flag.Parse()
	if *predictionProg == "" || *labelFile == "" || *modelDir == "" || *epoch == "" {
		flag.Usage()
		return
	}

	cpus := runtime.NumCPU()
	runtime.GOMAXPROCS(cpus)

	labelPaths, err := readLabels(*labelFile)
	if err != nil {
		log.Fatal(err)
	}
	_ = labelPaths
	//fmt.Printf("%v\n", labelPaths)
	//return
	// In order to use our pool of workers we need to send
	// them work and collect their results. We make 2
	// channels for this.
	jobs := make(chan labelPath, len(labelPaths))
	results := make(chan labelPath, len(labelPaths))

	// This starts up cpus workers, initially blocked
	// because there are no jobs yet.
	for w := 0; w < cpus; w++ {
		go worker(w, jobs, *predictionProg, *modelDir, *epoch, results)
	}

	// Here we send cpus `jobs` and then `close` that
	// channel to indicate that's all the work we have.
	for j := 0; j < len(labelPaths); j++ {

		jobs <- labelPaths[j]
	}
	close(jobs)

	// Finally we collect all the results of the work.
	for a := 0; a < len(labelPaths); a++ {
		res := <-results
		fmt.Printf("%s %d %d\n", res.Path, res.Label, res.Prediction)
	}

}

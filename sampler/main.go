package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
)

type InputParameters struct {
	dataset_path   string
	sampling_rate  string
	recursive_step string
	seed           int64
	limit          string
	prompt_name    string
	maxDrift       int
	upscale        string
	result_path    string
	url            string
}

func InitializeFrontEnd(inputParameters *InputParameters) (interface{}, map[string]interface{}, error) {
	datasetPath := inputParameters.dataset_path
	samplingRate := inputParameters.sampling_rate
	recursiveStep := inputParameters.recursive_step
	seed := inputParameters.seed
	datasetLimit := inputParameters.limit
	upScale := inputParameters.upscale
	promptName := inputParameters.prompt_name
	FrontendInstance := NewFrontend(upScale)

	tracedata, commonHeader, err := FrontendInstance.Preprocessor(datasetPath, seed, datasetLimit, upScale, samplingRate, recursiveStep, promptName)
	if err != nil {
		log.Fatal(err)
	}
	return tracedata, commonHeader, err
}

// Example use: ./LLMLoadgen -dataset_path	 datasets/test_azure.json -sampling_rate 100 -recur_step 10 -upscale ars -url http://localhost:8000/v1/completions -limit max -max_drift 100 -result_path results/result.json
func parser() (*InputParameters, error) {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage of %s:\n", os.Args[0])
		flag.PrintDefaults()
	}

	var help bool

	// Define flags
	var _dataset_path string
	var _sampling_rate string
	var _recursive_step string
	var _limit string
	var _maxDrift int
	var _seed int64
	var _prompt_name string
	var _upscale string
	var _result_path string
	var _url string

	flag.StringVar(&_dataset_path, "dataset_path", "datasets/test_azure.json", `Specify the path of the dataset`)
	flag.StringVar(&_sampling_rate, "sampling_rate", "100", `Specify the sampling rate, support from 0 to 1000, "0-100" is downsampling and "100-1000" is upsampling`)
	flag.StringVar(&_recursive_step, "recur_step", "10", `Specify the recursive step, support from 0 to 100`)
	flag.Int64Var(&_seed, "seed", 42, "The seed for the random number generator")
	flag.StringVar(&_upscale, "upscale", "ars", "Indicates the up scaling method, support 'ars' and 'trace'")
	flag.StringVar(&_url, "url", "http://localhost:8000/v1/completions", "Specify the url for liquid controller")
	flag.StringVar(&_limit, "limit", "max", "The maximum length limit of dataset")
	flag.StringVar(&_prompt_name, "prompt_name", "prompt", "The attribute for target controller downsampling in wasserstein distance, containing the prompt context is named `prompt` for Liquid and OpenAI. If no argument is passed, it defaults to None, resulting in random downsampling.")
	flag.IntVar(&_maxDrift, "max_drift", 100, "The maximum drift time for request sending, an exceed will cause generator error")
	flag.StringVar(&_result_path, "result_path", "results/result.json", "The path to save the result")
	flag.BoolVar(&help, "help", false, "this is the load generator part of Liquid LLM and amun-alloc project")

	flag.Parse()

	// Custom help handling
	if flag.NFlag() == 0 || flag.Lookup("help").Value.String() != "false" {
		flag.Usage()
		return nil, fmt.Errorf("help mode")
	}

	// Additional logic based on input parameters
	fmt.Printf("Dataset Path: %s\n", _dataset_path)

	inputParameters := InputParameters{
		dataset_path:   _dataset_path,
		sampling_rate:  _sampling_rate,
		recursive_step: _recursive_step,
		limit:          _limit,
		seed:           _seed,
		maxDrift:       _maxDrift,
		prompt_name:    _prompt_name,
		upscale:        _upscale,
		result_path:    _result_path,
		url:            _url,
	}
	return &inputParameters, nil

}

func main() {
	//parse input parameters
	inputParameters, ParserErr := parser()
	if ParserErr != nil || inputParameters == nil {
		log.Printf("Error: %v", ParserErr)
		return
	}
	tracedata, _, ProcessErr := InitializeFrontEnd(inputParameters)
	if ProcessErr != nil {
		log.Printf("Error: Reading and Processing")
	}
	//send the request array and delay array to the generator
	log.Printf("start generation")
	tracedata = tracedata.([]GlobalIDEntry)
	jsonData, err := json.MarshalIndent(tracedata, "", " ")
	if err != nil {
		log.Printf("Error marshling JSON: %v", err)
		return
	}

	err = os.WriteFile(inputParameters.result_path, jsonData, 0644)
	if err != nil {
		log.Printf("Error writing to file: %v", err)
		return
	}
	log.Printf("JSON written to: %v", inputParameters.result_path)
}

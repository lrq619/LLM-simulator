package main

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"
	"sync"
	"time"
)

type Frontend struct {
	upScale string
}

type TraceEntry struct {
	Timestamp  string                 `json:"timestamp"`
	Data       map[string]interface{} `json:"data"`
	Attributes map[string]interface{} `json:"attributes"`
}

type ProcessedEntry struct {
	TimeDelta  int                    `json:"timestamp"`
	Data       map[string]interface{} `json:"data"`
	Attributes map[string]interface{} `json:"attributes"`
}

type GlobalIDEntry struct {
	GlobalID int
	Request  ProcessedEntry
}

type Preprocessor struct {
	maximumLimit     string
	datasetPath      string
	samplingRate     int
	recursiveStep    int
	upscale          string
	promptName       string
	localDatasetPath string
	traceData        interface{} // Will store the loaded data
	commonHeader     map[string]interface{}
}

func NewFrontend(_upScale string) *Frontend {
	return &Frontend{
		upScale: _upScale,
	}
}

func (f *Frontend) Preprocessor(_datasetPath string, _seed int64, _datasetLimit string, _upscale string, _samplingRate string, _recursiveStep string, _promptName string) (interface{}, map[string]interface{}, error) {
	config := map[string]interface{}{
		"limit":          _datasetLimit,
		"dataset_path":   _datasetPath,
		"sampling_rate":  _samplingRate,
		"recursive_step": _recursiveStep,
		"upscale":        _upscale,
		"prompt_name":    _promptName,
		"seed":           _seed,
	}
	PreprocessorInstance := NewPreprocessor(config)
	err := PreprocessorInstance.Load_raw_trace(_seed)
	if err != nil {
		log.Fatal(err)
		panic("Error happened when preprocessing dataset! See above log for information.")
	}
	tracedata := PreprocessorInstance.traceData
	commonHeader := PreprocessorInstance.commonHeader
	return tracedata, commonHeader, nil
}

func NewPreprocessor(config map[string]interface{}) *Preprocessor {
	p := &Preprocessor{}

	// Set configuration values
	if limit, ok := config["limit"].(string); ok {
		p.maximumLimit = limit
	} else {
		p.maximumLimit = "max"
	}

	if path, ok := config["dataset_path"].(string); ok {
		p.datasetPath = path
	} else {
		p.datasetPath = "datasets/test_azure.json"
	}

	if samplingRate, ok := config["sampling_rate"].(string); ok {
		p.samplingRate, _ = strconv.Atoi(samplingRate)
	} else {
		p.samplingRate = 100
	}

	if recursiveStep, ok := config["recursive_step"].(string); ok {
		p.recursiveStep, _ = strconv.Atoi(recursiveStep)
	} else {
		p.recursiveStep = 10
	}

	if promptName, ok := config["prompt_name"].(string); ok {
		p.promptName = promptName
	} else {
		p.promptName = "prompt"
	}

	if upscale, ok := config["upscale"].(string); ok {
		p.upscale = upscale
	}

	p.localDatasetPath = p.datasetPath
	return p
}

func (p *Preprocessor) SampleDownDataset(dataframe []GlobalIDEntry, seed int64) ([]GlobalIDEntry, error) {
	trace_log, err := NewTraceManager(dataframe, seed, p.promptName)
	if err != nil {
		return nil, fmt.Errorf("failed to create trace log: %w", err)
	}

	sample_rate := float64(p.samplingRate) / 100
	recursiveStep := p.recursiveStep
	dataframe = trace_log.YieldTargetSample(sample_rate, recursiveStep)
	if p.maximumLimit != "max" {
		limit, _ := strconv.Atoi(p.maximumLimit)
		if len(dataframe) < limit {
			log.Printf("The total sample size must be greater than or equal to limit")
			return nil, fmt.Errorf("the total sample size must be greater than or equal to limit")
		}
	}
	return dataframe, nil
}

func (p *Preprocessor) SampleUpDataset(dataframe []GlobalIDEntry, seed int64) ([]GlobalIDEntry, error) {
	trace_log, err := NewTraceManager(dataframe, seed, p.promptName)
	if err != nil {
		return nil, fmt.Errorf("failed to create trace log: %w", err)
	}

	upscale_factor := float64(p.samplingRate) / 100
	log.Printf("The upscale rate will be %f", upscale_factor)
	if p.upscale == "ars" {
		dataframe = trace_log.UpscaleAvgRate(upscale_factor, seed)
	} else if p.upscale == "trace" {
		dataframe = trace_log.UpscalerTrace(upscale_factor, seed)
	}
	return dataframe, nil
}

func (p *Preprocessor) Load_raw_trace(seed int64) error {
	file, err := os.ReadFile(p.localDatasetPath)
	if err != nil {
		return fmt.Errorf("failed to read file: %w", err)
	}

	var rawData []map[string]interface{}
	if err := json.Unmarshal(file, &rawData); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}
	p.commonHeader = rawData[0]

	entrys := []TraceEntry{}
	if err := json.Unmarshal(file, &entrys); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}
	entrys = entrys[1:]
	initialTime, err := time.Parse("15:04:05.000000", entrys[0].Timestamp)
	if err != nil {
		return fmt.Errorf("failed to parse timestamp: %w", err)
	}
	var wg sync.WaitGroup

	processedData := make([]GlobalIDEntry, len(entrys))

	numWorkers := 4
	chunkSize := (len(entrys) + numWorkers - 1) / numWorkers

	worker := func(entries []TraceEntry, startIdx int) {
		defer wg.Done()
		for i, entry := range entries {
			ts, err := time.Parse("15:04:05.000000", entry.Timestamp)
			if err != nil {
				log.Printf("Skipping invalid timestamp: %s", entry.Timestamp)
				continue
			}
			timeDelta := int(ts.Sub(initialTime).Milliseconds())

			processedData[startIdx+i] = GlobalIDEntry{
				GlobalID: startIdx + i,
				Request: ProcessedEntry{
					TimeDelta:  timeDelta,
					Data:       entry.Data,
					Attributes: entry.Attributes,
				},
			}
		}
	}

	for i := 0; i < numWorkers; i++ {
		start := i * chunkSize
		end := start + chunkSize
		if end > len(entrys) {
			end = len(entrys)
		}
		if start >= len(entrys) {
			break
		}
		wg.Add(1)
		go worker(entrys[start:end], start)
	}

	wg.Wait()

	if p.samplingRate < 100 {
		log.Printf("Sampling down Trace, sample rate will be %d", p.samplingRate)
		processedData, err = p.SampleDownDataset(processedData, seed)
		if err != nil {
			return fmt.Errorf("failed to sample down dataset: %w", err)
		}
	}

	if p.samplingRate > 100 {
		log.Printf("Sampling up Trace by %s and sample rate is %d", p.upscale, p.samplingRate)
		processedData, err = p.SampleUpDataset(processedData, seed)
		log.Println("The number of sampled requests is", len(processedData))
		if err != nil {
			return fmt.Errorf("failed to sample up dataset: %w", err)
		}
	}

	if p.maximumLimit != "max" {
		limit, err := strconv.Atoi(p.maximumLimit)
		if err == nil && limit < len(processedData) {
			processedData = processedData[:limit]
		}
	}

	p.traceData = processedData

	return nil
}

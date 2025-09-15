package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"runtime"
	"sort"
	"sync"

)

type TraceManager struct {
	data           []GlobalIDEntry
	randomSeed     int64
	promptName     string
	promptTokenMap map[int]float64
}

type task struct {
	idx   int
	value interface{}
}

func NewTraceManager(data []GlobalIDEntry, seed int64, promptName string) (*TraceManager, error) {
	t := &TraceManager{
		data:           data,
		promptName:     promptName,
		randomSeed:     seed,
		promptTokenMap: make(map[int]float64),
	}

	t.data = data
	t.promptName = promptName
	t.randomSeed = seed
	if len(t.data) == 0 {
		return nil, fmt.Errorf("data is empty")
	}

	if t.promptName == "None" {
		r := rand.New(rand.NewSource(t.randomSeed))
		for i := range t.data {
			t.promptTokenMap[i] = float64(r.Intn(4000) + 1) // Random token count between 1 and 4000
		}
	} else {
		var err error
		t.promptTokenMap, err = CountPromptTokens(t.data, t.promptName)
		if err != nil {
			return nil, fmt.Errorf("failed to count prompt tokens: %v", err)
		}
	}

	return t, nil
}

func (t *TraceManager) SampleData(data []GlobalIDEntry, size int, seed int64) ([]GlobalIDEntry, error) {
	if size == 0 {
		return nil, fmt.Errorf("sample size is zero")
	}
	if size > len(data) {
		return data, nil
	}

	r := rand.New(rand.NewSource(seed))
	indices := r.Perm(len(data))[:size]
	sampled := make([]GlobalIDEntry, size)
	for i, idx := range indices {
		sampled[i] = data[idx]
	}
	return sampled, nil
}

func computeWassersteinDistance(x, y []float64) float64 {
	if len(x) == 0 || len(y) == 0 {
		return 0.0
	}
	sort.Float64s(x)
	sort.Float64s(y)
	xCDF := make([]float64, len(x))
	yCDF := make([]float64, len(y))

	for i := range x {
		xCDF[i] = float64(i+1) / float64(len(x))
	}
	for i := range y {
		yCDF[i] = float64(i+1) / float64(len(y))
	}

	// Merge x and y to find all unique values
	allValues := append(x, y...)
	sort.Float64s(allValues)

	// Calculate the Wasserstein distance
	var distance float64
	var xIndex, yIndex int
	for i := 1; i < len(allValues); i++ {
		delta := allValues[i] - allValues[i-1]
		for xIndex < len(x) && x[xIndex] <= allValues[i-1] {
			xIndex++
		}
		for yIndex < len(y) && y[yIndex] <= allValues[i-1] {
			yIndex++
		}
		xCDFValue := float64(xIndex) / float64(len(x))
		yCDFValue := float64(yIndex) / float64(len(y))
		distance += math.Abs(xCDFValue-yCDFValue) * delta
	}

	return distance
}

func CountPromptTokens(entries []GlobalIDEntry, promptName string) (map[int]float64, error) {
	numWorkers := runtime.NumCPU()
	taskCh := make(chan task, numWorkers*2)
	result := make(map[int]float64, len(entries))
	errCh := make(chan error, 1)
	var wg sync.WaitGroup
	var mu sync.Mutex

	// start up the worker pool
	for w := 0; w < numWorkers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for t := range taskCh {
				var count float64
				switch v := t.value.(type) {
				case string:
					count = float64(len(v))
				case []interface{}:
					count = float64(len(v))
				default:
					// handle unexpected type
				}
				// count := float64(strings.Count(t.value, " ")) + 1
				mu.Lock()
				result[t.idx] = count
				mu.Unlock()
			}
		}()
	}

	// dispatch tasks
	for i, entry := range entries {
		valueRaw, exists := entry.Request.Data[promptName]
		if !exists {
			errCh <- fmt.Errorf("prompt name '%s' not found in entry at index %d", promptName, i)
			close(taskCh)
			wg.Wait()
			return nil, <-errCh
		}
		switch v := valueRaw.(type) {
		case string:
			taskCh <- task{idx: i, value: v} // v is string
		case []interface{}:
			taskCh <- task{idx: i, value: v} // v is []int
		default:
			errCh <- fmt.Errorf("prompt value at index %d is not string or []int (got %T)", i, valueRaw)
			close(taskCh)
			wg.Wait()
			return nil, <-errCh
		}
	}

	close(taskCh)
	wg.Wait()

	select {
	case err := <-errCh:
		return nil, err
	default:
	}

	return result, nil
}

func (t *TraceManager) ComputeWasserstein(original, sample []int) ([]float64, error) {
	// pass the key in and we should give the value by looking it up in the promptTokenMap
	origPrompt := make([]float64, len(original))
	for i, key := range original {
		origPrompt[i] = t.promptTokenMap[key]
	}

	samplePrompt := make([]float64, len(sample))
	for i, key := range sample {
		samplePrompt[i] = t.promptTokenMap[key]
	}

	distances := []float64{computeWassersteinDistance(origPrompt, samplePrompt)}
	return distances, nil
}

func (t *TraceManager) GetMostRepresentativeSample(candidates [][]GlobalIDEntry, original []GlobalIDEntry) []GlobalIDEntry {
	var bestSample []GlobalIDEntry
	bestDistance := math.Inf(1)
	originalKeys := make([]int, len(original))

	for i, entry := range original {
		originalKeys[i] = entry.GlobalID
	}

	for _, candidate := range candidates {
		candidateKeys := make([]int, len(candidate))
		for i, cand := range candidate {
			candidateKeys[i] = cand.GlobalID
		}
		distances, err := t.ComputeWasserstein(originalKeys, candidateKeys)
		if err != nil {
			log.Fatalf("failed to compute wasserstein distance: %v", err)
		}
		meanDistance := distances[0]
		if meanDistance < bestDistance {
			bestDistance = meanDistance
			bestSample = candidate
		}
	}

	return bestSample
}

func (t *TraceManager) RecursiveRollDownSampling(trialsNum int, maxFraction, minFraction, fractionStep float64) map[float64][]GlobalIDEntry {
	results := make(map[float64][]GlobalIDEntry)
	totalEntries := len(t.data)
	maxSize := int(float64(totalEntries) * maxFraction)
	minSize := int(float64(totalEntries) * minFraction)
	stepSize := int(float64(totalEntries) * fractionStep)

	currentSample, _ := t.SampleData(t.data, maxSize, t.randomSeed)
	results[maxFraction] = currentSample
	previousSample := currentSample

	for size := maxSize - stepSize; size >= minSize; size -= stepSize {
		candidates := make([][]GlobalIDEntry, trialsNum)
		for i := 0; i < trialsNum; i++ {
			candidates[i], _ = t.SampleData(previousSample, size, int64(i))
		}

		mostRepSample := t.GetMostRepresentativeSample(candidates, previousSample)
		rounded_key := roundToTwoDecimals(float64(size) / float64(totalEntries))
		results[rounded_key] = mostRepSample
		previousSample = mostRepSample
	}

	return results
}

func roundToTwoDecimals(value float64) float64 {
	return math.Round(value*100) / 100
}

func (t *TraceManager) analyzeSamples(samples map[float64][]GlobalIDEntry) map[float64][]float64 {
	original := t.data
	originalKeys := make([]int, len(t.data))
	for i, entry := range original {
		originalKeys[i] = entry.GlobalID
	}

	stats := make(map[float64][]float64)
	for fraction, sample := range samples {
		sampleKeys := make([]int, len(sample))
		for i, samp := range sample {
			sampleKeys[i] = samp.GlobalID
		}
		wDist, err := t.ComputeWasserstein(originalKeys, sampleKeys)
		if err != nil {
			log.Fatalf("failed to compute wasserstein distance: %v", err)
		}

		stats[fraction] = wDist

	}
	return stats
}


func (t *TraceManager) RandomDownSampling(minSize float64) []GlobalIDEntry {
	currentSample, _ := t.SampleData(t.data, int(minSize*float64(len(t.data))), t.randomSeed)
	return currentSample
}

func (t *TraceManager) YieldTargetSample(minSize float64, trialsNum int) []GlobalIDEntry {
	var finalSample []GlobalIDEntry
	if t.promptName == "None" {
		finalSample = t.RandomDownSampling(minSize)
	} else {
		samples := t.RecursiveRollDownSampling(trialsNum, 1.0, minSize, 0.01)

		finalSample = samples[minSize]
	}
	sort.Slice(finalSample, func(i, j int) bool {
		return finalSample[i].Request.TimeDelta < finalSample[j].Request.TimeDelta
	})
	return finalSample
}

func (t *TraceManager) UpscalerTrace(upscaleFactor float64, seed int64) []GlobalIDEntry {
	if upscaleFactor <= 1.00 {
		fmt.Println("Invalid upscale factor!")
		return nil
	}
	data_frame := make([]GlobalIDEntry, len(t.data))
	copy(data_frame, t.data)
	oldTimestamps := make([]int, len(data_frame))

	for i, entry := range data_frame {
		oldTimestamps[i] = entry.Request.TimeDelta
	}

	totalEntries := len(oldTimestamps)
	floorValue := int(math.Floor(upscaleFactor))
	fraction := upscaleFactor - float64(floorValue)

	var newTimestamps []int
	r := rand.New(rand.NewSource(seed))
	for _, idx := range oldTimestamps {
		newCount := floorValue
		if r.Float64() < fraction {
			newCount++
		}
		for i := 0; i < newCount; i++ {
			newTimestamps = append(newTimestamps, idx)
			if len(newTimestamps) >= totalEntries {
				break
			}
		}
	}
	newData := make([]GlobalIDEntry, totalEntries)
	for i, entry := range data_frame {
		newData[i] = GlobalIDEntry{
			Request: ProcessedEntry{
				TimeDelta:  int(newTimestamps[i]),
				Data:       entry.Request.Data,
				Attributes: entry.Request.Attributes,
			},
		}
	}

	return newData
}

func (t *TraceManager) UpscaleAvgRate(upscaleFactor float64, seed int64) []GlobalIDEntry {
	if upscaleFactor <= 1.00 {
		log.Println("invalid upscale factor!")
		return t.data
	}
	// window size 10000ms
	data_frame := make([]GlobalIDEntry, len(t.data))
	copy(data_frame, t.data)
	const dataWindow = 10000
	maxTimestamp := t.getMaxTimestamp()
	lastWindowStart := (maxTimestamp / dataWindow) * dataWindow
	lastWindowEnd := maxTimestamp + 1

	windowGroups := make(map[int][]GlobalIDEntry)
	for _, entry := range data_frame {
		windowStart := (entry.Request.TimeDelta / dataWindow) * dataWindow
		windowGroups[windowStart] = append(windowGroups[windowStart], entry)
	}

	var windowKeys []int
	for k := range windowGroups {
		windowKeys = append(windowKeys, k)
	}
	sort.Ints(windowKeys)

	var newRows []GlobalIDEntry
	r := rand.New(rand.NewSource(seed))
	for _, windowStart := range windowKeys {
		group := windowGroups[windowStart]
		windowSize := dataWindow
		if windowStart == lastWindowStart {
			windowSize = lastWindowEnd - lastWindowStart
		}

		numNewReq := int(math.Round(float64(len(group)) * upscaleFactor))
		if numNewReq == 0 {
			continue
		}

		arrivalIntervals := make([]float64, numNewReq)
		for i := 0; i < numNewReq; i++ {
			arrivalIntervals[i] = r.ExpFloat64() * float64(dataWindow) / float64(numNewReq)
		}

		sort.Float64s(arrivalIntervals)
		timestamps := make([]int, numNewReq)
		cumsum := 0.0
		for i, interval := range arrivalIntervals {
			cumsum += interval
			timestamps[i] = windowStart + int(math.Round(cumsum))
			if timestamps[i] >= (windowStart + windowSize) {
				numNewReq = i
				break
			}
		}
		if numNewReq == 0 {
			continue
		}

		sampledRequests := sampleEntries(r, group, numNewReq)
		for i := 0; i < numNewReq; i++ {
			sampledRequests[i].Request.TimeDelta = timestamps[i]
			newRows = append(newRows, sampledRequests[i])
		}
	}

	sort.Slice(newRows, func(i, j int) bool {
		return newRows[i].Request.TimeDelta < newRows[j].Request.TimeDelta
	})
	fmt.Println("newRows len:", len(newRows))
	return newRows
}

func (t *TraceManager) getMaxTimestamp() int {
	maxTimestamp := 0
	for _, entry := range t.data {
		if entry.Request.TimeDelta > maxTimestamp {
			maxTimestamp = entry.Request.TimeDelta
		}
	}
	return maxTimestamp
}

func sampleEntries(r *rand.Rand, entries []GlobalIDEntry, n int) []GlobalIDEntry {
	sampled := make([]GlobalIDEntry, n)
	for i := 0; i < n; i++ {
		idx := r.Intn(len(entries))
		sampled[i] = entries[idx]
	}
	return sampled
}
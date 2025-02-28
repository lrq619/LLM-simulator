package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"
	"sort"

	"strings"

)

type TraceManager struct {
	data       []ProcessedEntry
	randomSeed int64
	promptName string
}

func NewTraceManager(data []ProcessedEntry, seed int64, promptName string) (*TraceManager, error) {
	t := &TraceManager{
		data:       data,
		promptName: promptName,
		randomSeed: seed,
	}

	t.data = data
	t.promptName = promptName
	t.randomSeed = seed
	if len(t.data) == 0 {
		return nil, fmt.Errorf("data is empty")
	}

	return t, nil

}

func (t *TraceManager) SampleData(data []ProcessedEntry, size int) ([]ProcessedEntry, error) {
	if size == 0 {
		return nil, fmt.Errorf("sample size is zero")
	}
	if size > len(data) {
		return data, nil
	}

	r := rand.New(rand.NewSource(t.randomSeed))
	indices := r.Perm(len(data))[:size]
	sampled := make([]ProcessedEntry, size)
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

func (t *TraceManager) ComputeWasserstein(original, sample []ProcessedEntry) ([]float64, error) {
	origPrompt := make([]float64, len(original))
	samplePrompt := make([]float64, len(sample))
	promptName := t.promptName

	for i, entry := range original {
		value, exists := entry.Data[promptName]
		if !exists {
			return nil, fmt.Errorf("prompt name '%s' not found in original entry", promptName)
		}
		prompt := strings.Fields(value.(string))
		origPrompt[i] = float64(len(prompt) + 1)
	}

	for i, entry := range sample {
		value, exists := entry.Data[promptName]
		if !exists {
			return nil, fmt.Errorf("prompt name '%s' not found in sample entry", promptName)
		}
		prompt := strings.Fields(value.(string))
		samplePrompt[i] = float64(len(prompt) + 1)
	}

	distances := []float64{computeWassersteinDistance(origPrompt, samplePrompt)}

	return distances, nil
}

func (t *TraceManager) GetMostRepresentativeSample(candidates [][]ProcessedEntry, original []ProcessedEntry) []ProcessedEntry {
	var bestSample []ProcessedEntry
	bestDistance := math.Inf(1)

	for _, candidate := range candidates {
		distances, err := t.ComputeWasserstein(original, candidate)
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

func (t *TraceManager) RecursiveRollDownSampling(trialsNum int, maxFraction, minFraction, fractionStep float64) map[float64][]ProcessedEntry {
	results := make(map[float64][]ProcessedEntry)
	totalEntries := len(t.data)
	maxSize := int(float64(totalEntries) * maxFraction)
	minSize := int(float64(totalEntries) * minFraction)
	stepSize := int(float64(totalEntries) * fractionStep)

	currentSample, _ := t.SampleData(t.data, maxSize)
	results[maxFraction] = currentSample
	previousSample := currentSample

	for size := maxSize - stepSize; size >= minSize; size -= stepSize {
		candidates := make([][]ProcessedEntry, trialsNum)
		for i := 0; i < trialsNum; i++ {
			candidates[i], _ = t.SampleData(previousSample, size)
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

func (t *TraceManager) analyzeSamples(samples map[float64][]ProcessedEntry) map[float64][]float64 {
	original := t.data
	stats := make(map[float64][]float64)
	for fraction, sample := range samples {
		wDist, err := t.ComputeWasserstein(original, sample)
		if err != nil {
			log.Fatalf("failed to compute wasserstein distance: %v", err)
		}

		stats[fraction] = wDist

	}
	return stats
}

// func (t *TraceManager) plotWasserstein(stats map[float64][]float64) {
// 	var fractions []float64
// 	for fraction := range stats {
// 		fractions = append(fractions, fraction)
// 	}
// 	sort.Float64s(fractions)

// 	contextTokens := make(plotter.XYs, len(fractions))

// 	for i, fraction := range fractions {
// 		contextTokens[i].X = fraction
// 		contextTokens[i].Y = stats[fraction][0]
// 	}

// 	p := plot.New()
// 	p.Title.Text = "Wasserstein Distance Across Sample Fractions for Tokens"
// 	p.X.Label.Text = "Sample Fraction"
// 	p.Y.Label.Text = "Wasserstein Distance"

// 	lineContext, err := plotter.NewLine(contextTokens)
// 	if err != nil {
// 		log.Fatalf("failed to create ContextTokens line: %v", err)
// 	}
// 	lineContext.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}

// 	scatterContext, err := plotter.NewScatter(contextTokens)
// 	if err != nil {
// 		log.Fatalf("failed to create ContextTokens scatter: %v", err)
// 	}
// 	scatterContext.GlyphStyle.Color = color.RGBA{R: 255, G: 0, B: 255, A: 255}
// 	scatterContext.GlyphStyle.Radius = vg.Points(5)
// 	scatterContext.GlyphStyle.Shape = draw.CircleGlyph{}

// 	p.Add(lineContext, scatterContext)

// 	p.Legend.Add("ContextTokens", lineContext)

// 	plotPath := "./plots/wasserstein_tokens.png"
// 	if err := p.Save(12*vg.Inch, 6*vg.Inch, plotPath); err != nil {
// 		log.Fatalf("failed to save image: %v", err)
// 	}
// 	fmt.Println("image saved to:", plotPath)
// }

func (t *TraceManager) RandomDownSampling(minSize float64) []ProcessedEntry {
	currentSample, _ := t.SampleData(t.data, int(minSize*float64(len(t.data))))
	return currentSample
}

func (t *TraceManager) YieldTargetSample(minSize float64, trialsNum int) []ProcessedEntry {
	var finalSample []ProcessedEntry
	if t.promptName == "None" {
		finalSample = t.RandomDownSampling(minSize)
	} else {
		samples := t.RecursiveRollDownSampling(trialsNum, 1.0, minSize, 0.01)

		// stats := t.analyzeSamples(samples)
		// t.plotWasserstein(stats)
		finalSample = samples[minSize]
	}
	sort.Slice(finalSample, func(i, j int) bool {
		return finalSample[i].TimeDelta < finalSample[j].TimeDelta
	})
	return finalSample
}

func (t *TraceManager) UpscalerTrace(upscaleFactor float64) []ProcessedEntry {
	if upscaleFactor <= 1.00 {
		fmt.Println("Invalid upscale factor!")
		return nil
	}
	data_frame := make([]ProcessedEntry, len(t.data))
	copy(data_frame, t.data)
	oldTimestamps := make([]int, len(data_frame))

	for i, entry := range data_frame {
		oldTimestamps[i] = entry.TimeDelta
	}

	totalEntries := len(oldTimestamps)
	floorValue := int(math.Floor(upscaleFactor))
	fraction := upscaleFactor - float64(floorValue)

	var newTimestamps []int
	for _, idx := range oldTimestamps {
		newCount := floorValue
		if rand.Float64() < fraction {
			newCount++
		}
		for i := 0; i < newCount; i++ {
			newTimestamps = append(newTimestamps, idx)
			if len(newTimestamps) >= totalEntries {
				break
			}
		}
	}
	newData := make([]ProcessedEntry, totalEntries)
	for i, entry := range data_frame {
		newData[i] = ProcessedEntry{
			TimeDelta: int(newTimestamps[i]),
			Data:      entry.Data,
		}
	}

	return newData
}

func (t *TraceManager) UpscaleAvgRate(upscaleFactor float64) []ProcessedEntry {
	if upscaleFactor <= 1.00 {
		log.Println("invalid upscale factor!")
		return t.data
	}
	// window size 10000ms
	data_frame := make([]ProcessedEntry, len(t.data))
	copy(data_frame, t.data)
	const dataWindow = 10000
	maxTimestamp := t.getMaxTimestamp()
	lastWindowStart := (maxTimestamp / dataWindow) * dataWindow
	lastWindowEnd := maxTimestamp + 1

	windowGroups := make(map[int][]ProcessedEntry)
	for _, entry := range data_frame {
		windowStart := (entry.TimeDelta / dataWindow) * dataWindow
		windowGroups[windowStart] = append(windowGroups[windowStart], entry)
	}
	var newRows []ProcessedEntry

	for windowStart, group := range windowGroups {
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
			arrivalIntervals[i] = rand.ExpFloat64() * float64(dataWindow) / float64(numNewReq)
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

		sampledRequests := sampleEntries(group, numNewReq)
		for i := 0; i < numNewReq; i++ {
			sampledRequests[i].TimeDelta = timestamps[i]
			newRows = append(newRows, sampledRequests[i])
		}
	}
	// sort newRows by timeDelta
	sort.Slice(newRows, func(i, j int) bool {
		return newRows[i].TimeDelta < newRows[j].TimeDelta
	})
	fmt.Println("newRows len:", len(newRows))
	return newRows
}

func (t *TraceManager) getMaxTimestamp() int {
	maxTimestamp := 0
	for _, entry := range t.data {
		if entry.TimeDelta > maxTimestamp {
			maxTimestamp = entry.TimeDelta
		}
	}
	return maxTimestamp
}

func sampleEntries(entries []ProcessedEntry, n int) []ProcessedEntry {
	sampled := make([]ProcessedEntry, n)
	for i := 0; i < n; i++ {
		idx := rand.Intn(len(entries))
		sampled[i] = entries[idx]
	}
	return sampled
}
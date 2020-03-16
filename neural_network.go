package main

import (
	"errors"
	"math"
	"math/rand"
)

const (
	Sigmoid = iota
	//Binary
	//TanH
)

func SigmoidFunc(x float64) float64 {
	return 1 / (1 + math.Pow(math.E, -x))
}

func BinaryFunc(x float64) float64 {
	if x >= 0 {
		return 1
	}
	return 0
}

func TanHFunc(x float64) float64 {
	return math.Tanh(x)
}

func DerivativeSigmoidFunc(x float64) float64 {
	return x * (1 - x)
}

func DerivativeBinaryFunc(x float64) float64 {
	return 0 * x
}

func DerivativeTanHFunc(x float64) float64 {
	return 1 - math.Pow(x, 2)
}

var ActivationFunctions = [3]func(float64) float64{
	SigmoidFunc,
	BinaryFunc,
	TanHFunc,
}

var DerivativeFunctions = [3]func(float64) float64{
	DerivativeSigmoidFunc,
	DerivativeBinaryFunc,
	DerivativeTanHFunc,
}

const (
	NeuronTypeInput = iota
	NeuronTypeOutput
	NeuronTypeHidden
	NeuronTypeBias
)

type Neuron struct {
	ActivationFunc func(float64) float64
	DerivativeFunc func(float64) float64
	NeuronType     int
	Output         float64
	weights        []float64 // outgoing
	error          float64
}

type NeuralNetworkConfig struct {
	Layers         []int
	Bias           bool
	ActivationFunc int
	LearningStep   float64
}

type NeuralNetwork struct {
	Config  *NeuralNetworkConfig
	neurons []*Neuron
}

func (neuralNetwork *NeuralNetwork) getNeuron(layer int, number int) *Neuron {
	lastLayer := len(neuralNetwork.Config.Layers) - 1
	if layer > lastLayer {
		panic(errors.New("layer out of range"))
	}
	layerNeurons := neuralNetwork.Config.Layers[layer] - 1
	if neuralNetwork.Config.Bias && layerNeurons != lastLayer {
		layerNeurons += 1
	}
	if number > layerNeurons {
		panic(errors.New("number out of range"))
	}
	neuronsNumber := 0
	for _, layerNeuronsNumber := range neuralNetwork.Config.Layers[:layer] {
		neuronsNumber += layerNeuronsNumber
	}
	return neuralNetwork.neurons[neuronsNumber+number]
}

func (neuralNetwork *NeuralNetwork) GetOutput() []float64 {
	outputLayer := len(neuralNetwork.Config.Layers) - 1
	neuronsNumber := neuralNetwork.Config.Layers[outputLayer]
	output := make([]float64, neuronsNumber)
	for neuronNumber := 0; neuronNumber < neuronsNumber; neuronNumber++ {
		output[neuronNumber] = neuralNetwork.getNeuron(outputLayer, neuronNumber).Output
	}
	return output
}

func (neuralNetwork *NeuralNetwork) New() {
	neuronsNumber := 0
	for _, layerNeuronsNumber := range neuralNetwork.Config.Layers {
		neuronsNumber += layerNeuronsNumber
	}
	if neuralNetwork.Config.Bias {
		neuronsNumber += len(neuralNetwork.Config.Layers) - 1
	}
	neuralNetwork.neurons = make([]*Neuron, neuronsNumber)

	neuronsCounter := 0
	for layerNumber, layerNeuronsNumber := range neuralNetwork.Config.Layers {
		// define neurons type
		var nextLayerNeuronsNumber int
		var NeuronType int
		switch layerNumber {
		case 0:
			NeuronType = NeuronTypeInput
			nextLayerNeuronsNumber = neuralNetwork.Config.Layers[layerNumber+1]
		case len(neuralNetwork.Config.Layers) - 1:
			NeuronType = NeuronTypeOutput
			nextLayerNeuronsNumber = 0
		default:
			NeuronType = NeuronTypeHidden
			nextLayerNeuronsNumber = neuralNetwork.Config.Layers[layerNumber+1]
		}

		if neuralNetwork.Config.Bias && layerNumber != len(neuralNetwork.Config.Layers)-1 {
			neuron := &Neuron{
				ActivationFunc: ActivationFunctions[neuralNetwork.Config.ActivationFunc],
				DerivativeFunc: DerivativeFunctions[neuralNetwork.Config.ActivationFunc],
				NeuronType:     NeuronTypeBias,
				weights:        make([]float64, nextLayerNeuronsNumber),
				Output:         1,
			}
			for nextLayerNeuronNumber := 0; nextLayerNeuronNumber < nextLayerNeuronsNumber; nextLayerNeuronNumber++ {
				neuron.weights[nextLayerNeuronNumber] = rand.Float64()
			}
			neuralNetwork.neurons[neuronsCounter] = neuron
			neuronsCounter += 1
		}

		// init neurons for current layer with defaults weight for next layer
		for layerNeuronNumber := 0; layerNeuronNumber < layerNeuronsNumber; layerNeuronNumber++ {
			neuron := &Neuron{
				ActivationFunc: ActivationFunctions[neuralNetwork.Config.ActivationFunc],
				DerivativeFunc: DerivativeFunctions[neuralNetwork.Config.ActivationFunc],
				NeuronType:     NeuronType,
				weights:        make([]float64, nextLayerNeuronsNumber),
			}
			for nextLayerNeuronNumber := 0; nextLayerNeuronNumber < nextLayerNeuronsNumber; nextLayerNeuronNumber++ {
				neuron.weights[nextLayerNeuronNumber] = rand.Float64()
			}
			// set neuron
			neuralNetwork.neurons[neuronsCounter] = neuron
			neuronsCounter += 1
		}
	}
}

func (neuralNetwork *NeuralNetwork) Learn(inputs []float64, answers []float64) {
	neuralNetwork.Run(inputs)

	// calculate error
	// do not calculate errors for the input (0) layer
	lastLayer := len(neuralNetwork.Config.Layers) - 1
	for layer := lastLayer; layer > 0; layer-- {
		for neuronNumber := 0; neuronNumber < neuralNetwork.Config.Layers[layer]; neuronNumber++ {
			neuron := neuralNetwork.getNeuron(layer, neuronNumber)
			switch layer {
			case lastLayer:
				neuron.error = neuron.DerivativeFunc(neuron.Output) * (answers[neuronNumber] - neuron.Output)
				//neuron.error = answers[neuronNumber] - neuron.Output
			default:
				var outputError float64 = 0
				for nextLayerNeuronNumber := range neuron.weights {
					nextLayerNeuron := neuralNetwork.getNeuron(layer+1, nextLayerNeuronNumber)
					outputError += nextLayerNeuron.error * neuron.weights[nextLayerNeuronNumber]
				}
				neuron.error = outputError * neuron.DerivativeFunc(neuron.Output)
				//neuron.error = outputError
			}
		}
	}

	// update weights
	for layer, neuronsNumber := range neuralNetwork.Config.Layers[:lastLayer] {
		for neuronNumber := 0; neuronNumber < neuronsNumber; neuronNumber++ {
			neuron := neuralNetwork.getNeuron(layer, neuronNumber)
			for nextLayerNeuronNumber := range neuron.weights {
				nextLayerNeuron := neuralNetwork.getNeuron(layer+1, nextLayerNeuronNumber)
				neuron.weights[nextLayerNeuronNumber] += neuralNetwork.Config.LearningStep *
					nextLayerNeuron.error *
					//neuron.DerivativeFunc(nextLayerNeuron.Output) *
					neuron.Output
			}
		}
	}
}

func (neuralNetwork *NeuralNetwork) Run(inputs []float64) {
	if len(inputs) != neuralNetwork.Config.Layers[0] {
		panic(errors.New("inputs mismatch"))
	}
	for layer, neuronsNumber := range neuralNetwork.Config.Layers {
		switch layer {
		case 0:
			for i := 0; i < neuronsNumber; i++ {
				neuron := neuralNetwork.getNeuron(layer, i)
				neuron.Output = inputs[i]
			}
		default:
			for currentLayerNeuron := 0; currentLayerNeuron < neuronsNumber; currentLayerNeuron++ {
				previousLayer := layer - 1
				previousLayerNeurons := neuralNetwork.Config.Layers[previousLayer]
				if neuralNetwork.Config.Bias {
					previousLayerNeurons += 1
				}
				var currentNeuronOutput float64 = 0
				for previousLayerNeuron := 0; previousLayerNeuron < previousLayerNeurons; previousLayerNeuron++ {
					neuron := neuralNetwork.getNeuron(previousLayer, previousLayerNeuron)
					currentNeuronOutput += neuron.Output * neuron.weights[currentLayerNeuron]
				}
				neuron := neuralNetwork.getNeuron(layer, currentLayerNeuron)
				neuron.Output = neuron.ActivationFunc(currentNeuronOutput)
			}
		}
	}
}

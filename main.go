package main

import (
	"fmt"
	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
	"math"
)

const (
	WIDTH    = 700
	HEIGHT   = 700
	CellSide = 7
)

const (
	MaxEpochs    = 2000
	DrawStep     = 1
	AllowedError = .01
)

type Dot struct {
	X, Y float64
}

var LearnData = map[Dot][]float64{
	// red & black
	Dot{11, 8}:    {1, 0},
	Dot{13, 10}:   {1, 0},
	Dot{15, 12}:   {1, 0},
	Dot{18, 15}:   {1, 0},
	Dot{19, 17}:   {1, 0},
	Dot{22, 20}:   {1, 0},
	Dot{-11, 8}:   {1, 0},
	Dot{-13, 10}:  {1, 0},
	Dot{-15, 12}:  {1, 0},
	Dot{-18, 15}:  {1, 0},
	Dot{-19, 17}:  {1, 0},
	Dot{-22, 20}:  {1, 0},
	Dot{-11, -8}:  {1, 0},
	Dot{-13, -10}: {1, 0},
	Dot{-15, -12}: {1, 0},
	Dot{-18, -15}: {1, 0},
	Dot{-19, -17}: {1, 0},
	Dot{-22, -20}: {1, 0},
	//Dot{-1, -20}: {1, 0},
	//Dot{-1, -25}: {1, 0},

	// blue & white
	Dot{11, -32}: {0, 1},
	Dot{3, -18}:  {0, 1},
	Dot{5, -20}:  {0, 1},
	Dot{18, -36}: {0, 1},
	Dot{9, -38}:  {0, 1},
	Dot{10, -39}: {0, 1},

	Dot{21, -32}: {0, 1},
	Dot{13, -18}: {0, 1},
	Dot{15, -20}: {0, 1},
	Dot{28, -36}: {0, 1},
	Dot{19, -38}: {0, 1},
	Dot{30, -39}: {0, 1},

	Dot{31, -32}: {0, 1},
	Dot{23, -18}: {0, 1},
	Dot{25, -20}: {0, 1},
	Dot{38, -36}: {0, 1},
	Dot{29, -38}: {0, 1},
	Dot{40, -39}: {0, 1},

	Dot{1, 32}:  {0, 1},
	Dot{1, 18}:  {0, 1},
	Dot{-1, 20}: {0, 1},
	Dot{0, 36}:  {0, 1},
	Dot{-1, 38}: {0, 1},
	Dot{0, 9}:   {0, 1},
}

func (dot *Dot) GetDisplayPosition() (float64, float64) {
	return WIDTH/2 + dot.X*CellSide, HEIGHT/2 + dot.Y*CellSide
}

func run() {
	cfgErr := pixelgl.WindowConfig{
		Title:  "Neural Network Error",
		Bounds: pixel.R(0, 0, MaxEpochs, HEIGHT/1.5),
		VSync:  true,
	}
	winErr, err := pixelgl.NewWindow(cfgErr)
	if err != nil {
		panic(err)
	}
	cfg := pixelgl.WindowConfig{
		Title:  "Neural Network",
		Bounds: pixel.R(0, 0, WIDTH, HEIGHT),
		VSync:  true,
	}
	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		panic(err)
	}

	imdErr := imdraw.New(nil)
	imd := imdraw.New(nil)

	neuralNetworkConfig := NeuralNetworkConfig{
		Layers: []int{
			2, // input layer
			6,
			2, // output layer
		},
		Bias:           false,
		ActivationFunc: Sigmoid,
		LearningStep:   .1,
	}
	neuralNetwork := NeuralNetwork{Config: &neuralNetworkConfig}
	neuralNetwork.New()

	Epoch := 0
	for !win.Closed() && !winErr.Closed() {
		imdErr.Clear()
		imd.Clear()

		var x, y float64
		for x = -(WIDTH / CellSide) / 2; x < (WIDTH/CellSide)/2; x++ {
			for y = -(HEIGHT / CellSide) / 2; y < (HEIGHT/CellSide)/2; y++ {
				dot := Dot{x, y}

				neuralNetwork.Run([]float64{dot.X, dot.Y})
				output := neuralNetwork.GetOutput()

				switch len(output) {
				case 1:
					if output[0] >= .5 {
						imd.Color = pixel.RGB(1, 0, 0)
					} else {
						imd.Color = pixel.RGB(0, 0, 1)
					}

				default:
					imd.Color = pixel.RGB(output[0], 0, output[1])
				}

				rightBotX, rightBotY := dot.GetDisplayPosition()
				imd.Push(pixel.V(rightBotX, rightBotY))
				imd.Push(pixel.V(rightBotX+CellSide, rightBotY+CellSide))
				imd.Rectangle(0)
			}
		}

		for dot, t := range LearnData {
			if Epoch <= MaxEpochs {
				neuralNetwork.Learn([]float64{dot.X, dot.Y}, t)
				output := neuralNetwork.GetOutput()

				var learningError float64 = 0
				for i := range output {
					learningError += math.Pow(t[i]-output[i], 2)
				}
				learningError /= float64(len(output))

				if learningError > AllowedError {
					fmt.Println(Epoch, "Still learning...")
					imdErr.Color = pixel.RGB(1, 0, 0)
				} else {
					imdErr.Color = pixel.RGB(0, 1, 0)
				}
				imdErr.Push(pixel.V(float64(Epoch), winErr.Bounds().H()*learningError))
				imdErr.Circle(1, 0)
			}
			if t[0] == 1 {
				imd.Color = pixel.RGB(0, 0, 0)
			} else {
				imd.Color = pixel.RGB(1, 1, 1)
			}
			imd.Push(pixel.V(dot.GetDisplayPosition()))
			imd.Circle(4, 0)
		}

		Epoch += 1

		// axises
		imd.Color = pixel.RGB(1, 1, 1)
		imd.Push(pixel.V(0, HEIGHT/2), pixel.V(WIDTH, HEIGHT/2))
		imd.Line(2)
		imd.Color = pixel.RGB(1, 1, 1)
		imd.Push(pixel.V(WIDTH/2, 0), pixel.V(WIDTH/2, HEIGHT))
		imd.Line(2)

		// draw
		if Epoch%DrawStep == 0 {
			imdErr.Draw(winErr)
			winErr.Update()
			imd.Draw(win)
			win.Update()
		}
	}
}

func main() {
	pixelgl.Run(run)
}

package main

import (
	"fmt"
	"github.com/faiface/pixel"
	"github.com/faiface/pixel/imdraw"
	"github.com/faiface/pixel/pixelgl"
)

const (
	WIDTH = 500
	HEIGHT = 500
	CellSide = 5
)

type Dot struct {
	X, Y	float64
}

var LearnData = map[Dot][]float64{
	// red & black
	Dot{11, 8}: {1, 0},
	Dot{13, 10}: {1, 0},
	Dot{15, 12}: {1, 0},
	Dot{18, 15}: {1, 0},
	Dot{19, 17}: {1, 0},
	Dot{22, 20}: {1, 0},
	Dot{-11, 8}: {1, 0},
	Dot{-13, 10}: {1, 0},
	Dot{-15, 12}: {1, 0},
	Dot{-18, 15}: {1, 0},
	Dot{-19, 17}: {1, 0},
	Dot{-22, 20}: {1, 0},
	Dot{-11, -8}: {1, 0},
	Dot{-13, -10}: {1, 0},
	Dot{-15, -12}: {1, 0},
	Dot{-18, -15}: {1, 0},
	Dot{-19, -17}: {1, 0},
	Dot{-22, -20}: {1, 0},
	//Dot{-1, -20}: {1, 0},
	//Dot{-1, -25}: {1, 0},

	// blue & white
	Dot{11, -32}: {0, 1},
	Dot{3, -18}: {0, 1},
	Dot{5, -20}: {0, 1},
	Dot{18, -36}: {0, 1},
	Dot{9, -38}: {0, 1},
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

	Dot{1, 32}: {0, 1},
	Dot{1, 18}: {0, 1},
	Dot{-1, 20}: {0, 1},
	Dot{0, 36}: {0, 1},
	Dot{-1, 38}: {0, 1},
	Dot{0, 9}: {0, 1},
}

func (dot *Dot) GetDisplayPosition() (float64, float64) {
	return WIDTH / 2 + dot.X * CellSide, HEIGHT / 2 + dot.Y * CellSide
}

func run() {
	cfg := pixelgl.WindowConfig{
		Title:  "Neural Network",
		Bounds: pixel.R(0, 0, WIDTH, HEIGHT),
		VSync:  true,
	}
	win, err := pixelgl.NewWindow(cfg)
	if err != nil {
		panic(err)
	}

	imd := imdraw.New(nil)

	neuralNetworkConfig := NeuralNetworkConfig{
		Layers:   		[]int{2, 6, 2},
		Bias:			false,
		ActivationFunc: Sigmoid,
		LearningStep:   .1,
	}
	neuralNetwork := NeuralNetwork{Config: &neuralNetworkConfig}
	neuralNetwork.New()

	STEP := 0
	for !win.Closed() {
		imd.Clear()

		fmt.Println(STEP, "Running...")
		var x, y float64
		for x = -(WIDTH / CellSide) / 2; x < (WIDTH / CellSide) / 2; x++ {
			for y = -(HEIGHT / CellSide) / 2; y < (HEIGHT / CellSide) / 2; y++ {
				dot := Dot{x, y}

				neuralNetwork.Run([]float64{dot.X, dot.Y})
				output := neuralNetwork.GetOutput()

				switch len(output){
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
				imd.Push(pixel.V(rightBotX + CellSide, rightBotY + CellSide))
				imd.Rectangle(0)
			}
		}

		fmt.Println(STEP, "Learning...")
		for dot, t := range LearnData {
			neuralNetwork.Learn([]float64{dot.X, dot.Y}, t)

			if t[0] == 1 {
				imd.Color = pixel.RGB(0, 0, 0)
			} else {
				imd.Color = pixel.RGB(1, 1, 1)
			}
			imd.Push(pixel.V(dot.GetDisplayPosition()))
			imd.Circle(4, 0)
		}

		STEP += 1

		// axises
		imd.Color = pixel.RGB(1, 1, 1)
		imd.Push(pixel.V(0, HEIGHT / 2), pixel.V(WIDTH, HEIGHT / 2))
		imd.Line(2)
		imd.Color = pixel.RGB(1, 1, 1)
		imd.Push(pixel.V(WIDTH / 2, 0), pixel.V(WIDTH / 2, HEIGHT))
		imd.Line(2)

		// draw
		if STEP % 1 == 0 {
			imd.Draw(win)
			win.Update()
		}
	}
}

func main() {
	pixelgl.Run(run)
}

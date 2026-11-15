module neural_net::types;
import std::core::mem::alloc;

alias Value @feat( BF16 & !F64) = bfloat;
alias Value @feat(!BF16 & !F64) = float;
alias Value @feat(!BF16 &  F64) = double;

alias Weight = Value;
alias Bias = Value;

// alias Weight @feat( WEIGHT_BF16 & !WEIGHT_F64) = bfloat;
// alias Weight @feat(!WEIGHT_BF16 & !WEIGHT_F64) = float;
// alias Weight @feat( WEIGHT_F64  & !WEIGHT_BF16) = double;

// alias Bias @feat( BIAS_BF16 & !BIAS_F64) = bfloat;
// alias Bias @feat(!BIAS_BF16 & !BIAS_F64) = float;
// alias Bias @feat( BIAS_F64  & !BIAS_BF16) = double;

alias Transform = fn Value (Value);
alias Input = fn Value ();
alias Output = fn void (Value);

interface Layer 
{
	fn Value[]? getValues(Allocator);
	fn void? evaluate(Value[]);
}

// struct Neuron @feat(!COMP_TIME)
// {
// 	Bias input_bias;
// 	Weight[] input_weights;
// }

// struct Layer @feat(!COMP_TIME)
// {
// 	Neuron[] neurons;
// 	Value[] results;
// }

struct Neuron <PREV_LAYER_SIZE>
{
	Bias input_bias;
	Weight[PREV_LAYER_SIZE]* input_weights;
}

struct HiddenLayer (Layer) <SIZE, PREV_SIZE>
{
	Neuron{PREV_SIZE}[SIZE]* neurons;
	Value[SIZE]* results;
}


struct InputLayer (Layer) <SIZE>
{
	Input[SIZE] results;
}
fn Value[]? InputLayer.getValues(InputLayer* self, Allocator allocator) @dynamic
{
	Value[] val = alloc::alloc_array_try(
		allocator, 
		Value, 
		SIZE
	)!;

	foreach(i, Input input : self.results)
	{
		val[i] = input();
	}

	return val;
}
fn void? InputLayer.evaluate(InputLayer* self, Value[]) @dynamic => (void)null;


struct OutputLayer (Layer) <SIZE>
{
	Output[SIZE] results;
}
fn void? OutputLayer.evaluate(OutputLayer* self, Value[] inputs) @dynamic
{
	foreach(i, Output output : self.results)
	{
		output(inputs[i]);
	}
}
fn Value[]? OutputLayer.getValues(OutputLayer* self, Allocator) @dynamic => (Value[]){};


struct NeuralNetwork <SIZE>
{
	Layer[SIZE] layers;
}


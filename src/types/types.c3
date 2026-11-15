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




// struct NeuralNetwork <SIZE>
// {
// 	InputLayer input;
// 	HiddenLayer[SIZE] layers;
// 	OutputLayer output;
// }

struct NeuralNetwork <SIZE>      
{                                
	Layer[SIZE] layers;    
}                                
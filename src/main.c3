module neural_net;
import std::io;

const ushort[*] LAYER_PROFILE = 
{
	1,
	3,
	1
};


fn Value const1() => 1;

fn void print(Value in) => io::printn(in);

const Input[*] INPUTS =
{
	fn Value() => 1,
};
const InputLayer{INPUTS.len} INPUT_LAYER = { .results = INPUTS };

const Output[*] OUTPUTS =
{
	fn void(Value in) => io::printn(in),
};
const OutputLayer{OUTPUTS.len} OUTPUT_LAYER = { .results = OUTPUTS };

alias MainNet = NeuralNetwork{LAYER_PROFILE.len};

fn int main(String[] args) => @pool()
{
	const MainNet NETWORK = setup_network(LAYER_PROFILE, &INPUT_LAYER, &OUTPUT_LAYER);
	io::printn(NETWORK);
	NETWORK.run()!!;
	io::printn("Hello, World!");
	return 0;
}

macro MainNet setup_network(ushort[*] $profile, Layer $inputs, Layer $outputs) @const
{
	Layer[$profile.len] $layers;
	$layers[0] = $inputs;

	$for var $i = 1; $i < ($profile.len - 1); $i++:
		HiddenLayer{$profile[$i], $profile[$i-1]} $hidden_layer;
		$layers[$i] = &&$hidden_layer;
	$endfor

	$layers[^1] = $outputs;

	MainNet $neural_net = { .layers = $layers };

	return $neural_net;
}
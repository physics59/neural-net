module neural_net::types::neurons;
import neural_net::transforms;

struct Neuron <PREV_LAYER_SIZE>
{
	Bias input_bias;
	Weight[PREV_LAYER_SIZE]* input_weights;
}
<*
 
*>
fn Value Neuron.eval(Neuron* self, Value[PREV_LAYER_SIZE]* previous)
{
	Value val = 0;
	for(sz i = 0; i < previous.len; i++)
	{
		val += previous.[i] * self.input_weights.[i];
	}
	val += self.input_bias;

	val = transforms::TRANSFORM(val);

	return val;
}
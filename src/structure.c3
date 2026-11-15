module neural_net::structure;
import neural_net::types;
import neural_net::transforms;
import std::core::mem::alloc;

const Transform TRANSFORM = &transforms::relu;
const Transform TRANSFORM_DERIV = &transforms::relu_deriv;

<*
 
*>
fn Value Neuron.eval(Neuron* self, Value[PREV_LAYER_SIZE] previous)
{
	Value val = 0;
	for(sz i = 0; i < previous.len; i++)
	{
		val += previous[i] * self.input_weights[i];
	}
	val += self.input_bias;

	val = TRANSFORM(val);

	return val;
}

<*
 @require previous::SIZE == self::PREV_SIZE : "layer size mismatch"
*>
fn void HiddenLayer.eval(Layer* self, Layer* previous, Allocator allocator)
{
	self.results = alloc::alloc_array(allocator, Value, self.neurons.len);
	foreach(Neuron* &neuron : self.neurons)
	{
		Neuron.eval(neuron, previous.results);
	}
	alloc::free(previous.results);
}
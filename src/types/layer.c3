module neural_net::types::layers;
import neural_net::types::neurons;



struct HiddenLayer (Layer) <SIZE, PREV_SIZE>
{
	Neuron{PREV_SIZE}[SIZE] neurons;
	Value[SIZE] results;
}
fn void? HiddenLayer.evaluate(HiddenLayer* self, Value[] values) @dynamic
{
	for(int i = 0; i < SIZE; i++)
	{
		self.neurons[i].eval((Value[PREV_SIZE]*)values);
	}
}
fn Value[]? HiddenLayer.getValues(HiddenLayer* self, Allocator) @dynamic => &self.results;


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
module neural_net::forward_pass;
import neural_net::types;

fn void? NeuralNetwork.run(&self)
{
	for(ushort i = 1; i < self.layers.len; i++)
	{
		Layer layer = self.layers[i];
		Layer prev_layer = self.layers[i-1];
		@pool() {
			layer.evaluate(prev_layer.getValues(tmem))!;
		};
	}
}
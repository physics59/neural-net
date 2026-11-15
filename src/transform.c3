module neural_net::transforms;
import neural_net::types;

fn Value relu(Value in) 
{
	if(in > 0) {
		return in;
	} else {
		return -0.002 * in;
	}
}

fn Value relu_deriv(Value in) 
{
	if(in > 0) {
		return 1;
	} else {
		return -0.002;
	}
}
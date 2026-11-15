module neural_net::transforms;
import neural_net::types;

const Transform TRANSFORM = &transforms::relu;
const Transform TRANSFORM_DERIV = &transforms::relu_deriv;

fn Value relu(Value in) 
{
	if(in > 0) {
		return in;
	} else {
		return 0;
	}
}

fn Value relu_deriv(Value in) 
{
	if(in > 0) {
		return 1;
	} else {
		return 0;
	}
}
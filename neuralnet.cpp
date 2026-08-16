#include<iostream>
#include<cmath>
#include <iterator>
#include<vector>
#include<random>


double sigmoid(double z)
{
	return (1.00/(1.00+exp(-z)));
}

double sigmoid_derivative(double z)
{
	return sigmoid(z)*(1-sigmoid(z));
}

std::vector<double> matrixM(int row1, int col1, int row2, int col2, const std::vector<std::vector<double>>& matA, const std::vector<double>& matB)
{

	unsigned int i,j,k;
	double tempsum=0.0;
	

	std::vector<double> c(row1*col2);
	
	for(int i=0;i<row1;i++)
	{
		for(int j=0;j<row2;j++)
		{
			tempsum+=matA[i][j]*matB[j];
		}
		c[i] = tempsum;
		tempsum = 0.0;
	}

	return c;

}



class network
{
public:
	int num_layers;
	std::vector<int> sizes;
	
	std::vector<std::vector<double>> activations;
	std::vector<std::vector<double>> zs; 
	std::vector<std::vector<std::vector<double>>> nabla_w;
	std::vector<std::vector<double>> nabla_b;




	//bias vector
	std::vector<std::vector<double>> biases;
	
	//weights vector 
	std::vector<std::vector<std::vector<double>>> weights;
	
	//constructor
	network(const std::vector<int>& sizes)
	{
		this->sizes = sizes;
		this->num_layers = sizes.size();

		std::random_device rd;
		std::mt19937 gen(rd());

		std::normal_distribution<> distribution(0.0, 1.0);

		biases = std::vector<std::vector<double>>(num_layers-1);
		
		//assigning empty spaces to store the biases
		for(int i=0;i<num_layers-1;i++)
		{
			biases[i] = std::vector<double>(sizes[i+1]);
		}

		//assigning randomly generated double values
		
		for(int i=1;i<num_layers;i++)
		{
			for(int j =0;j<sizes[i];j++)
			{
			biases[i-1][j] = distribution(gen);
			}
		}
		
		//yeta chai weights ko lagi vectors banaudai
		
		weights = std::vector<std::vector<std::vector<double>>>(num_layers-1);
		

		for(int i=0;i<num_layers-1;i++)
		{
			weights[i]=std::vector<std::vector<double>>(sizes[i+1]);
		}

		for(int i=0;i<num_layers-1;i++)
		{
			for(int j=0;j<sizes[i+1];j++)
			{
			weights[i][j]=std::vector<double>(sizes[i]);
			}
		}
		
		for(int i=0;i<num_layers-1;i++)
		{
			for(int j=0;j<sizes[i+1];j++)
			{
				for(int k=0;k<sizes[i];k++)
				{
					weights[i][j][k]=distribution(gen);
				}

			}
		}
	}


		//feedforward wala yeta bata suru ))
	void FeedForward()
	{
		std::vector<double> activation_vector(sizes[0]);
				

	
		activations = std::vector<std::vector<double>>(num_layers);
		
		zs = std::vector<std::vector<double>>(num_layers-1);
		
		for(int i = 0; i <num_layers;i++)
		{
			activations[i] = std::vector<double>(sizes[i]);
		}
		
		for(int i=0;i < num_layers-1;i++)
		{
			zs[i] = std::vector<double>(sizes[i+1]);
		}
		

		// //ok, backpropagation wala variables haru yetai siddyo )) 
		//
		// feedforward loop yeta bata xa haii !!!
		
		activations[0] = activation_vector;

		for(int i=0;i<num_layers-1;i++)
		{
			activation_vector = matrixM(sizes[i+1], sizes[i], sizes[i], 1, weights[i], activation_vector);

			for(int j=0;j<sizes[i+1];j++)
			{
				activation_vector[j]+=biases[i][j];
			}

			zs[i] = activation_vector;

			for(int j=0;j<sizes[i+1];j++)
			{				
				activation_vector[j] = sigmoid(activation_vector[j]);
			}
			activations[i+1] = activation_vector;
		}
		
	}
	void BackPropagation()
	{
		//yeta bata backpropagation ko implementation suru   	
		std::vector<double> output_errors(sizes[num_layers-1]);
		
		//equation 1 implementaion
		for(int i=0;i<sizes[num_layers-1];i++)
		{
			output_errors[i] = ((activations[num_layers-1][i]-req_output[i]) * sigmoid_derivative(zs[num_layers-2][i]));
		}

		//equation 2 implementation

		std::vector<std::vector<double>> bckprop(num_layers-1);
		
		for(int i=0;i<num_layers-1;i++)
		{
			bckprop[i] = std::vector<double>(sizes[i+1]);
		}

		bckprop[num_layers-2] = output_errors;

		std::vector<std::vector<double>> transposed(sizes[num_layers-2],std::vector<double>(sizes[num_layers-1]));

		//transpose of the weight matrix	
		for(int i = 0; i < sizes[num_layers - 1]; i++)
		{
		    for(int j = 0; j < sizes[num_layers - 2]; j++)
		    {
        		transposed[j][i] = weights[num_layers - 2][i][j];
		    }
		}	

		std::vector<double> tempvec(sizes[num_layers-1]);
		
		//tempvec = W^T * delta
		tempvec =matrixM(sizes[num_layers - 2],sizes[num_layers - 1],sizes[num_layers - 1],1,transposed,output_errors);
		
		//hadamard_product(tempvec*sigmoid_derivative(z)

		for(int j = 0; j < sizes[num_layers - 2]; j++)
		{
			bckprop[num_layers-3][j] = tempvec[j]*sigmoid_derivative(zs[num_layers - 3][j]);
		}


		//equation 3 implementation
		//
		nabla_b = std::vector<std::vector<double>>(num_layers-1);
		for(int i=0;i<num_layers-1;i++)
		{
			nabla_b[i] = std::vector<double>(sizes[i+1]);
		}

		for(int i=0;i<num_layers-1;i++)
		{
			nabla_b[i] = bckprop[i];
		}
		
		

		//equation 4 implementation::

		nabla_w = std::vector<std::vector<std::vector<double>>>(num_layers-1);

		for(int i=0;i<num_layers-1;i++)
		{
			nabla_w[i]=std::vector<std::vector<double>>(sizes[i+1]);
		}

		for(int i=0;i<num_layers-1;i++)
		{
			for(int j=0;j<sizes[i+1];j++)
			{
				nabla_w[i][j] = std::vector<double>(sizes[i]);
			}
		}

		for(int i=0;i<num_layers-1;i++)
		{
			for(int j=0;j<sizes[i+1];j++)
			{
				for(int k=0;k<sizes[i];k++)
				{
					nabla_w[i][j][k] = bckprop[i][j] * activations[i][k];
				}
			}
		}
		

	}


};



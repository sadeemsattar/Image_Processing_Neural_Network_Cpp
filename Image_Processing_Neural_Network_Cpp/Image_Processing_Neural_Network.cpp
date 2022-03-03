#include<iostream>
#include<string>
#include<string.h>
#include<cmath>
#include<time.h>
#include<sstream>
#include<vector>
#include<fstream>
using namespace std;

// count no of object 
int destroy=0;

// random for trainning
int get_randoms(int lower, int upper) 
{ 
	int num;
	num = (rand() %(upper - lower + 1)) + lower; 
    return num;
} 

// ---------------------------------------------------------------------------------------------
// sigmod function for mapping 1 ::::::::::::::::::  activation function
double sigmod(double x)
{
    return (1/(1+exp(-x)));
}


// --------------------------------------------------------------------------------------------
//  for maping 2 
double derivative_sigmod(double x)
{
    return x*(1-x);
}

// ----------------------------------------------------------------------------------------------
// 2d matrix class
template<class T>
class matrix1
{
    private:
    T **data;
    public:
    int row;
    int col;
    
    // default constructor
    matrix1()
    {
        row=0;
        col=0;
        data=NULL;
    }

    // parameterize constructor
    matrix1(int row,int col)
    {
        this->row=row;
        this->col=col;
        data=new T*[row];
        for(int i=0;i<row;i++)
        {
            *(data+i)=new T[col];
            for(int j=0;j<col;j++)
            {
                *(*(data+i)+j)=0;
            }
        }
    }

    // copy constructor
    matrix1(const matrix1<T> &rhs)
    {
        this->row=rhs.row;
        this->col=rhs.col;
        this->data=new T*[row];
        for(int i=0;i<row;i++)
        {
            *(data+i)=new T[col];
            for(int j=0;j<col;j++)
            {
                *(* (data+i)+j)=*(* (rhs.data+i)+j);
            }
        }        
    }

    // equal to operator
    matrix1<T>& operator=(const matrix1<T> &rhs)
    {
        if(this!=&rhs)
        {
            this->row=rhs.row;
            this->col=rhs.col;
            this->data=new T*[row];
            for(int i=0;i<row;i++)
            {
                *(data+i)=new T[col];
                for(int j=0;j<col;j++)
                {
                    *(* (data+i)+j)=*(* (rhs.data+i)+j);
                }
            }
        }
        return *this;
    }


    // calculate matrix multiplication 
    void cross_product(const matrix1<T> &a,const matrix1<T> &b)
    {
        if(a.col!=b.row)
        {
            cout<<"Error In Multiplication"<<endl;
        }
        else if(a.col==b.row)
        {
            for(int i=0;i<a.row;i++)
            {
                for(int j=0;j<b.col;j++)
                {
                    double sum=0;
                    for(int k=0;k<a.col;k++)
                    {
                        sum=sum+(*(*(a.data+i)+k))*(*(*(b.data+k)+j));
                    }
                    *(*(this->data+i)+j)=sum;
                }
            }
        }
    }

    // scalar product mean multiply matrix with number
    void scalar_product(const double num)
    {
        for(int i=0;i<this->row;i++)
        {
            for(int j=0;j<this->col;j++)
            {
                *(*(this->data+i)+j)= *(*(this->data+i)+j) * num;
            }
        }
    }
    

    // dot product
    void dot_product(const matrix1<T> &a)
    {
        for(int i=0;i<this->row;i++)
        {
            for(int j=0;j<this->col;j++)
            {
                *(*(this->data+i)+j)= (*(*(this->data+i)+j)) * (*(*(a.data+i)+j));
            }
        }
    }


    // add two matrix
    void add(const matrix1<T> &a)
    {
        if(this->row==a.row && this->col==a.col)
        {
            for(int i=0;i<a.row;i++)
            {
                for(int j=0;j<a.col;j++)
                {
                    *(*(this->data+i)+j)= *(*(this->data+i)+j) + *(*(a.data+i)+j);
                }
            }
        }
        else
        {
            cout<<"Error In Adding"<<endl;
        }
    } 

    // matrix subtraction
    void subtract(const matrix1<T> &a,const matrix1<T> &b)
    {
        for(int i=0;i<a.row;i++)
        {
            for(int j=0;j<a.col;j++)
            {
                *(*(this->data+i)+j)= *(*(a.data+i)+j) - *(*(b.data+i)+j);
            }
        }
    } 

    // maping1 (sigmoid) 
    void maping1()
    {
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                double val=*(*(this->data+i)+j);
                *(*(this->data+i)+j)=sigmod(val);
            }
        }
    }

    // maping2 (sigmoid derivative) 
    void maping2()
    {
        for(int i=0;i<row;i++)
        {
            for(int j=0;j<col;j++)
            {
                double val=*(*(this->data+i)+j);
                *(*(this->data+i)+j)=derivative_sigmod(val);
            }
        }
    }

    // adjust random weight
    void randomize()
    {
        srand((unsigned)time(NULL));
        for(int i=0;i<this->row;i++)
        {
            for(int j=0;j<this->col;j++)
            {
                *(*(data+i)+j)=((double)rand()/RAND_MAX)*2-1;
            }
        }
    }
    
    // transpose of matrix
    matrix1<T> transpose(const matrix1<T> &rhs)
    {
        matrix1 ans(rhs.col,rhs.row);
        for(int i=0;i<rhs.row;i++)
        {
            for(int j=0;j<rhs.col;j++)
            *(*(this->data+j)+i)=(*(*(rhs.data+i)+j));
        }
        return ans;
    }


    // set target matrix
    void set_target(const double val)
    {
        for(int i=0;i<this->row;i++)
        {
            if(i==val)
            {
                *(*(data+i)+0)=1;   
            }
            else
            {
                *(*(data+i)+0)=0;
            }
            
        }
    }

    // set input data read from mnist data set
    void copy_data_vel(const vector<T> &rhs)
    {
        
        this->row=rhs.size();
        this->col=1;
        data=new T*[row];
        for(int i=0;i<row;i++)
        {
            *(data+i)=new T[col];
            for(int j=0;j<col;j++)
            {
                *(*(data+i)+j)=rhs[i]/225;
            }
        }
    }


    // get max (soft_max)
    int max()
    {
        T val=*(*(this->data+0)+0);
        int num=0;
        for(int i=0;i<this->row;i++)
        {
            if(*(*(this->data+i)+0)>val)
            {
                val=*(*(this->data+i)+0);
                num=i;
            }
        }
        cout<<"     Predicted: "<<num<<"     : "<<(val/1)*100 <<" %";
		return num;
    }

    // get space in memory
    void Initiallize(int row,int col)
    {
        this->row=row;
        this->col=col;
        this->data=new T*[row];
        for(int i=0;i<row;i++)
        {
            *(data+i)=new T[col];
            for(int j=0;j<col;j++)
            {
                *(*(data+i)+j)=0;
            }
        }
        
    }

    // free data pointer 
    void memoryfree()
    {
        if(data!=0)
        {
            for(int i=row-1;i>0;i--)
            {
                if(*(data+i)!=0)
                {
                    delete [] *(data+i);
                    *(data+i)=0;
                }
            }
        }
        delete [] data;
        data=0;
    }

    // save weights and bias in file 
    void save(string name)
    {
        ofstream myfile (name);
        if (myfile.is_open())
        {
            int i=0;
            while(i<this->row)
            {
                for(int j=0;j<this->col;j++)
                {
                    myfile<<data[i][j];
                    myfile<<",";
                }
                myfile<<"\n";
                i++;            
            }
            myfile.close();
        }	
	}


    // read weight and bias in file
    void read(string name)
    {
        ifstream input(name);
        int roww=0,coll=0;
        while(input.good())
        {
            string val;
            getline(input,val);
            
            stringstream ss(val);
            string inp;
            while(getline(ss,inp,','))
            {    
                data[roww][coll]=stod(inp);
                coll++;
            }
           coll=0;
           roww++;
        
        }
        input.close();
    }


    // destructor
    ~matrix1()
    {
        if(data!=0)
        {
            for(int i=row-1;i>0;i--)
            {
                if(*(data+i)!=0)
                {
                    delete [] *(data+i);
                    *(data+i)=0;
                }
            }
        }
        delete [] data;
        data=0;
        destroy++;
    }

    
};

// ----------------------------------------------------------------------------------------------

// create network class
template<class TT>
class neural_network
{
    private:
		int input_nodes,hidden_nodes,output_node;
		matrix1<TT> weight_ih;
        matrix1<TT> weight_ho;
		matrix1<TT> bias_h;
        matrix1<TT> bias_o;
		double learning_rate;


	public:
    neural_network()
    {
        input_nodes=0;
		hidden_nodes=0;
		output_node=0;

        weight_ih.Initiallize(hidden_nodes,input_nodes);
		weight_ho.Initiallize(output_node,hidden_nodes);

		bias_h.Initiallize(hidden_nodes,1);
		bias_o.Initiallize(output_node,1);

		learning_rate=0.1;
    }
	
    // parameterize constructor and initiallize weights and bias with random value
	neural_network(int input_n,int hidden_n,int output_n)
    {
		input_nodes=input_n;
		hidden_nodes=hidden_n;
		output_node=output_n;
		
		weight_ih.Initiallize(hidden_nodes,input_nodes);
		weight_ho.Initiallize(output_node,hidden_nodes);

        weight_ih.randomize();
        weight_ho.randomize();

		bias_h.Initiallize(hidden_nodes,1);
		bias_o.Initiallize(output_node,1);

		bias_h.randomize();
		bias_o.randomize();

		learning_rate=0.1;
        
	}
	
	
	matrix1<TT> feedforward(matrix1<TT> &input)
    {
		matrix1<TT> hidden(weight_ih.row,input.col);
		hidden.cross_product(weight_ih,input);
		hidden.add(bias_h);
		hidden.maping1();

		matrix1<TT> outputt(weight_ho.row,hidden.col);
		outputt.cross_product(weight_ho,hidden);
		outputt.add(bias_o);
		outputt.maping1();
		return outputt;
	}

    // data training
	void train(matrix1<TT> &input,matrix1<TT> &targets)
    {
		// first layer
		matrix1<TT> hidden(weight_ih.row,input.col);
		hidden.cross_product(weight_ih,input);
        hidden.add(bias_h);
        hidden.maping1();
	

		// second layer
		matrix1<TT> output(weight_ho.row,hidden.col);
		output.cross_product(weight_ho,hidden);
    	output.add(bias_o);
		output.maping1();
		
		
		//calculating output error
		matrix1<TT> output_errors(targets.row,targets.col);
		output_errors.subtract(targets,output);
		

		// calculate gradint   
		output.maping2();
		output.dot_product(output_errors);
		output.scalar_product(learning_rate);


        // calculate deltas	    
        matrix1<TT> h= h.transpose(hidden);
		matrix1<TT> weight_ho_delta(output.row,h.col);
        weight_ho_delta.cross_product(output,h);
       
        

		// adjust weights
	    weight_ho.add(weight_ho_delta);

		// adjust biases
	    bias_o.add(output);
		

   
		// calculating hidden erros:
		matrix1<TT> w=w.transpose(weight_ho);
		w.cross_product(w,output_errors);
        //calculating gradient
		hidden.maping2();
		hidden.dot_product(w);
		hidden.scalar_product(learning_rate);

		//calculating  gradient delta of input layer
		matrix1<TT> in=in.transpose(input);
        matrix1<TT> dot(hidden.row,in.col);
		dot.cross_product(hidden,in);
        

		//adjust weights
		weight_ih.add(dot);
		//adjust bias
		bias_h.add(hidden);

	}

    // save network ...
    void save_Data()
    {
        weight_ih.save("weight_ih.csv");
        weight_ho.save("weight_ho.csv");
        bias_h.save("bias_h.csv");
        bias_o.save("bias_o.csv");
    }

    // read network ...
    void feed_data()
    {
        weight_ih.read("weight_ih.csv");
        weight_ho.read("weight_ho.csv");
        bias_h.read("bias_h.csv");
        bias_o.read("bias_o.csv");
    }
    ~neural_network()
    {
        destroy++;
    }
 
};



// ---------------------------------------------------------------------------------------------------

// data test 
vector< vector<double> > data_test;


// ---------------------------------------------------------------------------------------------------
// read test data ... 
vector<double> read_data()
{
    ifstream input("mnist_test.csv");
    vector<double> label_test;
	int row=0,col=0;

	// e.o.f
    while(input.good())
    {
		string val;
		getline(input,val);
		if(row!=0)
        {
			stringstream ss(val);
			string inp;
			vector<double> dat_row;
			while(getline(ss,inp,',')){
				if(col==0)
					label_test.push_back(stod(inp));
				else
					dat_row.push_back(stod(inp));
				col++;
			}

			data_test.push_back(dat_row);				
			if(col==785){
				col=0;
				row++;
			}
		}
		else
        {
			row++;
		}
	}
	input.close();
    return label_test;
}

// ----------------------------------------------------------------------------------------------------

// trianing data and test 
void test_run(vector<vector<double>> & data_train,vector<double> &label_train,vector<double> &label_test)
{
    	//create network
        neural_network<double> nn(784,30,10);
        matrix1<double> input_arr(1,1);
        matrix1<double> target_arr(10,1);
        fflush(stdin);	
        int t=0;
        int y=0;
        for(int	i=0;i<60000;i++)
        {			
            //y=get_randoms(0,59999);
			input_arr.copy_data_vel(data_train[i]);
			target_arr.set_target(label_train[i]);
			nn.train(input_arr,target_arr);
            input_arr.memoryfree();
            target_arr.memoryfree();
            input_arr.Initiallize(1,1);
            target_arr.Initiallize(10,1);
			cout<<i<<endl;
            t++;
		}


        for(int	i=0;i<30000;i++)
        {	
            //y=get_randoms(0,59999);		
			input_arr.copy_data_vel(data_train[i]);
			target_arr.set_target(label_train[i]);
			nn.train(input_arr,target_arr);
            input_arr.memoryfree();
            target_arr.memoryfree();
            input_arr.Initiallize(1,1);
            target_arr.Initiallize(10,1);
			cout<<t<<endl;
            t++;
		}

        // save in file after training ...
        nn.save_Data();
        float success=0;
	    for(int i=0;i<10000;i++)
        {
			cout<<"Target: "<<label_test[i];
			input_arr.copy_data_vel(data_test[i]);
			if(nn.feedforward(input_arr).max()==label_test[i])
            {
                success++;
            }
            input_arr.memoryfree();
            input_arr.Initiallize(1,1);
			cout<<endl;

	}
	   // cout<<endl<<"     Total Successfull Prediction: "<<success<<endl<<"     In Percentage : "<<((success)/10000)*100<<" %"<<endl;
       
}



// ----------------------------------------------------------------------------------------------------------
// test on previous traning ...
void test_run2(vector<double> &label_test)
{
        neural_network<double> nnn(784,30,10);
        nnn.feed_data();
        matrix1<double> input_arr(1,1);
        float success=0;
	    for(int i=0;i<10000;i++)
        {
			
			cout<<"Target: "<<label_test[i];
			
			input_arr.copy_data_vel(data_test[i]);
			if(nnn.feedforward(input_arr).max()==label_test[i])
            {
                success++;
            }
            input_arr.memoryfree();
            input_arr.Initiallize(1,1);
			cout<<endl;

	}

	    //cout<<endl<<"     Total Successfull Prediction: "<<success<<endl<<"     In Percentage : "<<((success)/10000)*100<<" %"<<endl;

}

// ------------------------------------------------------------------------------------------------------
// read training data
void read_train(vector<double>& label_test,int check)
{
    ifstream input2("mnist_train.csv");
	vector< vector<double> > data_train;
    vector<double> label_train;
		
		int row=0,col=0;
		while(input2.good()){
			
			string val2;
			getline(input2,val2);
			if(row!=0){
				stringstream ss2(val2);
				string inp2;
				vector<double> dat_row2;
				while(getline(ss2,inp2,',')){
					if(col==0)
						label_train.push_back(stod(inp2));
					else
						dat_row2.push_back(stod(inp2));
					col++;
				}

				data_train.push_back(dat_row2);
					
				if(col==785){
					col=0;
					row++;
				}
			}
			else{
				row++;
			}
		}
		input2.close();
        if(check==1)
        {
            test_run(data_train,label_train,label_test);
        }
        else if( check==2)
        {
            test_run2(label_test);
        }
}




// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
int main()
{
    
    int choice;
    cout<<endl<<endl<<endl<<"       Enter Choice      "<<endl;
    cout<<"Enter [1] To Train Again "<<endl;
    cout<<"Enter [2] To Predict On Previous Trainng "<<endl;
    cout<<"Enter [3] To Exit Code "<<endl<<endl;
    cin>>choice;
    
    switch (choice)
    {
        case 1:
        { 
            cout<<endl<<"loading..."<<endl;
            vector<double> label_testt=read_data();
            read_train(label_testt,1);
            break;
        
        }
        case 2:
        {
            cout<<endl<<"loading..."<<endl;
            vector<double> label_testt=read_data();
            read_train(label_testt,2);
            break;
        }
        case 3:
        {
            exit(0);
            break;
        }
        default:
        {
            cout<<" Wrong Choice "<<endl;
            break;
        }
    }

 cout<<endl<<" Total Object "<<destroy<<endl;   
	

	
    return 0;
}
`timescale 1ns/100fs
`define No0fKernels 2

module FCNeuron_tb;

logic             clk;
logic [63:0]      weight;
logic [3:0] [7:0] pooledPixelArray[1:0];
logic [7:0]       result;

FC_neuron I_FCNeuron(
.clk,
.pooledPixelArray({pooledPixelArray[0],pooledPixelArray[1]}),
.weight,
.result
);

initial begin
clk = 1'b0;
end

always begin
	// for X
	weight[63:32] = 32'h01ffff01 ;
	weight[31: 0] = 32'hff0101ff ;
@(negedge clk);
	// for O
	weight[63:32] = 32'hff0101ff ;
	weight[31: 0] = 32'h01ffff01 ;

@(negedge clk);
	// for /
	weight[63:32] = 32'hffffffff ;
	weight[31: 0] = 32'hff0101ff ;
@(negedge clk);
	
	weight[63:32] = 32'h01ffff01 ;
	weight[31: 0] = 32'hffffffff ;

@(negedge clk);
end

initial begin 
//X
pooledPixelArray[0] = 32'h01ffff01;
pooledPixelArray[1] = 32'hff0101ff;

repeat(4) @(negedge clk);
// '\'
pooledPixelArray[0] = 32'hff0101ff;
pooledPixelArray[1] = 32'h01ffff01;

repeat(4) @(negedge clk);
//"0"
pooledPixelArray[0] = 32'hffffffff; 
pooledPixelArray[1] = 32'hff0101ff;

repeat(4) @(negedge clk);
// '/'
pooledPixelArray[0] = 32'h01ffff01;
pooledPixelArray[1] = 32'hffffffff;

repeat(4) @(negedge clk);
#20 $finish; 
end

always begin
#5 clk=~clk;
end

endmodule


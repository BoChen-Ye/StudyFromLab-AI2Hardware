`timescale 1ns/100fs
`define No0fKernels 2

module FCNeuron_tb;

logic             clk;
logic [63:0]      weight;
logic [3:0] [7:0] pooledPixelArray;
logic [7:0]       result;

FCNeuron I_FCNeuron(
.clk,
.pooledPixelArray({pooledPixelArray[0],pooledPixelArray[1]}),
.weight,
.result
);

initial begin
clk = 1'b0;
end

always begin
// weight for X
@(negedge clk);
// weight for 0
@(negedge clk);
// weight for /
@(negedge clk);
// weight for \
@(negedge clk);
end

initial begin
// pooledPixelArray X
repeat(4) @(negedge clk);
// pooledPixelArray 0
repeat(4) @(negedge clk);
// pooledPixelArray /
repeat(4) @(negedge clk);
// pooledPixelArray "\"
repeat(4) @(negedge clk);
#20 $finish
end

always begin
#5 clk=~clk;
end

endmodule


`timescale 1ns/100fs

module FCNeuron( 
input clk,
input [7:0][7:0] pooledPixelArray,
input [63:0] weight,
output logic [7:0] result
);

logic [7:0] sum;

integer i;
always_comb begin

endmodule

always_ff@(posedge clk) begin
	convResult <= sum;
end

endmodule


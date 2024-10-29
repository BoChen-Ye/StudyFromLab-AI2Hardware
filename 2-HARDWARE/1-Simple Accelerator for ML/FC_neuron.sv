`timescale 1ns/100fs

module FC_neuron( 
input clk,
input [7:0][7:0] pooledPixelArray,
input [63:0] weight,
output logic [7:0] result
);

logic [7:0] sum;

integer i;
always_comb begin
	sum =0;
	for (i = 0; i < 8; i = i + 1) begin
		sum = sum + pooledPixelArray[i] * weight[(i*8) +: 8];
	end
end

always_ff@(posedge clk) begin
	result <= sum;
end

endmodule


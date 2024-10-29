`timescale 1ns/100fs

module Pooling_tb;

logic clk, En;
logic [7:0] convResult;
logic [3:0][7:0] pooledPixels;

Pooling I_Pooling(.*);

initial begin
    clk = 1'b0 ;
end

initial
begin

	convResult = 8'h31;
	@(negedge clk);
	convResult = 8'h32;
	@(negedge clk);
	convResult = 8'h38;
	@(negedge clk);
	convResult = 8'h07;
	@(negedge clk);
	convResult = 8'h01;
	@(negedge clk);
	convResult = 8'h00;
	@(negedge clk);
	convResult = 8'h33;
	@(negedge clk);
	convResult = 8'hfc;

	#120 $finish; 
end

initial
begin
	En = 1'b0;
	repeat(3) @(posedge clk);
	En = 1'b1;
	@(posedge clk);
	En = 1'b0;
	repeat(3) @(posedge clk);
	En = 1'b1;
	@(posedge clk);
	En = 1'b0;
end

always
begin
    #5 clk = ~clk;
end

endmodule




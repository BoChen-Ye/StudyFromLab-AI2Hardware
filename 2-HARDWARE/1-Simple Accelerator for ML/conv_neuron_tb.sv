`timescale 1ns/100fs

module conv_neuron_tb;

logic clk;
logic [3:0] [7:0] pixels;
logic [7:0] convResult;
logic [31:0] kernel;

conv_neuron  I_CNeuron(
.clk, 
.kernel, 
.pixels, 
.convResult
);

initial
begin
    clk = 1'b0 ;
    @(negedge clk);
    kernel = 32'h01ff01ff;
    pixels = {8'h01, 8'hff, 8'h8f, 8'h01};
    // setting kernel and pixels
    #20 $finish;
end

always
    #5 clk = ~clk;

endmodule

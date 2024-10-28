`timescale 1ns/100fs

module Pooling( 
input clk, En,
input [7:0] convResult,
output logic [3:0][7:0] pooledPixels
);

integer i;
logic [2:0][7:0] convolution;
logic [3:0][7:0] pooledReg;

always_ff @(posedge clk)begin
    // 代码，用于累加卷积结果到 "convolution" 寄存器中
end

always_ff @(negedge clk)begin
    if (En == 1'b1)
    begin
        // 代码，用于将卷积结果加载到 "pooledReg" 中
    end
end

assign pooledPixels = pooledReg;

endmodule



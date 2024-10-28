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
    convResult = 8'h84;
    @(negedge clk);
    // 添加代码，分配其他6个值，每个值一个时钟周期
    #80 $finish;
end

initial
begin
    // 设置 En 的代码
end

always
begin
    #5 clk = ~clk;
end

endmodule




`timescale 1ns/100fs

`define numAddr 5
`define numOut 32
`define wordDepth 32

module dpram32x32_cb_test;

    logic [`numAddr-1:0] A1;
    logic [`numAddr-1:0] A2;
    logic CEB1, WEB1, OEB1, CEB2, WEB2, OEB2, CSB2;
    logic [`numOut-1:0] I1, I2;
    logic [`numOut-1:0] O1, O2;

    dpram32x32_cb RAM_U1(.*);

    initial begin
        // 初始化 CEB1 和 CEB2
        @(posedge CEB1);
        // SRAM 信号的代码
        #30 $finish;
    end

    always begin
        // CEB1 的代码
    end

    always begin
        // CEB2 的代码
    end

endmodule

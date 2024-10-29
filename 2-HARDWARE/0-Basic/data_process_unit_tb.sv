`timescale 1ns / 1ps

module data_process_unit_tb;

localparam WIDTH=8;
localparam DEPTH=4;

logic clk,rstn,en;
logic [WIDTH-1:0] data_in1,data_in2;
logic data_valid;
logic [2*WIDTH*DEPTH-1:0] data_out;

data_process_unit #(
.WIDTH(WIDTH),
.DEPTH(DEPTH)
)
u_dpu(
.clk,
.rstn,
.en,
.data_in1,
.data_in2,

.data_valid,
.data_out
);

always #5 clk = ~clk;

initial begin
	clk      = 0;
    rstn     = 0;
    en       = 0;
	data_in1 = 0;
    data_in2 = 0;
	
	#10 rstn= 1;
		en   = 1;
	
	#10 data_in1 = 'd1;
        data_in2 = 'd2;
		
	#10 data_in1 = 'd3;
        data_in2 = 'd4;
		
	#10 data_in1 = 'd5;
        data_in2 = 'd6;
		
	#10 data_in1 = 'd7;
        data_in2 = 'd8;
		
	#10 data_in1 = 'd0;
        data_in2 = 'd0;
	
	#10 en       = 0;
end

endmodule
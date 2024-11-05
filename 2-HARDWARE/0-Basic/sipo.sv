`timescale 1ns / 1ps

module sipo #(
	parameter WIDTH = 8,
	parameter MAX_NUM  = 2
)(
    input                                rstn         ,
    input                                clk          ,
    input [WIDTH-1:0]                    din_serial   ,
    input                                din_valid    ,
    output  logic  [WIDTH*MAX_NUM-1:0]   dout_parallel,
    output  logic                        dout_valid
);
 
logic [WIDTH*MAX_NUM-1:0] dout_temp;
logic [$clog2(MAX_NUM):0] count;
 
always @(posedge clk)begin
    if(!rstn)begin
        dout_temp  <= 'd0;
        count      <= 'd0;
    end
    else if(din_valid)begin
        dout_temp <= {din_serial,dout_temp[WIDTH*MAX_NUM-1:WIDTH]};
        count     <= count + 1'b1;
    end
    else begin

        dout_temp  <= 'd0;
        count      <= 'd0;
    end
end
 
always @(posedge clk)begin
	if(!rstn)begin
		dout_parallel<='d0;
		dout_valid <= 1'b0;
	end
    else if(count == MAX_NUM)begin
        dout_parallel <= dout_temp;
        dout_valid    <= 1'b1;
    end
	else begin
		dout_valid <= 1'b0;
		dout_parallel<=dout_parallel;
	end
end
endmodule
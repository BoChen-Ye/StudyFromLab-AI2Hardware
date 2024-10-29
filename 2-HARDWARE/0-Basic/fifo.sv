`timescale 1ns / 1ps

module fifo
#(parameter depth=16,width=32)
(
	input clk,rstn,//clock and reset
	input wr_en,rd_en,//enable
	input  [width-1:0]data_in,//input data
	output logic [width-1:0]data_out,//output data
	output logic empty,full//flag
);


logic   [width-1:0] ram[depth-1:0];//dual port　RAM
logic   [$clog2(depth):0] wr_ptr,rd_ptr;//pointer
logic   [$clog2(depth):0] counter;

always_ff@(posedge clk or negedge rstn)
begin
	if(!rstn)
	begin
		counter<=0;
		data_out<=0;
		wr_ptr<=0;
		rd_ptr<=0;
	end
	else
	begin
		case({wr_en,rd_en})
		2'b00: begin
			counter<=counter;
			data_out<='d0;
		end
		2'b01: 
        begin
			data_out<=ram[rd_ptr];//first in first out
		    counter<=counter-1;
		    rd_ptr<=(rd_ptr==depth-1)?0:rd_ptr+1;
		end
		2'b10:
        begin
		    ram[wr_ptr]<=data_in;//write operation
		    counter<=counter+1;
		    wr_ptr<=(wr_ptr==depth-1)?0:wr_ptr+1;
		end
		2'b11:
        begin
		    ram[wr_ptr]<=data_in;//write and read at same time, counter constant
		    data_out<=ram[rd_ptr];
		    wr_ptr<=(wr_ptr==depth-1)?0:wr_ptr+1;
		    rd_ptr<=(rd_ptr==depth-1)?0:rd_ptr+1;
		end 
		endcase
	end
end

assign empty=(counter==0)?1:0;
assign full =(counter==depth-1)?1:0;

endmodule





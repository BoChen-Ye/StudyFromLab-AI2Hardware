`timescale 1ns / 1ps

module data_process_unit#(
	parameter WIDTH = 8,
	parameter DEPTH = 4
)(
input clk,
input rstn,
input en,
input logic [WIDTH-1:0] data_in1,
input logic [WIDTH-1:0] data_in2, 

//output [WIDTH-1:0] addr, 
output data_valid,
output logic [2*WIDTH*DEPTH-1:0] data_out   
);

//======================================================================
// Variable Definition
//======================================================================
logic fifo_rd_en;
logic fifo_empty_1,fifo_empty_2;
logic fifo_full_1,fifo_full_2;
logic [WIDTH-1:0] fifo_out_1,fifo_out_2;
logic [2*WIDTH-1:0] mult_result;
logic [2*WIDTH-1:0] acc_result;

logic valid0,valid1,valid2,valid3;
logic ready1,ready2,ready3;
logic en_s1,en_s2;
logic en_fifo;
logic [2*WIDTH*DEPTH-1:0] dout_parallel;

typedef enum logic [1:0] {
  IDLE,
  LOAD,
  PROC
} STATE_DPU;
STATE_DPU state, state_next;

//======================================================================
// Control FSM
//======================================================================
always_ff@(posedge clk or negedge rstn)begin
   if(!rstn)begin
      state <= IDLE;
   end
   else begin
      state <= state_next;
   end
end

always_comb begin
state_next = state;
en_fifo = 1'd0;
fifo_rd_en=1'd0;
	case(state)
		IDLE: begin		
			if(en)begin
				state_next=LOAD;
			end
		end
		LOAD:begin
			en_fifo = 1'b1;
			if(fifo_full_1&&fifo_full_2)begin
				state_next=PROC;
			end
		end
		PROC:begin
			fifo_rd_en=1'd1;
			if(valid3)begin
				state_next=IDLE;
			end
		end

		default:begin
			state_next = state_next;
		end
	endcase
end


//======================================================================
// Pipeline Stage 0: fetch data from memory
//======================================================================
fifo #(
.depth(DEPTH), 
.width(WIDTH)
) 
u_fifo_1(
.clk,
.rstn,
.wr_en(en_fifo),
.rd_en(fifo_rd_en&& !fifo_empty_1),
.data_in(data_in1),

.data_out(fifo_out_1),
.empty(fifo_empty_1),
.full(fifo_full_1)
);

fifo #(
.depth(DEPTH), 
.width(WIDTH)
) 
u_fifo_2(
.clk,
.rstn,
.wr_en(en_fifo),
.rd_en(fifo_rd_en&& !fifo_empty_2),
.data_in(data_in2),

.data_out(fifo_out_2),
.empty(fifo_empty_2),
.full(fifo_full_2)
);

always_ff @(posedge clk or posedge rstn) begin
    if (!rstn) begin
		valid0<=1'd0;
    end 
	else begin
		valid0 = !fifo_empty_1 && !fifo_empty_2;
    end
end

//assign valid0 = !fifo_empty_1 && !fifo_empty_2;
assign en_s0  = valid0 && ready1 && fifo_rd_en;
//======================================================================
// Pipeline Stage 1: Multiplication
//======================================================================
always_ff @(posedge clk or posedge rstn) begin
    if (!rstn) begin
        mult_result <= 1'b0;
		valid1 <= 1'b0;
    end else if (en_s0) begin
        mult_result <= fifo_out_1 * fifo_out_2;
		valid1 <= 1'b1;
    end
end

assign ready1 = ready2 | !valid0;
assign en_s1  = valid1 && ready2;

//======================================================================
// Pipeline Stage 2: Accumulation
//======================================================================
always_ff @(posedge clk or posedge rstn) begin
    if (!rstn) begin
        acc_result <= 0;
		valid2 <= 1'b0;
    end else if (en_s1) begin
        acc_result <= acc_result + mult_result;
		valid2 <= 1'b1;
    end
end
assign ready2 = ready3 | !valid1;
assign en_s2= valid2 && ready3;
//======================================================================
// Pipeline Stage 2: Shift out
//======================================================================
sipo #(
.WIDTH(2*WIDTH), 
.MAX_NUM(DEPTH)
)
u_sipo(
.clk,
.rstn,
.din_serial(acc_result),
.din_valid(en_s2),

.dout_parallel,
.dout_valid(valid3)
);
always_ff @(posedge clk or posedge rstn) begin
    if (!rstn) begin
		ready3<=1'd0;
    end 
	else begin
		ready3 = !valid3;
    end
end
always_ff @(posedge clk or posedge rstn) begin
    if (!rstn) begin
        data_out <= 'd0;
    end else if (en_s1) begin
        data_out <= dout_parallel;
    end
end

assign data_valid = valid3;

endmodule
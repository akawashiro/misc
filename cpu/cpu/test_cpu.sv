`include "cpu.sv"

module test_pc;
    logic clk;
    logic [31:0] pc_in;
    logic [31:0] pc_out;

    pc pc_inst (
        .clk(clk),
        .pc_in(pc_in),
        .pc_out(pc_out)
    );

    initial begin
        clk = 0;
        pc_in = 4;
        clk = 1;
        #10 $display("pc_out = %d", pc_out);
        clk = 0;
        pc_in = 8;
        clk = 1;
        #10 $display("pc_out = %d", pc_out);
    end
endmodule

module test_pc_plus_4;
    logic [31:0] pc_in;
    logic [31:0] pc_out;

    pc_plus_4 pc_plus_4_inst (
        .pc_in(pc_in),
        .pc_out(pc_out)
    );

    initial begin
        pc_in = 4;
        #10 $display("pc_out = %d", pc_out);
        pc_in = 8;
        #10 $display("pc_out = %d", pc_out);
    end
endmodule

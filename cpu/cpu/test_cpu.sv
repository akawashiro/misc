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

module test_instruction_memory;
    logic [31:0] pc;
    logic [31:0] instruction;

    instruction_memory instruction_memory_inst (
        .pc(pc),
        .instruction(instruction)
    );

    initial begin
        pc = 0;
        #10 $display("instruction = %h", instruction);
        pc = 4;
        #10 $display("instruction = %h", instruction);
    end
endmodule

module test_register_file;
    logic [4:0] rs1;
    logic [4:0] rs2;
    logic [4:0] rd;
    logic [31:0] data_in;
    logic clk;
    logic write_enable;
    logic [31:0] data_out1;
    logic [31:0] data_out2;

    register_file register_file_inst (
        .rs1(rs1),
        .rs2(rs2),
        .rd(rd),
        .data_in(data_in),
        .clk(clk),
        .write_enable(write_enable),
        .data_out1(data_out1),
        .data_out2(data_out2)
    );

    initial begin
        rs1 = 1;
        rs2 = 3;
        rd = 3;
        data_in = 32'hdeadbeef;
        clk = 0;
        write_enable = 0;
        #10 $display("data_out1 = %h, data_out2 = %h", data_out1, data_out2);
        clk = 1;
        write_enable = 1;
        #20 $display("data_out1 = %h, data_out2 = %h", data_out1, data_out2);
    end
endmodule

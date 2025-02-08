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
        #10 assert(pc_out == 4) else $error("pc_out = %d", pc_out);
        clk = 0;
        pc_in = 8;
        clk = 1;
        #20 assert(pc_out == 8) else $error("pc_out = %d", pc_out);
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
        assert(pc_out == 8) else $error("pc_out = %d", pc_out);
        pc_in = 8;
        assert(pc_out == 12) else $error("pc_out = %d", pc_out);
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
        #10 assert(instruction == 32'h005303b3) else $error("instruction = %h", instruction);
        pc = 4;
        #10 assert(instruction == 32'h00000000) else $error("instruction = %h", instruction);
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
        clk = 1;
        write_enable = 1;
        #20 assert(data_out2 == 32'hdeadbeef) else $error("data_out2 = %h", data_out2);
    end
endmodule

module test_alu;
    logic [31:0] a;
    logic [31:0] b;
    logic [2:0] alu_op;
    logic [31:0] result;

    alu alu_inst (
        .a(a),
        .b(b),
        .alu_op(alu_op),
        .result(result)
    );

    initial begin
        a = 4;
        b = 2;
        alu_op = ADD;
        #10 assert(result == 6) else $error("result = %d", result);
        alu_op = SUB;
        #10 assert(result == 2) else $error("result = %d", result);
        alu_op = AND;
        #10 assert(result == 0) else $error("result = %d", result);
        alu_op = OR;
        #10 assert(result == 6) else $error("result = %d", result);
        alu_op = XOR;
        #10 assert(result == 6) else $error("result = %d", result);
        alu_op = SLL;
        #10 assert(result == 16) else $error("result = %d", result);
        alu_op = SRL;
        #10 assert(result == 1) else $error("result = %d", result);
        alu_op = SLT;
        #10 assert(result == 0) else $error("result = %d", result);
    end
endmodule

module test_sign_extend;
    logic [11:0] imm;
    logic [31:0] imm_ext;

    sign_extend sign_extend_inst (
        .imm(imm),
        .imm_ext(imm_ext)
    );

    initial begin
        imm = 12'b101010101010;
        #10 assert(imm_ext == 32'b11111111111111111111101010101010) else $error("imm_ext = %h", imm_ext);
        imm = 12'b010101010101;
        #10 assert(imm_ext == 32'b00000000000000000000010101010101) else $error("imm_ext = %h", imm_ext);
    end
endmodule

module test_cpu;
    logic clk;
    logic [31:0] pc_out_check;
    logic [31:0] instruction_check;

    cpu cpu_inst (
        .clk(clk),
        .pc_out_check(pc_out_check),
        .instruction_check(instruction_check)
    );

    initial begin
        clk = 0;
        #10 assert(pc_out_check == 0) else $error("pc_out_check = %d", pc_out_check);
        #10 assert(instruction_check == 32'h005303b3) else $error("instruction_check = %h", instruction_check);
    end
endmodule

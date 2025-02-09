`include "cpu.sv"

module test_pc;
    logic clk;
    logic reset;
    logic [31:0] pc_in;
    logic [31:0] pc_out;

    pc pc_inst (
        .clk(clk),
        .reset(reset),
        .pc_in(pc_in),
        .pc_out(pc_out)
    );

    initial begin
        clk = 0;
        pc_in = 4;
        clk = 1;
        #10
        assert(pc_out == 4) else $error("pc_out = %d", pc_out);
        clk = 0;
        pc_in = 8;
        clk = 1;
        #10
        assert(pc_out == 8) else $error("pc_out = %d", pc_out);
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
    logic [31:0] initial_instructions [31:0];

    assign initial_instructions[0] = 32'b0000000_00101_00110_000_00111_0110011;
    assign initial_instructions[1] = 32'b0100000_01000_01001_000_01010_0110011;
    assign initial_instructions[2] = 32'b000000000001_01100_000_01101_0010011;
    genvar i;
    generate
        for (i = 3; i < 32; i = i + 1) begin: fill_rom
            assign initial_instructions[i] = 32'b0;
        end
    endgenerate

    instruction_memory instruction_memory_inst (
        .pc(pc),
        .instruction(instruction),
        .initial_instructions(initial_instructions)
    );

    initial begin
        pc = 0;
        #10
        assert(instruction == 32'b0000000_00101_00110_000_00111_0110011) else $error("instruction = %h", instruction);
        pc = 4;
        #10
        assert(instruction == 32'b0100000_01000_01001_000_01010_0110011) else $error("instruction = %h", instruction);
    end
endmodule

module test_register_file;
    logic [4:0] rs1;
    logic [4:0] rs2;
    logic [4:0] rd;
    logic [31:0] data_in;
    logic clk;
    logic reset;
    logic write_enable;
    logic [31:0] data_out1;
    logic [31:0] data_out2;
    // You must use wire instead of logic for the register_check because of
    // https://github.com/steveicarus/iverilog/issues/1001.
    wire [31:0] register_check[0:31];

    register_file register_file_0 (
        .rs1(rs1),
        .rs2(rs2),
        .rd(rd),
        .data_in(data_in),
        .clk(clk),
        .reset(reset),
        .write_enable(write_enable),
        .data_out1(data_out1),
        .data_out2(data_out2),
        .register_check(register_check)
    );

    initial begin
        reset = 0;
        clk = 0;
        #10
        reset = 1;
        clk = 1;
        #10
        clk = 0;
        reset = 0;
        rs1 = 1;
        rs2 = 2;
        rd = 3;
        data_in = 32'hdeadbeef;
        write_enable = 0;
        clk = 1;
        write_enable = 1;
        #10
        assert(data_out1 == 3001) else $error("data_out1 = %h", data_out1);
        assert(data_out2 == 3002) else $error("data_out2 = %h", data_out2);
        assert(register_check[1] == 3001) else $error("register_check[1] = %h", register_check[1]);
        assert(register_check[2] == 3002) else $error("register_check[2] = %h", register_check[2]);
        assert(register_check[3] == 32'hdeadbeef) else $error("register_check[3] = %h", register_check[3]);
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

// See https://riscvasm.lucasteske.dev/
//
// add x7, x6, x5 # x7 <- x6 + x5
// 0x005303b3 in hex
// 0000000_00101_00110_000_00111_0110011 in binary
// opcode: 0110011
// funct3: 000
// funct7: 0000000
// rd:  111 (7)
// rs1: 110 (6)
// rs2: 101 (5)
logic [31:0] add_x7_x6_x5 = 32'b0000000_00101_00110_000_00111_0110011;
// sub x10, x9, x8 # x10 <- x9 - x8
// 0x40848533
// 0100000_01000_01001_000_01010_0110011
// opcode: 0110011
// funct3: 000
// funct7: 0100000
// rd:  01010 (10)
// rs1: 01001 (9)
// rs2: 01000 (8)
logic [31:0] sub_x10_x9_x8 = 32'b0100000_01000_01001_000_01010_0110011;
// addi x13, x12, 0x1 # x13 <- x12 + 1
// 0x00160693
// 000000000001_01100_000_01101_0010011
// opcode: 0010011
// funct3: 000
// rd:  01101 (13)
// rs1: 01100 (12)
// imm: 000000000001
logic [31:0] addi_x13_x12_1 = 32'b000000000001_01100_000_01101_0010011;

module test_cpu;
    logic clk;
    logic reset;
    logic [31:0] pc_out_check;
    logic [31:0] instruction_check;
    logic [2:0] alu_op_check;
    logic [31:0] register_data_out1_check;
    logic [31:0] register_data_out2_check;
    logic [31:0] b_input_check;
    logic [31:0] register_data_in_check;
    logic [31:0] alu_result_check;
    logic [0:0] reg_write_enable_check;
    logic [31:0] register_check [31:0];
    logic [31:0] imm_ext_check;
    logic use_imm_check;
    logic [31:0] initial_instructions [31:0];

    // Fill the ROM with RV32I instructions
    assign initial_instructions[0] = add_x7_x6_x5;
    assign initial_instructions[1] = sub_x10_x9_x8;
    assign initial_instructions[2] = addi_x13_x12_1;

    // Fill the rest of the ROM with 0s
    genvar i;
    generate
        for (i = 3; i < 32; i = i + 1) begin: fill_rom
            assign initial_instructions[i] = 32'b0;
        end
    endgenerate

    cpu cpu_inst (
        .clk(clk),
        .reset(reset),
        .pc_out_check(pc_out_check),
        .instruction_check(instruction_check),
        .alu_op_check(alu_op_check),
        .register_data_out1_check(register_data_out1_check),
        .register_data_out2_check(register_data_out2_check),
        .b_input_check(b_input_check),
        .register_data_in_check(register_data_in_check),
        .alu_result_check(alu_result_check),
        .reg_write_check(reg_write_enable_check),
        .imm_ext_check(imm_ext_check),
        .use_imm_check(use_imm_check),
        .initial_instructions(initial_instructions),
        .register_check_arg(register_check)
    );

    initial begin
        clk = 0;
        reset = 0;
        #10
        clk = 1;
        reset = 1;
        #10
        reset = 0;
        assert(pc_out_check == 0) else $error("pc_out_check = %d", pc_out_check);
        assert(instruction_check == 32'h005303b3) else $error("instruction_check = %h", instruction_check);
        assert(alu_op_check == ADD) else $error("alu_op_check = %d", alu_op_check);
        assert(register_data_out1_check == 3006) else $error("register_data_out1_check = %d", register_data_out1_check);
        assert(register_data_out2_check == 3005) else $error("register_data_out2_check = %d", register_data_out2_check);
        assert(register_check[6] == 3006) else $error("register_check[6] = %d", register_check[6]);
        assert(alu_result_check == 6011) else $error("alu_result_check = %d", alu_result_check);
        assert(use_imm_check == 0) else $error("use_imm_check = %d", use_imm_check);
        assert(register_check[7] == 6011) else $error("register_check[7] = %d", register_check[7]);
        #10
        clk = 0;
        #10
        clk = 1;
        #10
        assert(pc_out_check == 4) else $error("pc_out_check = %d", pc_out_check);
        assert(instruction_check == 32'h40848533) else $error("instruction_check = %h", instruction_check);
        assert(alu_op_check == SUB) else $error("alu_op_check = %d", alu_op_check);
        assert(register_data_out1_check == 3009) else $error("register_data_out1_check = %d", register_data_out1_check);
        assert(register_data_out2_check == 3008) else $error("register_data_out2_check = %d", register_data_out2_check);
        assert(alu_result_check == 1) else $error("alu_result_check = %d", alu_result_check);
        assert(use_imm_check == 0) else $error("use_imm_check = %d", use_imm_check);
        #10
        clk = 0;
        #10
        clk = 1;
        #10
        assert(pc_out_check == 8) else $error("pc_out_check = %d", pc_out_check);
        assert(instruction_check == 32'b000000000001_01100_000_01101_0010011) else $error("instruction_check = %h", instruction_check);
        assert(alu_op_check == ADD) else $error("alu_op_check = %d", alu_op_check);
        assert(imm_ext_check == 12'b000000000001) else $error("imm_ext_check = %b", imm_ext_check);
        assert(use_imm_check == 1) else $error("use_imm_check = %d", use_imm_check);
        assert(register_data_out1_check == 3012) else $error("register_data_out1_check = %d", register_data_out1_check);
        assert(b_input_check == 1) else $error("b_input_check = %d", b_input_check);
        assert(alu_result_check == 3013) else $error("alu_result_check = %d", alu_result_check);
    end
endmodule

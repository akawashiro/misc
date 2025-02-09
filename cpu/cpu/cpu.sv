module pc (
    input logic clk,
    input logic reset,
    input logic [31:0] pc_in,
    output logic [31:0] pc_out
);
    always_ff @(posedge clk)
    begin
        if (reset) begin
            pc_out <= 0;
        end else begin
            pc_out <= pc_in;
        end
    end
endmodule

module pc_plus_4 (
    input logic [31:0] pc_in,
    output logic [31:0] pc_out
);
    assign pc_out = pc_in + 4;
endmodule

module instruction_memory (
    input logic [31:0] pc,
    output logic [31:0] instruction,
    input logic [31:0] initial_instructions [0:31]
);
    logic [31:0] rom [0:31];

    genvar i;
    generate
        for (i = 0; i < 32; i = i + 1) begin: fill_rom
            assign rom[i] = initial_instructions[i];
        end
    endgenerate
   
    assign instruction = rom[pc[6:2]];
endmodule

module register_file (
    input logic [4:0] rs1,
    input logic [4:0] rs2,
    input logic [4:0] rd,
    input logic [31:0] data_in,
    input logic clk,
    input logic reset,
    input logic write_enable,
    output logic [31:0] data_out1,
    output logic [31:0] data_out2,
    input logic [31:0] initial_values [0:31],
    output logic [31:0] register_check [0:31]
);
    logic [31:0] registers [0:31];

    always_comb begin
        data_out1 = registers[rs1];
        data_out2 = registers[rs2];
        for (int j = 0; j < 32; j = j + 1) begin
            register_check[j] = registers[j];
        end
    end
   
    always_ff @(posedge clk)
        if (reset) begin
            for (int i = 0; i < 32; i = i + 1) begin
                registers[i] <= initial_values[i];
            end
        end
        else begin
        if (write_enable) begin
            registers[rd[4:0]] <= data_in;
        end
    end
endmodule

typedef enum logic [2:0] {ADD, SUB, AND, OR, XOR, SLL, SRL, SLT} alu_op_t;

module alu (
    input logic [31:0] a,
    input logic [31:0] b,
    input logic [2:0] alu_op,
    output logic [31:0] result
);
    always_comb begin
        case (alu_op)
            ADD: result = a + b;
            SUB: result = a - b;
            AND: result = a & b;
            OR: result = a | b;
            XOR: result = a ^ b;
            SLL: result = a << b;
            SRL: result = a >> b;
            SLT: result = (a < b) ? 1 : 0;
            default: result = 0;
        endcase
    end
endmodule

module sign_extend (
    input logic [11:0] imm,
    output logic [31:0] imm_ext
);
    assign imm_ext = {{20{imm[11]}}, imm};
endmodule

typedef enum logic [6:0] {
    ALU_WITH_TWO_REGISTERS = 7'b0110011,
    ALU_WITH_IMMEDIATE = 7'b0010011
} OPCODE_TYPE;

module control_unit (
    input logic [6:0] opcode,
    input logic [2:0] funct3,
    input logic [6:0] funct7,
    output logic [2:0] alu_op,
    output logic [0:0] reg_write,
    output logic use_imm
);
    always_comb begin
        case (opcode)
            ALU_WITH_TWO_REGISTERS: begin
                case (funct3)
                    3'b000: begin
                        case (funct7)
                            7'b0000000: alu_op = ADD;
                            7'b0100000: alu_op = SUB;
                        endcase
                    end
                    3'b001: alu_op = SLL;
                    3'b010: alu_op = SLT;
                    3'b011: alu_op = SLT;
                    3'b100: alu_op = XOR;
                    3'b101: alu_op = SRL;
                    3'b110: alu_op = OR;
                    3'b111: alu_op = AND;
                    default: alu_op = ADD;
                endcase
                reg_write = 1;
                use_imm = 0;
            end
            ALU_WITH_IMMEDIATE: begin
                case (funct3)
                    3'b000: alu_op = ADD;
                    3'b001: alu_op = SLL;
                    3'b100: alu_op = XOR;
                    3'b101: alu_op = SRL;
                    3'b110: alu_op = OR;
                    3'b111: alu_op = AND;
                endcase
                reg_write = 1;
                use_imm = 1;
            end
        endcase
    end
endmodule


module b_input_mux (
    input logic [31:0] register_data_out2,
    input logic [31:0] imm_ext,
    input logic use_imm,
    output logic [31:0] b_input
);
    assign b_input = use_imm ? imm_ext : register_data_out2;
endmodule


module cpu (
    input logic clk,
    input logic reset,
    input logic [31:0] initial_instructions [0:31],
    input logic [31:0] initial_register_values [0:31],
    output logic [31:0] pc_out_check,
    output logic [31:0] instruction_check,
    output logic [2:0] alu_op_check,
    output logic [31:0] register_data_out1_check,
    output logic [31:0] register_data_out2_check,
    output logic [31:0] b_input_check,
    output logic [31:0] register_data_in_check,
    output logic [31:0] alu_result_check,
    output logic [0:0] reg_write_check,
    output logic [31:0] imm_ext_check,
    output logic [0:0] use_imm_check,
    output wire [31:0] register_check [0:31]
);
    logic [31:0] pc_in;
    logic [31:0] pc_out;
    logic [31:0] instruction;
    logic [2:0] alu_op;
    logic [31:0] register_data_out1;
    logic [31:0] register_data_out2;
    logic [31:0] register_data_in;
    logic [0:0] reg_write;
    logic [31:0] alu_result;
    logic [31:0] imm_ext;
    logic [0:0] use_imm;

    pc pc_0 (
        .clk(clk),
        .reset(reset),
        .pc_in(pc_in),
        .pc_out(pc_out)
    );
    assign pc_out_check = pc_out;

    pc_plus_4 pc_plus_4_0 (
        .pc_in(pc_out),
        .pc_out(pc_in)
    );

    instruction_memory instruction_memory_0 (
        .pc(pc_out),
        .instruction(instruction),
        .initial_instructions(initial_instructions)
    );
    assign instruction_check = instruction;

    control_unit control_unit_0 (
        .opcode(instruction[6:0]),
        .funct3(instruction[14:12]),
        .funct7(instruction[31:25]),
        .alu_op(alu_op),
        .reg_write(reg_write),
        .use_imm(use_imm)
    );
    assign alu_op_check = alu_op;
    assign reg_write_check = reg_write;
    assign use_imm_check = use_imm;

    sign_extend sign_extend_0 (
        .imm(instruction[31:20]),
        .imm_ext(imm_ext)
    );
    assign imm_ext_check = imm_ext;

    register_file register_file_0 (
        .rs1(instruction[19:15]),
        .rs2(instruction[24:20]),
        .rd(instruction[11:7]),
        .data_in(register_data_in),
        .clk(clk),
        .reset(reset),
        .write_enable(reg_write),
        .data_out1(register_data_out1),
        .data_out2(register_data_out2),
        .register_check(register_check),
        .initial_values(initial_register_values)
    );
    assign register_data_out1_check = register_data_out1;
    assign register_data_out2_check = register_data_out2;
    assign register_data_in_check = register_data_in;

    logic [31:0] b_input;
    b_input_mux b_input_mux_0 (
        .register_data_out2(register_data_out2),
        .imm_ext(imm_ext),
        .use_imm(use_imm),
        .b_input(b_input)
    );
    assign b_input_check = b_input;

    alu alu_0 (
        .a(register_data_out1),
        .b(b_input),
        .alu_op(alu_op),
        .result(alu_result)
    );
    assign alu_result_check = alu_result;
    assign register_data_in = alu_result;
endmodule

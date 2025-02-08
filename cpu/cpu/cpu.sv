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
    output logic [31:0] instruction
);
    logic [31:0] rom [0:31];
   
    // Fill the ROM with RV32I instructions
    //
    // add x7, x6, x5 # x7 <- x6 + x5
    // 0x005303b3 in hex
    // 000000_00101_00110_000_00111_0110011 in binary
    // opcode: 0110011
    // funct3: 000
    // funct7: 0000000
    // rd: 111
    // rs1: 110
    // rs2: 101
    assign rom[0] = 32'b000000_00101_00110_000_00111_0110011;

    // Fill the rest of the ROM with 0s
    genvar i;
    generate
        for (i = 1; i < 32; i = i + 1) begin: fill_rom
            assign rom[i] = 32'b0;
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
    output logic [31:0] data_out2
);
    logic [31:0] registers [0:31];
   
    always_comb begin
        data_out1 = registers[rs1];
        data_out2 = registers[rs2];
    end
   
    always_ff @(posedge clk)
        if (reset) begin
            for (int i = 0; i < 32; i = i + 1) begin
                // TODO: Initialize the registers to 0. Now 3 is used for
                // testing.
                registers[i] <= 3;
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

module cpu (
    input logic clk,
    input logic reset,
    output logic [31:0] pc_out_check,
    output logic [31:0] instruction_check
);
    logic [31:0] pc_in;
    logic [31:0] pc_out;
    logic [31:0] instruction;

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
        .instruction(instruction)
    );
    assign instruction_check = instruction;
endmodule

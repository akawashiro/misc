module pc (
    input logic clk,
    input logic [31:0] pc_in,
    output logic [31:0] pc_out
);
    always_ff @(posedge clk)
    begin
        pc_out <= pc_in;
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

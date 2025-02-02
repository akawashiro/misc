`include "popcnt.sv"

module test_popcnt;
  localparam int BitWidth = 5;
  reg [BitWidth-1:0] data;
  reg [$clog2(BitWidth+1)-1:0] count;
  popcnt #(
      .WIDTH(BitWidth)
  ) _popcnt (
      .i_data (data),
      .o_count(count)
  );
  initial begin
    for (int i = 0; i < (1<<BitWidth); ++i) begin
      data = i;
      #10 $display("%b, %d", data, count);
    end
  end
endmodule

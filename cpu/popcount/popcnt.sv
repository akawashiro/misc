module popcnt #(
    parameter int WIDTH = 64
) (
    input wire [WIDTH-1:0] i_data,
    output wire [$clog2(WIDTH+1)-1:0] o_count
);
  wire [$clog2((WIDTH >> 1) + 1)-1:0] left_count;
  wire [$clog2(WIDTH - (WIDTH >> 1) + 1)-1:0] right_count;
  generate
    if (WIDTH == 1) assign o_count = i_data;
    else begin : gen_count_tree
      assign o_count = left_count + right_count;
      popcnt #(
          .WIDTH(WIDTH >> 1)
      ) left_popcnt (
          .i_data (i_data[(WIDTH>>1)-1:0]),
          .o_count(left_count)
      );
      popcnt #(
          .WIDTH(WIDTH - (WIDTH >> 1))
      ) right_popcnt (
          .i_data (i_data[WIDTH-1:(WIDTH>>1)]),
          .o_count(right_count)
      );
    end
  endgenerate
endmodule

`timescale 1ns/1ps
`default_nettype none
// Rate-equality comparator eq_rc: y_t = 1{ VALUE_GAIN*|count| / t <= tolerance },
// with count = sum(i_input_0 - i_input_1) and t the elapsed timestep count, both
// updated with this timestep's spikes, so the output is combinational (pp_delay=0).
// The float tolerance is carried as the exact fraction TOL_NUM/TOL_DEN, giving the
// bit-exact integer relation VALUE_GAIN*TOL_DEN*|count| <= TOL_NUM*t.
// One posedge == one Python timestep; active-low i_rst_n clears count and t to 0.


module eq_rc #(
    parameter VALUE_GAIN = 1,    // rate-code value gain: 1 unipolar, 2 bipolar
    parameter TOL_NUM    = 1,    // tolerance numerator   (tolerance = TOL_NUM/TOL_DEN)
    parameter TOL_DEN    = 10,   // tolerance denominator
    parameter TW         = 16    // width of the elapsed-timestep counter
) (
    input  wire i_clk,
    input  wire i_rst_n,
    input  wire i_input_0,
    input  wire i_input_1,
    output wire o_output
);
    // compare width: TW plus headroom for the VALUE_GAIN*TOL_DEN and TOL_NUM factors.
    localparam PW = TW + 8;

    reg signed [TW:0]   count;   // running spike-count difference, range [-t, t]
    reg        [TW-1:0] t;       // elapsed timestep count since reset

    // count and t reflect this timestep's spikes (output is post-update).
    wire signed [TW:0]   count_next = count + $signed({{TW{1'b0}}, i_input_0})
                                            - $signed({{TW{1'b0}}, i_input_1});
    wire        [TW-1:0] t_next     = t + 1'b1;

    wire [TW:0] abs_count = count_next[TW] ? (~count_next + 1'b1) : count_next;

    wire [PW-1:0] lhs = VALUE_GAIN * TOL_DEN * abs_count;
    wire [PW-1:0] rhs = TOL_NUM * t_next;
    assign o_output = (lhs <= rhs);

    always @(posedge i_clk or negedge i_rst_n) begin
        if (!i_rst_n) begin
            count <= {(TW+1){1'b0}};
            t     <= {TW{1'b0}};
        end else begin
            count <= count_next;
            t     <= t_next;
        end
    end
endmodule
`default_nettype wire

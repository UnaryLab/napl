`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH mirrors the Python model configuration.
`include "mul_ugemm/vec/mul_ugemm_params.vh"
// Python golden rows are <rst> <in_0> <in_1u> <out_u> <in_1b> <out_b>.
// Outputs use pre-update ROM indices; rst=1 first clears both counters.
// Co-sim: make test OP=mul_ugemm


module mul_ugemm_tb;
    reg                 i_clk;
    reg                 i_rst_n;
    reg                 i_input_0;
    reg  [`GEN_WIDTH:0] i_input_1u;   // operand bus = WIDTH+1 bits
    reg  [`GEN_WIDTH:0] i_input_1b;
    wire                o_output_uni;
    wire                o_output_bi;

    mul_ugemm_unipolar #(.WIDTH(`GEN_WIDTH)) dut_u (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input_0  (i_input_0),
        .i_input_1  (i_input_1u),
        .o_output   (o_output_uni)
    );

    mul_ugemm_bipolar #(.WIDTH(`GEN_WIDTH)) dut_b (
        .i_clk   (i_clk),
        .i_rst_n (i_rst_n),
        .i_input_0  (i_input_0),
        .i_input_1  (i_input_1b),
        .o_output   (o_output_bi)
    );

    integer fd, code, n, fails;
    reg          rst;
    reg          a;
    integer      in1u, out_u, in1b, out_b;
    reg [1023:0] hdr_line;

    initial begin
        i_clk   = 1'b0;
        i_rst_n = 1'b1;
        i_input_0  = 1'b0;
        i_input_1u = {(`GEN_WIDTH+1){1'b0}};
        i_input_1b = {(`GEN_WIDTH+1){1'b0}};

        fd = $fopen("vec/mul_ugemm.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/mul_ugemm.vec (run `make vectors` first)");
            $finish;
        end

        // skip the header line
        code = $fgets(hdr_line, fd);

        n = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(fd, "%d %d %d %d %d %d\n", rst, a, in1u, out_u, in1b, out_b);
            if (code == 6) begin
                // reset boundary: reload the index counters to 0 before this cycle
                if (rst == 1) begin
                    i_rst_n = 1'b0;
                    #1;
                    i_rst_n = 1'b1;
                    #1;
                end

                i_input_0  = a;
                i_input_1u = in1u[`GEN_WIDTH:0];
                i_input_1b = in1b[`GEN_WIDTH:0];
                #1;

                n = n + 1;
                if (o_output_uni !== out_u[0]) begin
                    $display("FAIL uni n=%0d in_0=%b in_1=%0d : got %b exp %0d",
                             n, a, in1u, o_output_uni, out_u);
                    fails = fails + 1;
                end
                if (o_output_bi !== out_b[0]) begin
                    $display("FAIL bi  n=%0d in_0=%b in_1=%0d : got %b exp %0d",
                             n, a, in1b, o_output_bi, out_b);
                    fails = fails + 1;
                end

                // clock edge advances the counters for the next cycle
                i_clk = 1'b1; #1;
                i_clk = 1'b0; #1;
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS mul_ugemm: %0d/%0d vectors (unipolar+bipolar)", n, n);
        else
            $display("FAIL mul_ugemm: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire

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

    // LINE_BYTES sizes the line buffer, which holds LINE_BYTES bytes, so a
    // newline-terminated row carries at most LINE_BYTES-1 payload bytes, and
    // DATA_COLS is the column count every data row carries. Neither constant
    // can be set wrong and still let a malformed row through, because the
    // trailing sentinel makes the column check two-sided. A DATA_COLS set too
    // large fatals on the first clean row. A LINE_BYTES set too large costs
    // guard precision rather than correctness, because the length assertion
    // can then fire only at the buffer boundary instead of near the true row
    // width, so both constants are measured from the vec file rather than
    // guessed.
    localparam LINE_BYTES = 64;
    localparam DATA_COLS  = 6;

    // The operand columns are consumed through a WIDTH+1 bit bus, so a token
    // wider than that bus is rejected below rather than silently truncated at
    // the bus. OPERAND_MAX is the widest value the bus can carry; the two
    // one-bit columns are the same rule at a one-bit sink.
    localparam OPERAND_MAX = (1 << (`GEN_WIDTH + 1)) - 1;

    integer fd, code, chars, n, fails;
    reg          rst;
    reg          a;
    integer      out_u, out_b;
    integer      in1u, in1b;
    // The one-bit columns are scanned as text so a token outside {0, 1} always
    // reaches the guard below; the reg is wider than a legal token so an
    // over-wide token reads back over-wide instead of saturating.
    reg [8*8-1:0]          tok_rst;
    reg [8*8-1:0]          tok_a;
    reg [8*8-1:0]          extra;
    reg [8*LINE_BYTES-1:0] line;

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

        // HEADER GUARD: the header is read through the same buffer as a data
        // row, so it takes the same truncation check.
        chars = $fgets(line, fd);
        if (chars == 0)
            $fatal(1, "mul_ugemm: vec file has no header line");
        if (chars == LINE_BYTES && line[7:0] !== "\n")
            $fatal(1, "mul_ugemm: header line fills the %0d byte line buffer", LINE_BYTES);

        n = 0;
        fails = 0;
        // Each row is compared before the posedge, so this testbench can only drive a
        // pp_delay of 0; that pre-posedge sampling is what enforces it. This check
        // rejects a model whose pp_delay stopped agreeing with that assumption.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL mul_ugemm: testbench samples combinationally, model pp_delay is %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // A row carrying any other column count, a row that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES && line[7:0] !== "\n")
                    $fatal(1, "mul_ugemm: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES);
                // The field count of this format string must match DATA_COLS.
                code = $sscanf(line, "%s %s %d %d %d %d %s", tok_rst, tok_a, in1u, out_u, in1b, out_b, extra);
                if (code == 0) begin
                    // A blank line is benign only at end of file.
                    if ($fgets(line, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "mul_ugemm: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS)
                        $fatal(1, "mul_ugemm: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS);
                    // INPUT COLUMN GUARDS: a stimulus column wider than the port
                    // that carries it is a corrupt file, and is named as one
                    // rather than truncated into a value the DUT would accept as
                    // stimulus. rst and in_0 drive one-bit ports; the operand
                    // columns drive the WIDTH+1 bit bus.
                    if (tok_rst !== "0" && tok_rst !== "1")
                        $fatal(1, "mul_ugemm: row %0d input field rst is \"%0s\", expected a single 0 or 1", n + 1, tok_rst);
                    rst = (tok_rst == "1");
                    if (tok_a !== "0" && tok_a !== "1")
                        $fatal(1, "mul_ugemm: row %0d input field in_0 is \"%0s\", expected a single 0 or 1", n + 1, tok_a);
                    a = (tok_a == "1");
                    // The operand buses are multi-bit, so their unknown check is a
                    // reduction over every bit rather than a two-value compare.
                    if ((^in1u) === 1'bx)
                        $fatal(1, "mul_ugemm: row %0d field in_1u is unknown (%b)", n + 1, in1u);
                    if (in1u < 0 || in1u > OPERAND_MAX)
                        $fatal(1, "mul_ugemm: row %0d field in_1u is out of range (%0d), the %0d bit operand holds 0 to %0d",
                               n + 1, in1u, `GEN_WIDTH + 1, OPERAND_MAX);
                    if ((^in1b) === 1'bx)
                        $fatal(1, "mul_ugemm: row %0d field in_1b is unknown (%b)", n + 1, in1b);
                    if (in1b < 0 || in1b > OPERAND_MAX)
                        $fatal(1, "mul_ugemm: row %0d field in_1b is out of range (%0d), the %0d bit operand holds 0 to %0d",
                               n + 1, in1b, `GEN_WIDTH + 1, OPERAND_MAX);
                    // EXPECTED COLUMN GUARDS: the expected-output columns are read
                    // as integers so a golden value outside {0, 1} is named as a
                    // malformed field rather than truncated into a bit that can
                    // match the DUT.
                    if ((^out_u) === 1'bx)
                        $fatal(1, "mul_ugemm: row %0d field out_uni is unknown (%b)", n + 1, out_u);
                    if (out_u != 0 && out_u != 1)
                        $fatal(1, "mul_ugemm: row %0d field out_uni is out of range (%0d)", n + 1, out_u);
                    if ((^out_b) === 1'bx)
                        $fatal(1, "mul_ugemm: row %0d field out_bi is unknown (%b)", n + 1, out_b);
                    if (out_b != 0 && out_b != 1)
                        $fatal(1, "mul_ugemm: row %0d field out_bi is out of range (%0d)", n + 1, out_b);

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
                    if (o_output_uni !== out_u) begin
                        $display("FAIL uni n=%0d in_0=%b in_1=%0d : got %b exp %0d",
                                 n, a, in1u, o_output_uni, out_u);
                        fails = fails + 1;
                    end
                    if (o_output_bi !== out_b) begin
                        $display("FAIL bi  n=%0d in_0=%b in_1=%0d : got %b exp %0d",
                                 n, a, in1b, o_output_bi, out_b);
                        fails = fails + 1;
                    end

                    // clock edge advances the counters for the next cycle
                    i_clk = 1'b1; #1;
                    i_clk = 1'b0; #1;
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL mul_ugemm: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS mul_ugemm: %0d/%0d vectors (unipolar+bipolar)", n, n);
        else
            $display("FAIL mul_ugemm: %0d mismatch(es) over %0d vectors", fails, n);
        $finish;
    end
endmodule
`default_nettype wire

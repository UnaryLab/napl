`timescale 1ns/1ps
`default_nettype none
// Generated WIDTH/FRAC mirror the Python model configuration.
`include "encode/vec/encode_params.vh"
// Python golden rows are <rst> <dim> <i_input> <o_spike>; the spike is checked in
// the arrival cycle, before the posedge advances the Sobol generator. rst=1
// requests an active-low reset to the sequence start before its row. One encoder
// per covered Sobol dimension runs in lockstep and the row's dim selects which
// one is compared.
// Co-sim: make test OP=encode


module encode_tb;
    localparam WIDTH = `GEN_WIDTH;
    localparam FRAC  = `GEN_FRAC;
    localparam DIMS  = `GEN_DIMS;

    reg  clk, rst_n;
    reg  [FRAC:0] value;
    wire [DIMS:1] spike;

    // One encoder per dimension, each on the generator's table for that dim.
    // The name is built from the genvar, so it tracks `GEN_DIMS: the character
    // count matches the module's default, which a longer override would be
    // truncated to.
    genvar d;
    generate
        for (d = 1; d <= DIMS; d = d + 1) begin : g_dut
            localparam [7:0] DIGIT = 8'd48 + d;
            encode #(
                .WIDTH       (WIDTH),
                .FRAC        (FRAC),
                .DIRVEC_FILE ({"vec/encode_dirvec_d", DIGIT, ".hex"})
            ) dut (
                .i_clk   (clk),
                .i_rst_n (rst_n),
                .i_input (value),
                .o_spike (spike[d])
            );
        end
    endgenerate

    // Sequence-generator block: the three remaining hardware number-sequence
    // generators plus a WIDTH=1 encode, which is what walks sobol.v's two-entry
    // table branch. They are held in reset through the encode phase above and
    // walked afterwards against vec/encode_gen.vec.
    localparam WIDTH_W1 = `GEN_WIDTH_W1;
    localparam FRAC_W1  = `GEN_FRAC_W1;

    reg  rst_n_g;
    reg  [FRAC_W1:0] value_w1;
    reg  [FRAC:0]    value_g;
    wire [WIDTH-1:0] rand_lfsr, rand_lfsr_ext, rand_tc;
    wire spike_w1;
    // One full encoder per non-Sobol GENERATOR mode, on the values the mapping
    // resolves for that generator: this is what pins the selection in encode.v.
    wire [3:1] spike_mode;

    encode #(
        .WIDTH     (WIDTH),
        .FRAC      (FRAC),
        .GENERATOR (`GEN_MODE_LFSR),
        .TAPS      (`GEN_TAPS),
        .SEED      (`GEN_SEED)
    ) u_encode_lfsr (
        .i_clk   (clk),
        .i_rst_n (rst_n_g),
        .i_input (value_g),
        .o_spike (spike_mode[1])
    );

    encode #(
        .WIDTH     (WIDTH),
        .FRAC      (FRAC),
        .GENERATOR (`GEN_MODE_LFSR_EXT),
        .TAPS      (`GEN_TAPS),
        .SEED      (`GEN_SEED)
    ) u_encode_lfsr_ext (
        .i_clk   (clk),
        .i_rst_n (rst_n_g),
        .i_input (value_g),
        .o_spike (spike_mode[2])
    );

    encode #(
        .WIDTH     (WIDTH),
        .FRAC      (FRAC),
        .GENERATOR (`GEN_MODE_TC)
    ) u_encode_tc (
        .i_clk   (clk),
        .i_rst_n (rst_n_g),
        .i_input (value_g),
        .o_spike (spike_mode[3])
    );

    lfsr #(
        .WIDTH (WIDTH),
        .TAPS  (`GEN_TAPS),
        .SEED  (`GEN_SEED)
    ) u_lfsr (
        .i_clk   (clk),
        .i_rst_n (rst_n_g),
        .i_en    (1'b1),
        .o_rand  (rand_lfsr)
    );

    lfsr_ext #(
        .WIDTH (WIDTH),
        .TAPS  (`GEN_TAPS),
        .SEED  (`GEN_SEED)
    ) u_lfsr_ext (
        .i_clk   (clk),
        .i_rst_n (rst_n_g),
        .i_en    (1'b1),
        .o_rand  (rand_lfsr_ext)
    );

    temporalseq #(
        .WIDTH (WIDTH)
    ) u_temporalseq (
        .i_clk   (clk),
        .i_rst_n (rst_n_g),
        .i_en    (1'b1),
        .o_rand  (rand_tc)
    );

    encode #(
        .WIDTH       (WIDTH_W1),
        .FRAC        (FRAC_W1),
        .DIRVEC_FILE ("vec/encode_dirvec_d1.hex")
    ) u_encode_w1 (
        .i_clk   (clk),
        .i_rst_n (rst_n_g),
        .i_input (value_w1),
        .o_spike (spike_w1)
    );

    // Width sweep: one instance of every sequence circuit at every table width
    // the mapping accepts, so a width-dependent elaboration fault turns this
    // target red. sobol and temporalseq carry WIDTH from 1, the narrowest
    // sequence with a point; lfsr and lfsr_ext carry it from 2, the narrowest
    // width with a feedback polynomial. The top is mapping.yaml's tap-table
    // ceiling of 12, which the mapping applies to lfsr and lfsr_ext alone;
    // sobol and temporalseq carry a lower guard only and no upper bound.
    // The sweep instances hold i_en low, so each stays on its post-reset value:
    // 0 for sobol and temporalseq, SEED for lfsr and lfsr_ext. Their sequences
    // are walked at `GEN_WIDTH and WIDTH=1 by the phases above; here the width
    // is what is covered. The generator emits one dimension-1 table per swept
    // width, named vec/encode_dv_w<NN>.hex, so each instance's $readmemb range
    // matches its file exactly.
    localparam SWEEP_HI = `GEN_SWEEP_HI;

    wire [SWEEP_HI:1] sweep_ok;

    genvar w;
    generate
        for (w = 1; w <= SWEEP_HI; w = w + 1) begin : g_sweep
            wire [w-1:0] rand_sobol_w, rand_tc_w;

            // Two ASCII digits, so the name tracks the genvar up to SWEEP_HI 99.
            // WIDTH 1 takes sobol's table branch and reads no file.
            localparam [7:0] TENS = 8'd48 + w / 10;
            localparam [7:0] ONES = 8'd48 + w % 10;

            sobol #(
                .WIDTH       (w),
                .DIRVEC_FILE ({"vec/encode_dv_w", TENS, ONES, ".hex"})
            ) u_sobol (
                .i_clk   (clk),
                .i_rst_n (rst_n),
                .i_en    (1'b0),
                .o_rand  (rand_sobol_w)
            );

            temporalseq #(
                .WIDTH (w)
            ) u_temporalseq (
                .i_clk   (clk),
                .i_rst_n (rst_n),
                .i_en    (1'b0),
                .o_rand  (rand_tc_w)
            );

            if (w >= 2) begin : g_lfsr_forms
                wire [w-1:0] rand_lfsr_w, rand_lfsr_ext_w;

                lfsr #(
                    .WIDTH (w),
                    .SEED  (1)
                ) u_lfsr (
                    .i_clk   (clk),
                    .i_rst_n (rst_n),
                    .i_en    (1'b0),
                    .o_rand  (rand_lfsr_w)
                );

                lfsr_ext #(
                    .WIDTH (w),
                    .SEED  (1)
                ) u_lfsr_ext (
                    .i_clk   (clk),
                    .i_rst_n (rst_n),
                    .i_en    (1'b0),
                    .o_rand  (rand_lfsr_ext_w)
                );

                assign sweep_ok[w] = (rand_sobol_w    === {w{1'b0}})
                                  && (rand_tc_w       === {w{1'b0}})
                                  && (rand_lfsr_w     === {{(w-1){1'b0}}, 1'b1})
                                  && (rand_lfsr_ext_w === {{(w-1){1'b0}}, 1'b1});
            end else begin : g_no_lfsr_forms
                assign sweep_ok[w] = (rand_sobol_w === {w{1'b0}})
                                  && (rand_tc_w    === {w{1'b0}});
            end
        end
    endgenerate

    // This file has two read loops, so it carries one constant pair per phase
    // rather than a single LINE_BYTES and DATA_COLS.
    // The encode phase reads encode.vec into a LINE_BYTES_ENC-byte buffer, so
    // one of its newline-terminated rows carries at most LINE_BYTES_ENC-1
    // payload bytes, and DATA_COLS_ENC is the 4-column width every encode.vec
    // data row carries: that loop's format string scans one trailing %s
    // sentinel past DATA_COLS_ENC, which makes its code != DATA_COLS_ENC fatal
    // two-sided, so a DATA_COLS_ENC set too large fatals on the first clean
    // encode.vec row while a LINE_BYTES_ENC set too large costs guard
    // precision rather than correctness, and both are measured from encode.vec
    // rather than guessed.
    // The generator phase reads encode_gen.vec into a LINE_BYTES_GEN-byte
    // buffer, so one of its newline-terminated rows carries at most
    // LINE_BYTES_GEN-1 payload bytes, and DATA_COLS_GEN is the 10-column width
    // every encode_gen.vec data row carries: that loop's format string scans
    // one trailing %s sentinel past DATA_COLS_GEN, which makes its code !=
    // DATA_COLS_GEN fatal two-sided, so a DATA_COLS_GEN set too large fatals on
    // the first clean encode_gen.vec row while a LINE_BYTES_GEN set too large
    // costs guard precision rather than correctness, and both are measured from
    // encode_gen.vec rather than guessed.
    localparam LINE_BYTES_ENC = 32;
    localparam DATA_COLS_ENC  = 4;
    localparam LINE_BYTES_GEN = 64;
    localparam DATA_COLS_GEN  = 10;

    // Every column is consumed through a sink of a fixed width: a value bus, a
    // sequence bus, or a single spike bit. A token wider than its sink is
    // rejected below rather than silently truncated at the sink, so these are
    // the widest values each sink can carry. The one-bit columns are the same
    // rule at a one-bit sink, and dim is an index rather than a value, so its
    // bound is the instantiated dimension range.
    localparam VALUE_MAX    = (1 << (FRAC + 1)) - 1;
    localparam VALUE_W1_MAX = (1 << (FRAC_W1 + 1)) - 1;
    localparam RAND_MAX     = (1 << WIDTH) - 1;

    integer fd, code, chars, n, fails, sw;
    integer dim, in_value;
    integer fd_g, m, exp_lfsr, exp_lfsr_ext, exp_tc, in_w1;
    integer in_mode;
    reg rst, exp_spike, exp_spike_w1;
    reg exp_mode_lfsr, exp_mode_lfsr_ext, exp_mode_tc;
    reg got;
    // The one-bit columns are scanned as text so a token outside {0, 1} always
    // reaches the guard below; each reg is wider than a legal token so an
    // over-wide token reads back over-wide instead of saturating.
    reg [8*8-1:0]              tok_rst;
    reg [8*8-1:0]              tok_spike;
    reg [8*8-1:0]              tok_spike_w1;
    reg [8*8-1:0]              tok_mode_lfsr;
    reg [8*8-1:0]              tok_mode_lfsr_ext;
    reg [8*8-1:0]              tok_mode_tc;
    reg [8*8-1:0]              extra;
    reg [8*LINE_BYTES_ENC-1:0] line_enc;
    reg [8*LINE_BYTES_GEN-1:0] line_gen;

    // 10 ns clock period.
    initial clk = 1'b0;
    always #5 clk = ~clk;

    initial begin
        value    = {(FRAC+1){1'b0}};
        value_w1 = {(FRAC_W1+1){1'b0}};
        value_g  = {(FRAC+1){1'b0}};
        rst_n    = 1'b1;
        // The sequence generators stay in reset until their own phase starts.
        rst_n_g  = 1'b0;
        n        = 0;
        m        = 0;
        fails    = 0;

        fd = $fopen("vec/encode.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/encode.vec (run `make vectors` first)");
            $finish;
        end

        // pp_delay is 0: the spike is combinational in the arrival cycle.
        if (`GEN_PP_DELAY != 0) begin
            $display("FAIL encode: observed latency 0, expected pp_delay %0d", `GEN_PP_DELAY);
            fails = fails + 1;
        end

        // Reset to the model's post-reset() state (sequence value 0). Hold rst_n
        // low across a posedge, then deassert on the falling edge so the first
        // driven vector sees that state with no stray advancing edge in between.
        @(negedge clk);
        rst_n = 1'b0;
        @(posedge clk);
        @(negedge clk);
        rst_n = 1'b1;

        // Width sweep: each swept width holds its post-reset sequence value.
        for (sw = 1; sw <= SWEEP_HI; sw = sw + 1) begin
            if (sweep_ok[sw] !== 1'b1) begin
                $display("FAIL encode width sweep: sequence generators at WIDTH=%0d do not hold their post-reset value",
                         sw);
                fails = fails + 1;
            end
        end

        // A row carrying any other column count, a row that fills the line
        // buffer, or an x or z data field ends the run with a fatal at that row,
        // so a row can never borrow a column from its neighbour. A blank line is
        // benign only at end of file.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line_enc, fd);
            if (chars != 0) begin
                if (chars == LINE_BYTES_ENC && line_enc[7:0] !== "\n")
                    $fatal(1, "encode: row %0d fills the %0d byte line buffer", n + 1, LINE_BYTES_ENC);
                // The field count of this format string must match DATA_COLS_ENC.
                // The trailing %s catches a row with a column to spare.
                code = $sscanf(line_enc, "%s %d %d %s %s", tok_rst, dim, in_value, tok_spike, extra);
                if (code == 0) begin
                    if ($fgets(line_enc, fd) == 0)
                        chars = 0;
                    else
                        $fatal(1, "encode: row %0d is blank", n + 1);
                end else begin
                    if (code != DATA_COLS_ENC)
                        $fatal(1, "encode: row %0d scanned %0d columns, expected %0d", n + 1, code, DATA_COLS_ENC);
                    // The wide numeric columns land in integers, so their unknown
                    // check is a reduction over every bit.
                    // INPUT COLUMN GUARDS: a stimulus column wider than the sink
                    // that carries it is a corrupt file, and is named as one
                    // rather than truncated into a value the DUT would accept as
                    // stimulus.
                    if (tok_rst !== "0" && tok_rst !== "1")
                        $fatal(1, "encode: row %0d input field rst is \"%0s\", expected a single 0 or 1", n + 1, tok_rst);
                    rst = (tok_rst == "1");
                    if ((^dim) === 1'bx)
                        $fatal(1, "encode: row %0d field dim is unknown (%b)", n + 1, dim);
                    if (dim < 1 || dim > DIMS)
                        $fatal(1, "encode: row %0d field dim is out of range (%0d), %0d encoders are instantiated",
                               n + 1, dim, DIMS);
                    if ((^in_value) === 1'bx)
                        $fatal(1, "encode: row %0d field i_input is unknown (%b)", n + 1, in_value);
                    if (in_value < 0 || in_value > VALUE_MAX)
                        $fatal(1, "encode: row %0d field i_input is out of range (%0d), the %0d bit value holds 0 to %0d",
                               n + 1, in_value, FRAC + 1, VALUE_MAX);
                    // EXPECTED COLUMN GUARD: the spike is compared through a
                    // one-bit sink, so a golden spike outside {0, 1} is named as
                    // a corrupt answer rather than truncated into a bit that can
                    // match the DUT.
                    if (tok_spike !== "0" && tok_spike !== "1")
                        $fatal(1, "encode: row %0d expected field o_spike is \"%0s\", expected a single 0 or 1", n + 1, tok_spike);
                    exp_spike = (tok_spike == "1");

                    if (rst != 0) begin
                        rst_n = 1'b0;
                        @(posedge clk);
                        @(negedge clk);
                        rst_n = 1'b1;
                    end
                    // Drive from the negedge: the sequence register samples on the
                    // posedge, so an input changing at that edge would race it.
                    value = in_value[FRAC:0];
                    #1;
                    n = n + 1;
                    got = spike[dim];
                    if (got !== exp_spike) begin
                        $display("FAIL t=%0d dim=%0d value=%0d : got %b exp %0d",
                                 n, dim, in_value, got, exp_spike);
                        fails = fails + 1;
                    end
                    @(posedge clk);
                    @(negedge clk);
                end
            end
        end
        $fclose(fd);

        // Every generated row must have been compared, so a vec file that lost
        // rows to a format drift fails instead of passing on the rows it kept.
        if (n != `GEN_VECTORS) begin
            $display("FAIL encode: compared %0d vectors, generator wrote %0d",
                     n, `GEN_VECTORS);
            fails = fails + 1;
        end

        // Sequence-generator phase. Each row carries all three generators plus the
        // WIDTH=1 encode, so they run in lockstep off one row stream.
        fd_g = $fopen("vec/encode_gen.vec", "r");
        if (fd_g == 0) begin
            $display("ERROR: cannot open vec/encode_gen.vec (run `make vectors` first)");
            $finish;
        end

        // Same row discipline as the phase above, on this file's own row shape.
        chars = 1;
        while (chars != 0) begin
            chars = $fgets(line_gen, fd_g);
            if (chars != 0) begin
                if (chars == LINE_BYTES_GEN && line_gen[7:0] !== "\n")
                    $fatal(1, "encode: generator row %0d fills the %0d byte line buffer", m + 1, LINE_BYTES_GEN);
                // The field count of this format string must match DATA_COLS_GEN.
                // The trailing %s catches a row with a column to spare.
                code = $sscanf(line_gen, "%s %d %d %d %d %s %d %s %s %s %s",
                               tok_rst, exp_lfsr, exp_lfsr_ext, exp_tc, in_w1, tok_spike_w1,
                               in_mode, tok_mode_lfsr, tok_mode_lfsr_ext, tok_mode_tc, extra);
                if (code == 0) begin
                    if ($fgets(line_gen, fd_g) == 0)
                        chars = 0;
                    else
                        $fatal(1, "encode: generator row %0d is blank", m + 1);
                end else begin
                    if (code != DATA_COLS_GEN)
                        $fatal(1, "encode: generator row %0d scanned %0d columns, expected %0d", m + 1, code, DATA_COLS_GEN);
                    // The wide numeric columns land in integers, so their unknown
                    // check is a reduction over every bit.
                    // INPUT COLUMN GUARDS: a stimulus column wider than the sink
                    // that carries it is a corrupt file, and is named as one
                    // rather than truncated into a value the DUT would accept as
                    // stimulus.
                    if (tok_rst !== "0" && tok_rst !== "1")
                        $fatal(1, "encode: generator row %0d input field rst is \"%0s\", expected a single 0 or 1", m + 1, tok_rst);
                    rst = (tok_rst == "1");
                    if ((^in_w1) === 1'bx)
                        $fatal(1, "encode: generator row %0d field i_input_w1 is unknown (%b)", m + 1, in_w1);
                    if (in_w1 < 0 || in_w1 > VALUE_W1_MAX)
                        $fatal(1, "encode: generator row %0d field i_input_w1 is out of range (%0d), the %0d bit value holds 0 to %0d",
                               m + 1, in_w1, FRAC_W1 + 1, VALUE_W1_MAX);
                    if ((^in_mode) === 1'bx)
                        $fatal(1, "encode: generator row %0d field i_input_mode is unknown (%b)", m + 1, in_mode);
                    if (in_mode < 0 || in_mode > VALUE_MAX)
                        $fatal(1, "encode: generator row %0d field i_input_mode is out of range (%0d), the %0d bit value holds 0 to %0d",
                               m + 1, in_mode, FRAC + 1, VALUE_MAX);
                    // EXPECTED COLUMN GUARDS: the three sequence columns are
                    // compared through a WIDTH bit sink and the four spike
                    // columns through a one-bit sink, so a golden value wider
                    // than its sink is named as a corrupt answer rather than
                    // truncated into a value that can match the DUT.
                    if ((^exp_lfsr) === 1'bx)
                        $fatal(1, "encode: generator row %0d field lfsr is unknown (%b)", m + 1, exp_lfsr);
                    if (exp_lfsr < 0 || exp_lfsr > RAND_MAX)
                        $fatal(1, "encode: generator row %0d field lfsr is out of range (%0d), the %0d bit sequence holds 0 to %0d",
                               m + 1, exp_lfsr, WIDTH, RAND_MAX);
                    if ((^exp_lfsr_ext) === 1'bx)
                        $fatal(1, "encode: generator row %0d field lfsr_ext is unknown (%b)", m + 1, exp_lfsr_ext);
                    if (exp_lfsr_ext < 0 || exp_lfsr_ext > RAND_MAX)
                        $fatal(1, "encode: generator row %0d field lfsr_ext is out of range (%0d), the %0d bit sequence holds 0 to %0d",
                               m + 1, exp_lfsr_ext, WIDTH, RAND_MAX);
                    if ((^exp_tc) === 1'bx)
                        $fatal(1, "encode: generator row %0d field temporalseq is unknown (%b)", m + 1, exp_tc);
                    if (exp_tc < 0 || exp_tc > RAND_MAX)
                        $fatal(1, "encode: generator row %0d field temporalseq is out of range (%0d), the %0d bit sequence holds 0 to %0d",
                               m + 1, exp_tc, WIDTH, RAND_MAX);
                    if (tok_spike_w1 !== "0" && tok_spike_w1 !== "1")
                        $fatal(1, "encode: generator row %0d expected field o_spike_w1 is \"%0s\", expected a single 0 or 1", m + 1, tok_spike_w1);
                    exp_spike_w1 = (tok_spike_w1 == "1");
                    if (tok_mode_lfsr !== "0" && tok_mode_lfsr !== "1")
                        $fatal(1, "encode: generator row %0d expected field o_spike_lfsr is \"%0s\", expected a single 0 or 1", m + 1, tok_mode_lfsr);
                    exp_mode_lfsr = (tok_mode_lfsr == "1");
                    if (tok_mode_lfsr_ext !== "0" && tok_mode_lfsr_ext !== "1")
                        $fatal(1, "encode: generator row %0d expected field o_spike_lfsr_ext is \"%0s\", expected a single 0 or 1", m + 1, tok_mode_lfsr_ext);
                    exp_mode_lfsr_ext = (tok_mode_lfsr_ext == "1");
                    if (tok_mode_tc !== "0" && tok_mode_tc !== "1")
                        $fatal(1, "encode: generator row %0d expected field o_spike_tc is \"%0s\", expected a single 0 or 1", m + 1, tok_mode_tc);
                    exp_mode_tc = (tok_mode_tc == "1");

                    if (rst != 0) begin
                        rst_n_g = 1'b0;
                        @(posedge clk);
                        @(negedge clk);
                        rst_n_g = 1'b1;
                    end
                    value_w1 = in_w1[FRAC_W1:0];
                    value_g  = in_mode[FRAC:0];
                    #1;
                    m = m + 1;
                    if (rand_lfsr !== exp_lfsr[WIDTH-1:0]) begin
                        $display("FAIL lfsr t=%0d : got %0d exp %0d", m, rand_lfsr, exp_lfsr);
                        fails = fails + 1;
                    end
                    if (rand_lfsr_ext !== exp_lfsr_ext[WIDTH-1:0]) begin
                        $display("FAIL lfsr_ext t=%0d : got %0d exp %0d", m, rand_lfsr_ext, exp_lfsr_ext);
                        fails = fails + 1;
                    end
                    if (rand_tc !== exp_tc[WIDTH-1:0]) begin
                        $display("FAIL temporalseq t=%0d : got %0d exp %0d", m, rand_tc, exp_tc);
                        fails = fails + 1;
                    end
                    if (spike_w1 !== exp_spike_w1) begin
                        $display("FAIL encode WIDTH=1 t=%0d value=%0d : got %b exp %0d",
                                 m, in_w1, spike_w1, exp_spike_w1);
                        fails = fails + 1;
                    end
                    if (spike_mode[1] !== exp_mode_lfsr) begin
                        $display("FAIL encode GENERATOR=lfsr t=%0d value=%0d : got %b exp %0d",
                                 m, in_mode, spike_mode[1], exp_mode_lfsr);
                        fails = fails + 1;
                    end
                    if (spike_mode[2] !== exp_mode_lfsr_ext) begin
                        $display("FAIL encode GENERATOR=lfsr_ext t=%0d value=%0d : got %b exp %0d",
                                 m, in_mode, spike_mode[2], exp_mode_lfsr_ext);
                        fails = fails + 1;
                    end
                    if (spike_mode[3] !== exp_mode_tc) begin
                        $display("FAIL encode GENERATOR=tc t=%0d value=%0d : got %b exp %0d",
                                 m, in_mode, spike_mode[3], exp_mode_tc);
                        fails = fails + 1;
                    end
                    @(posedge clk);
                    @(negedge clk);
                end
            end
        end
        $fclose(fd_g);

        if (m != `GEN_SEQ_VECTORS) begin
            $display("FAIL encode: compared %0d generator vectors, generator wrote %0d",
                     m, `GEN_SEQ_VECTORS);
            fails = fails + 1;
        end

        if (fails == 0)
            $display("PASS encode: %0d/%0d vectors (unipolar+bipolar probabilities, %0d Sobol dimensions) + %0d/%0d generator vectors (lfsr, lfsr_ext, temporalseq, WIDTH=1 encode, and encode at GENERATOR 1/2/3) + sequence generators elaborated at WIDTH 1..%0d",
                     n, n, DIMS, m, m, SWEEP_HI);
        else
            $display("FAIL encode: %0d mismatch(es) over %0d + %0d vectors", fails, n, m);
        $finish;
    end
endmodule
`default_nettype wire

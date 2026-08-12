`timescale 1ns/1ps
`default_nettype none


module or_sat_tb;
    reg i_input_0;
    reg i_input_1;
    wire o_output;

    or_sat dut (
        .i_input_0(i_input_0),
        .i_input_1(i_input_1),
        .o_output(o_output)
    );

    integer fd;
    integer code;
    integer count;
    integer fails;
    reg exp_out;

    initial begin
        fd = $fopen("vec/or_sat.vec", "r");
        if (fd == 0) begin
            $display("ERROR: cannot open vec/or_sat.vec");
            $finish;
        end

        count = 0;
        fails = 0;
        while (!$feof(fd)) begin
            code = $fscanf(
                fd, "%b %b %b\n",
                i_input_0, i_input_1, exp_out
            );
            if (code == 3) begin
                #1;
                count = count + 1;
                if (o_output !== exp_out)
                    fails = fails + 1;
            end
        end
        $fclose(fd);

        if (fails == 0)
            $display("PASS or_sat: %0d/%0d vectors", count, count);
        else
            $display(
                "FAIL or_sat: %0d mismatches over %0d vectors",
                fails, count
            );
        $finish;
    end
endmodule

`default_nettype wire

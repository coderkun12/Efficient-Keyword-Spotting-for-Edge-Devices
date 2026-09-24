// ---------------------------------------------------------------------------
// wmem_probe2 -- second round. The first probe's answer did not transfer.
//
// WHAT ROUND ONE SAID
//   g_b  2048 deep, generate scope, PORT read address      -> inferred
//   g_a  1152 deep, generate scope, PORT read address      -> not inferred
//   mem_c 1152 deep, MODULE scope,  PORT read address      -> inferred
// so "non-power-of-two depth inside a generate" looked like the rule.
//
// WHAT HAPPENED NEXT
// Rounding layer_top's wmem to 2048 changed nothing except making it bigger:
// registers 182,873 -> 297,561, exactly +114,688 = 16 x 896 x 8, and memory
// bits still 589,824. So depth was not the whole story.
//
// THE UNTESTED VARIABLE
// Every memory in round one was addressed by an input PORT. layer_top's read
// address is a COMPUTED EXPRESSION over registers:
//
//     w_rd_addr = WW_AW'({8'd0, wkt} * K + (K-1-wcnt))
//
// A port is an unconstrained free input. An expression over registers is
// something the tool can analyse -- and possibly decide it cannot build a RAM
// address from. That difference is what this probe isolates.
//
// FOUR VARIANTS, 4 lanes each so the run stays fast whichever way it goes:
//
//   E  generate + reg array, 2048, PORT address       control; should infer
//   F  generate + reg array, 2048, COMPUTED address   is the address the cause?
//   G  MODULE instance,      1152, COMPUTED address   the proposed fix
//   H  MODULE instance,      2048, COMPUTED address   fix + power-of-two depth
//
// READING IT: whichever of G/H infers is what layer_top should become. If F
// infers too, the cause is something else again and neither fix is needed.
// If NOTHING infers, the computed address is fatal to inference and the
// answer is to instantiate altsyncram explicitly behind an `ifdef.
// ---------------------------------------------------------------------------

`default_nettype none

// One lane of byte-wide memory, at MODULE scope -- the shape round one showed
// inferring as "mem_c". A generate can instantiate this many times without
// ever declaring a reg array inside generate scope.
module byte_ram #(
    parameter int DEPTH = 1152,
    parameter int AW    = 11
) (
    input  wire          clk,
    input  wire          we,
    input  wire [AW-1:0] waddr,
    input  wire [7:0]    wdata,
    input  wire [AW-1:0] raddr,
    output reg  [7:0]    q
);
    reg [7:0] mem [0:DEPTH-1];
    always @(posedge clk) begin
        if (we) mem[waddr] <= wdata;
        q <= mem[raddr];
    end
endmodule


module wmem_probe2 #(
    parameter int LANES = 4,
    parameter int AW    = 11
) (
    input  wire            clk,
    input  wire            rst_n,

    input  wire            wr_en,
    input  wire [3:0]      wr_lane,
    input  wire [AW-1:0]   wr_addr,
    input  wire [7:0]      wr_data,

    input  wire [AW-1:0]   rd_addr_port,   // free input, as in round one
    input  wire            step,

    output wire [LANES*8-1:0] q_e,
    output wire [LANES*8-1:0] q_f,
    output wire [LANES*8-1:0] q_g,
    output wire [LANES*8-1:0] q_h
);

    // A read address COMPUTED from registers, mirroring layer_top's
    //     w_rd_addr = WW_AW'({8'd0, wkt} * K + (K-1-wcnt))
    reg [7:0] kt;
    reg [3:0] cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            kt  <= 8'd0;
            cnt <= 4'd0;
        end else if (step) begin
            cnt <= cnt + 4'd1;
            if (cnt == 4'd15) kt <= kt + 8'd1;
        end
    end
    wire [AW-1:0] rd_addr_calc = AW'({4'd0, kt} * 16 + (4'd15 - cnt));

    // ---- E: generate + reg array, 2048, PORT address ---------------------
    genvar ge;
    generate
        for (ge = 0; ge < LANES; ge = ge + 1) begin : g_e
            reg [7:0] mem [0:(1<<AW)-1];
            reg [7:0] qr;
            always @(posedge clk) begin
                if (wr_en && (wr_lane == ge)) mem[wr_addr] <= wr_data;
                qr <= mem[rd_addr_port];
            end
            assign q_e[ge*8 +: 8] = qr;
        end
    endgenerate

    // ---- F: generate + reg array, 2048, COMPUTED address -----------------
    genvar gf;
    generate
        for (gf = 0; gf < LANES; gf = gf + 1) begin : g_f
            reg [7:0] mem [0:(1<<AW)-1];
            reg [7:0] qr;
            always @(posedge clk) begin
                if (wr_en && (wr_lane == gf)) mem[wr_addr] <= wr_data;
                qr <= mem[rd_addr_calc];
            end
            assign q_f[gf*8 +: 8] = qr;
        end
    endgenerate

    // ---- G: module instance, 1152, COMPUTED address ----------------------
    genvar gg;
    generate
        for (gg = 0; gg < LANES; gg = gg + 1) begin : g_g
            byte_ram #(.DEPTH(1152), .AW(AW)) u (
                .clk   (clk),
                .we    (wr_en && (wr_lane == gg)),
                .waddr (wr_addr),
                .wdata (wr_data),
                .raddr (rd_addr_calc),
                .q     (q_g[gg*8 +: 8])
            );
        end
    endgenerate

    // ---- H: module instance, 2048, COMPUTED address ----------------------
    genvar gh;
    generate
        for (gh = 0; gh < LANES; gh = gh + 1) begin : g_h
            byte_ram #(.DEPTH(1<<AW), .AW(AW)) u (
                .clk   (clk),
                .we    (wr_en && (wr_lane == gh)),
                .waddr (wr_addr),
                .wdata (wr_data),
                .raddr (rd_addr_calc),
                .q     (q_h[gh*8 +: 8])
            );
        end
    endgenerate

endmodule

`default_nettype wire

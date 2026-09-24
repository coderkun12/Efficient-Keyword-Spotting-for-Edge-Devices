// ---------------------------------------------------------------------------
// wmem_probe -- why will the weight memory not infer as block RAM?
//
// Two structural fixes have now failed to change anything: splitting the
// M*8-bit array into M byte-wide lanes, and replacing the computed write
// address with a bit-slice. Both produced a byte-identical result -- 329,027
// LEs, 182,873 registers, 589,824 memory bits -- and an explicit
// (* ramstyle = "M9K" *) drew no error, meaning the tool ignored it rather
// than trying and failing.
//
// Guessing again costs 15 minutes per attempt. So this asks FOUR questions in
// ONE run, each as a separate memory whose inference can be read off
// independently in the synthesis report:
//
//   A  exactly what layer_top has now: 1152 deep, generate-scoped, write
//      enable gated on a genvar comparison, read address shared by all lanes
//   B  same but depth rounded to 2048 -- tests whether the non-power-of-two
//      depth is what blocks it (band_sram's working arrays are 8192)
//   C  same but ONE lane, declared at module scope, no generate -- tests
//      whether generate scoping is what blocks it
//   D  same as A but with a per-lane read address -- tests whether sharing
//      one read address across sixteen arrays is what blocks it
//
// band_sram's nine arrays infer from a structurally identical shape, so one
// of these four differences is the cause. Whichever variants appear as
// altsyncram in the report answer it directly.
//
// Read the result as: memory bits reported vs the 4 x 16 x 9216 = 589,824
// bits declared here. Anything inferred shows up as altsyncram; anything not
// shows up as registers.
// ---------------------------------------------------------------------------

`default_nettype none

module wmem_probe #(
    parameter int M      = 16,
    parameter int WWORDS = 1152,        // MAX_KTILES(72) * K(16)
    parameter int AW     = 11,          // $clog2(1152)
    parameter int AW2    = 11           // power-of-two variant uses 2048
) (
    input  wire            clk,

    input  wire            wr_en,
    input  wire [3:0]      wr_lane,
    input  wire [AW-1:0]   wr_addr,
    input  wire [7:0]      wr_data,

    input  wire [AW-1:0]   rd_addr,                 // shared
    input  wire [M*AW-1:0] rd_addr_per_lane,        // independent

    output wire [M*8-1:0]  q_a,
    output wire [M*8-1:0]  q_b,
    output wire [7:0]      q_c,
    output wire [M*8-1:0]  q_d
);

    // ---- A: exactly what layer_top has now -------------------------------
    genvar ga;
    generate
        for (ga = 0; ga < M; ga = ga + 1) begin : g_a
            reg [7:0] mem [0:WWORDS-1];
            reg [7:0] qr;
            always @(posedge clk) begin
                if (wr_en && (wr_lane == ga)) mem[wr_addr] <= wr_data;
                qr <= mem[rd_addr];
            end
            assign q_a[ga*8 +: 8] = qr;
        end
    endgenerate

    // ---- B: depth rounded up to a power of two ---------------------------
    genvar gb;
    generate
        for (gb = 0; gb < M; gb = gb + 1) begin : g_b
            reg [7:0] mem [0:(1<<AW2)-1];
            reg [7:0] qr;
            always @(posedge clk) begin
                if (wr_en && (wr_lane == gb)) mem[wr_addr] <= wr_data;
                qr <= mem[rd_addr];
            end
            assign q_b[gb*8 +: 8] = qr;
        end
    endgenerate

    // ---- C: one lane, module scope, no generate --------------------------
    reg [7:0] mem_c [0:WWORDS-1];
    reg [7:0] q_c_r;
    always @(posedge clk) begin
        if (wr_en && (wr_lane == 4'd0)) mem_c[wr_addr] <= wr_data;
        q_c_r <= mem_c[rd_addr];
    end
    assign q_c = q_c_r;

    // ---- D: per-lane read address ----------------------------------------
    genvar gd;
    generate
        for (gd = 0; gd < M; gd = gd + 1) begin : g_d
            reg [7:0] mem [0:WWORDS-1];
            reg [7:0] qr;
            always @(posedge clk) begin
                if (wr_en && (wr_lane == gd)) mem[wr_addr] <= wr_data;
                qr <= mem[rd_addr_per_lane[gd*AW +: AW]];
            end
            assign q_d[gd*8 +: 8] = qr;
        end
    endgenerate

endmodule

`default_nettype wire

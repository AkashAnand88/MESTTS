// src/lib/sequenceMatcher.ts
//
// Minimal JS port of Python's difflib.SequenceMatcher, covering only what
// typing_module.py's analyze_errors() actually uses: get_matching_blocks(),
// get_opcodes(), and ratio(). This is NOT a generic diff — it specifically
// replicates the Ratcliff-Obershelp longest-matching-block algorithm so that
// "replace" opcodes line up the same way Python's difflib would, which is
// what the transposition/mirror-error detection in the backend was written
// against. A Myers-diff library (e.g. npm "diff") gives different opcode
// boundaries and would silently change these counts.

export type Opcode = {
  tag: "equal" | "replace" | "delete" | "insert";
  i1: number;
  i2: number;
  j1: number;
  j2: number;
};

export class SequenceMatcher {
  private a: string;
  private b: string;
  private b2j: Map<string, number[]>;

  constructor(a: string, b: string) {
    this.a = a;
    this.b = b;
    this.b2j = new Map();
    for (let i = 0; i < b.length; i++) {
      const ch = b[i];
      if (!this.b2j.has(ch)) this.b2j.set(ch, []);
      this.b2j.get(ch)!.push(i);
    }
  }

  private findLongestMatch(alo: number, ahi: number, blo: number, bhi: number) {
    const { a, b, b2j } = this;
    let besti = alo,
      bestj = blo,
      bestsize = 0;
    let j2len = new Map<number, number>();

    for (let i = alo; i < ahi; i++) {
      const newj2len = new Map<number, number>();
      const indices = b2j.get(a[i]) || [];
      for (const j of indices) {
        if (j < blo) continue;
        if (j >= bhi) break;
        const k = (j2len.get(j - 1) || 0) + 1;
        newj2len.set(j, k);
        if (k > bestsize) {
          besti = i - k + 1;
          bestj = j - k + 1;
          bestsize = k;
        }
      }
      j2len = newj2len;
    }
    return { besti, bestj, bestsize };
  }

  getMatchingBlocks(): Array<[number, number, number]> {
    const queue: Array<[number, number, number, number]> = [
      [0, this.a.length, 0, this.b.length],
    ];
    const matchingBlocks: Array<[number, number, number]> = [];

    while (queue.length) {
      const [alo, ahi, blo, bhi] = queue.pop()!;
      const { besti: i, bestj: j, bestsize: k } = this.findLongestMatch(alo, ahi, blo, bhi);
      if (k > 0) {
        matchingBlocks.push([i, j, k]);
        if (alo < i && blo < j) queue.push([alo, i, blo, j]);
        if (i + k < ahi && j + k < bhi) queue.push([i + k, ahi, j + k, bhi]);
      }
    }

    matchingBlocks.sort((x, y) => x[0] - y[0] || x[1] - y[1]);

    // Merge adjacent equal blocks (mirrors difflib's cleanup pass)
    const merged: Array<[number, number, number]> = [];
    let i1 = 0,
      j1 = 0,
      k1 = 0;
    for (const [i2, j2, k2] of matchingBlocks) {
      if (i1 + k1 === i2 && j1 + k1 === j2) {
        k1 += k2;
      } else {
        if (k1 > 0) merged.push([i1, j1, k1]);
        i1 = i2;
        j1 = j2;
        k1 = k2;
      }
    }
    if (k1 > 0) merged.push([i1, j1, k1]);
    merged.push([this.a.length, this.b.length, 0]);
    return merged;
  }

  getOpcodes(): Opcode[] {
    const opcodes: Opcode[] = [];
    let i = 0,
      j = 0;
    for (const [ai, bj, size] of this.getMatchingBlocks()) {
      let tag: Opcode["tag"] | null = null;
      if (i < ai && j < bj) tag = "replace";
      else if (i < ai) tag = "delete";
      else if (j < bj) tag = "insert";

      if (tag) opcodes.push({ tag, i1: i, i2: ai, j1: j, j2: bj });
      i = ai + size;
      j = bj + size;
      if (size > 0) opcodes.push({ tag: "equal", i1: ai, i2: i, j1: bj, j2: j });
    }
    return opcodes;
  }

  ratio(): number {
    const matches = this.getMatchingBlocks().reduce((sum, [, , k]) => sum + k, 0);
    const total = this.a.length + this.b.length;
    return total === 0 ? 1 : (2.0 * matches) / total;
  }
}

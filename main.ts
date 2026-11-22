// main.ts is mend, the CLI SPEC.md specifies. Node 22, the standard library alone; run it as `node main.ts [options] [PATCH]`.
//
// Architecture (variant 3): a canonical unified-hunk pipeline. Every patch format is first translated into
// canonical hunks {oldStart, oldLines, newLines} and one shared hunk applier applies them: it walks the file
// once, finds each hunk's old lines at the first matching tier, and splices the new lines in. The formats
// differ only in how their hunks are read and which occurrence they take, never in how a hunk is applied.
//
// In a trial the grader runs `node /bot/main.ts` with MAKEFUN_ENV_URL set, and then this file hands over to
// runner.ts (the last lines below): keep that hand-over and runner.ts as they are, and write the CLI.

import { mkdirSync, readFileSync, rmSync, statSync, writeFileSync, writeSync } from 'node:fs';
import { dirname } from 'node:path';

const FORMATS = ['auto', 'unified', 'search-replace', 'v4a', 'json'];

class Exit2 {
  readonly line: string;
  constructor(line: string) {
    this.line = line;
  }
}
class Fail {
  readonly code: string;
  readonly path: string;
  constructor(code: string, path: string) {
    this.code = code;
    this.path = path;
  }
}

const blank = (s: string): boolean => s.trim() === '';

function patchLines(text: string): string[] {
  if (text === '') return [];
  const lines = text.split('\n');
  if (text.endsWith('\n')) lines.pop();
  return lines.map((l) => (l.endsWith('\r') ? l.slice(0, -1) : l));
}

function safe(p: string): boolean {
  return p !== '' && !p.startsWith('/') && !p.includes('\\') && p.split('/').every((s) => s !== '' && s !== '.' && s !== '..');
}

/** A file as lines: its line break, its lines, and whether it ends with a line break (SPEC.md section 3). */
interface FileState {
  br: string;
  lines: string[];
  final: boolean;
}

function parseFile(text: string): FileState {
  if (text === '') return { br: '\n', lines: [], final: false };
  const nl = text.indexOf('\n');
  const br = nl > 0 && text[nl - 1] === '\r' ? '\r\n' : '\n';
  const parts = text.split(br);
  const final = text.endsWith(br);
  if (final) parts.pop();
  return { br, lines: parts, final };
}

function renderFile(f: FileState): string {
  if (f.lines.length === 0) return '';
  return f.lines.join(f.br) + (f.final ? f.br : '');
}

/** The tree as the edits leave it: changed paths in `now` (null: deleted), the rest read from disk. */
class Tree {
  private readonly orig = new Map<string, string | null>();
  readonly now = new Map<string, string | null>();
  /** Moves, by the path a file now sits at, to the path it came from (transitive, for `R` lines). */
  readonly moves = new Map<string, string>();
  original(p: string): string | null {
    if (!this.orig.has(p)) {
      let v: string | null = null;
      try {
        if (statSync(p).isFile()) v = readFileSync(p, 'utf8');
      } catch {
        v = null;
      }
      this.orig.set(p, v);
    }
    return this.orig.get(p)!;
  }
  get(p: string): string | null {
    return this.now.has(p) ? this.now.get(p)! : this.original(p);
  }
}

function recordMove(tree: Tree, oldP: string, newP: string): void {
  const origin = tree.moves.get(oldP) ?? oldP;
  tree.moves.delete(oldP);
  tree.moves.set(newP, origin);
}

type Edit = (tree: Tree) => void;

/** One body line of a hunk: context, old, new, or a backslash line. */
interface BodyLine {
  kind: 'ctx' | 'old' | 'new' | 'bs';
  text: string;
}

/** One canonical hunk: old lines O replaced by new lines N. `ctxOf[i]` is the index into `old` of new
 * line i when it is a context line, else -1. `sel` says which occurrence of O the hunk takes. */
interface Hunk {
  old: string[];
  neu: string[];
  ctxOf: number[];
  sel: { mode: 'nearest'; hint: number } | { mode: 'unique' } | { mode: 'firstAfter'; eof: boolean };
  noNL: boolean;
  a: number;
}

/** Matching lines (SPEC.md section 5): tier 0 exact, tier 1 trailing spaces/tabs removed, tier 2 both ends. */
const stripTrail = (s: string): string => s.replace(/[ \t]+$/, '');
const stripBoth = (s: string): string => s.replace(/^[ \t]+/, '').replace(/[ \t]+$/, '');

function lineEq(a: string, b: string, tier: number): boolean {
  if (tier === 0) return a === b;
  if (tier === 1) return stripTrail(a) === stripTrail(b);
  return stripBoth(a) === stripBoth(b);
}

/** Every 0-based position at or after `from` where O occurs in `lines` at `tier` (occurrences may overlap). */
function occurrences(O: string[], lines: string[], from: number, tier: number): number[] {
  const out: number[] = [];
  if (O.length === 0) return out;
  const last = lines.length - O.length;
  for (let i = Math.max(0, from); i <= last; i++) {
    let ok = true;
    for (let k = 0; k < O.length; k++) {
      if (!lineEq(O[k]!, lines[i + k]!, tier)) {
        ok = false;
        break;
      }
    }
    if (ok) out.push(i);
  }
  return out;
}

/** The first tier at which O occurs at least once at or after `from`, or -1. */
function firstTier(O: string[], lines: string[], from: number, strict: boolean): number {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const t of tiers) {
    if (occurrences(O, lines, from, t).length > 0) return t;
  }
  return -1;
}

/** The first line at or after `from` equal to `anchor` (a single line, section 5 tiers), or -1. */
function findAnchor(anchor: string, lines: string[], from: number, strict: boolean): number {
  const tiers = strict ? [0] : [0, 1, 2];
  for (const t of tiers) {
    for (let i = Math.max(0, from); i < lines.length; i++) {
      if (lineEq(anchor, lines[i]!, t)) return i;
    }
  }
  return -1;
}

/** The new lines of a hunk as written: context lines are the file's own matched lines; at tier 2 every
 * other line is re-indented from O's first line's leading whitespace to the file's (SPEC.md section 5). */
function buildNew(h: Hunk, matched: string[], tier: number): string[] {
  if (tier !== 2 || h.old.length === 0) {
    return h.neu.map((l, i) => (h.ctxOf[i]! >= 0 ? matched[h.ctxOf[i]!]! : l));
  }
  const F = /^[ \t]*/.exec(matched[0]!)![0];
  const P = /^[ \t]*/.exec(h.old[0]!)![0];
  return h.neu.map((l, i) => {
    if (h.ctxOf[i]! >= 0) return matched[h.ctxOf[i]!]!;
    if (l !== '' && l.trim() !== '' && l.startsWith(P)) return F + l.slice(P.length);
    return l;
  });
}

/** Apply one canonical hunk to a file's lines. Returns where it landed and whether its matched lines
 * ran to the end of the file (or, for an empty O, the insertion was at the end). */
function applyHunk(f: FileState, h: Hunk, strict: boolean, from: number, path: string): { pos: number; endBefore: boolean } {
  const lines = f.lines;
  if (h.old.length === 0) {
    let at: number;
    if (h.sel.mode === 'nearest') at = h.sel.hint;
    else if (h.sel.mode === 'firstAfter') at = h.sel.eof ? lines.length : from;
    else at = from;
    at = Math.min(Math.max(at, 0), lines.length);
    const neu = buildNew(h, [], 0);
    const endBefore = at === lines.length;
    lines.splice(at, 0, ...neu);
    return { pos: at, endBefore };
  }
  if (h.sel.mode === 'firstAfter' && h.sel.eof) {
    const p = lines.length - h.old.length;
    if (p < from) throw new Fail('no-match', path);
    const tiers = strict ? [0] : [0, 1, 2];
    let tier = -1;
    for (const t of tiers) {
      let ok = true;
      for (let k = 0; k < h.old.length; k++) {
        if (!lineEq(h.old[k]!, lines[p + k]!, t)) {
          ok = false;
          break;
        }
      }
      if (ok) {
        tier = t;
        break;
      }
    }
    if (tier < 0) throw new Fail('no-match', path);
    const matched = lines.slice(p, p + h.old.length);
    const neu = buildNew(h, matched, tier);
    lines.splice(p, h.old.length, ...neu);
    return { pos: p, endBefore: true };
  }
  const tier = firstTier(h.old, lines, from, strict);
  if (tier < 0) throw new Fail('no-match', path);
  const occ = occurrences(h.old, lines, from, tier);
  let pos: number;
  if (h.sel.mode === 'nearest') {
    const hint = h.sel.hint;
    pos = occ[0]!;
    for (const p of occ) {
      if (Math.abs(p - hint) < Math.abs(pos - hint)) pos = p;
    }
  } else if (h.sel.mode === 'unique') {
    if (occ.length !== 1) throw new Fail('ambiguous', path);
    pos = occ[0]!;
  } else {
    pos = occ[0]!;
  }
  const endBefore = pos + h.old.length === lines.length;
  const matched = lines.slice(pos, pos + h.old.length);
  const neu = buildNew(h, matched, tier);
  lines.splice(pos, h.old.length, ...neu);
  return { pos, endBefore };
}

function detect(lines: string[]): string | null {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) return null;
  if (lines[first]!.trim() === '*** Begin Patch') return 'v4a';
  const c = lines[first]!.trimStart()[0];
  if (c === '[' || c === '{') return 'json';
  if (lines.some((l) => l.trimEnd() === '<<<<<<< SEARCH')) return 'search-replace';
  if (lines.some((l, i) => l.startsWith('diff --git ') || (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')))) return 'unified';
  return null;
}

/** One JSON edit object (SPEC.md section 9), or null when it is not one. */
function jsonEdit(v: unknown): Edit | null {
  if (v === null || typeof v !== 'object' || Array.isArray(v)) return null;
  const o = v as Record<string, unknown>;
  const p = o.path;
  if (typeof p !== 'string') return null;
  const exists = (tree: Tree): string => {
    if (!safe(p)) throw new Fail('unsafe', p);
    const t = tree.get(p);
    if (t === null) throw new Fail('missing', p);
    return t;
  };
  if (o.command === 'create' && typeof o.file_text === 'string') {
    const text = o.file_text;
    return (tree) => {
      if (!safe(p)) throw new Fail('unsafe', p);
      if (tree.get(p) !== null) throw new Fail('exists', p);
      tree.now.set(p, text);
    };
  }
  if (o.command === 'str_replace' && typeof o.old_str === 'string' && o.old_str !== '' && (o.new_str === undefined || typeof o.new_str === 'string')) {
    const oldStr = o.old_str;
    const newStr = (o.new_str as string | undefined) ?? '';
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const crlf = nl > 0 && before[nl - 1] === '\r';
      const lf = (s: string): string => (crlf ? s.replaceAll('\r\n', '\n') : s);
      const t = lf(before);
      const old = lf(oldStr);
      let count = 0;
      for (let k = t.indexOf(old); k >= 0; k = t.indexOf(old, k + 1)) count += 1;
      if (count === 0) throw new Fail('no-match', p);
      if (count > 1) throw new Fail('ambiguous', p);
      const at = t.indexOf(old);
      const out = t.slice(0, at) + lf(newStr) + t.slice(at + old.length);
      tree.now.set(p, crlf ? out.replaceAll('\n', '\r\n') : out);
    };
  }
  if (o.command === 'insert' && typeof o.insert_line === 'number' && Number.isInteger(o.insert_line) && typeof o.new_str === 'string') {
    const line = o.insert_line;
    const add = patchLines(o.new_str);
    return (tree) => {
      const before = exists(tree);
      const nl = before.indexOf('\n');
      const br = nl > 0 && before[nl - 1] === '\r' ? '\r\n' : '\n';
      const lines = before === '' ? [] : before.split(br);
      const final = before !== '' && before.endsWith(br);
      if (final) lines.pop();
      if (line < 0 || line > lines.length) throw new Fail('no-match', p);
      lines.splice(line, 0, ...add);
      tree.now.set(p, lines.length === 0 ? '' : lines.join(br) + (final ? br : ''));
    };
  }
  return null;
}

function jsonEdits(text: string, lines: string[]): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  const edits: Edit[] = [];
  if (lines[first]!.trimStart().startsWith('[')) {
    let v: unknown;
    try {
      v = JSON.parse(text);
    } catch {
      throw new Exit2(`error: parse ${first + 1}`);
    }
    if (!Array.isArray(v)) throw new Exit2(`error: parse ${first + 1}`);
    for (const x of v) {
      const e = jsonEdit(x);
      if (e === null) throw new Exit2(`error: parse ${first + 1}`);
      edits.push(e);
    }
    return edits;
  }
  lines.forEach((l, i) => {
    if (blank(l)) return;
    let e: Edit | null = null;
    try {
      e = jsonEdit(JSON.parse(l));
    } catch {
      e = null;
    }
    if (e === null) throw new Exit2(`error: parse ${i + 1}`);
    edits.push(e);
  });
  return edits;
}

/* ---------------- unified diffs (SPEC.md section 6) ---------------- */

interface UniHunkParsed {
  a: number;
  old: string[];
  neu: string[];
  ctxOf: number[];
  noNL: boolean;
  plusNoNL: boolean;
}

interface UniSection {
  oldPath: string | null;
  newPath: string | null;
  isCreate: boolean;
  isDelete: boolean;
  hunks: UniHunkParsed[];
}

/** A `--- ` / `+++ ` path: text after the four characters, up to the first tab, surrounding spaces
 * removed; `/dev/null` means no file, a leading `a/` or `b/` is removed. */
function uniPath(raw: string): string | null {
  let p = raw.split('\t')[0]!.trim();
  if (p === '/dev/null') return null;
  if (p.startsWith('a/') || p.startsWith('b/')) p = p.slice(2);
  return p;
}

function uniHunkBody(body: BodyLine[]): { old: string[]; neu: string[]; ctxOf: number[] } {
  const old: string[] = [];
  const neu: string[] = [];
  const ctxOf: number[] = [];
  for (const b of body) {
    if (b.kind === 'ctx') {
      const oi = old.length;
      old.push(b.text);
      neu.push(b.text);
      ctxOf.push(oi);
    } else if (b.kind === 'old') {
      old.push(b.text);
    } else if (b.kind === 'new') {
      neu.push(b.text);
      ctxOf.push(-1);
    }
  }
  return { old, neu, ctxOf };
}

function applyUniHunks(f: FileState, hunks: UniHunkParsed[], strict: boolean, path: string): void {
  let d = 0;
  for (const h of hunks) {
    const hh: Hunk = {
      old: h.old,
      neu: h.neu,
      ctxOf: h.ctxOf,
      sel: { mode: 'nearest', hint: h.old.length > 0 ? h.a - 1 + d : h.a + d },
      noNL: h.noNL,
      a: h.a,
    };
    const res = applyHunk(f, hh, strict, 0, path);
    if (res.endBefore) f.final = !h.noNL;
    d += h.neu.length - h.old.length;
  }
}

function uniSectionEdit(sec: UniSection, strict: boolean): Edit {
  return (tree) => {
    const oldP = sec.oldPath;
    const newP = sec.newPath;
    if (oldP !== null && !safe(oldP)) throw new Fail('unsafe', oldP);
    if (newP !== null && !safe(newP)) throw new Fail('unsafe', newP);
    if (sec.isCreate) {
      const p = newP ?? '';
      if (!safe(p)) throw new Fail('unsafe', p);
      if (tree.get(p) !== null) throw new Fail('exists', p);
      const plus: string[] = [];
      for (const h of sec.hunks) {
        for (let k = 0; k < h.neu.length; k++) if (h.ctxOf[k] === -1) plus.push(h.neu[k]!);
      }
      let finalBreak = true;
      for (let k = sec.hunks.length - 1; k >= 0; k--) {
        const h = sec.hunks[k]!;
        if (h.neu.some((_, idx) => h.ctxOf[idx] === -1)) {
          finalBreak = !h.plusNoNL;
          break;
        }
      }
      tree.now.set(p, plus.length === 0 ? '' : plus.join('\n') + (finalBreak ? '\n' : ''));
      return;
    }
    if (sec.isDelete) {
      const p = oldP ?? '';
      if (!safe(p)) throw new Fail('unsafe', p);
      if (tree.get(p) === null) throw new Fail('missing', p);
      tree.now.set(p, null);
      return;
    }
    if (oldP !== newP) {
      const o = oldP ?? '';
      const nw = newP ?? '';
      if (!safe(o)) throw new Fail('unsafe', o);
      if (!safe(nw)) throw new Fail('unsafe', nw);
      if (tree.get(o) === null) throw new Fail('missing', o);
      if (tree.get(nw) !== null) throw new Fail('exists', nw);
      const f = parseFile(tree.get(o)!);
      applyUniHunks(f, sec.hunks, strict, o);
      tree.now.set(nw, renderFile(f));
      tree.now.set(o, null);
      recordMove(tree, o, nw);
      return;
    }
    const p = oldP ?? '';
    if (!safe(p)) throw new Fail('unsafe', p);
    if (tree.get(p) === null) throw new Fail('missing', p);
    const f = parseFile(tree.get(p)!);
    applyUniHunks(f, sec.hunks, strict, p);
    tree.now.set(p, renderFile(f));
  };
}

/** Parse one unified section starting at index `s`; returns the index after the section. */
function parseUniSection(lines: string[], s: number, edits: Edit[], strict: boolean): number {
  const n = lines.length;
  let oldPath: string | null = null;
  let newPath: string | null = null;
  let oldNamed = false;
  let newNamed = false;
  let dgOld: string | null = null;
  let dgNew: string | null = null;
  let isCreate = false;
  let isDelete = false;
  let j = s;
  if (lines[j]!.startsWith('diff --git ')) {
    const rest = lines[j]!.slice('diff --git '.length);
    const m = /^a\/(.*) b\/(.+)$/.exec(rest);
    if (!m || m[1] === '') throw new Exit2(`error: parse ${j + 1}`);
    dgOld = m[1]!;
    dgNew = m[2]!;
    j++;
  }
  const hunks: UniHunkParsed[] = [];
  while (j < n) {
    const l = lines[j]!;
    if (l.startsWith('@@')) break;
    if (l.startsWith('diff --git ')) break;
    if (l.startsWith('--- ')) {
      if ((lines[j + 1] ?? '').startsWith('+++ ')) {
        oldPath = uniPath(l.slice(4));
        newPath = uniPath(lines[j + 1]!.slice(4));
        oldNamed = true;
        newNamed = true;
        j += 2;
      } else {
        throw new Exit2(`error: parse ${j + 2}`);
      }
    } else if (l.startsWith('+++ ')) {
      newPath = uniPath(l.slice(4));
      newNamed = true;
      j++;
    } else if (l.startsWith('rename from ')) {
      if (!oldNamed) {
        oldPath = l.slice('rename from '.length).trim();
        oldNamed = true;
      }
      j++;
    } else if (l.startsWith('rename to ')) {
      if (!newNamed) {
        newPath = l.slice('rename to '.length).trim();
        newNamed = true;
      }
      j++;
    } else if (l.startsWith('new file mode')) {
      isCreate = true;
      j++;
    } else if (l.startsWith('deleted file mode')) {
      isDelete = true;
      j++;
    } else {
      j++;
    }
  }
  while (j < n) {
    const l = lines[j]!;
    if (l.startsWith('diff --git ')) break;
    if (l.startsWith('--- ') && (lines[j + 1] ?? '').startsWith('+++ ')) break;
    if (l.startsWith('@@')) {
      const hm = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(?=$|\s)/.exec(l);
      if (!hm) throw new Exit2(`error: parse ${j + 1}`);
      const a = parseInt(hm[1]!, 10);
      j++;
      const body: BodyLine[] = [];
      while (j < n) {
        const b = lines[j]!;
        if (b.startsWith('diff --git ')) break;
        if (b.startsWith('--- ') && (lines[j + 1] ?? '').startsWith('+++ ')) break;
        if (b === '' || b.startsWith(' ') || b.startsWith('-') || b.startsWith('+') || b.startsWith('\\')) {
          body.push({ kind: b === '' ? 'ctx' : b[0] === ' ' ? 'ctx' : b[0] === '-' ? 'old' : b[0] === '+' ? 'new' : 'bs', text: b === '' ? '' : b.slice(1) });
          j++;
        } else break;
      }
      while (body.length > 0) {
        const last = body[body.length - 1]!;
        if (last.kind === 'ctx' && last.text === '') body.pop();
        else break;
      }
      let lastCP = -1;
      for (let k = 0; k < body.length; k++) if (body[k]!.kind === 'ctx' || body[k]!.kind === 'new') lastCP = k;
      const noNL = lastCP >= 0 && body[lastCP + 1] !== undefined && body[lastCP + 1]!.kind === 'bs';
      let lastPlus = -1;
      for (let k = 0; k < body.length; k++) if (body[k]!.kind === 'new') lastPlus = k;
      const plusNoNL = lastPlus >= 0 && body[lastPlus + 1] !== undefined && body[lastPlus + 1]!.kind === 'bs';
      const { old, neu, ctxOf } = uniHunkBody(body);
      hunks.push({ a, old, neu, ctxOf, noNL, plusNoNL });
    } else {
      j++;
    }
  }
  if (!oldNamed && dgOld !== null) oldPath = dgOld;
  if (!newNamed && dgNew !== null) newPath = dgNew;
  edits.push(uniSectionEdit({ oldPath, newPath, isCreate, isDelete, hunks }, strict));
  return j;
}

function parseUnified(lines: string[], strict: boolean): Edit[] {
  const edits: Edit[] = [];
  const n = lines.length;
  let i = 0;
  while (i < n) {
    const l = lines[i]!;
    if (l.startsWith('@@')) throw new Exit2(`error: parse ${i + 1}`);
    if (l.startsWith('diff --git ')) {
      i = parseUniSection(lines, i, edits, strict);
    } else if (l.startsWith('--- ') && (lines[i + 1] ?? '').startsWith('+++ ')) {
      i = parseUniSection(lines, i, edits, strict);
    } else {
      i++;
    }
  }
  return edits;
}

/* ---------------- SEARCH/REPLACE blocks (SPEC.md section 7) ---------------- */

function srBlockEdit(path: string, oldLines: string[], newLines: string[], strict: boolean): Edit {
  return (tree) => {
    if (!safe(path)) throw new Fail('unsafe', path);
    if (oldLines.length === 0) {
      if (tree.get(path) !== null) throw new Fail('exists', path);
      tree.now.set(path, newLines.length === 0 ? '' : newLines.join('\n') + '\n');
      return;
    }
    const cur = tree.get(path);
    if (cur === null) throw new Fail('missing', path);
    const f = parseFile(cur);
    const h: Hunk = { old: oldLines, neu: newLines, ctxOf: newLines.map(() => -1), sel: { mode: 'unique' }, noNL: false, a: 0 };
    applyHunk(f, h, strict, 0, path);
    tree.now.set(path, renderFile(f));
  };
}

function parseSearchReplace(lines: string[], strict: boolean): Edit[] {
  const edits: Edit[] = [];
  const n = lines.length;
  let i = 0;
  let prevPath: string | null = null;
  let prevEnd = 0;
  while (i < n) {
    if (lines[i]!.trimEnd() !== '<<<<<<< SEARCH') {
      i++;
      continue;
    }
    const searchIdx = i;
    let eq = -1;
    for (let k = i + 1; k < n; k++) {
      if (lines[k]!.trimEnd() === '=======') {
        eq = k;
        break;
      }
      if (lines[k]!.trimEnd() === '>>>>>>> REPLACE' || lines[k]!.trimEnd() === '<<<<<<< SEARCH') break;
    }
    if (eq < 0) throw new Exit2(`error: parse ${searchIdx + 1}`);
    let rep = -1;
    for (let k = eq + 1; k < n; k++) {
      if (lines[k]!.trimEnd() === '>>>>>>> REPLACE') {
        rep = k;
        break;
      }
      if (lines[k]!.trimEnd() === '<<<<<<< SEARCH') break;
    }
    if (rep < 0) throw new Exit2(`error: parse ${searchIdx + 1}`);
    let path: string | null = null;
    for (let k = searchIdx - 1; k >= prevEnd; k--) {
      const t = lines[k]!.trim();
      if (t === '') continue;
      if (t.startsWith('```')) continue;
      path = t;
      break;
    }
    if (path === null) {
      if (prevPath === null) throw new Exit2(`error: parse ${searchIdx + 1}`);
      path = prevPath;
    }
    prevPath = path;
    const oldLines = lines.slice(searchIdx + 1, eq);
    const newLines = lines.slice(eq + 1, rep);
    edits.push(srBlockEdit(path, oldLines, newLines, strict));
    prevEnd = rep + 1;
    i = rep + 1;
  }
  return edits;
}

/* ---------------- V4A patches (SPEC.md section 8) ---------------- */

interface VChunkParsed {
  anchor: string | null;
  old: string[];
  neu: string[];
  ctxOf: number[];
  eof: boolean;
}

function v4aChunkToCanonical(c: { anchor: string | null; body: BodyLine[]; eof: boolean }): VChunkParsed {
  const old: string[] = [];
  const neu: string[] = [];
  const ctxOf: number[] = [];
  for (const b of c.body) {
    if (b.kind === 'ctx') {
      const oi = old.length;
      old.push(b.text);
      neu.push(b.text);
      ctxOf.push(oi);
    } else if (b.kind === 'old') {
      old.push(b.text);
    } else if (b.kind === 'new') {
      neu.push(b.text);
      ctxOf.push(-1);
    }
  }
  return { anchor: c.anchor, old, neu, ctxOf, eof: c.eof };
}

function v4aSectionEdit(
  kind: 'add' | 'del' | 'upd',
  path: string,
  raw: { text: string; line: number }[],
  secLine: number,
  strict: boolean,
): Edit {
  if (kind === 'add') {
    const fileLines: string[] = [];
    for (const r of raw) {
      if (r.text.startsWith('+')) fileLines.push(r.text.slice(1));
      else if (r.text === '') fileLines.push('');
      else throw new Exit2(`error: parse ${r.line}`);
    }
    return (tree) => {
      if (!safe(path)) throw new Fail('unsafe', path);
      if (tree.get(path) !== null) throw new Fail('exists', path);
      tree.now.set(path, fileLines.length === 0 ? '' : fileLines.join('\n') + '\n');
    };
  }
  if (kind === 'del') {
    if (raw.length > 0) throw new Exit2(`error: parse ${raw[0]!.line}`);
    return (tree) => {
      if (!safe(path)) throw new Fail('unsafe', path);
      if (tree.get(path) === null) throw new Fail('missing', path);
      tree.now.set(path, null);
    };
  }
  let moveTo: string | null = null;
  let idx = 0;
  if (raw.length > 0 && raw[0]!.text.startsWith('*** Move to: ')) {
    moveTo = raw[0]!.text.slice('*** Move to: '.length).trim();
    idx = 1;
  }
  const chunks: VChunkParsed[] = [];
  let cur: { anchor: string | null; body: BodyLine[]; eof: boolean } | null = null;
  const flush = (): void => {
    if (cur) {
      chunks.push(v4aChunkToCanonical(cur));
      cur = null;
    }
  };
  for (; idx < raw.length; idx++) {
    const r = raw[idx]!;
    const t = r.text;
    if (t.startsWith('@@')) {
      flush();
      const anchor = t.slice(2).trim();
      cur = { anchor: anchor === '' ? null : anchor, body: [], eof: false };
    } else if (t === '*** End of File') {
      if (!cur) throw new Exit2(`error: parse ${r.line}`);
      cur.eof = true;
      flush();
    } else if (t === '' || t.startsWith(' ') || t.startsWith('-') || t.startsWith('+')) {
      if (!cur) cur = { anchor: null, body: [], eof: false };
      cur.body.push({ kind: t === '' ? 'ctx' : t[0] === ' ' ? 'ctx' : t[0] === '-' ? 'old' : 'new', text: t === '' ? '' : t.slice(1) });
    } else {
      throw new Exit2(`error: parse ${r.line}`);
    }
  }
  flush();
  if (chunks.length === 0 && moveTo === null) throw new Exit2(`error: parse ${secLine}`);
  return (tree) => {
    if (!safe(path)) throw new Fail('unsafe', path);
    if (moveTo !== null && !safe(moveTo)) throw new Fail('unsafe', moveTo);
    const curText = tree.get(path);
    if (curText === null) throw new Fail('missing', path);
    if (moveTo !== null && tree.get(moveTo) !== null) throw new Fail('exists', moveTo);
    const f = parseFile(curText);
    let cursor = 0;
    for (const ch of chunks) {
      if (ch.anchor !== null) {
        const a = findAnchor(ch.anchor, f.lines, cursor, strict);
        if (a < 0) throw new Fail('no-match', path);
        cursor = a + 1;
      }
      const h: Hunk = { old: ch.old, neu: ch.neu, ctxOf: ch.ctxOf, sel: { mode: 'firstAfter', eof: ch.eof }, noNL: false, a: 0 };
      const res = applyHunk(f, h, strict, cursor, path);
      cursor = res.pos + h.neu.length;
    }
    const out = renderFile(f);
    if (moveTo !== null) {
      tree.now.set(moveTo, out);
      tree.now.set(path, null);
      recordMove(tree, path, moveTo);
    } else {
      tree.now.set(path, out);
    }
  };
}

function parseV4a(lines: string[], strict: boolean): Edit[] {
  const first = lines.findIndex((l) => !blank(l));
  if (first < 0) throw new Exit2('error: parse 1');
  if (lines[first]!.trim() !== '*** Begin Patch') throw new Exit2(`error: parse ${first + 1}`);
  let last = lines.length - 1;
  while (last >= 0 && blank(lines[last]!)) last--;
  if (last < 0 || lines[last]!.trim() !== '*** End Patch') throw new Exit2(`error: parse ${lines.length + 1}`);
  const edits: Edit[] = [];
  let i = first + 1;
  const end = last;
  while (i < end) {
    if (blank(lines[i]!)) {
      i++;
      continue;
    }
    const l = lines[i]!;
    let kind: 'add' | 'del' | 'upd';
    let path: string;
    if (l.startsWith('*** Add File: ')) {
      kind = 'add';
      path = l.slice('*** Add File: '.length).trim();
    } else if (l.startsWith('*** Delete File: ')) {
      kind = 'del';
      path = l.slice('*** Delete File: '.length).trim();
    } else if (l.startsWith('*** Update File: ')) {
      kind = 'upd';
      path = l.slice('*** Update File: '.length).trim();
    } else {
      throw new Exit2(`error: parse ${i + 1}`);
    }
    const secLine = i + 1;
    i++;
    const raw: { text: string; line: number }[] = [];
    while (i < end) {
      const m = lines[i]!;
      if (m.startsWith('*** Add File: ') || m.startsWith('*** Delete File: ') || m.startsWith('*** Update File: ')) break;
      raw.push({ text: m, line: i + 1 });
      i++;
    }
    while (raw.length > 0 && blank(raw[raw.length - 1]!.text)) raw.pop();
    edits.push(v4aSectionEdit(kind, path, raw, secLine, strict));
  }
  return edits;
}

/* ---------------- run ---------------- */

function run(args: string[]): { code: number; out: string[] } {
  let format: string | null = null;
  let check = false;
  let strict = false;
  const pos: string[] = [];
  let ended = false;
  for (const a of args) {
    if (!ended && a === '--') ended = true;
    else if (!ended && a.startsWith('-') && a !== '-') {
      if (a.startsWith('--format=') && format === null && FORMATS.includes(a.slice(9))) format = a.slice(9);
      else if (a === '--check' && !check) check = true;
      else if (a === '--strict' && !strict) strict = true;
      else return { code: 2, out: ['error: usage'] };
    } else pos.push(a);
  }
  if (pos.length > 1) return { code: 2, out: ['error: usage'] };
  const file = pos.length === 0 || pos[0] === '-' ? null : pos[0]!;
  let text = '';
  try {
    text = readFileSync(file ?? 0, 'utf8');
  } catch {
    if (file !== null) return { code: 2, out: ['error: no-patch'] };
  }
  const lines = patchLines(text);
  const kind = format === null || format === 'auto' ? detect(lines) : format;
  if (kind === null) return { code: 2, out: ['error: format'] };
  let edits: Edit[];
  try {
    if (kind === 'json') edits = jsonEdits(text, lines);
    else if (kind === 'unified') edits = parseUnified(lines, strict);
    else if (kind === 'search-replace') edits = parseSearchReplace(lines, strict);
    else edits = parseV4a(lines, strict);
    if (edits.length === 0) throw new Exit2('error: parse 1');
  } catch (e) {
    if (e instanceof Exit2) return { code: 2, out: [e.line] };
    throw e;
  }
  const tree = new Tree();
  const fails: Fail[] = [];
  for (const edit of edits) {
    const before = new Map(tree.now);
    const movesBefore = new Map(tree.moves);
    try {
      edit(tree);
    } catch (e) {
      if (!(e instanceof Fail)) throw e;
      tree.now.clear();
      for (const [k, v] of before) tree.now.set(k, v);
      tree.moves.clear();
      for (const [k, v] of movesBefore) tree.moves.set(k, v);
      fails.push(e);
    }
  }
  if (fails.length > 0) return { code: 1, out: [...fails.map((f) => `error: ${f.code} ${f.path}`), `failed: ${fails.length} of ${edits.length} edits`] };
  const rows: { key: string; line: string }[] = [];
  const rTargets = new Set<string>();
  const covered = new Set<string>();
  for (const [nw, origin] of tree.moves) {
    const now = tree.now.get(nw);
    if (now === undefined || now === null) continue;
    if (tree.original(nw) !== null) continue;
    if (tree.original(origin) === null) continue;
    rows.push({ key: nw, line: `R ${origin} -> ${nw}` });
    rTargets.add(nw);
    covered.add(origin);
  }
  for (const p of [...tree.now.keys()].sort((a, b) => Buffer.compare(Buffer.from(a), Buffer.from(b)))) {
    if (rTargets.has(p)) continue;
    const was = tree.original(p);
    const now = tree.now.get(p)!;
    if (was === now) continue;
    if (was === null) rows.push({ key: p, line: `A ${p}` });
    else if (now === null) {
      if (!covered.has(p)) rows.push({ key: p, line: `D ${p}` });
    } else rows.push({ key: p, line: `M ${p}` });
  }
  rows.sort((a, b) => Buffer.compare(Buffer.from(a.key), Buffer.from(b.key)));
  const outLines = rows.map((r) => r.line);
  if (!check) {
    for (const p of [...tree.now.keys()]) {
      const was = tree.original(p);
      const now = tree.now.get(p)!;
      if (was === now) continue;
      if (now === null) rmSync(p, { force: true });
      else {
        mkdirSync(dirname(p), { recursive: true });
        writeFileSync(p, now);
      }
    }
  }
  return { code: 0, out: [...outLines, `ok: ${outLines.length} ${outLines.length === 1 ? 'file' : 'files'}`] };
}

const trialEnv = process.env.MAKEFUN_ENV_URL;
if (trialEnv !== undefined && trialEnv !== '') {
  // A trial: the kit's runner reads the checks from the env and runs this file on each one, without this variable.
  const { runTrial } = await import('./runner.ts');
  await runTrial(trialEnv);
} else {
  const r = run(process.argv.slice(2));
  writeSync(1, r.out.map((l) => `${l}\n`).join(''));
  process.exitCode = r.code;
}

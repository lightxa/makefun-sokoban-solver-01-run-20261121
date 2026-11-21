// Room-by-room push-optimal Sokoban solver.
//
// Idea: every level in this competition is a hallway with one to four small
// rooms off it (3-4 boxes each). A box that leaves its room can never be
// pushed back cheaply, so the level's push optimum is the sum of the rooms'
// optima. This program therefore splits each level into rooms, solves every
// room push-optimally with an A* search over pushes (minimum-matching
// heuristic, dead-square and frozen-block pruning), and stitches the room
// solutions together with plain walks through the hallway. Walk moves are
// free: only pushes are scored, so stitching never costs points.
//
// Determinism: the search budget is counted in expanded states, never in
// wall-clock time; tie-breaking is by fixed insertion order. No clock, no
// randomness, no environment-dependent behaviour. Two runs on the same input
// produce byte-identical output.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Directions in fixed order: up, down, left, right. The order is part of the
// deterministic tie-breaking, never changed at runtime.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const LETTER = ['u', 'd', 'l', 'r'];

// Work budgets (state counts, never wall clock). The generator guarantees a
// room is solvable well inside a few hundred thousand push states.
const MAX_EXPANSIONS_PER_ROOM = 500_000;
const MAX_EXPANSIONS_WHOLE_LEVEL = 800_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** Parse XSB text (# wall, space floor, . goal, $ box, * box on goal, @ player). */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * Dead-square analysis: a square is "live" when a box standing on it can
 * still be pushed onto some goal. Computed by pulling boxes backwards from
 * the goals: a box on square x can be pulled to y = x - d when the player
 * could have stood at y - d, i.e. both y and y - d are open floor. Any box
 * on a dead square is doomed, so such pushes are pruned outright.
 */
function liveSquares(wall: Uint8Array, goals: number[], w: number): Uint8Array {
  const live = new Uint8Array(wall.length);
  const queue: number[] = [];
  for (const g of goals) {
    live[g] = 1;
    queue.push(g);
  }
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi]!;
    for (let k = 0; k < 4; k++) {
      const d = DY[k]! * w + DX[k]!;
      const y = x - d;
      if (y < 0 || y >= wall.length) continue;
      if (wall[y] || wall[y - d] || live[y]) continue;
      live[y] = 1;
      queue.push(y);
    }
  }
  return live;
}

/**
 * Shortest-path distances over open floor from every goal (boxes ignored).
 * dist[g][cell] is the fewest steps from cell to goal g, or -1 if unreachable.
 * A box's distance to a goal never grows when other boxes are ignored, so a
 * minimum matching over these distances is an admissible push heuristic.
 */
function goalDistances(wall: Uint8Array, goals: number[], w: number): Int32Array[] {
  return goals.map((g) => {
    const dist = new Int32Array(wall.length).fill(-1);
    const queue = [g];
    dist[g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi]!;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * w + DX[k]!;
        if (q < 0 || q >= wall.length || wall[q] || dist[q] !== -1) continue;
        dist[q] = dist[p]! + 1;
        queue.push(q);
      }
    }
    return dist;
  });
}

interface Room {
  boxes: number[];
  goals: number[];
}

/**
 * Split a level into rooms. The player starts in the hallway; every hallway
 * square a box can never reach a goal from is cut out of the floor, and each
 * connected component left that holds boxes is one room. Returns null when
 * the level does not split cleanly (a box in the hallway, a goal stranded
 * outside any room, or a room whose box and goal counts differ).
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(wall, goals, w);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let x = 0; x < w; x++) {
    const i = row * w + x;
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      const p = squares[i]!;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * w + DX[k]!;
        if (q < 0 || q >= wall.length || wall[q] || cut[q] || part[q] >= 0) continue;
        part[q] = rooms.length;
        squares.push(q);
      }
    }
    rooms.push({ boxes: [], goals: [] });
  }
  for (const b of boxes) {
    const r = part[b]!;
    if (r < 0) return null;
    rooms[r]!.boxes.push(b);
  }
  for (const g of goals) {
    const r = part[g]!;
    if (r < 0) return null;
    rooms[r]!.goals.push(g);
  }
  if (rooms.some((r) => r.boxes.length === 0 || r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/**
 * Push-space A* over one (sub)level. A state is the sorted box squares plus
 * the player's reachable region (named by its lowest square, so where the
 * player happens to stand inside the region does not multiply states). Each
 * step is one push (cost 1); the first solved state popped is push-optimal
 * because the heuristic below never overestimates the pushes still needed.
 *
 * Returns the push list as {from, to} pairs plus the pushes count, or null
 * when the state budget runs out or the level is unsolvable.
 */
function solvePushes(
  wall: Uint8Array,
  goals: number[],
  boxes0: number[],
  player0: number,
  w: number,
  maxExpansions: number,
): { pushes: { from: number; to: number }[] } | null {
  const size = wall.length;
  const n = boxes0.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const live = liveSquares(wall, goals, w);
  if (boxes0.some((b) => !live[b])) return null;
  const dists = goalDistances(wall, goals, w);
  const solved = (boxes: readonly number[]): boolean => boxes.every((b) => goalSet[b]);
  if (solved(boxes0)) return { pushes: [] };

  // Minimum matching: the cheapest way to pair each box with a distinct
  // goal, over the per-goal floor distances. Enumerating the n! pairings
  // (n <= 4 here) is exact and cheap; any pairing a real solution uses
  // costs at least this, so the estimate is admissible.
  const perms: number[][] = [];
  const buildPerm = (used: boolean[], cur: number[]): void => {
    if (cur.length === n) {
      perms.push(cur.slice());
      return;
    }
    for (let j = 0; j < n; j++) {
      if (used[j]) continue;
      used[j] = true;
      cur.push(j);
      buildPerm(used, cur);
      cur.pop();
      used[j] = false;
    }
  };
  buildPerm(new Array<boolean>(n).fill(false), []);
  const heuristic = (boxes: readonly number[]): number => {
    let best = Infinity;
    for (const perm of perms) {
      let sum = 0;
      for (let i = 0; i < n; i++) {
        const d = dists[perm[i]!]![boxes[i]!]!;
        if (d < 0) {
          sum = Infinity;
          break;
        }
        sum += d;
      }
      if (sum < best) best = sum;
    }
    return best;
  };

  // State store in flat typed arrays (memory-safe for hundreds of thousands
  // of states): boxes as n columns, the player's square, parent, the push.
  let cap = 65536;
  let boxesFlat = new Int32Array(cap * n);
  let playerArr = new Int32Array(cap);
  let parentArr = new Int32Array(cap);
  let pushFromArr = new Int32Array(cap);
  let pushToArr = new Int32Array(cap);
  let gArr = new Int32Array(cap);
  let keyArr = new Float64Array(cap);
  let count = 0;
  const boxesAt = (s: number): number[] => {
    const out: number[] = [];
    for (let i = 0; i < n; i++) out.push(boxesFlat[s * n + i]!);
    return out;
  };

  // Reachable region from `from` with `occupied` in the way, stamped into
  // `seen2`; returns the region's lowest square (the state's player key).
  // Uses its own stamp array: the expansion loop marks the parent's region
  // in `seen`, and this must never clobber those marks mid-expansion.
  const seen = new Int32Array(size);
  const seen2 = new Int32Array(size);
  const stack = new Int32Array(size);
  const stack2 = new Int32Array(size);
  let stamp = 0;
  let stamp2 = 0;
  const regionLow = (occupied: Uint8Array, from: number): number => {
    stamp2++;
    let low = from;
    let sp = 0;
    stack2[sp++] = from;
    seen2[from] = stamp2;
    while (sp > 0) {
      const p = stack2[--sp]!;
      if (p < low) low = p;
      for (let k = 0; k < 4; k++) {
        const q = p + DY[k]! * w + DX[k]!;
        if (q < 0 || q >= size || wall[q] || occupied[q] || seen2[q] === stamp2) continue;
        seen2[q] = stamp2;
        stack2[sp++] = q;
      }
    }
    return low;
  };

  // A box is frozen when it sits in a 2x2 block of walls and boxes holding a
  // box off its goal: no push can ever move any box of that block again.
  const frozenAt = (occupied: Uint8Array, t: number): boolean => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (a < 0 || d >= size) continue;
      if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b])) continue;
      if (!(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
      if ((occupied[a] && !goalSet[a]) || (occupied[b] && !goalSet[b])) return true;
      if ((occupied[c] && !goalSet[c]) || (occupied[d] && !goalSet[d])) return true;
    }
    return false;
  };

  const keyOf = (boxes: readonly number[], low: number): number => {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };

  const occupied = new Uint8Array(size);
  for (const b of boxes0) occupied[b] = 1;
  const rootLow = regionLow(occupied, player0);
  for (const b of boxes0) occupied[b] = 0;

  // Binary heap on (f, -g, seq): lowest estimated total first, deeper first
  // on ties, then earliest inserted. The sequence counter makes every pop
  // order fully deterministic.
  const heapF: number[] = [];
  const heapG: number[] = [];
  const heapSeq: number[] = [];
  const heapIdx: number[] = [];
  let seq = 0;
  const heapLess = (a: number, b: number): boolean =>
    heapF[a]! < heapF[b]! ||
    (heapF[a]! === heapF[b]! &&
      (heapG[a]! > heapG[b]! || (heapG[a]! === heapG[b]! && heapSeq[a]! < heapSeq[b]!)));
  const heapPush = (f: number, g: number, idx: number): void => {
    heapF.push(f);
    heapG.push(g);
    heapSeq.push(seq++);
    heapIdx.push(idx);
    let i = heapF.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(i, p)) break;
      for (const arr of [heapF, heapG, heapSeq, heapIdx] as const) {
        const t = arr[i]!;
        (arr as number[])[i] = arr[p]!;
        (arr as number[])[p] = t;
      }
      i = p;
    }
  };
  const heapPop = (): number => {
    const top = heapIdx[0]!;
    const last = heapF.length - 1;
    heapF[0] = heapF[last]!;
    heapG[0] = heapG[last]!;
    heapSeq[0] = heapSeq[last]!;
    heapIdx[0] = heapIdx[last]!;
    heapF.pop();
    heapG.pop();
    heapSeq.pop();
    heapIdx.pop();
    for (let i = 0; ; ) {
      let c = 2 * i + 1;
      if (c >= heapF.length) break;
      if (c + 1 < heapF.length && heapLess(c + 1, c)) c++;
      if (!heapLess(c, i)) break;
      for (const arr of [heapF, heapG, heapSeq, heapIdx] as const) {
        const t = arr[i]!;
        (arr as number[])[i] = arr[c]!;
        (arr as number[])[c] = t;
      }
      i = c;
    }
    return top;
  };

  // The open set: key -> state index of the best-known state with that key.
  const best = new Map<number, number>();
  const addState = (
    boxes: number[],
    player: number,
    parent: number,
    from: number,
    to: number,
    g: number,
    low: number,
  ): number => {
    if (count >= cap) {
      // Grow every store at once (doubling). Typed arrays cannot resize, so
      // new ones are allocated and the old contents copied over.
      const nc = cap * 2;
      const nb = new Int32Array(nc * n);
      nb.set(boxesFlat.subarray(0, count * n));
      boxesFlat = nb;
      const np = new Int32Array(nc);
      np.set(playerArr.subarray(0, count));
      playerArr = np;
      const npar = new Int32Array(nc);
      npar.set(parentArr.subarray(0, count));
      parentArr = npar;
      const nf = new Int32Array(nc);
      nf.set(pushFromArr.subarray(0, count));
      pushFromArr = nf;
      const nt = new Int32Array(nc);
      nt.set(pushToArr.subarray(0, count));
      pushToArr = nt;
      const ng = new Int32Array(nc);
      ng.set(gArr.subarray(0, count));
      gArr = ng;
      const nk = new Float64Array(nc);
      nk.set(keyArr.subarray(0, count));
      keyArr = nk;
      cap = nc;
    }
    const s = count++;
    for (let i = 0; i < n; i++) boxesFlat[s * n + i] = boxes[i]!;
    playerArr[s] = player;
    parentArr[s] = parent;
    pushFromArr[s] = from;
    pushToArr[s] = to;
    gArr[s] = g;
    keyArr[s] = keyOf(boxes, low);
    return s;
  };

  const root = addState(boxes0.slice(), player0, -1, -1, -1, 0, rootLow);
  best.set(keyArr[root]!, root);
  heapPush(heuristic(boxes0), 0, root);

  let expansions = 0;
  let found = -1;
  const curBoxes: number[] = new Array<number>(n);
  while (heapF.length > 0 && expansions < maxExpansions) {
    const s = heapPop();
    if (best.get(keyArr[s]!) !== s) continue;
    for (let i = 0; i < n; i++) curBoxes[i] = boxesFlat[s * n + i]!;
    if (solved(curBoxes)) {
      found = s;
      break;
    }
    expansions++;
    for (const b of curBoxes) occupied[b] = 1;
    const here = ++stamp;
    // Mark the player's reachable region for this state (same flood fill as
    // regionLow, but keeping the stamp so push squares can be tested).
    {
      let sp = 0;
      stack[sp++] = playerArr[s]!;
      seen[playerArr[s]!] = here;
      while (sp > 0) {
        const p = stack[--sp]!;
        for (let k = 0; k < 4; k++) {
          const q = p + DY[k]! * w + DX[k]!;
          if (q < 0 || q >= size || wall[q] || occupied[q] || seen[q] === here) continue;
          seen[q] = here;
          stack[sp++] = q;
        }
      }
    }
    for (let bi = 0; bi < n; bi++) {
      const b = curBoxes[bi]!;
      for (let k = 0; k < 4; k++) {
        const d = DY[k]! * w + DX[k]!;
        const t = b + d;
        const behind = b - d;
        if (behind < 0 || behind >= size || t < 0 || t >= size) continue;
        if (seen[behind] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenAt(occupied, t);
        let nextLow = -1;
        let next: number[] | null = null;
        if (!frozen) {
          next = curBoxes.slice();
          next[bi] = t;
          next.sort((a, c) => a - c);
          nextLow = regionLow(occupied, b);
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || next === null) continue;
        const key = keyOf(next, nextLow);
        const prev = best.get(key);
        if (prev !== undefined && gArr[prev]! <= gArr[s]! + 1) continue;
        if (solved(next)) {
          const ns = addState(next, b, s, b, t, gArr[s]! + 1, nextLow);
          best.set(key, ns);
          found = ns;
          break;
        }
        const ns = addState(next, b, s, b, t, gArr[s]! + 1, nextLow);
        best.set(key, ns);
        heapPush(gArr[s]! + 1 + heuristic(next), gArr[s]! + 1, ns);
      }
      if (found >= 0) break;
    }
    for (const b of curBoxes) occupied[b] = 0;
    if (found >= 0) break;
  }
  if (found < 0) return null;
  const pushes: { from: number; to: number }[] = [];
  for (let s = found; parentArr[s]! >= 0; s = parentArr[s]!) {
    pushes.push({ from: pushFromArr[s]!, to: pushToArr[s]! });
  }
  pushes.reverse();
  return { pushes };
}

/** Shortest walk from `from` to `to` around walls and boxes (lowercase letters). */
function walkTo(
  wall: Uint8Array,
  occupied: Uint8Array,
  from: number,
  to: number,
  w: number,
): string {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Array<string>(size);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi]!;
    if (p === to) break;
    for (let k = 0; k < 4; k++) {
      const q = p + DY[k]! * w + DX[k]!;
      if (q < 0 || q >= size || wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = LETTER[k]!;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return '';
  let s = '';
  for (let p = to; p !== from; p = prev[p]!) s = how[p]! + s;
  return s;
}

/**
 * Solve one whole level. Rooms are solved in scan order; while a room is
 * being solved every other room's boxes count as walls (a solved room's boxes
 * sit on its goals, an unsolved room's on their starting squares). The
 * player's walk between pushes is filled in by BFS, so the stitched move
 * string replays exactly. Falls back to a whole-level push search when the
 * level does not split into rooms, and to an empty answer when nothing
 * solves it (an empty answer is valid and scores 0, never illegal).
 */
function solveLevel(level: Level): string {
  const { w, wall, goals, boxes, player } = level;
  const goalSet = new Set(goals);
  if (boxes.every((b) => goalSet.has(b))) return '';
  const rooms = roomsOf(level);
  if (rooms !== null) {
    let moves = '';
    let curPlayer = player;
    let curBoxes = boxes.slice();
    const done = new Array<boolean>(rooms.length).fill(false);
    for (let ri = 0; ri < rooms.length; ri++) {
      const room = rooms[ri]!;
      const subWall = wall.slice();
      for (let oj = 0; oj < rooms.length; oj++) {
        if (oj === ri) continue;
        const other = rooms[oj]!;
        for (const b of done[oj]! ? other.goals : other.boxes) subWall[b] = 1;
      }
      const solved = solvePushes(
        subWall,
        room.goals.slice().sort((a, b) => a - b),
        room.boxes.slice().sort((a, b) => a - b),
        curPlayer,
        w,
        MAX_EXPANSIONS_PER_ROOM,
      );
      if (solved === null) return '';
      // Fill in the walks: from where the player stands to each push square.
      const occupied = new Uint8Array(wall.length);
      for (const b of curBoxes) occupied[b] = 1;
      let roomBoxes = room.boxes.slice().sort((a, b) => a - b);
      for (const push of solved.pushes) {
        const d = push.to - push.from;
        const k = d === -w ? 0 : d === w ? 1 : d === -1 ? 2 : 3;
        moves += walkTo(subWall, occupied, curPlayer, push.from - d, w);
        moves += LETTER[k]!.toUpperCase();
        occupied[push.from] = 0;
        occupied[push.to] = 1;
        curPlayer = push.from;
        roomBoxes = roomBoxes.map((b) => (b === push.from ? push.to : b)).sort((a, b) => a - b);
      }
      curBoxes = curBoxes.filter((b) => !room.boxes.includes(b)).concat(roomBoxes);
      done[ri] = true;
    }
    return moves;
  }
  // No clean room split: one push search over the whole level.
  const solved = solvePushes(wall, goals, boxes, player, w, MAX_EXPANSIONS_WHOLE_LEVEL);
  if (solved === null) return '';
  const occupied = new Uint8Array(wall.length);
  for (const b of boxes) occupied[b] = 1;
  let moves = '';
  let curPlayer = player;
  for (const push of solved.pushes) {
    const d = push.to - push.from;
    const k = d === -w ? 0 : d === w ? 1 : d === -1 ? 2 : 3;
    moves += walkTo(wall, occupied, curPlayer, push.from - d, w);
    moves += LETTER[k]!.toUpperCase();
    occupied[push.from] = 0;
    occupied[push.to] = 1;
    curPlayer = push.from;
  }
  return moves;
}

for (const id of readdirSync(INPUT_DIR).sort()) {
  const levelPath = join(INPUT_DIR, id, 'level.txt');
  let moves = '';
  try {
    moves = solveLevel(parseLevel(readFileSync(levelPath, 'utf8')));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUTPUT_DIR, id), { recursive: true });
  writeFileSync(join(OUTPUT_DIR, id, 'solution.txt'), `${moves}\n`);
}

#!/usr/bin/env python3
"""All-levels sprites -> ROM-ready sets sharing one compact 8x8 tile dictionary.

Input: gen/full_sprites.pkl from animconv.py --sets, whose entries are keyed
(set, anim, mirror) with parts (8x16 tile id, dx, dy).  Sets:
  0 Conrad, 1..4 the four monster types, 10 the (global) level objects.

Output (for mkdata.py --sprsets):
  * tiles: 8x8 tiles (64 palette indices each), deduplicated - an 8x16 sprite
    is its top and bottom half, and halves repeat far more than whole sprites
  * sets: {set_slot: {(anim, mirror): [(top_id, bottom_id, dx, dy), ...]}}
    with set_slot 0..5 (objects are slot 5)
  * monster_of: per level part, per object, which monster set (0..3) it uses,
    or 0xFF - worked out exactly as the engine's loadMonsterSprites does
  * palette: the sprite palette

Entries with more than MAX_PARTS sprites are dropped: they are large pieces
of machinery drawn as objects, and could not be shown with the SMS's 64
hardware sprites (8 per line) anyway.
"""
import argparse, os, pickle, struct, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import tilepack as TP
from levelconv import read_fbl

MAX_PARTS = 40
# sprite set slots: 0 Conrad, 1..4 the monsters, then level objects per level
# part and palette slot: 5 + 4 * part + slot (the engine draws an object in the
# 16-colour palette slot its flags select, so a frame has up to 4 colourings)
NUM_SETS = 5 + 4 * 7


def object_frames(ani):
    """animation numbers the level's object records can show (records whose
    'object' word, at +4, is set; frames are u16 anim|mirror, i8 dx, i8 dy)"""
    ntypes = struct.unpack_from('<H', ani, 2)[0] // 2
    offs = sorted({2 + struct.unpack_from('<H', ani, 2 + n * 2)[0] for n in range(ntypes)})
    ends = offs[1:] + [len(ani)]
    out = set()
    for o, e in zip(offs, ends):
        if e - o < 6 or not struct.unpack_from('<H', ani, o + 4)[0]:
            continue
        count = struct.unpack_from('<H', ani, o)[0]
        for i in range(count):
            q = o + 6 + i * 4
            if q + 2 > e:
                break
            w = struct.unpack_from('<H', ani, q)[0]
            if w != 0xFFFF:
                out.add(w & 0x7FFF)
    return out

# the engine's _monsterListLevels: (object node, monster set) pairs per level part
MONSTER_LISTS = [
    {0x22: 0, 0x23: 0},
    {0x22: 0, 0x23: 0, 0x4B: 0, 0x49: 1, 0x4D: 1, 0x76: 2},
    {0x76: 2},
    {0x4D: 1, 0x76: 2},
    {0x76: 2, 0xAC: 2, 0xD7: 3},
    {0xB0: 3, 0xD7: 3},
    {0xB0: 3, 0xD7: 3, 0xD8: 3},
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('sprites')
    ap.add_argument('--levels', nargs='+', required=True, help='level_L*.fbl in part order')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    d = pickle.load(open(a.sprites, 'rb'))

    halves, hindex = [], {}
    def half_id(t32):
        k = bytes(t32)
        if k not in hindex:
            hindex[k] = len(halves)
            halves.append(k)
        return hindex[k]

    by_part = {int(os.path.basename(p).split('_L')[1].split('.')[0]): p for p in a.levels}
    levels = {part: read_fbl(path) for part, path in by_part.items()}
    obj_anims = {part: object_frames(lv['ani']) for part, lv in levels.items()}
    obj_slots = {part: {(f & 0x60) >> 5 for f in lv['live_flags']} for part, lv in levels.items()}

    def slot_of(setid, anim):
        if setid < 10:
            return setid
        part, k = (setid - 10) % 16, (setid - 10) // 16
        if part not in levels or k not in obj_slots[part] or anim not in obj_anims[part]:
            return None                     # no object of that level can show it so
        return 5 + 4 * part + k

    sets = {}
    dropped = 0
    for (setid, anim, mirror), parts in d['entries'].items():
        slot = slot_of(setid, anim)
        if slot is None:
            continue
        if len(parts) > MAX_PARTS:
            dropped += 1
            continue
        out = []
        for tid, dx, dy in parts:
            t = d['tiles'][tid]
            out.append((half_id(t[:32]), half_id(t[32:]), dx, dy))
        sets.setdefault(slot, {})[(anim, mirror)] = out

    tiles64 = [TP.unplanar(h) for h in halves]
    raw = len(tiles64) * 32
    packed = sum(TP.SIZE[TP.tile_type(t)] for t in tiles64)

    # rows are indexed by level part (level_L<part>.fbl); a part the data
    # does not have (the demo ships three) gets an empty row
    monster_of = [b''] * (max(by_part) + 1)
    for part, lv in sorted(levels.items()):
        mlist = MONSTER_LISTS[part]
        row = []
        for i in range(lv['npges']):
            p = i * 31
            node = struct.unpack_from('<H', lv['pges'], p + 6)[0]
            otype = lv['pges'][p + 18]
            if (node == 0x49 or otype == 10) and node in mlist:
                row.append(mlist[node])
            else:
                row.append(0xFF)
        monster_of[part] = bytes(row)

    n_entries = sum(len(v) for v in sets.values())
    print(f'{n_entries} entries in {len(sets)} sets ({dropped} oversized dropped); '
          f'{len(halves)} distinct 8x8 tiles: {raw/1024:.0f} KB raw -> {packed/1024:.0f} KB compact')
    for slot in sorted(sets):
        print(f'  set {slot}: {len(sets[slot])} entries')
    pickle.dump(dict(tiles=tiles64, sets=sets, nsets=NUM_SETS, monster_of=monster_of,
                     palette=d['palette']), open(a.out, 'wb'))


if __name__ == '__main__':
    main()

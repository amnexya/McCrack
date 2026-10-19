#!/usr/bin/env python3

import math
import numpy as np
import os
import sys

from cubiomes import Generator, MCVersion, Dimension, BiomeID
from multiprocessing import Pool, cpu_count 

_G = None
_POINTS = None

# make logging look pretty
class c_fg:
    # foreground text colours
    BLACK   = '\033[30m'
    RED     = '\033[31m'
    GREEN   = '\033[32m'
    YELLOW  = '\033[33m'
    BLUE    = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN    = '\033[36m'
    WHITE   = '\033[37m'
    RESET   = '\033[39m'

class j_random():
    # set constants
    MULT = 0x5DEECE66D
    ADD = 0xB
    MASK = (1 << 48) - 1

    def __init__(self, seed):
        self.set_seed(seed)

    def set_seed(self, seed):
        """Set the new seed according to how Java would do it.
        Args:
            self (Self): the class
            seed (int): the provided seed
        """
        # Java XOR's your seed with MULT before actually doing anything
        # so we need to recreate that here, and then use `& MASK` to
        # make sure it's exactly what Java would store.
        self.state = (seed ^ self.MULT) & self.MASK

    def next(self, bits):
        """Advance LCG and return top bits.
        Args:
            self (Self): the class
            bits (int): number of top bits to returm
        """
        self.state = (self.state * self.MULT + self.ADD) & self.MASK
        return self.state >> (48 - bits)

    def nextInt(self, n):
        if n > 0 and (n & (n - 1)) == 0: # is n a power of two?
            return (n * self.next(31)) >> 31
        else:
            while True:
                bits = self.next(31)
                val = bits % n
                if bits - val + (n - 1) < 2**31:
                    return val

def info(msg: str) -> None:
    print(f"{c_fg.BLUE}[INFO]{c_fg.RESET} {msg}")

def warn(msg: str) -> None:
    print(f"{c_fg.YELLOW}[WARN]{c_fg.RESET} {msg}")

def error(msg: str) -> None:
    print(f"{c_fg.RED}[ERROR]{c_fg.RESET} {msg}")

def die(msg: str="Exiting...") -> None:
    error(msg)
    sys.exit()

def check_coord_file(filename: str) -> str:
    """Checks if the file containing co-ordinates to analyse exists.

    Args:
        filename (str): Path to the co-ordinates file, by default "co.txt".
    
    Returns:
        filename (str): New filename, or unchanged in the event the file is found.
    """
    if not os.path.exists(filename):
        while True:
            warn(f"Could not find {filename}, please enter path to filename below (e.g. /path/to/co.txt)")
            filename = input("Path: ")
            
            if os.path.exists(filename):
                return filename
    info(f"Valid data file found at {filename}")
    return filename

def offset_validation(coords: list) -> tuple:
    """Validates the offsets of all co-ordinates provided.

    Args:
        coords (list): List of tuples containing all co-ordinates to analyse.
    
    Returns:
        regions, offsets (tuple): Returns the validated chunks, regions, and offsets
    """
    chunks, regions, offsets = [], [], []
    for coord in coords:
        x, z = coord # set x and z with coord tuple
        print("\n================\n")
        info(f"Checking validity of structure at ({x}, {z})...")

        info(f"Processing {x}")
        # get chunk, region, and offset vals
        chunk_x = math.floor(x / 16)
        chunk_z = math.floor(z / 16)
        region_x = math.floor(chunk_x / 32)
        region_z = math.floor(chunk_z / 32)
        offset_x = chunk_x - region_x * 32
        offset_z = chunk_z - region_z * 32

        info(f"Chunk: ({chunk_x}, {chunk_z}), Region: ({region_x}, {region_z}), Offset: ({offset_x}, {offset_z})")
        
        info("Checking offsets...")
        # check if x and y offset is between 0-23
        if 0 <= offset_x <= 23 and 0 <= offset_z <= 23:
            info(f"Valid structure at ({x}, {z})")
            chunks.append((int(chunk_x), int(chunk_z)))
            regions.append((int(region_x), int(region_z)))
            offsets.append((int(offset_x), int(offset_z)))
        else:
            error("Offset validation failed! This is either not a real structure, or you entered your co-ordinates incorrectly.")

    return regions, offsets

def open_coord_file(filename: str) -> str:
    """Opens the co-ordinates file and reads.

    Args:
        filename (str): Path to the co-ordinates file, by default "co.txt".

    Returns:
        content (str): Contents of the provided file.
    """
    try:
        with open(filename, "r") as f:
            content = f.read()
            return content
    except Exception as e:
        error(str(e))
        die(f"Something went wrong opening {filename}. Please ensure this file is readable by your user, and is a plain text file.")
        return ""

def check_regions(worldSeed: int, region: tuple, offset: tuple) -> bool:
    """This checks whether the seed is correct by recreating the structure placements and checking them.
    
    Args:
        worldSeed (int): the calculated world seed
        region (tuple): the originally provided region
        offset (tuple): the originally provided offset

    Returns:
        bool: True or False if its correct or not.
    """
    rx, rz = region
    ox, oz = offset
    regionSeed = rx * 341873128712 + rz * 132897987541 + worldSeed + 14357617
    rng = j_random(regionSeed)
    offsetX = rng.nextInt(24)
    offsetZ = rng.nextInt(24)
    return ox == offsetX and oz == offsetZ

def low_bits_match(low: int, region: tuple, offset: tuple) -> bool:
    """This is basically the same as check_regions, except we are now checking low bits.
    
    Args:
        worldSeed (int): the calculated world seed
        region (tuple): the originally provided region
        offset (tuple): the originally provided offset

    Returns:
        bool: True or False if its correct or not.
    """
    rx, rz = region
    ox, oz = offset
    regionSeed = rx * 341873128712 + rz * 132897987541 + low + 14357617
    rng = j_random(regionSeed)
    offsetX = rng.nextInt(24)
    offsetZ = rng.nextInt(24)
    return ox % 8 == offsetX % 8 and oz % 8 == offsetZ % 8

def brute_force_upper_bits(regions: tuple, offsets: tuple, survivors: list):

    # numpy needs these as a 64 bit unsigned int
    MULT64 = np.uint64(0x5DEECE66D)
    ADD64 = np.uint64(0xB)
    MASK = np.uint64((1 << 48) - 1)

    consts = []
    for (rx, rz), (ox, oz) in zip(regions, offsets):
        # same equation as regionSeed, masked bc no worldSeed
        c = (rx * 341873128712 + rz * 132897987541 + 14357617) & ((1 << 48) - 1)
        consts.append((np.uint64(c), np.uint64(ox), np.uint64(oz)))

    results=[]

    # outer loop (usually only one in survivors)
    for low in survivors:
        # low to low64
        low64 = np.uint64(low)

        # middle loop (for batching, reduces memory usage)
        for start in range(0, 1<<28, 1<<22):
            hi = np.arange(start, start + (1<<22), dtype=np.uint64)
            seeds = (hi << np.uint64(20)) | low64

            # inner loop (big boy)
            for c64, ox, oz in consts:
                state = ((seeds + c64) & MASK ^ MULT64) # region seed + set seed scramble

                state = (state * MULT64 + ADD64) & MASK # next
                px = (state >> np.uint64(17)) % np.uint64(24) # predicted x offset

                state = (state * MULT64 + ADD64) & MASK # next
                pz = (state >> np.uint64(17)) % np.uint64(24) # predicted z offset

                keep = (px == ox) & (pz == oz) # if these match, this is a seed we keep
                seeds = seeds[keep]

                if seeds.size == 0:
                    break

            results.extend(seeds.tolist())

    return results

def to_signed(seed: int) -> int:
    """Convert an unsigned 64-bit seed to the signed form /seed shows."""
    return seed - (1 << 64) if seed >= (1 << 63) else seed

def brute_force_biomes(structure_seeds: list, samples: list) -> list:
    """Tries all 65536 upper 16 bits for each structure seed and keeps
    the full seeds whose biomes match every sample point.

    Args:
        structure_seeds (list): 48-bit structure seeds from brute_force_upper_bits
        samples (list): (x, z, biome_name) tuples taken from the map

    Returns:
        list: full signed 64-bit world seeds that match every sample
    """
    # convert biome names to IDs once, so typos fail early instead of mid-search
    try:
        points = [(x, z, int(BiomeID[name])) for x, z, name in samples]
    except KeyError as e:
        error(f"Unknown biome name {e}. Valid names are:")
        print(", ".join(b.name for b in BiomeID))
        die()

    g = Generator(MCVersion.MC_1_12)  # create once, reuse for every seed
    matches = []

    for i, s48 in enumerate(structure_seeds, 1):
        info(f"Structure seed {i}/{len(structure_seeds)}: {s48}")
        for upper in range(1 << 16):
            seed = to_signed((upper << 48) | s48)
            g.apply_seed(Dimension.DIM_OVERWORLD, seed)
            if all(g.get_biome_at(x, 63, z, 1) == biome for x, z, biome in points):
                info(f"{c_fg.GREEN}Match: {seed}{c_fg.RESET}")
                matches.append(seed)

    return matches

def diagnose_low_bits(regions: list, offsets: list, coords: list = None) -> None:
    """Called when no low-bit value fits every structure. Finds values that fit
    all but ONE structure, to show which structure is the odd one out."""
    info("No full match. Looking for values that fit all but one structure...")
    misses = [[] for _ in regions]
    for low in range(1 << 20):
        failed = []
        for i, (r, o) in enumerate(zip(regions, offsets)):
            if not low_bits_match(low, r, o):
                failed.append(i)
                if len(failed) > 1:
                    break
        if len(failed) == 1:
            misses[failed[0]].append(low)
    found = False
    for i, lows in enumerate(misses):
        if lows:
            found = True
            where = f" at {coords[i]}" if coords else ""
            warn(f"{len(lows)} value(s) fit everything except structure {i + 1}{where}: {lows[:5]}")
    if not found:
        warn("Nothing fits even all-but-one. At least two structures are wrong, or these don't follow the vanilla 32/24 grid.")
    else:
        warn("With 5 structures, ~0.3 such values appear by chance in total, so a hit is a lead, not proof. "
             "If several structures show hits, add 2-3 more structures and rerun: a real hit keeps fitting, a chance one drops out.")

def vet_candidate(x: int, z: int, low: int) -> None:
    """Tests one candidate structure against the low bits you already trust,
    trying the two most likely systematic mistakes. Hits are HINTS: any single
    variant passes by chance 1 time in 64, so look for the same pattern on 2+ candidates."""
    cx, cz = x >> 4, z >> 4
    def fits(cx, cz, shift=0):
        region, offset = (cx >> 5, cz >> 5), (cx & 31, cz & 31)
        return offset[0] <= 23 and offset[1] <= 23 and low_bits_match(low + shift, region, offset)

    hits = []
    if fits(cx, cz):
        hits.append("as given")
    # 1) wrong corner: big structures (desert temples) span two chunks
    for dx, dz in ((-1, 0), (0, -1), (-1, -1)):
        if fits(cx + dx, cz + dz):
            hits.append(f"start chunk is ({dx:+d},{dz:+d}) from the one you used")
    # 2) a different structure type using a different salt (1.13-style: base salt + 1/2/3)
    for d in (1, 2, 3, -1, -2, -3):
        if fits(cx, cz, d):
            hits.append(f"fits if the salt is shifted by {d:+d}")
    print(f"({x}, {z}): " + ("; ".join(hits) if hits else "no variant fits"))

def fit_mask(lows, region: tuple, offset: tuple):
    """numpy version of low_bits_match: which low-bit values fit this structure?"""
    M, A, MASK = np.uint64(0x5DEECE66D), np.uint64(0xB), np.uint64((1 << 48) - 1)
    rx, rz = region
    c = np.uint64((rx * 341873128712 + rz * 132897987541 + 14357617) & ((1 << 48) - 1))
    s = ((lows + c) & MASK) ^ M
    s = (s * M + A) & MASK
    ok = ((s >> np.uint64(17)) & np.uint64(7)) == np.uint64(offset[0] % 8)
    s = (s * M + A) & MASK
    return ok & (((s >> np.uint64(17)) & np.uint64(7)) == np.uint64(offset[1] % 8))

def best_low_bits(regions: list, offsets: list, coords: list, min_fit: int = 4, trials: int = 20) -> None:
    """Finds the low-bit values that fit the MOST structures, without assuming
    which ones are good, and compares against shuffled (junk) data so you can
    see whether a group of agreeing structures is real or just chance."""
    import random
    n = len(regions)
    lows = np.arange(1 << 20, dtype=np.uint64)
    masks = [fit_mask(lows, r, o) for r, o in zip(regions, offsets)]
    counts = np.sum(masks, axis=0, dtype=np.int32)
    obs = np.bincount(counts, minlength=n + 1)

    rng, null = random.Random(0), np.zeros(n + 1)
    for _ in range(trials):
        junk = sum(fit_mask(lows, r, (rng.randrange(24), rng.randrange(24))).astype(np.int32) for r in regions)
        null += np.bincount(junk, minlength=n + 1)
    null /= trials

    info(f"{n} structures. Low-bit values that fit k of them (real data vs. random junk):")
    for k in range(min_fit, n + 1):
        flag = "  <-- more than chance" if obs[k] > 0 and obs[k] > 3 * null[k] + 1 else ""
        print(f"  k={k}: {obs[k]:6d} real   vs  {null[k]:8.2f} expected by chance{flag}")
    top = np.argsort(-counts, kind="stable")[:8]
    print("  best values:")
    for low in top:
        fits = [coords[i] for i in range(n) if masks[i][low]]
        print(f"    low={int(low):7d} fits {int(counts[low])}/{n}: {fits}")

def _init_worker(points):
    """Runs once per worker process: each gets its own generator."""
    global _G, _POINTS
    _G = Generator(MCVersion.MC_1_12)
    _POINTS = points

def _check_structure_seed(s48: int) -> list:
    found = []
    for upper in range(1 << 16):
        seed = to_signed((upper << 48) | s48)
        _G.apply_seed(Dimension.DIM_OVERWORLD, seed)
        if all(_G.get_biome_at(x, 63, z, 1) == biome for x, z, biome in _POINTS):
            found.append(seed)
    return found

def brute_force_biomes_parallel(structure_seeds: list, samples: list, workers: int = None) -> list:
    """Same search as brute_force_biomes, split across CPU cores (one structure seed per task)."""
    try:
        points = [(x, z, int(BiomeID[name])) for x, z, name in samples]
    except KeyError as e:
        error(f"Unknown biome name {e}. Valid names are:")
        print(", ".join(b.name for b in BiomeID))
        die()

    workers = workers or cpu_count()
    info(f"Searching {len(structure_seeds)} structure seeds on {workers} worker(s)...")
    matches = []
    with Pool(workers, initializer=_init_worker, initargs=(points,)) as pool:
        for done, found in enumerate(pool.imap_unordered(_check_structure_seed, structure_seeds), 1):
            for seed in found:
                info(f"{c_fg.GREEN}Match: {seed}{c_fg.RESET}")
                matches.append(seed)
            if done % 50 == 0 or done == len(structure_seeds):
                info(f"{done}/{len(structure_seeds)} structure seeds done")
    return matches

def main():
    # default filename
    filename="coordinates.txt"
    biomes_filename="biomes.txt"
    # init lists
    survivors, coords, samples = [], [], []

    g = Generator(MCVersion.MC_1_12_2)

    # Check if coord and biome file exists
    # Function replaces filename with user input if file is not found
    filename = check_coord_file(filename)
    biomes_filename = check_coord_file(biomes_filename)

    # get file content
    content = open_coord_file(filename)
    biome_content = open_coord_file(biomes_filename)

    # move coords from file into list of tuples
    try:
        for line in content.splitlines():
            if not line.strip():
                continue
            x, z = line.split(",")
            coords.append((int(x), int(z)))
    except Exception as e:
        error(str(e))
        die("YOUR CO-ORDINATES ARE FORMATTED INCORRECTLY! Please ensure your co-ordinates are written correctly, consult the README and the above traceback for more information.")

    # and the same for the biomes
    try:
        for line in biome_content.splitlines():
            if not line.strip():
                continue
            x, z, biome = line.split(",")
            samples.append((int(x), int(z), biome.strip()))
    except Exception as e:
        error(str(e))
        die("YOUR BIOME SAMPLES ARE FORMATTED INCORRECTLY! Please ensure your co-ordinates and biome are written correctly, consult the README and the above traceback for more information.")

    # calculate offsets and validate
    regions, offsets = offset_validation(coords)

    for low in range(1 << 20):
        if all(low_bits_match(low, r, o) for r, o in zip(regions, offsets)):
            survivors.append(low)

    print("\n================\n")
    info(f"{len(survivors)} candidates (low bit candidates): {survivors}")

    if len(survivors) == 0:
        diagnose_low_bits(regions, offsets, coords)

        die("""Couldn't find any seeds matching the co-ordinates provided. 
            You should use co-ordinates for Desert Temples, Jungle Temples, 
            and Witch Huts ONLY.""")

    info(f"Brute forcing upper bits, this may take a while...")
    results = brute_force_upper_bits(regions, offsets, survivors)

    if len(results) == 0:
        die("Could not find any seeds matching all brute forced upper bits. You should check every co-ordinate entered, as you will be close.")

    print("\n================\n")

    info(f"{len(results)} potential seeds (high bit brute forced)")
    if len(results) <= 8:
        print(results)
    info(f"{len(regions)} structures, {len(set(regions))} unique regions")

    print("\n================\n")

    info("Brute forcing upper 16 bits with biomes, this may take a few minutes...")
    seeds = brute_force_biomes_parallel(results, samples)

    print("\n================\n")
    info(f"{len(seeds)} world seed(s) found: {seeds}")

if __name__ == "__main__":
    main()
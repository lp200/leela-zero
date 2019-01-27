/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Michael O and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "config.h"
#include <functional>
#include <memory>

#include "NNCache.h"
#include "Utils.h"
#include "UCTSearch.h"
#include "GTP.h"

const int NNCache::MAX_CACHE_COUNT;
const int NNCache::MIN_CACHE_COUNT;
const size_t NNCache::ENTRY_SIZE;

bool NNCache::load_cachefile(std::string filename, bool read_only) {
    bool file_did_not_exist = false;

    m_outfile_map.clear();
    m_outfile.close();

    m_filename = filename;

    std::ifstream ifs(filename);

    // if read-only, the file should exist and should be readible.
    // Otherwise there is something worong.
    if (!ifs.good()) {
        file_did_not_exist = true;
        if (read_only) {
            m_filename = "";
            return false;
        }
    }

    // prepare memory.  this will prune out some of the (normal) cache entries
    // if the memory map is too big
    resize(m_size, true);

    if (!file_did_not_exist) {
        // if the file pre-existed, check if the first four bytes are "\xfeLNC"
        // which is the magic number for this file
        char b[4];
        ifs.read(b, 4);
        if (b[0] == '\xfe' && b[1] == 'L' && b[2] == 'N' && b[3] == 'C') {

        } else {
            Utils::myprintf("File '%s' does not seems to be a leela-zero NNCache file\n",
                filename.c_str()
            );
            return false;
        }

        // sixteen 0xff values are inserted periodically to mark a new start of a net.
        // This is to be more resillient to data corruptions
        auto skip_guard = [&ifs] () {
            char c;
            auto count = 0;
            // find sixteen consecutive 0xff bytes
            while(ifs.good() && count < 16) {
                ifs.read(&c, 1);
                if( c == '\xff') {
                    count++;
                } else {
                    count = 0;
                }
            }
        };


        while (ifs.good()) {
            skip_guard();

            CompressedEntry e;
            while (ifs.good()) {
                size_t pos = ifs.tellg();
                try {
                    Netresult dummy;
                    auto hash = e.read(ifs);
                    e.test_get();
                    m_outfile_map[hash] = pos;
                } catch (...) {
                    // something went wrong, we should skip until the next magic number to re-sync with stream,
                    // ...but rewind read pointer so that we can properly skip magic number
                    ifs.seekg(pos);
                    break;
                }
            }
        }
    }

    if (!m_outfile_map.empty()) {
        Utils::myprintf("Loaded %d entries from disk based NNCache (%s)\n",
            m_outfile_map.size(), m_filename.c_str());
    } else if (read_only) {
        // a read-only mode and zero records. hmm...
        m_filename = "";
        return false;
    }

    if (!read_only) {
        m_outfile.open(filename, std::ios_base::out | std::ios_base::app);
        if (file_did_not_exist) {
            constexpr char magic_number[4] = { '\xfe', 'L', 'N', 'C' };
            m_outfile.write(magic_number, 4);
            if (m_outfile.good()) {
                Utils::myprintf("Created new disk based NNCache (%s)\n", m_filename.c_str());
            }
        }

        constexpr char start_header[16] = {
            '\xff', '\xff', '\xff', '\xff',
            '\xff', '\xff', '\xff', '\xff',
            '\xff', '\xff', '\xff', '\xff',
            '\xff', '\xff', '\xff', '\xff'
        };
        m_outfile.write(start_header, 16);

        // if we are bad at this point, then we probably should consider failure
        if (!m_outfile.good()) {
            m_outfile.close();
            Utils::myprintf("Failed to start writing file\n");
            return false;
        }
    }

    return true;
}


std::uint64_t NNCache::CompressedEntry::read(std::ifstream & ifs, std::uint64_t expected_hash) {
    unsigned char c;
    std::uint64_t hash_read;
    ifs.read((char*)&(hash_read), sizeof(std::uint64_t));
    if(expected_hash != 0xffff'ffff'ffff'ffffLL && hash_read != expected_hash) {
        // hash mismatch - something wrong with file?
        throw std::runtime_error("Unexpected hash value");
    }

    ifs.read((char*)&(policy_pass), sizeof(float));
    ifs.read((char*)&(winrate), sizeof(float));
    ifs.read((char*)&c, 1);
    compressed_policy.clear();
    compressed_policy.expand(c*8);
    for (size_t i = 0; i < c; i++) {
        unsigned char v;
        ifs.read((char*)&v, 1);
        compressed_policy.push_bits(8, v);
    }

    return hash_read;
}

bool NNCache::lookup(std::uint64_t hash, Netresult & result) {
    RLOCK(m_mutex, lock);
    ++m_lookups;

    auto iter = m_cache.find(hash);
    if (iter == m_cache.end()) {
        auto iter2 = m_outfile_map.find(hash);
        if(iter2 == m_outfile_map.end()) {
            return false;  // Not found.
        }

        // at this point the file should open and run.
        std::ifstream ifs;
        ifs.open(m_filename);
        ifs.seekg(iter2->second);

        try {
            // throws and exception if ended up being parse error
            CompressedEntry e;
            e.read(ifs, hash);
            e.get(result);
        } catch (...) {
            return false;
        }

        ++m_file_hits;
        return true;
    }

    const auto& entry = iter->second;

    // Found it.
    ++m_hits;
    result = entry->result;
    return true;
}

void NNCache::insert(std::uint64_t hash,
                     const Netresult& result) {
    WLOCK(m_mutex, lock);

    if (m_cache.find(hash) != m_cache.end()) {
        return;  // Already in the cache.
    }

    CompressedEntry ce(result);
    auto size_in_bytes = ce.compressed_policy.size();
    size_in_bytes = (size_in_bytes + 7) / 8; // ceil operator

    // on an unlikely case where compression result is larger than 255 bytes,
    // or if the hash REALLY is 0xffff'ffff'ffff'ffffLL
    // we give up saving.
    if(size_in_bytes < 256 && hash != 0xffff'ffff'ffff'ffffLL
        && m_outfile.is_open() && m_outfile.good())
    {
        char c = (char)size_in_bytes;
        size_t pos = m_outfile.tellp();

        m_outfile.write((const char*)(&hash), 8);
        m_outfile.write((const char*)&(ce.policy_pass), sizeof(float));
        m_outfile.write((const char*)&(ce.winrate), sizeof(float));
        m_outfile.write(&c, 1);
        for (size_t i = 0; i < ce.compressed_policy.size(); i += 8) {
            unsigned char v = ce.compressed_policy.read_bits(i, 8);
            m_outfile.write((char*)&v, 1);
        }

        m_outfile_map[hash] = pos;

        if (m_outfile_map.size() % 1024 == 0) {
            constexpr char start_header[16] = {
                '\xff', '\xff', '\xff', '\xff',
                '\xff', '\xff', '\xff', '\xff',
                '\xff', '\xff', '\xff', '\xff',
                '\xff', '\xff', '\xff', '\xff'
            };
            m_outfile.write(start_header, 16);
        }

        // this will somewhat randomly erase from filemap.
        // this simply makes the cache entry 'inaccessible', and
        // nothing will be lost from the file itself.
        // consider this to be a way to prevent the memory from blowing up...
        if (m_outfile_map.size() > m_max_outfile_map_size) {
            m_outfile_map.erase(m_outfile_map.begin());
        }
    }

    m_cache.emplace(hash, std::make_unique<Entry>(result));
    m_order.push_back(hash);
    ++m_inserts;

    // If the cache is too large, remove the oldest entry.
    if (m_order.size() > m_max_cache_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

void NNCache::resize(int size, bool reserve_filecache) {
    m_size = size;

    assert(size >= MIN_CACHE_COUNT);

    // if we have a file-backed cache:
    // - Allocate the first MIN_CACHE_COUNT entries to normal cache
    // - Until we hit MAX_CACHE_COUNT we allocate half to normal cache, other half to file backed cache
    // - Anything beyond MAX_CACHE_COUNT goes to file backed cache
    if (reserve_filecache || m_outfile.good() || !m_outfile_map.empty()) {
        if (size < MIN_CACHE_COUNT) {
            size = MIN_CACHE_COUNT;
        }

        size = MIN_CACHE_COUNT + (size - MIN_CACHE_COUNT) / 2;

        if (size > MAX_CACHE_COUNT) {
            size = MAX_CACHE_COUNT;
        }
    }
    
    m_max_cache_size = size;
    m_max_outfile_map_size = (m_size - size) * ENTRY_SIZE / 32;

    Utils::myprintf("NNCache budgeting : %d cache, %d filemap\n",
        m_max_cache_size,
        m_max_outfile_map_size
    );
    while (m_order.size() > m_max_cache_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }

    // this will somewhat randomly erase from filemap.
    // this simply makes the cache entry 'inaccessible'
    while (m_outfile_map.size() > m_max_outfile_map_size) {
        m_outfile_map.erase(m_outfile_map.begin());
    }
    m_outfile_map.reserve(m_max_outfile_map_size);
}

void NNCache::dump_stats() {
    Utils::myprintf(
        "NNCache memory: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, %u size\n",
        m_hits, m_lookups, 100. * m_hits / (m_lookups + 1),
        m_inserts, m_cache.size());

    Utils::myprintf(
        "NNCache file: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, %u size\n",
        m_file_hits, m_lookups, 100. * m_file_hits / (m_lookups + 1),
        m_inserts, m_outfile_map.size());
}

size_t NNCache::get_estimated_size() {
    return m_order.size() * NNCache::ENTRY_SIZE + m_outfile_map.size() * 32;
}

// symbols
// V0...V63 : single value of 0 ~ 63
// Z0...Z15 : 2 ~ 17 consecutive zeros
// X0...X31 : If previous code was a V, then add 64 * (n+1) to previous value
//            If previous code was a Z, then append 16 more zeros

constexpr int V_BASE = 0;
constexpr int Z_BASE = 64;
constexpr int X_BASE = 80;

// Encoding table rule
struct NNCompressEncodeTable {

    // The value of the code
    std::uint16_t code;

    // The bit width of the code
    std::uint16_t width;

    // The number of codewords matching this entry
    std::uint16_t count;
};

static constexpr NNCompressEncodeTable encode_table[18] = {
    { 0x4, 4, 1 }, // V0
    { 0x0, 3, 1 }, // V1
    { 0xc, 4, 2 }, // V2 ~ V3
    { 0x2, 4, 4 }, // V4 ~ V7
    { 0xa, 4, 8 }, // V8 ~ V15
    { 0x6, 4, 16}, // V16 ~ V31
    { 0xe, 4, 32}, // V32 ~ V63
    { 0x1, 4, 1 }, // Z0
    { 0x9, 4, 1 }, // Z1
    { 0x5, 4, 2 }, // Z2 ~ Z3
    { 0xd, 4, 4 }, // Z4 ~ Z7
    { 0x3, 4, 8 }, // Z8 ~ Z15
    { 0xb, 4, 1 }, // X0
    { 0x7, 5, 1 }, // X1
    { 0x17, 5, 2}, // X2 ~ X3
    { 0xf, 5, 4 }, // X4 ~ X7
    { 0x1f, 6, 8}, // X8 ~ X15
    { 0x3f, 6, 16}, // X16 ~ X31
};

constexpr int bit_log2 (int x) {
    if (x == 1) return 0;
    if (x == 2) return 1;
    if (x == 4) return 2;
    if (x == 8) return 3;
    if (x == 16) return 4;
    if (x == 32) return 5;
    if (x == 64) return 6;
    return 7;
};


void NNCache::CompressedEntry::test_get() const {
    size_t iptr = 0;
    int optr = 0;
    int prev_type = -1;
    while(optr < NUM_INTERSECTIONS) {
        int symbol = 0;

        size_t lower_bits = compressed_policy.read_bits(iptr, 10);
        int symbol_base = 0;
        for(auto & entry : encode_table) {
            if(entry.code == (lower_bits & ( (1LL << entry.width) - 1))) {
                symbol = symbol_base + ((lower_bits >> entry.width) % entry.count);
                iptr += entry.width + bit_log2(entry.count);
                break;
            } else {
                symbol_base += entry.count;
            }
        }

        if(symbol < Z_BASE) {
            optr++;
            prev_type = 0;
        }
        else if(symbol < X_BASE) {
            optr += symbol-Z_BASE + 2;
            if (optr > NUM_INTERSECTIONS) {
                throw std::runtime_error("Buffer overflow");
            }
            prev_type = 1;
        } else {
            int bias = symbol - X_BASE + 1;
            if(prev_type == 0) {
                // empty
            } else if(prev_type == 1) {
		optr += bias * 16;
                if (optr > NUM_INTERSECTIONS) {
                    throw std::runtime_error("Buffer overflow");
                }
            } else {
                throw std::runtime_error("Didn't expect X type symbol");
            }
            prev_type = -1;
        }
    }

    if(iptr > compressed_policy.size() || iptr < compressed_policy.size() - 8) {
        // A 8-bit margin due to how serialization-to-disk works
        throw std::runtime_error("Unexpected size");
    }
}


void NNCache::CompressedEntry::get(NNCache::Netresult & ret) const {
    size_t iptr = 0;
    int optr = 0;
    int prev_type = -1;
    while(optr < NUM_INTERSECTIONS) {
        int symbol = 0;

        size_t lower_bits = compressed_policy.read_bits(iptr, 10);
        int symbol_base = 0;
        for(auto & entry : encode_table) {
            if(entry.code == (lower_bits & ( (1LL << entry.width) - 1))) {
                symbol = symbol_base + ((lower_bits >> entry.width) % entry.count);
                iptr += entry.width + bit_log2(entry.count);
                break;
            } else {
                symbol_base += entry.count;
            }
        }

        if(symbol < Z_BASE) {
            ret.policy[optr++] = symbol / 2048.0;
            prev_type = 0;
        }
        else if(symbol < X_BASE) {
            for(int i=0; i<symbol-Z_BASE + 2; i++) {
                if (optr >= NUM_INTERSECTIONS) {
                    throw std::runtime_error("Buffer overflow");
                }
                ret.policy[optr++] = 0;
            }
            prev_type = 1;
        } else {
            int bias = symbol - X_BASE + 1;
            if(prev_type == 0) {
                ret.policy[optr-1] += 64 / 2048.0 * bias;
            } else if(prev_type == 1) {
                for(int i=0; i<bias * 16; i++) {
                    if (optr >= NUM_INTERSECTIONS) {
                        throw std::runtime_error("Buffer overflow");
                    }
                    ret.policy[optr++] = 0;
                }
            } else {
                throw std::runtime_error("Didn't expect X type symbol");
            }
            prev_type = -1;
        }
    }

    ret.policy_pass = policy_pass;
    ret.winrate = winrate;

    if(iptr > compressed_policy.size() || iptr < compressed_policy.size() - 8) {
        // A 8-bit margin due to how serialization-to-disk works
        throw std::runtime_error("Unexpected size");
    }
}

NNCache::CompressedEntry::CompressedEntry(const Netresult & r) {
    policy_pass = r.policy_pass;
    winrate = r.winrate;
    int iptr = 0;

    auto push_symbol = [this](int symbol) {
        int symbol_base = 0;
        for(auto & entry : encode_table) {
            if(symbol >= symbol_base && symbol < symbol_base + entry.count) {
                size_t code = entry.code;
                code |= (symbol % entry.count) << entry.width;
                compressed_policy.push_bits(entry.width + bit_log2(entry.count), code);
                return;
            } else {
                symbol_base += entry.count;
            }
        }
    };

    while(iptr < NUM_INTERSECTIONS) {
        int v = static_cast<int>(r.policy[iptr] * 2048.0);
        if(v == 0) {
            int count = 0;
            while(iptr < NUM_INTERSECTIONS && static_cast<int>(r.policy[iptr] * 2048.0) == 0) {
                iptr++;
                count++;
            }
            if(count == 1) {
                push_symbol(V_BASE);
            }
            else {
                int bias = (count-2)/16;
                int offset = (count-2)%16;
                push_symbol(offset + Z_BASE);
                if(bias != 0) {
                    push_symbol(bias - 1 + X_BASE);
                }
            }
        }
        else {
            int bias = v / 64;
            int offset = v % 64;
            push_symbol(V_BASE + offset);
            if(bias != 0) {
                push_symbol(X_BASE + bias - 1);
            }
            iptr++;
        }
    }
}

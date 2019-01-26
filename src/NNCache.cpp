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

const auto NNCACHE_FILE_LOCAL = std::string("leelaz_nncache_local");

NNCache::NNCache(int size) : m_size(size)
{
    m_outfile.open(Utils::leelaz_file(NNCACHE_FILE_LOCAL), std::ios_base::out | std::ios_base::app); 
    
    constexpr char start_header[16] = {
        '\xff', '\xff', '\xff', '\xff',
        '\xff', '\xff', '\xff', '\xff',
        '\xff', '\xff', '\xff', '\xff',
        '\xff', '\xff', '\xff', '\xff'
    };
    m_outfile.write(start_header, 16);
}

bool NNCache::lookup(std::uint64_t hash, Netresult & result) {
    RLOCK(m_mutex, lock);
    ++m_lookups;
    if(m_lookups % 100000 == 0) {
        dump_stats();
    }

    auto iter = m_cache.find(hash);
    if (iter == m_cache.end()) {
        auto iter2 = m_outfile_map.find(hash);
        if(iter2 == m_outfile_map.end()) {
            return false;  // Not found.
        }

        std::ifstream ifs;
        ifs.open(Utils::leelaz_file(NNCACHE_FILE_LOCAL)); 
        ifs.seekg(iter2->second);

        Entry e;
        unsigned char c;
        ifs.read((char*)&(e.policy_pass), sizeof(float));
        ifs.read((char*)&(e.winrate), sizeof(float));
        ifs.read((char*)&c, 1);
        e.compressed_policy.expand(c*8);
        for (size_t i = 0; i < c-1; i++) {
            unsigned char v;
            ifs.read((char*)&v, 1);
            e.compressed_policy.push_bits(8, v);
        }
        e.get(result);

        ++m_file_hits;
        return true;
    }

    const auto& entry = iter->second;

    // Found it.
    ++m_hits;
    entry->get(result);
    return true;
}

void NNCache::insert(std::uint64_t hash,
                     const Netresult& result) {
    WLOCK(m_mutex, lock);

    if (m_cache.find(hash) != m_cache.end()) {
        return;  // Already in the cache.
    }

    auto entry = std::make_unique<Entry>(result);
    auto size_in_bytes = entry->compressed_policy.size();
    size_in_bytes += 8; // size of the header itself
    size_in_bytes = (size_in_bytes + 7) / 8; // ceil operator

    // on an unlikely case where compression result is larger than 255 bytes,
    // we give up saving.
    if(size_in_bytes < 256 && m_outfile.good()) {
        char c = (char)size_in_bytes;
    
        m_outfile.write((const char*)(&hash), 8);

        size_t pos = m_outfile.tellp();
        m_outfile.write((const char*)&(entry->policy_pass), sizeof(float));
        m_outfile.write((const char*)&(entry->winrate), sizeof(float));
        m_outfile.write(&c, 1);
        for (size_t i = 0; i < entry->compressed_policy.size(); i += 8) {
            unsigned char v = entry->compressed_policy.read_bits(i, 8);
            m_outfile.write((char*)&v, 1);
        }

        m_outfile_map[hash] = pos;
    }

    //
    // constexpr char start_header[16] = {
    //     '\xff', '\xff', '\xff', '\xff',
    //     '\xff', '\xff', '\xff', '\xff',
    //     '\xff', '\xff', '\xff', '\xff',
    //     '\xff', '\xff', '\xff', '\xff'
    // };
    // m_outfile.write(start_header, 16);
    //

    m_cache.emplace(hash, std::move(entry));
    m_order.push_back(hash);
    ++m_inserts;

    // If the cache is too large, remove the oldest entry.
    if (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

void NNCache::resize(int size) {
    m_size = size;
    while (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

void NNCache::set_size_from_playouts(int max_playouts) {
    // cache hits are generally from last several moves so setting cache
    // size based on playouts increases the hit rate while balancing memory
    // usage for low playout instances. 150'000 cache entries is ~208 MiB
    constexpr auto num_cache_moves = 3;
    auto max_playouts_per_move =
        std::min(max_playouts,
                 UCTSearch::UNLIMITED_PLAYOUTS / num_cache_moves);
    auto max_size = num_cache_moves * max_playouts_per_move;
    max_size = std::min(MAX_CACHE_COUNT, std::max(MIN_CACHE_COUNT, max_size));
    resize(max_size);
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
    return m_order.size() * NNCache::ENTRY_SIZE;
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


void NNCache::Entry::get(NNCache::Netresult & ret) const {
    size_t iptr = 0;
    int optr = 0;
    int prev_type = 0;
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
                ret.policy[optr++] = 0;
            }
            prev_type = 1;
        } else {
            int bias = symbol - X_BASE + 1;
            if(prev_type == 0) {
                ret.policy[optr-1] += 64 / 2048.0 * bias;
            } else if(prev_type == 1) {
                for(int i=0; i<bias * 16; i++) {
                    ret.policy[optr++] = 0;
                }
            }
        }
    }

    ret.policy_pass = policy_pass;
    ret.winrate = winrate;
}

NNCache::Entry::Entry(const Netresult & r) 
{
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



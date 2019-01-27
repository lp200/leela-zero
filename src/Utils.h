/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

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

#ifndef UTILS_H_INCLUDED
#define UTILS_H_INCLUDED

#include "config.h"

#include <atomic>
#include <limits>
#include <string>

#include "ThreadPool.h"

extern Utils::ThreadPool thread_pool;

namespace Utils {
    void myprintf_error(const char *fmt, ...);
    void myprintf(const char *fmt, ...);
    void gtp_printf(int id, const char *fmt, ...);
    void gtp_printf_raw(const char *fmt, ...);
    void gtp_fail_printf(int id, const char *fmt, ...);
    void log_input(const std::string& input);
    bool input_pending();

    template<class T>
    void atomic_add(std::atomic<T> &f, T d) {
        T old = f.load();
        while (!f.compare_exchange_weak(old, old + d));
    }

    template<typename T>
    T rotl(const T x, const int k) {
        return (x << k) | (x >> (std::numeric_limits<T>::digits - k));
    }

    inline bool is7bit(int c) {
        return c >= 0 && c <= 127;
    }

    size_t ceilMultiple(size_t a, size_t b);

    const std::string leelaz_file(std::string file);


    class bitstream {
    private:
        size_t _bitcount = 0;
        size_t _capacity = 0;
        std::unique_ptr<std::uint64_t[]> _ptr;
    public:
        void clear() {
            _bitcount = 0;
            _capacity = 0;
            _ptr.reset();
        }

        size_t size() const {
            return _bitcount;
        }

        void expand(size_t count) {
            // make count the next largest multiple-of-64
            count = (count + 63) / 64;
            count *= 64;

            if (_capacity >= count) {
                return;
            }

            auto newptr = new std::uint64_t[count/64];
            for (size_t i=0; i < _capacity/64; i++) {
                newptr[i] = _ptr[i];
            }
            for(size_t i = _capacity/64; i < count/64; i++) {
                newptr[i] = 0;
            }
            _capacity = count;
            _ptr.reset(newptr);
        }

        void push_bits(size_t count, size_t value) {
            if (_bitcount + count > _capacity) {
                expand(_bitcount + count * 2);
            }
            while (count > 0) {
                auto bits_to_add = 64 - _bitcount % 64;
                if (bits_to_add > count) {
                    bits_to_add = count;
                }

                auto masked_value = value & ((1LL << bits_to_add)-1);
                _ptr[_bitcount/64] = _ptr[_bitcount/64] | (masked_value << (_bitcount % 64));

                _bitcount += bits_to_add;
                count -= bits_to_add;
                value = value >> bits_to_add;
            }
        }
        size_t read_bits(size_t start_loc, size_t count) const {
            if (start_loc >= _capacity) {
                return 0;
            }
            auto start_loc_offset = start_loc % 64;
            if (count > 64 - start_loc_offset) {
                return
                    // upper bits
                    read_bits(start_loc + 64 - start_loc_offset, count - 64 + start_loc_offset) << (64 - start_loc_offset)
                    // lower 64-start_loc_offset bits
                    | (_ptr[start_loc/64] >> start_loc_offset);
            } else {
                return (_ptr[start_loc/64] >> start_loc_offset) & ( (1LL << count) - 1);
            }
        }
    };
}

#endif

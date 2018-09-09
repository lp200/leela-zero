/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2018 Junhee Yoo and contributors

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
*/

#ifndef DISTRIBUTED_NETWORK_H_INCLUDED
#define DISTRIBUTED_NETWORK_H_INCLUDED

#include "config.h"

#include <boost/array.hpp>
#include <boost/bind.hpp>
#include <boost/asio.hpp>
#include <boost/algorithm/string.hpp>
#include <algorithm>
#include <vector>

#include "Utils.h"
#include "SMP.h"

#include "Network.h"

class DistributedClientNetwork : public Network
{
private:
    boost::asio::io_service m_io_service;
    std::deque<boost::asio::ip::tcp::socket> m_sockets;
    std::atomic<size_t> m_active_socket_count{0};
    SMP::Mutex m_socket_mutex;
    std::vector<std::string> m_serverlist;

    std::vector<float> get_output_from_socket(const std::vector<bool> & input_data,
                                              const int symmetry, boost::asio::ip::tcp::socket & socket);

public:
    void initialize(int playouts, const std::vector<std::string> & serverlist, std::uint64_t hash);
    void init_servers(const std::vector<std::string> & serverlist, std::uint64_t hash);

protected:
    virtual Netresult get_output_internal(const std::vector<bool> & input_data,
                                          const int symmetry, bool selfcheck = false);
};


class DistributedServerNetwork : public Network
{
private:
    boost::asio::io_service m_io_service;
public:
    void listen(int portnum, std::uint64_t hash);
};

#endif

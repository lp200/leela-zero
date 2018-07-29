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

#include "SMP.h"

#include "Network.h"

class DistributedNetwork : public Network
{

private:
    std::deque<boost::asio::ip::tcp::socket> m_sockets;
    SMP::Mutex m_socket_mutex;
public:
    void initialize(int playouts, const std::vector<std::string> & serverlist) {
        using boost::asio::ip::tcp;

        Network::initialize(playouts, "");

        for (auto x : serverlist) {
            std::vector<std::string> x2;
            boost::split(x2, x, boost::is_any_of(":"));
            if(x2.size() != 2) {
                printf("Error in --nn-client argument parsing : Expecting [server]:[port] syntax\n");
                printf("(got %s\n", x.c_str());
                throw std::runtime_error("Malformed --nn-client argument ");
            }
        
            auto addr = x2[0];
            auto port = x2[1];

            boost::asio::io_service io_service;
    
            tcp::resolver resolver(io_service);
            tcp::resolver::query query(addr, port);
            auto endpoints = resolver.resolve(query);
    
            const auto num_threads = (cfg_num_threads + serverlist.size() - 1) / serverlist.size();

            for(auto i=size_t{0}; i<num_threads; i++) {
                tcp::socket socket(io_service);
                boost::asio::connect(socket, endpoints);
                m_sockets.emplace_back(std::move(socket));
            }
        }
    }
protected:
    virtual Netresult get_output_internal(const std::vector<bool> & input_data,
                                          const int symmetry, bool selfcheck = false) {
        if (selfcheck) {
            return Network::get_output_internal(input_data, symmetry, true);
        }
        using boost::asio::ip::tcp;

        LOCK(m_socket_mutex, lock);
        assert(!m_sockets.empty());

        auto socket = std::move(m_sockets.front());
        m_sockets.pop_front();
        lock.unlock();

        std::vector<char> input_data_ch(input_data.size() + 1); // input_data (18*361) + symmetry
        assert(input_data_ch.size() == INPUT_CHANNELS * BOARD_SQUARES + 1);
        std::copy(begin(input_data), end(input_data), begin(input_data_ch));
        input_data_ch[input_data_ch.size()-1] = symmetry;
    
        std::vector<float> output_data_f(BOARD_SQUARES + 2);
        try {
            boost::system::error_code error;
            boost::asio::write(socket, boost::asio::buffer(input_data_ch), error);
            if (error)
                throw boost::system::system_error(error); // Some other error.

            boost::asio::read(socket, boost::asio::buffer(output_data_f), error);
            if (error)
                throw boost::system::system_error(error); // Some other error.
        } catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
            throw;
        }

        {
            LOCK(m_socket_mutex, lock2);
            m_sockets.push_back(std::move(socket));
        }

        Netresult ret;
        std::copy(begin(output_data_f), begin(output_data_f) + BOARD_SQUARES, begin(ret.policy));
        ret.policy_pass = output_data_f[BOARD_SQUARES];
        ret.winrate = output_data_f[BOARD_SQUARES + 1];

        return ret;
    }
public:
    void listen(int portnum) {
        using boost::asio::ip::tcp;
        try {
            boost::asio::io_service io_service;
    
            tcp::acceptor acceptor(io_service, tcp::endpoint(tcp::v4(), portnum));
    
            for (;;)
            {
                tcp::socket socket(io_service);
                acceptor.accept(socket);
    
                std::thread t(
                    std::bind(
                        [this](tcp::socket & socket) {
                            while (true) {
                                std::array<char,  INPUT_CHANNELS * BOARD_SQUARES + 1> buf;
    
                                boost::system::error_code error;
                                boost::asio::read(socket, boost::asio::buffer(buf), error);
                                if (error == boost::asio::error::eof)
                                    break; // Connection closed cleanly by peer.
                                else if (error)
                                    throw boost::system::system_error(error); // Some other error.
                                std::vector<bool> input_data(INPUT_CHANNELS * BOARD_SQUARES);
                                std::copy(begin(buf), end(buf)-1, begin(input_data));
                                int symmetry = buf[INPUT_CHANNELS * BOARD_SQUARES];
                                
                                auto result = Network::get_output_internal(input_data, symmetry);

                                std::array<float, BOARD_SQUARES+2> obuf;
                                std::copy(begin(result.policy), end(result.policy), begin(obuf));
                                obuf[BOARD_SQUARES] = result.policy_pass;
                                obuf[BOARD_SQUARES+1] = result.winrate;
                                boost::asio::write(socket, boost::asio::buffer(obuf), error);
                                if (error == boost::asio::error::eof)
                                    break; // Connection closed cleanly by peer.
                                else if (error)
                                    throw boost::system::system_error(error); // Some other error.
                            }
                        },
                        std::move(socket)
                    )
                );
                t.detach();
            }
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }


    }
};

#endif

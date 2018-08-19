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

#include "GTP.h"
#include "DistributedNetwork.h"

using boost::asio::ip::tcp;

std::vector<float> DistributedClientNetwork::get_output_from_socket(const std::vector<bool> & input_data,
                                          const int symmetry, boost::asio::ip::tcp::socket & socket) {

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
    return output_data_f;
}
void DistributedClientNetwork::initialize(int playouts, const std::vector<std::string> & serverlist) {
    m_serverlist = serverlist;
    Network::initialize(playouts, "");

    // if this didn't create enough threads, the background thread will retry creating more and more
    // if it never creates enough threads, local capability (be it CPU or GPU) will be used
    init_servers(serverlist);

    // create a background thread which tries to create new connectins if some are dead.
    // thread stays active forever, hence if somebody wants to have capability of destroying
    // hets in the middle of a run, this thread should also be safely killed...
    std::thread t(
        [this]() {
            while (true) {
                std::this_thread::sleep_for(
                    std::chrono::seconds(1)
                );
                if (m_active_socket_count.load() < static_cast<size_t>(cfg_num_threads)) {
                    LOCK(m_socket_mutex, lock);
                    init_servers(m_serverlist);
                }
            }
        }
    );
    t.detach();
}

void DistributedClientNetwork::init_servers(const std::vector<std::string> & serverlist) {

    const auto num_threads = (cfg_num_threads - m_sockets.size() + serverlist.size() - 1) / serverlist.size();
    for (auto x : serverlist) {
        std::vector<std::string> x2;
        boost::split(x2, x, boost::is_any_of(":"));
        if (x2.size() != 2) {
            printf("Error in --nn-client argument parsing : Expecting [server]:[port] syntax\n");
            printf("(got %s\n", x.c_str());
            throw std::runtime_error("Malformed --nn-client argument ");
        }
    
        auto addr = x2[0];
        auto port = x2[1];

        tcp::resolver resolver(m_io_service);

        // these are deprecated in latest boost but still a quite recent Ubuntu distribution
        // doesn't support the alternative newer interface.
        decltype(resolver)::iterator endpoints;
        decltype(resolver)::query query(addr, port);
        try {
            endpoints = resolver.resolve(query);
        } catch (...) {
            Utils::myprintf("Cannot resolve server address %s port %s\n", addr.c_str(), port.c_str());
            // cannot resolve server - probably server dead
            break;
        }


        for (auto i=size_t{0}; i<num_threads; i++) {
            tcp::socket socket(m_io_service);

            // try a dummy call
            try {
                boost::asio::connect(socket, endpoints);
                GameState gs;
                gs.init_game(BOARD_SIZE, 7.5);
                std::vector<bool> dummy_input = gather_features(&gs, 0);
                get_output_from_socket(dummy_input, 0, socket);
            } catch (...) {
                // doesn't work. Probably remote side ran out of threads.
                // drop socket.
                Utils::myprintf("NN client dropped to server %s port %s (thread %d)\n", addr.c_str(), port.c_str(), i);
                continue;
            }
            m_sockets.emplace_back(std::move(socket));
            m_active_socket_count++;

            Utils::myprintf("NN client connected to server %s port %s (thread %d)\n", addr.c_str(), port.c_str(), i);
        }

    }
}

Network::Netresult DistributedClientNetwork::get_output_internal(
                                      const std::vector<bool> & input_data,
                                      const int symmetry, bool selfcheck) {
    if (selfcheck) {
        return Network::get_output_internal(input_data, symmetry, true);
    }

    LOCK(m_socket_mutex, lock);
    if (m_sockets.empty()) {
        // if we don't have enough sockets, use local machine capability as backup
        lock.unlock();
        return Network::get_output_internal(input_data, symmetry, selfcheck);
    }

    // XXX : moving a closed socket will segfault.  Think what we should do?
    auto socket = std::move(m_sockets.front());
    m_sockets.pop_front();
    lock.unlock();

    std::vector<float> output_data_f;

    try {
        output_data_f = get_output_from_socket(input_data, symmetry, socket);
    } catch (...) {
        // socket is dead for some reason.  Throw it away and use local machine
        // capability as a backup
        assert(m_active_socket_count.load() > 0);
        m_active_socket_count--;
        return Network::get_output_internal(input_data, symmetry, selfcheck);
    }

    {
        LOCK(m_socket_mutex, lock2);
        m_sockets.push_back(std::move(socket));
    }

    Network::Netresult ret;
    std::copy(begin(output_data_f), begin(output_data_f) + BOARD_SQUARES, begin(ret.policy));
    ret.policy_pass = output_data_f[BOARD_SQUARES];
    ret.winrate = output_data_f[BOARD_SQUARES + 1];

    return ret;
}


void DistributedServerNetwork::listen(int portnum) {
    try {
        std::atomic<int> num_threads{0};

        tcp::acceptor acceptor(m_io_service, tcp::endpoint(tcp::v4(), portnum));
        Utils::myprintf("NN server listening on port %d\n", portnum);

        for (;;)
        {
            tcp::socket socket(m_io_service);
            acceptor.accept(socket);

            int v = num_threads++;
            if (v >= cfg_num_threads) {
                --num_threads;
                Utils::myprintf("Dropping connection from %s due to too many threads\n",
                     socket.remote_endpoint().address().to_string().c_str()
                );
                socket.shutdown(tcp::socket::shutdown_send);
                socket.shutdown(tcp::socket::shutdown_receive);
                socket.close();
                continue;
            }

            Utils::myprintf("NN server connection established from %s (thread %d, max %d)\n",
                     socket.remote_endpoint().address().to_string().c_str(), v, cfg_num_threads
            );

            std::thread t(
                std::bind(
                    [&num_threads, this](tcp::socket & socket) {

                        auto remote_endpoint = socket.remote_endpoint().address().to_string();
                        while (true) {
                            std::array<char,  INPUT_CHANNELS * BOARD_SQUARES + 1> buf;

                            boost::system::error_code error;
                            boost::asio::read(socket, boost::asio::buffer(buf), error);
                            if (error == boost::asio::error::eof)
                                break; // Connection closed cleanly by peer.
                            else if (error) {
                                Utils::myprintf("Socket read failed with message : %s\n",
                                                error.message().c_str()
                                );
                                break;
                            }
                                
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
                            else if (error) {
                                Utils::myprintf("Socket write failed with message : %s\n",
                                                error.message().c_str()
                                );
                                break;
                            }
                        }

                        Utils::myprintf("NN server connection closed from %s\n", remote_endpoint.c_str());
                        num_threads--;
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

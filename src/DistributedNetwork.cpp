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

#include <iostream>

#include "GTP.h"
#include "DistributedNetwork.h"

using boost::asio::ip::tcp;

template <typename... T> static void netprintf(const char * fmt, T... params) {
    if (cfg_nn_client_verbose) {
        Utils::myprintf(fmt, params...);
    }
}
std::vector<float> DistributedClientNetwork::get_output_from_socket(const std::vector<float> & input_data,
                                                                    boost::asio::ip::tcp::socket & socket) {

    std::vector<char> input_data_ch(input_data.size()); // input_data (18*361)
    assert(input_data_ch.size() == INPUT_CHANNELS * NUM_INTERSECTIONS + 1);
    std::copy(begin(input_data), end(input_data), begin(input_data_ch));

    std::vector<float> output_data_f(NUM_INTERSECTIONS + 2);
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

void DistributedClientNetwork::initialize(int playouts, const std::vector<std::string> & serverlist, std::uint64_t hash) {
    m_serverlist = serverlist;
    Network::initialize(playouts, "");

    // if this didn't create enough threads, the background thread will retry creating more and more
    // if it never creates enough threads, local capability (be it CPU or GPU) will be used
    init_servers(serverlist, hash);

    // create a background thread which tries to create new connectins if some are dead.
    // thread stays active forever, hence if somebody wants to have capability of destroying
    // hets in the middle of a run, this thread should also be safely killed...
    std::thread t(
        [this, hash]() {
            while (true) {
                std::this_thread::sleep_for(
                    std::chrono::seconds(1)
                );
                if (m_active_socket_count.load() < static_cast<size_t>(cfg_num_threads)) {
                    LOCK(m_socket_mutex, lock);
                    init_servers(m_serverlist, hash);
                }
            }
        }
    );
    t.detach();
}

void DistributedClientNetwork::init_servers(const std::vector<std::string> & serverlist, std::uint64_t hash) {

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
            netprintf("Cannot resolve server address %s port %s\n", addr.c_str(), port.c_str());
            // cannot resolve server - probably server dead
            break;
        }


        for (auto i=size_t{0}; i<num_threads; i++) {
            tcp::socket socket(m_io_service);

            try {

                auto connect_task = [this, &addr, &port, &socket, &endpoints, hash] () {
                    boost::asio::connect(socket, endpoints);
                    std::array<std::uint64_t,1> my_hash {hash};
                    std::array<std::uint64_t,1> remote_hash {0};
    
                    boost::system::error_code error;
                    boost::asio::write(socket, boost::asio::buffer(my_hash), error);
                    if (error)
                        throw boost::system::system_error(error); // Some other error.
    
                    boost::asio::read(socket, boost::asio::buffer(remote_hash), error);
                    if (error)
                        throw boost::system::system_error(error); // Some other error.
    
                    if(my_hash[0] != remote_hash[0]) {
                        netprintf(
                            "NN client dropped to server %s port %s (hash mismatch, remote=%llx, local=%llx)\n",
                                addr.c_str(), port.c_str(), remote_hash[0], my_hash[0]);
                        throw std::exception();
                    }
                };
        
                auto f = std::async(std::launch::async, connect_task);
                auto res = f.wait_for(std::chrono::milliseconds(500));
                if (res == std::future_status::timeout) {
                    throw std::exception();
                }
                f.get();
            } catch (...) {
                // doesn't work. Probably remote side ran out of threads.
                // drop socket.
                netprintf("NN client dropped to server %s port %s (thread %d)\n", addr.c_str(), port.c_str(), i);
                continue;
            }
            m_sockets.emplace_back(std::move(socket));
            m_active_socket_count++;

            netprintf("NN client connected to server %s port %s (thread %d)\n", addr.c_str(), port.c_str(), i);
        }
    }

    m_socket_initialized = true;
}

void DistributedClientNetwork::initialize(int playouts, const std::string & weightsfile) {
    m_local_initialized = true;
    Network::initialize(playouts, weightsfile);
}

std::pair<std::vector<float>,float> DistributedClientNetwork::get_output_internal(
                                      const std::vector<float> & input_data,
                                      bool selfcheck) {
    if (selfcheck) {
        assert(m_local_initialized);
        return Network::get_output_internal(input_data, true);
    }

    if (!m_socket_initialized) {
        assert(m_local_initialized);
        return Network::get_output_internal(input_data, selfcheck);
    }


    LOCK(m_socket_mutex, lock);
    if (m_sockets.empty()) {
        lock.unlock();

        // if we don't have enough sockets, use local machine capability as backup
        if (m_local_initialized) {
            return Network::get_output_internal(input_data, selfcheck);
        } else {
            // no local resource, try again later
            std::this_thread::sleep_for(std::chrono::seconds(1));
            return get_output_internal(input_data, selfcheck);
        }
    }

    auto socket = std::move(m_sockets.front());
    m_sockets.pop_front();
    lock.unlock();

    std::vector<float> output_data_f;

    try {
        // try socket access as an asynchronous task.  
        auto f = std::async(
            std::launch::async, 
            [this, input_data, &socket]() { return get_output_from_socket(input_data, socket); }
        );

        auto res = f.wait_for(std::chrono::milliseconds(500));
        if (res == std::future_status::timeout) {
            throw std::exception();
        }
        output_data_f = f.get();
    } catch (...) {
        // socket is dead for some reason.  Throw it away and use local machine
        // capability as a backup
        assert(m_active_socket_count.load() > 0);
        m_active_socket_count--;
        if (m_local_initialized) {
            return Network::get_output_internal(input_data, selfcheck);
        } else {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            return get_output_internal(input_data, selfcheck);
        }
    }

    {
        LOCK(m_socket_mutex, lock2);
        m_sockets.push_back(std::move(socket));
    }

    std::vector<float> p(NUM_INTERSECTIONS + 1);
    float v;

    std::copy(begin(output_data_f), begin(output_data_f) + NUM_INTERSECTIONS, begin(p));
    v = output_data_f[NUM_INTERSECTIONS + 1];

    return {p, v};
}


NetServer::NetServer(Network & net) : m_net(net)
{
}

void NetServer::listen(int portnum, std::uint64_t hash) {
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
                    [&num_threads, this, hash](tcp::socket & socket) {

                        auto remote_endpoint = socket.remote_endpoint().address().to_string();

                        std::array<std::uint64_t, 1> my_hash{hash};
                        std::array<std::uint64_t, 1> remote_hash {0};
                        boost::system::error_code error;

                        boost::asio::read(socket, boost::asio::buffer(remote_hash), error);
                        if (error)
                            throw boost::system::system_error(error); // Some other error.

                        boost::asio::write(socket, boost::asio::buffer(my_hash), error);
                        if (error)
                            throw boost::system::system_error(error); // Some other error.


                        while (true) {
                            std::array<char,  INPUT_CHANNELS * NUM_INTERSECTIONS> buf;

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

                            std::vector<float> input_data(INPUT_CHANNELS * NUM_INTERSECTIONS);
                            std::copy(begin(buf), end(buf), begin(input_data));

                            auto result = m_net.get_output_internal(input_data, false);

                            std::array<float, NUM_INTERSECTIONS+2> obuf;
                            std::copy(begin(result.first), end(result.first), begin(obuf));
                            obuf[NUM_INTERSECTIONS+1] = result.second;
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

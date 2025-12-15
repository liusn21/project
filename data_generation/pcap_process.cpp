#include <pcap.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <filesystem>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ether.h>
#include <arpa/inet.h>
#include <cstring>
#include <memory>

namespace fs = std::filesystem;

// DLT (Data Link Type) constants
#ifndef DLT_IPV4
#define DLT_IPV4 228
#endif

#ifndef DLT_RAW
#define DLT_RAW 12
#endif

#ifndef DLT_LINUX_SLL
#define DLT_LINUX_SLL 113
#endif

// Flow key structure preserving first packet direction (client->server)
struct FlowKey {
    uint8_t protocol;
    uint32_t ip1;
    uint16_t port1;
    uint32_t ip2;
    uint16_t port2;
    
    FlowKey(uint8_t proto, uint32_t src_ip, uint16_t src_port, 
            uint32_t dst_ip, uint16_t dst_port) 
        : protocol(proto), ip1(src_ip), port1(src_port), 
          ip2(dst_ip), port2(dst_port) {}
    
    bool operator==(const FlowKey& other) const {
        return protocol == other.protocol && 
               ip1 == other.ip1 && port1 == other.port1 &&
               ip2 == other.ip2 && port2 == other.port2;
    }
    
    FlowKey reverse() const {
        return FlowKey(protocol, ip2, port2, ip1, port1);
    }
    
    std::string to_filename() const {
        char ip1_str[INET_ADDRSTRLEN], ip2_str[INET_ADDRSTRLEN];
        struct in_addr addr;
        
        addr.s_addr = ip1;
        inet_ntop(AF_INET, &addr, ip1_str, INET_ADDRSTRLEN);
        
        addr.s_addr = ip2;
        inet_ntop(AF_INET, &addr, ip2_str, INET_ADDRSTRLEN);
        
        std::string proto_str = (protocol == IPPROTO_TCP) ? "TCP" : "UDP";
        
        return proto_str + "_" + std::string(ip1_str) + "_" + 
               std::to_string(port1) + "_" + 
               std::string(ip2_str) + "_" + std::to_string(port2) + ".pcap";
    }
};

namespace std {
    template<>
    struct hash<FlowKey> {
        size_t operator()(const FlowKey& k) const {
            size_t h1 = hash<uint8_t>()(k.protocol);
            size_t h2 = hash<uint32_t>()(k.ip1);
            size_t h3 = hash<uint16_t>()(k.port1);
            size_t h4 = hash<uint32_t>()(k.ip2);
            size_t h5 = hash<uint16_t>()(k.port2);
            return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3) ^ (h5 << 4);
        }
    };
}

struct PacketData {
    std::vector<uint8_t> data;
    struct pcap_pkthdr header;
    bool has_payload;
    
    PacketData(const struct pcap_pkthdr* h, const uint8_t* packet, bool payload) {
        header = *h;
        data.assign(packet, packet + h->caplen);
        has_payload = payload;
    }
};

// Check if packet has transport layer payload
bool has_transport_payload(const uint8_t* packet, int len, int linktype) {
    int offset = 0;
    
    if (linktype == DLT_EN10MB) {
        if (len < sizeof(struct ether_header)) return false;
        const struct ether_header* eth = (struct ether_header*)packet;
        if (ntohs(eth->ether_type) != ETHERTYPE_IP) return false;
        offset = sizeof(struct ether_header);
    } else if (linktype == DLT_RAW || linktype == DLT_IPV4 || linktype == 101) {
        offset = 0;
    } else if (linktype == DLT_LINUX_SLL) {
        if (len < 16) return false;
        uint16_t proto = (packet[14] << 8) | packet[15];
        if (proto != ETHERTYPE_IP) return false;
        offset = 16;
    } else {
        return false;
    }
    
    if (len < offset + sizeof(struct ip)) return false;
    
    const struct ip* ip_hdr = (struct ip*)(packet + offset);
    if (ip_hdr->ip_v != 4) return false;
    
    uint8_t protocol = ip_hdr->ip_p;
    if (protocol != IPPROTO_TCP && protocol != IPPROTO_UDP) return false;
    
    uint16_t ip_total_len = ntohs(ip_hdr->ip_len);
    int ip_header_len = ip_hdr->ip_hl * 4;
    
    int payload_len = 0;
    
    if (protocol == IPPROTO_TCP) {
        if (len < offset + ip_header_len + sizeof(struct tcphdr)) return false;
        const struct tcphdr* tcp = (struct tcphdr*)(packet + offset + ip_header_len);
        int tcp_header_len = tcp->th_off * 4;
        payload_len = ip_total_len - ip_header_len - tcp_header_len;
    } else {
        if (len < offset + ip_header_len + sizeof(struct udphdr)) return false;
        payload_len = ip_total_len - ip_header_len - 8;
    }
    
    return payload_len > 0;
}

// Parse packet and extract flow key
std::unique_ptr<FlowKey> parse_packet(const uint8_t* packet, int len, int linktype) {
    const struct ip* ip_hdr = nullptr;
    int offset = 0;
    
    if (linktype == DLT_EN10MB) {
        if (len < sizeof(struct ether_header)) return nullptr;
        const struct ether_header* eth = (struct ether_header*)packet;
        if (ntohs(eth->ether_type) != ETHERTYPE_IP) return nullptr;
        offset = sizeof(struct ether_header);
    } else if (linktype == DLT_RAW || linktype == DLT_IPV4 || linktype == 101) {
        offset = 0;
    } else if (linktype == DLT_LINUX_SLL) {
        if (len < 16) return nullptr;
        uint16_t proto = (packet[14] << 8) | packet[15];
        if (proto != ETHERTYPE_IP) return nullptr;
        offset = 16;
    } else {
        return nullptr;
    }
    
    if (len < offset + sizeof(struct ip)) return nullptr;
    
    ip_hdr = (struct ip*)(packet + offset);
    if (ip_hdr->ip_v != 4) return nullptr;
    
    uint8_t protocol = ip_hdr->ip_p;
    if (protocol != IPPROTO_TCP && protocol != IPPROTO_UDP) return nullptr;
    
    uint32_t src_ip = ip_hdr->ip_src.s_addr;
    uint32_t dst_ip = ip_hdr->ip_dst.s_addr;
    
    int ip_header_len = ip_hdr->ip_hl * 4;
    const uint8_t* transport_hdr = packet + offset + ip_header_len;
    
    uint16_t src_port, dst_port;
    
    if (protocol == IPPROTO_TCP) {
        if (len < offset + ip_header_len + sizeof(struct tcphdr)) 
            return nullptr;
        const struct tcphdr* tcp = (struct tcphdr*)transport_hdr;
        src_port = ntohs(tcp->th_sport);
        dst_port = ntohs(tcp->th_dport);
    } else {
        if (len < offset + ip_header_len + sizeof(struct udphdr)) 
            return nullptr;
        const struct udphdr* udp = (struct udphdr*)transport_hdr;
        src_port = ntohs(udp->uh_sport);
        dst_port = ntohs(udp->uh_dport);
        
        if (src_port == 53 || dst_port == 53) {
            return nullptr;
        }
    }
    
    return std::make_unique<FlowKey>(protocol, src_ip, src_port, dst_ip, dst_port);
}

// Process a single pcap file
// output_subdir: in label mode, this is the label directory; in normal mode, this is pcap stem name
void process_pcap(const std::string& input_pcap, const std::string& output_dir, 
                  const std::string& output_subdir) {
    std::cout << "Processing: " << input_pcap << std::endl;
    
    char errbuf[PCAP_ERRBUF_SIZE];
    
    pcap_t* handle = pcap_open_offline(input_pcap.c_str(), errbuf);
    if (!handle) {
        std::cerr << "Error opening pcap file: " << input_pcap << " - " << errbuf << std::endl;
        return;
    }
    
    int linktype = pcap_datalink(handle);
    int snapshot = pcap_snapshot(handle);
    
    std::unordered_map<FlowKey, int> flow_counts;
    std::unordered_map<FlowKey, std::vector<PacketData>> flow_packets;
    
    struct pcap_pkthdr* header;
    const uint8_t* packet;
    int result;
    
    while ((result = pcap_next_ex(handle, &header, &packet)) >= 0) {
        if (result == 0) continue;
        
        auto flow_key = parse_packet(packet, header->caplen, linktype);
        if (flow_key) {
            bool has_payload = has_transport_payload(packet, header->caplen, linktype);
            
            auto it = flow_counts.find(*flow_key);
            if (it != flow_counts.end()) {
                if (has_payload) {
                    flow_counts[*flow_key]++;
                }
                flow_packets[*flow_key].emplace_back(header, packet, has_payload);
            } else {
                FlowKey reverse_key = flow_key->reverse();
                auto reverse_it = flow_counts.find(reverse_key);
                if (reverse_it != flow_counts.end()) {
                    if (has_payload) {
                        flow_counts[reverse_key]++;
                    }
                    flow_packets[reverse_key].emplace_back(header, packet, has_payload);
                } else {
                    if (has_payload) {
                        flow_counts[*flow_key] = 1;
                    } else {
                        flow_counts[*flow_key] = 0;
                    }
                    flow_packets[*flow_key].emplace_back(header, packet, has_payload);
                }
            }
        }
    }
    
    pcap_close(handle);
    
    std::unordered_map<FlowKey, std::vector<PacketData>> valid_flows;
    for (auto& [key, packets] : flow_packets) {
        if (flow_counts[key] >= 5) {
            valid_flows[key] = std::move(packets);
        }
    }
    
    std::cout << "  Found " << flow_counts.size() << " flows, " 
              << valid_flows.size() << " valid (>=5 packets with payload)" << std::endl;
    
    if (valid_flows.empty()) {
        std::cout << "  No valid flows, skipping output" << std::endl;
        return;
    }
    
    // Create output directory using the provided subdirectory
    std::string pcap_output_dir = output_dir + "/" + output_subdir;
    
    try {
        fs::create_directories(pcap_output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error creating directory: " << pcap_output_dir << " - " << e.what() << std::endl;
        return;
    }
    
    for (const auto& [flow_key, packets] : valid_flows) {
        std::string flow_filename = pcap_output_dir + "/" + flow_key.to_filename();
        
        pcap_t* out_handle = pcap_open_dead(linktype, snapshot);
        if (!out_handle) {
            std::cerr << "Error creating pcap handle for: " << flow_filename << std::endl;
            continue;
        }
        
        pcap_dumper_t* dumper = pcap_dump_open(out_handle, flow_filename.c_str());
        if (!dumper) {
            std::cerr << "Error opening output file: " << flow_filename << std::endl;
            pcap_close(out_handle);
            continue;
        }
        
        for (const auto& pkt : packets) {
            pcap_dump((u_char*)dumper, &pkt.header, pkt.data.data());
        }
        
        pcap_dump_close(dumper);
        pcap_close(out_handle);
    }
    
    std::cout << "  Completed: " << output_subdir << " (" << valid_flows.size() << " flows)" << std::endl;
}

// Thread pool worker
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
    
public:
    ThreadPool(size_t num_threads) : stop(false) {
        for (size_t i = 0; i < num_threads; ++i) {
            workers.emplace_back([this] {
                while (true) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { 
                            return this->stop || !this->tasks.empty(); 
                        });
                        
                        if (this->stop && this->tasks.empty()) return;
                        
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }
    
    template<class F>
    void enqueue(F&& f) {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            tasks.emplace(std::forward<F>(f));
        }
        condition.notify_one();
    }
    
    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread& worker : workers) {
            worker.join();
        }
    }
};

// Structure to hold pcap file info with its label (parent directory)
struct PcapFileInfo {
    std::string filepath;
    std::string label;  // Parent directory name (used in label mode)
    std::string stem;   // File stem without extension (used in normal mode)
};

// Recursively find all pcap files and extract label info
void find_pcap_files(const std::string& dir, std::vector<PcapFileInfo>& pcap_files, 
                     const std::string& base_dir) {
    try {
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".pcap" || ext == ".pcapng") {
                    PcapFileInfo info;
                    info.filepath = entry.path().string();
                    info.stem = entry.path().stem().string();
                    
                    // Get label from parent directory relative to base_dir
                    // e.g., base_dir/label/xxx.pcap -> label = "label"
                    fs::path rel_path = fs::relative(entry.path().parent_path(), base_dir);
                    info.label = rel_path.string();
                    
                    // Handle case where pcap is directly in base_dir (no label)
                    if (info.label == "." || info.label.empty()) {
                        info.label = "unlabeled";
                    }
                    
                    pcap_files.push_back(info);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning directory: " << dir << " - " << e.what() << std::endl;
    }
}

void print_usage(const char* prog_name) {
    std::cerr << "Usage: " << prog_name << " [OPTIONS] <input_pcap_dir> <output_pcap_dir>" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -l, --label-mode    Preserve label directory structure" << std::endl;
    std::cerr << "                      Input:  input_dir/label/xxx.pcap" << std::endl;
    std::cerr << "                      Output: output_dir/label/flow1.pcap, flow2.pcap, ..." << std::endl;
    std::cerr << std::endl;
    std::cerr << "Default mode (without -l):" << std::endl;
    std::cerr << "                      Output: output_dir/xxx/flow1.pcap, flow2.pcap, ..." << std::endl;
    std::cerr << "                      (Each input pcap gets its own subdirectory)" << std::endl;
}

int main(int argc, char* argv[]) {
    bool label_mode = false;
    std::vector<std::string> positional_args;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-l" || arg == "--label-mode") {
            label_mode = true;
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg[0] == '-') {
            std::cerr << "Unknown option: " << arg << std::endl;
            print_usage(argv[0]);
            return 1;
        } else {
            positional_args.push_back(arg);
        }
    }
    
    if (positional_args.size() != 2) {
        print_usage(argv[0]);
        return 1;
    }
    
    std::string input_dir = positional_args[0];
    std::string output_dir = positional_args[1];
    
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: Input directory does not exist: " << input_dir << std::endl;
        return 1;
    }
    
    try {
        fs::create_directories(output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error creating output directory: " << output_dir << " - " << e.what() << std::endl;
        return 1;
    }
    
    std::vector<PcapFileInfo> pcap_files;
    std::cout << "Scanning for pcap files in: " << input_dir << std::endl;
    find_pcap_files(input_dir, pcap_files, input_dir);
    
    std::cout << "Found " << pcap_files.size() << " pcap files" << std::endl;
    
    if (label_mode) {
        std::cout << "Mode: Label-preserving (output grouped by label directory)" << std::endl;
        
        // Print label statistics
        std::unordered_map<std::string, int> label_counts;
        for (const auto& info : pcap_files) {
            label_counts[info.label]++;
        }
        std::cout << "Labels found:" << std::endl;
        for (const auto& [label, count] : label_counts) {
            std::cout << "  " << label << ": " << count << " files" << std::endl;
        }
    } else {
        std::cout << "Mode: Default (output grouped by pcap filename)" << std::endl;
    }
    
    if (pcap_files.empty()) {
        std::cout << "No pcap files found. Exiting." << std::endl;
        return 0;
    }
    
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4;
    
    std::cout << "Using " << num_threads << " threads for parallel processing" << std::endl;
    
    ThreadPool pool(num_threads);
    
    for (const auto& pcap_info : pcap_files) {
        // Determine output subdirectory based on mode
        std::string output_subdir = label_mode ? pcap_info.label : pcap_info.stem;
        
        pool.enqueue([pcap_info, output_dir, output_subdir] {
            process_pcap(pcap_info.filepath, output_dir, output_subdir);
        });
    }
    
    std::cout << "Processing complete!" << std::endl;
    
    return 0;
}
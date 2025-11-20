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

// DLT (Data Link Type) constants - some may not be in older pcap.h
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
    uint8_t protocol;      // TCP or UDP
    uint32_t ip1;          // First packet source IP (typically client)
    uint16_t port1;        // First packet source port
    uint32_t ip2;          // First packet destination IP (typically server)
    uint16_t port2;        // First packet destination port
    
    // Constructor: preserve original packet direction (no normalization)
    FlowKey(uint8_t proto, uint32_t src_ip, uint16_t src_port, 
            uint32_t dst_ip, uint16_t dst_port) 
        : protocol(proto), ip1(src_ip), port1(src_port), 
          ip2(dst_ip), port2(dst_port) {}
    
    bool operator==(const FlowKey& other) const {
        return protocol == other.protocol && 
               ip1 == other.ip1 && port1 == other.port1 &&
               ip2 == other.ip2 && port2 == other.port2;
    }
    
    // Get reverse direction key (for bidirectional flow matching)
    FlowKey reverse() const {
        return FlowKey(protocol, ip2, port2, ip1, port1);
    }
    
    // Convert to filename string (first packet direction)
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

// Hash function for FlowKey
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

// Structure to store packet data for later writing
struct PacketData {
    std::vector<uint8_t> data;
    struct pcap_pkthdr header;
    bool has_payload;  // Whether packet contains transport layer payload
    
    PacketData(const struct pcap_pkthdr* h, const uint8_t* packet, bool payload) {
        header = *h;
        data.assign(packet, packet + h->caplen);
        has_payload = payload;
    }
};

// Result of packet parsing
struct ParseResult {
    std::unique_ptr<FlowKey> flow_key;
    bool has_payload;  // True if packet has transport layer payload data
    
    ParseResult() : flow_key(nullptr), has_payload(false) {}
    ParseResult(std::unique_ptr<FlowKey> key, bool payload) 
        : flow_key(std::move(key)), has_payload(payload) {}
};

// Check if packet has transport layer payload (data beyond headers)
// Returns true if packet contains actual data (not just control packets like SYN/ACK/FIN)
bool has_transport_payload(const uint8_t* packet, int len, int linktype) {
    int offset = 0;
    
    // Determine offset based on link type
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
    
    // Check minimum IP header length
    if (len < offset + sizeof(struct ip)) return false;
    
    const struct ip* ip_hdr = (struct ip*)(packet + offset);
    if (ip_hdr->ip_v != 4) return false;
    
    uint8_t protocol = ip_hdr->ip_p;
    if (protocol != IPPROTO_TCP && protocol != IPPROTO_UDP) return false;
    
    // Get total IP packet length
    uint16_t ip_total_len = ntohs(ip_hdr->ip_len);
    int ip_header_len = ip_hdr->ip_hl * 4;
    
    int payload_len = 0;
    
    if (protocol == IPPROTO_TCP) {
        if (len < offset + ip_header_len + sizeof(struct tcphdr)) return false;
        const struct tcphdr* tcp = (struct tcphdr*)(packet + offset + ip_header_len);
        int tcp_header_len = tcp->th_off * 4;
        
        // Calculate TCP payload length
        payload_len = ip_total_len - ip_header_len - tcp_header_len;
    } else { // UDP
        if (len < offset + ip_header_len + sizeof(struct udphdr)) return false;
        const struct udphdr* udp = (struct udphdr*)(packet + offset + ip_header_len);
        
        // Calculate UDP payload length
        // UDP header is 8 bytes
        payload_len = ip_total_len - ip_header_len - 8;
    }
    
    // Return true only if there's actual payload data
    return payload_len > 0;
}

// Parse packet and extract flow key, return nullptr if not TCP/UDP IPv4 or if it's DNS
// Supports both Ethernet frames and Raw IP packets
std::unique_ptr<FlowKey> parse_packet(const uint8_t* packet, int len, int linktype) {
    const struct ip* ip_hdr = nullptr;
    int offset = 0;
    
    // Determine offset based on link type
    if (linktype == DLT_EN10MB) {
        // Ethernet frame
        if (len < sizeof(struct ether_header)) return nullptr;
        
        const struct ether_header* eth = (struct ether_header*)packet;
        
        // Check if it's IPv4
        if (ntohs(eth->ether_type) != ETHERTYPE_IP) return nullptr;
        
        offset = sizeof(struct ether_header);
    } else if (linktype == DLT_RAW || linktype == DLT_IPV4 || linktype == 101) {
        // Raw IP packet (no link layer header)
        // DLT_RAW = 12, DLT_IPV4 = 228, or Linux cooked capture (101) raw IP mode
        offset = 0;
    } else if (linktype == DLT_LINUX_SLL) {
        // Linux cooked capture
        if (len < 16) return nullptr;
        // SLL header is 16 bytes, check protocol type at offset 14-15
        uint16_t proto = (packet[14] << 8) | packet[15];
        if (proto != ETHERTYPE_IP) return nullptr;
        offset = 16;
    } else {
        // Unsupported link type
        return nullptr;
    }
    
    // Check minimum IP header length
    if (len < offset + sizeof(struct ip)) return nullptr;
    
    ip_hdr = (struct ip*)(packet + offset);
    
    // Verify IP version is 4
    if (ip_hdr->ip_v != 4) return nullptr;
    
    uint8_t protocol = ip_hdr->ip_p;
    
    // Only process TCP and UDP
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
    } else { // UDP
        if (len < offset + ip_header_len + sizeof(struct udphdr)) 
            return nullptr;
        const struct udphdr* udp = (struct udphdr*)transport_hdr;
        src_port = ntohs(udp->uh_sport);
        dst_port = ntohs(udp->uh_dport);
        
        // Filter out DNS packets (port 53)
        if (src_port == 53 || dst_port == 53) {
            return nullptr;
        }
    }
    
    return std::make_unique<FlowKey>(protocol, src_ip, src_port, dst_ip, dst_port);
}

// Process a single pcap file
void process_pcap(const std::string& input_pcap, const std::string& output_dir) {
    std::cout << "Processing: " << input_pcap << std::endl;
    
    char errbuf[PCAP_ERRBUF_SIZE];
    
    // Open pcap to get link type first
    pcap_t* handle = pcap_open_offline(input_pcap.c_str(), errbuf);
    if (!handle) {
        std::cerr << "Error opening pcap file: " << input_pcap << " - " << errbuf << std::endl;
        return;
    }
    
    int linktype = pcap_datalink(handle);
    int snapshot = pcap_snapshot(handle);
    
    std::unordered_map<FlowKey, int> flow_counts;  // Count only packets with payload
    std::unordered_map<FlowKey, std::vector<PacketData>> flow_packets;  // Store all packets
    
    struct pcap_pkthdr* header;
    const uint8_t* packet;
    int result;
    
    // First pass: collect all packets, but count only those with payload
    // Check both forward and reverse directions for bidirectional flow aggregation
    while ((result = pcap_next_ex(handle, &header, &packet)) >= 0) {
        if (result == 0) continue; // Timeout
        
        auto flow_key = parse_packet(packet, header->caplen, linktype);
        if (flow_key) {
            // Check if packet has transport layer payload
            bool has_payload = has_transport_payload(packet, header->caplen, linktype);
            
            // Check if this flow key already exists
            auto it = flow_counts.find(*flow_key);
            if (it != flow_counts.end()) {
                // Forward direction exists, use it
                if (has_payload) {
                    flow_counts[*flow_key]++;  // Only count if has payload
                }
                flow_packets[*flow_key].emplace_back(header, packet);  // Save all packets
            } else {
                // Check if reverse direction exists
                FlowKey reverse_key = flow_key->reverse();
                auto reverse_it = flow_counts.find(reverse_key);
                if (reverse_it != flow_counts.end()) {
                    // Reverse direction exists, aggregate into it
                    if (has_payload) {
                        flow_counts[reverse_key]++;  // Only count if has payload
                    }
                    flow_packets[reverse_key].emplace_back(header, packet);  // Save all packets
                } else {
                    // New flow (first packet determines direction)
                    if (has_payload) {
                        flow_counts[*flow_key] = 1;  // Initialize count
                    } else {
                        flow_counts[*flow_key] = 0;  // Initialize with 0 for control packet
                    }
                    flow_packets[*flow_key].emplace_back(header, packet);  // Save all packets
                }
            }
        }
    }
    
    pcap_close(handle);
    
    // Filter flows with packet count >= 5
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
    
    // Create output directory for this pcap
    std::string pcap_name = fs::path(input_pcap).stem().string();
    std::string pcap_output_dir = output_dir + "/" + pcap_name;
    
    try {
        fs::create_directories(pcap_output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error creating directory: " << pcap_output_dir << " - " << e.what() << std::endl;
        return;
    }
    
    // Write each valid flow to separate file
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
        
        // Write all packets for this flow
        for (const auto& pkt : packets) {
            pcap_dump((u_char*)dumper, &pkt.header, pkt.data.data());
        }
        
        pcap_dump_close(dumper);
        pcap_close(out_handle);
    }
    
    std::cout << "  Completed: " << pcap_name << " (" << valid_flows.size() << " flows)" << std::endl;
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

// Recursively find all pcap files
void find_pcap_files(const std::string& dir, std::vector<std::string>& pcap_files) {
    try {
        for (const auto& entry : fs::recursive_directory_iterator(dir)) {
            if (entry.is_regular_file()) {
                std::string ext = entry.path().extension().string();
                // Convert to lowercase for comparison
                std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                if (ext == ".pcap" || ext == ".pcapng") {
                    pcap_files.push_back(entry.path().string());
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error scanning directory: " << dir << " - " << e.what() << std::endl;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_pcap_dir> <output_pcap_dir>" << std::endl;
        std::cerr << "  input_pcap_dir: Directory containing pcap files (will be recursively scanned)" << std::endl;
        std::cerr << "  output_pcap_dir: Output directory for filtered flows" << std::endl;
        return 1;
    }
    
    std::string input_dir = argv[1];
    std::string output_dir = argv[2];
    
    // Check if input directory exists
    if (!fs::exists(input_dir) || !fs::is_directory(input_dir)) {
        std::cerr << "Error: Input directory does not exist: " << input_dir << std::endl;
        return 1;
    }
    
    // Create output directory
    try {
        fs::create_directories(output_dir);
    } catch (const std::exception& e) {
        std::cerr << "Error creating output directory: " << output_dir << " - " << e.what() << std::endl;
        return 1;
    }
    
    // Find all pcap files recursively
    std::vector<std::string> pcap_files;
    std::cout << "Scanning for pcap files in: " << input_dir << std::endl;
    find_pcap_files(input_dir, pcap_files);
    
    std::cout << "Found " << pcap_files.size() << " pcap files" << std::endl;
    
    if (pcap_files.empty()) {
        std::cout << "No pcap files found. Exiting." << std::endl;
        return 0;
    }
    
    // Determine number of threads (use hardware concurrency)
    size_t num_threads = std::thread::hardware_concurrency();
    if (num_threads == 0) num_threads = 4; // Fallback
    
    std::cout << "Using " << num_threads << " threads for parallel processing" << std::endl;
    
    // Create thread pool and process files
    ThreadPool pool(num_threads);
    
    for (const auto& pcap_file : pcap_files) {
        pool.enqueue([pcap_file, output_dir] {
            process_pcap(pcap_file, output_dir);
        });
    }
    
    // Wait for all tasks to complete (ThreadPool destructor will join all threads)
    std::cout << "Processing complete!" << std::endl;
    
    return 0;
}
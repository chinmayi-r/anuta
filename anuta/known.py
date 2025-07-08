from bidict import bidict


known_ports = [
    20,   # FTP Data Transfer
    21,   # FTP Control
    22,   # SSH
    23,   # Telnet
    25,   # SMTP
    53,   # DNS
    67,   # DHCP Server
    68,   # DHCP Client
    69,   # TFTP
    80,   # HTTP
    110,  # POP3
    143,  # IMAP
    443,  # HTTPS
    514,  # Syslog
    993,  # IMAPS
    995,  # POP3S
    8080 # Alternative HTTP
]
UNINTERESTED_PORT = 60_000  # Default value for unknown ports

#************************* Domain Knowledge begins *************************
metadc_ints = ['IngressBytesAgg','EgressBytesAgg','InRxmitBytesAgg','OutRxmitBytesAgg','InCongestionBytesAgg','ConnectionsAgg','IngressBytes0','IngressBytes1','IngressBytes2','IngressBytes3','IngressBytes4','IngressBytes5','IngressBytes6','IngressBytes7','IngressBytes8','IngressBytes9','IngressBytes10','IngressBytes11','IngressBytes12','IngressBytes13','IngressBytes14','IngressBytes15','IngressBytes16','IngressBytes17','IngressBytes18','IngressBytes19','IngressBytes20','IngressBytes21','IngressBytes22','IngressBytes23','IngressBytes24','IngressBytes25','IngressBytes26','IngressBytes27','IngressBytes28','IngressBytes29','IngressBytes30','IngressBytes31','IngressBytes32','IngressBytes33','IngressBytes34','IngressBytes35','IngressBytes36','IngressBytes37','IngressBytes38','IngressBytes39','IngressBytes40','IngressBytes41','IngressBytes42','IngressBytes43','IngressBytes44','IngressBytes45','IngressBytes46','IngressBytes47','IngressBytes48','IngressBytes49']

#************************ Yatesbury ********************
yatesbury_categoricals = ["SrcIp", "DstIp", "SrcPt", "DstPt", 
                        "Proto", "FlowDir", "Decision", "FlowState", "Label"]
yatesbury_numericals = ["PktSent", "BytesSent", "PktRecv", "BytesRecv"]
#* Gurantee ordering (https://github.com/microsoft/Yatesbury?tab=readme-ov-file#dataset-description)
yatesbury_direction_conversion = bidict({direction: i for i, direction in enumerate(['I', 'O'])})
yatesbury_proto_conversion = bidict({proto: i for i, proto in enumerate(['T', 'U'])})
yatesbury_decision_conversion = bidict({decision: i for i, decision in enumerate(['A', 'D'])})
yatesbury_flowstate_conversion = bidict({state: i for i, state in enumerate(['B', 'C', 'E'])})
#* Taking into account every port for now.
yatesbury_port_conversion = bidict({port: port for port in range(1, 65536)})

def yatesbury_direction_map(direction: str):
    if direction in yatesbury_direction_conversion:
        return yatesbury_direction_conversion[direction]
    else:
        raise ValueError(f"Unknown flow direction: {direction}")
def yatesbury_proto_map(proto: str):
    if proto in yatesbury_proto_conversion:
        return yatesbury_proto_conversion[proto]
    else:
        raise ValueError(f"Unknown protocol: {proto}")
def yatesbury_decision_map(decision: str):
    if decision in yatesbury_decision_conversion:
        return yatesbury_decision_conversion[decision]
    else:
        raise ValueError(f"Unknown decision: {decision}")
def yatesbury_flowstate_map(state: str):
    if state in yatesbury_flowstate_conversion:
        return yatesbury_flowstate_conversion[state]
    else:
        raise ValueError(f"Unknown flow state: {state}")


def yatesbury_ip_map(ip: str):
    """
    Map IP addresses to integers for Yatesbury dataset.
    """
    assert ip.startswith('172.0.0.'), "IP address must start with '172.0.0.'"
    return int(ip.split('.')[-1])

yatesbury_ips = ['172.0.0.4', '172.0.0.5', '172.0.0.6', '172.0.0.7', '172.0.0.8', '172.0.0.9', '172.0.0.10', '172.0.0.11', '172.0.0.12', '172.0.0.13', '172.0.0.14', '172.0.0.15', '172.0.0.16', '172.0.0.17', '172.0.0.18', '172.0.0.19', '172.0.0.20',]
#******************** Netflix data Domain Knowledge begins ********************
netflix_flags = ['SYN', 'ACK-SYN', 'ACK', 'ACK-PSH', 'ACK-FIN']
netflix_seqnum_increaments = [1]
netflix_tcplen_limits = [0]

def netflix_ip_map(ip: str):
    if ip in ["192.168.43.72", "75.202.11.223", 
              3232246600, 2259302873, 3232246600, 3500680008, 970557286, 1176926361]:
        #^ Client IPs
        return 0
    elif ip in ["198.38.120.153", "28.111.112.209", 
                3324410009, 3381504333, 1401727068, 3325491353, 1062720695, 3799415797]:
        #^ Server IPs
        return 1
    else:
        raise ValueError(f"Unknown IP address for Netflix dataset: {ip}")

def netflix_port_map(port: int):
    if port in [49977, 22379]:
        #^ Client ports
        return 0
    elif port in [443, 40059]:
        #^ Server ports
        return 1
    elif port >= 49152:
        #^ Application ports
        return 0
    else:
        return 1

def netflix_flags_map(flags: str):
    if flags in netflix_flags:
        return netflix_flags.index(flags)
    else:
        return -1

def netflix_proto_map(proto: str):
    if proto == 'TCP':
        return 0
    else:
        return 1

#******************** Netflix data Domain Knowledge ends ********************

popular_ports = [
    # TCP Ports
    20, 21, 22, 23, 25, 53, 80, 110, 143, 443,
    # UDP Ports
    53, 67, 68, 69, 123, 161,
    # ICMP Types
    0, 3, 5, 8, 11,
    # IGMP Types
    0x11, 0x16, 0x17, 0x22
]

#******************** CIDDS-001 Domain Knowledge begins ********************
#? Should port be a categorical variable? Sometimes we need range values (i.e., application and dynamic ports).
cidds_categoricals = ['Flags', 'Proto', 'SrcIpAddr', 'DstIpAddr'] + ['SrcPt', 'DstPt']
cidds_numericals = ['Packets', 'Bytes', 'Duration']
cidds_ips = ['private_p2p', 'private_broadcast', 'any', 'public_p2p', 'dns']
cidds_ports = [0, 3, 8, 11, 22, 25, 
            #    23, #* Telnet
            #    8000, #* Seafile Server
               53, 67, 68, 80, 123, 137, 138, 443, 993, 8080]
cidds_ints = ['Packets', 'Bytes', 'Flows'] + cidds_categoricals
cidds_floats = ['Duration']

#* Map strings to integers
cidds_ip_conversion = bidict({ip: i for i, ip in enumerate(cidds_ips)})
cidds_flags_conversion = bidict({flag: i for i, flag in enumerate(['noflags', 'hasflags'])})
#TODO: Change the mapping to standard NetFlow codes: 
# https://www.iana.org/assignments/protocol-numbers/protocol-numbers.xhtml
cidds_proto_conversion = bidict({proto: i for i, proto in enumerate(['TCP', 'UDP', 'ICMP', 'IGMP'])})
cidds_port_conversion = bidict({port: port for port in cidds_ports + known_ports})
cidds_port_conversion.inverse[UNINTERESTED_PORT] = 'uninterested'
cidds_conversions = {
    'ip': cidds_ip_conversion,
    'flags': cidds_flags_conversion,
    'proto': cidds_proto_conversion
}
cidds_constants = {
	'ip': list(cidds_conversions['ip'].values()),
	'port': cidds_ports,
	'packet': [42, 64, 65_535], #* MTU
    'bytes': [1],
}

def cidds_port_map(port):
    if port not in set(cidds_ports) | set(known_ports):
        return UNINTERESTED_PORT
    else:
        return port
    
def cidds_proto_map(proto: str):
	return cidds_proto_conversion[proto]

def cidds_ip_map(ip: str):
    if isinstance(ip, int):
        return ip
    new_ip = ''
    if ip.startswith('192.168.'):
        new_ip += 'private_'
    else:        
        new_ip += 'public_'
    if '.255' in ip:
        new_ip += 'broadcast'
    else:
        new_ip += 'p2p'
    
    if ip == '0.0.0.0':
        new_ip = 'any'
    elif ip == '255.255.255.255':
        new_ip = 'private_broadcast'
    elif ip == 'DNS':
        new_ip = 'dns'
    
    return cidds_ip_conversion[new_ip]

def cidds_flag_map(flag: str):
	#! Don't consider the specific 'Flags' values for now
    new_flag = ''
    if flag == '......':
        new_flag = cidds_flags_conversion.inverse[0]
    else:
        new_flag = cidds_flags_conversion.inverse[1]
    
    return cidds_flags_conversion[new_flag]
#******************** CIDDS-001 Domain Knowledge ends ********************

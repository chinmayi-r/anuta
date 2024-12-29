from bidict import bidict


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

#******************** CIDDS-001 Domain Knowledge ********************
#? Should port be a categorical variable? Sometimes we need range values (i.e., application and dynamic ports).
cidds_categorical = ['Flags', 'Proto', 'SrcIpAddr', 'DstIpAddr'] + ['SrcPt', 'DstPt']
cidds_numerical = ['Packets', 'Bytes', 'Flows', 'Duration']
cidds_ips = ['private_p2p', 'private_broadcast', 'private_any', 'public_p2p', 'dns']
cidds_ports = [0, 3, 8, 11, 22, 25, 
            #    23, #* Telnet
            #    8000, #* Seafile Server
               53, 67, 68, 80, 123, 137, 138, 443, 8080]

#* Map strings to integers
cidds_ip_conversion = bidict({ip: i for i, ip in enumerate(cidds_ips)})
cidds_flags_conversion = bidict({flag: i for i, flag in enumerate(['noflags', 'flags'])})
cidds_proto_conversion = bidict({proto: i for i, proto in enumerate(['TCP', 'UDP', 'ICMP', 'IGMP'])})
cidds_conversions = {
    'ip': cidds_ip_conversion,
    'flags': cidds_flags_conversion,
    'proto': cidds_proto_conversion
}
cidds_constants = {
	'ip': list(cidds_conversions['ip'].values()),
	'port': cidds_ports,
	'packet': [42, 64, 65_535], #* MTU
}

def proto_map(proto: str):
	return cidds_proto_conversion[proto]

def ip_map(ip: str):
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
        new_ip = 'private_any'
    elif ip == '255.255.255.255':
        new_ip = 'private_broadcast'
    elif ip == 'DNS':
        new_ip = 'dns'
    
    return cidds_ip_conversion[new_ip]

def flag_map(flag: str):
	#! Don't consider the specific 'Flags' values for now
    new_flag = ''
    if flag == '......':
        new_flag = 'noflags'
    else:
        new_flag = 'flags'
    
    return cidds_flags_conversion[new_flag]
#******************** CIDDS-001 Domain Knowledge ********************

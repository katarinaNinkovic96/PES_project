import cv2
# import io
import socket
import struct
import time
import pickle
import zlib
import argparse
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='set the IP address.')
parser.add_argument('--IP', type=str, help='set the IP address of the rPi (server device)')
parser.add_argument('--mcast', type=str2bool, help='run on multicast mode', nargs='?', const=True, default=False)

args = parser.parse_args()

port = 21000

bMulticast = args.mcast 
if(False == bMulticast):
    host = str(args.IP) 
else:
    MCAST_GRP = "224.0.0.251"
    host = MCAST_GRP
    IS_ALL_GROUPS = True

if(False == bMulticast):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port) )
else:
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
    client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    if IS_ALL_GROUPS:
        # on this port, receives ALL multicast groups
        client_socket.bind(('', port))
    else:
        # on this port, listen ONLY to MCAST_GRP
        client_socket.bind((MCAST_GRP, port))
    mreq = struct.pack("4sl", socket.inet_aton(MCAST_GRP), socket.INADDR_ANY)

    client_socket.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)

client_socket.setblocking(0)

message = b""

MAX_PAYLOAD = 65535 #this is MAX possible payload for TCP or UDP packets

while True:
    try:
        try:
            message = client_socket.recv(MAX_PAYLOAD)
        except BlockingIOError:
            time.sleep(0.05)    # 50 milisec delay
            continue
    except KeyboardInterrupt:
        break

    frame=cv2.imdecode(np.ndarray(shape=(1, len(message)), dtype=np.uint8, buffer=message),cv2.IMREAD_COLOR)
    cv2.imshow('ImageWindow',frame)
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break

client_socket.close()
cv2.destroyAllWindows()

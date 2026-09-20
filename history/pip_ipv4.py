"""Use IPv4 for this pip process only; does not change host networking."""
import runpy, socket
original = socket.getaddrinfo
def ipv4(host, port, family=0, type=0, proto=0, flags=0):
    return original(host, port, socket.AF_INET, type, proto, flags)
socket.getaddrinfo = ipv4
runpy.run_module('pip', run_name='__main__')

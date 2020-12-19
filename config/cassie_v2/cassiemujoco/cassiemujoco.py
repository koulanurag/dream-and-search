# Copyright (c) 2018 Dynamic Robotics Laboratory
#
# Permission to use, copy, modify, and distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from .cassiemujoco_ctypes import *
import os
import ctypes
import numpy as np

# Get base directory
_dir_path = os.path.dirname(os.path.realpath(__file__))

# Initialize libcassiesim
cassie_mujoco_init(str.encode(_dir_path+"/cassie.xml"))

# Interface classes
class CassieSim:
    def __init__(self, modelfile):
        self.c = cassie_sim_init(modelfile.encode('utf-8'))
        self.nv = cassie_sim_nv(self.c)
        self.nbody = cassie_sim_nbody(self.c)
        self.nq = cassie_sim_nq(self.c)
        self.ngeom = cassie_sim_ngeom(self.c)

    def step(self, u):
        y = cassie_out_t()
        cassie_sim_step(self.c, y, u)
        return y

    def step_pd(self, u):
        y = state_out_t()
        cassie_sim_step_pd(self.c, y, u)
        return y

    def get_state(self):
        s = CassieState()
        cassie_get_state(self.c, s.s)
        return s

    def set_state(self, s):
        cassie_set_state(self.c, s.s)

    def time(self):
        timep = cassie_sim_time(self.c)
        return timep[0]

    def qpos(self):
        qposp = cassie_sim_qpos(self.c)
        return qposp[:self.nq]

    def qvel(self):
        qvelp = cassie_sim_qvel(self.c)
        return qvelp[:self.nv]

    def set_time(self, time):
        timep = cassie_sim_time(self.c)
        timep[0] = time

    def set_qpos(self, qpos):
        qposp = cassie_sim_qpos(self.c)
        for i in range(min(len(qpos), self.nq)):
            qposp[i] = qpos[i]

    def set_qvel(self, qvel):
        qvelp = cassie_sim_qvel(self.c)
        for i in range(min(len(qvel), self.nv)):
            qvelp[i] = qvel[i]

    def hold(self):
        cassie_sim_hold(self.c)

    def release(self):
        cassie_sim_release(self.c)

    def apply_force(self, xfrc, body=1):
        xfrc_array = (ctypes.c_double * 6)()
        for i in range(len(xfrc)):
            xfrc_array[i] = xfrc[i]
        cassie_sim_apply_force(self.c, xfrc_array, body)

    def foot_pos(self):
      pos = []
      pos_array = (ctypes.c_double * 6)()
      cassie_sim_foot_positions(self.c, pos_array)
      for i in range(6):
          pos.append(pos_array[i])
      return pos

    def clear_forces(self):
        cassie_sim_clear_forces(self.c)

    """
    def get_foot_forces(self):
        y = state_out_t()
        force = np.zeros(12)
        self.foot_force(force)
        return force[[2, 8]]
  """

    def get_foot_force(self):
        force = np.zeros(12)
        frc_array = (ctypes.c_double * 12)()
        cassie_sim_foot_forces(self.c, frc_array)
        for i in range(12):
            force[i] = frc_array[i]
        return force

    def get_dof_damping(self):
        ptr = cassie_sim_dof_damping(self.c)
        ret = np.zeros(self.nv)
        for i in range(self.nv):
          ret[i] = ptr[i]
        return ret
    
    def get_body_mass(self):
        ptr = cassie_sim_body_mass(self.c)
        ret = np.zeros(self.nbody)
        for i in range(self.nbody):
          ret[i] = ptr[i]
        return ret

    def get_body_ipos(self):
        nbody = self.nbody * 3
        ptr = cassie_sim_body_ipos(self.c)
        ret = np.zeros(nbody)
        for i in range(nbody):
          ret[i] = ptr[i]
        return ret

    def get_geom_friction(self):
        ptr = cassie_sim_geom_friction(self.c)
        ret = np.zeros(self.ngeom * 3)
        for i in range(self.ngeom * 3):
          ret[i] = ptr[i]
        return ret

    def get_geom_rgba(self):
        ptr = cassie_sim_geom_rgba(self.c)
        ret = np.zeros(self.ngeom * 4)
        for i in range(self.ngeom * 4):
          ret[i] = ptr[i]
        return ret

    def get_geom_quat(self):
        ptr = cassie_sim_geom_quat(self.c)
        ret = np.zeros(self.ngeom * 4)
        for i in range(self.ngeom * 4):
          ret[i] = ptr[i]
        return ret

    def set_dof_damping(self, data):
        c_arr = (ctypes.c_double * self.nv)()

        if len(data) != self.nv:
          print("SIZE MISMATCH SET_DOF_DAMPING()")
          exit(1)
        
        for i in range(self.nv):
          c_arr[i] = data[i]

        cassie_sim_set_dof_damping(self.c, c_arr)

    def set_body_mass(self, data):
        c_arr = (ctypes.c_double * self.nbody)()

        if len(data) != self.nbody:
          print("SIZE MISMATCH SET_BODY_MASS()")
          exit(1)
        
        for i in range(self.nbody):
          c_arr[i] = data[i]

        cassie_sim_set_body_mass(self.c, c_arr)

    def set_body_ipos(self, data):
        nbody = self.nbody * 3
        c_arr = (ctypes.c_double * nbody)()

        if len(data) != nbody:
          print("SIZE MISMATCH SET_BODY_IPOS()")
          exit(1)
        
        for i in range(nbody):
          c_arr[i] = data[i]

        cassie_sim_set_body_ipos(self.c, c_arr)

    def set_geom_friction(self, data):
        c_arr = (ctypes.c_double * (self.ngeom*3))()

        if len(data) != self.ngeom * 3:
           print("SIZE MISMATCH SET_GEOM_FRICTION()")
           exit(1)

        for i in range(self.ngeom*3):
          c_arr[i] = data[i]

        cassie_sim_set_geom_friction(self.c, c_arr)

    def set_geom_rgba(self, data):
        ngeom = self.ngeom * 4

        if len(data) != ngeom:
           print("SIZE MISMATCH SET_GEOM_RGBA()")
           exit(1)

        c_arr = (ctypes.c_float * ngeom)()

        for i in range(ngeom):
          c_arr[i] = data[i]

        cassie_sim_set_geom_rgba(self.c, c_arr)
    
    def set_geom_quat(self, data):
        ngeom = self.ngeom * 4

        if len(data) != ngeom:
           print("SIZE MISMATCH SET_GEOM_QUAT()")
           exit(1)

        c_arr = (ctypes.c_double * ngeom)()
        #print("SETTING:")
        #print(c_arr, data)

        for i in range(ngeom):
          c_arr[i] = data[i]

        cassie_sim_set_geom_quat(self.c, c_arr)

    def set_const(self):
        cassie_sim_set_const(self.c)

    def __del__(self):
        cassie_sim_free(self.c)

class CassieVis:
    def __init__(self, c, modelfile):
        self.v = cassie_vis_init(c.c, modelfile.encode('utf-8'))

    def draw(self, c):
        state = cassie_vis_draw(self.v, c.c)
        # print("vis draw state:", state)
        return state

    def valid(self):
        return cassie_vis_valid(self.v)

    def ispaused(self):
        return cassie_vis_paused(self.v)

    def __del__(self):
        cassie_vis_free(self.v)

class CassieState:
    def __init__(self):
        self.s = cassie_state_alloc()

    def time(self):
        timep = cassie_state_time(self.s)
        return timep[0]

    def qpos(self):
        qposp = cassie_state_qpos(self.s)
        return qposp[:35]

    def qvel(self):
        qvelp = cassie_state_qvel(self.s)
        return qvelp[:32]

    def set_time(self, time):
        timep = cassie_state_time(self.s)
        timep[0] = time

    def set_qpos(self, qpos):
        qposp = cassie_state_qpos(self.s)
        for i in range(min(len(qpos), 35)):
            qposp[i] = qpos[i]

    def set_qvel(self, qvel):
        qvelp = cassie_state_qvel(self.s)
        for i in range(min(len(qvel), 32)):
            qvelp[i] = qvel[i]

    def __del__(self):
        cassie_state_free(self.s)

class CassieUdp:
    def __init__(self, remote_addr='127.0.0.1', remote_port='25000',
                 local_addr='0.0.0.0', local_port='25001'):
        self.sock = udp_init_client(str.encode(remote_addr),
                                    str.encode(remote_port),
                                    str.encode(local_addr),
                                    str.encode(local_port))
        self.packet_header_info = packet_header_info_t()
        self.recvlen = 2 + 697
        self.sendlen = 2 + 58
        self.recvlen_pd = 2 + 493
        self.sendlen_pd = 2 + 476
        self.recvbuf = (ctypes.c_ubyte * max(self.recvlen, self.recvlen_pd))()
        self.sendbuf = (ctypes.c_ubyte * max(self.sendlen, self.sendlen_pd))()
        self.inbuf = ctypes.cast(ctypes.byref(self.recvbuf, 2),
                                 ctypes.POINTER(ctypes.c_ubyte))
        self.outbuf = ctypes.cast(ctypes.byref(self.sendbuf, 2),
                                  ctypes.POINTER(ctypes.c_ubyte))

    def send(self, u):
        pack_cassie_user_in_t(u, self.outbuf)
        send_packet(self.sock, self.sendbuf, self.sendlen, None, 0)

    def send_pd(self, u):
        pack_pd_in_t(u, self.outbuf)
        send_packet(self.sock, self.sendbuf, self.sendlen_pd, None, 0)

    def recv_wait(self):
        nbytes = -1
        while nbytes != self.recvlen:
            nbytes = get_newest_packet(self.sock, self.recvbuf, self.recvlen,
                                       None, None)
        process_packet_header(self.packet_header_info,
                              self.recvbuf, self.sendbuf)
        cassie_out = cassie_out_t()
        unpack_cassie_out_t(self.inbuf, cassie_out)
        return cassie_out

    def recv_wait_pd(self):
        nbytes = -1
        while nbytes != self.recvlen_pd:
            nbytes = get_newest_packet(self.sock, self.recvbuf, self.recvlen_pd,
                                       None, None)
        process_packet_header(self.packet_header_info,
                              self.recvbuf, self.sendbuf)
        state_out = state_out_t()
        unpack_state_out_t(self.inbuf, state_out)
        return state_out

    def recv_newest(self):
        nbytes = get_newest_packet(self.sock, self.recvbuf, self.recvlen,
                                   None, None)
        if nbytes != self.recvlen:
            return None
        process_packet_header(self.packet_header_info,
                              self.recvbuf, self.sendbuf)
        cassie_out = cassie_out_t()
        unpack_cassie_out_t(self.inbuf, cassie_out)
        return cassie_out

    def recv_newest_pd(self):
        nbytes = get_newest_packet(self.sock, self.recvbuf, self.recvlen_pd,
                                   None, None)
        if nbytes != self.recvlen_pd:
            return None
        process_packet_header(self.packet_header_info,
                              self.recvbuf, self.sendbuf)
        state_out = state_out_t()
        unpack_state_out_t(self.inbuf, state_out)
        return state_out

    def delay(self):
        return ord(self.packet_header_info.delay)

    def seq_num_in_diff(self):
        return ord(self.packet_header_info.seq_num_in_diff)

    def __del__(self):
        udp_close(self.sock)

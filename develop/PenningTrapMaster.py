import uci.BorisUpdater as BorisUpdater
import uci.CoulombAcc as CoulombAcc
import uci.Ptcls as Ptcls
import uci.TrapAcc as TrapAcc
import uci.TrapConfiguration as TrapConfiguration
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys
import os



sys.path.append(r'C:\Users\sbt\Desktop\calc\simulation\crystal_mode_code')
sys.path.append(r'C:\Users\sbt\Desktop\calc\simulation\itano_cooling_code')

import mode_analysis_code as mc
import Wayne_Itano_Code as ic


class UsageError(Exception):
    def __init__(self, msg):
        self.msg = msg


fund_charge = 1.602176565e-19
atomic_unit = 1.66053892e-27
ion_mass = 8.96 * atomic_unit


class penning_trap_master:
    """
    Contains methods to manage the Itano, Mode, and Dynamics codes all at once, and pass information between the three.

    """
    def __init__(self, writeto=None):

        if writeto is None:
            self.writeto = os.getcwd()

        self.itano = None
        self.crystal = None

        self.trap_config = None
        self.ptcls = None
        self.n_ions = 0
        self.accelerations = None

        self.mode_code_loaded = False
        self.itano_code_loaded = False
        self.dynamics_loaded = False

    def update_loads(self):
        if self.crystal is not None:
            self.mode_code_loaded = True

        if self.itano is not None:
            self.itano_code_loaded = True

        if self. ptcls is not None and self.trap_config is not None and self:
            self.dynamics_loaded = True

    def run_everything(self):

        pass # make this do everything

    def setup_crystal_and_trap(self, n_ions=19, Vtrap=(0.0, -1750.0, -2000.0), B=4.4588, frot=180., Vwall=5.):
        crystal = mc.ModeAnalysis(N=n_ions, Vtrap=Vtrap, frot=frot, B=B, Vwall=Vwall)
        crystal.run()
        trap_config = TrapConfiguration.TrapConfiguration()
        trap_config.Bz = B
        trap_config.kz = 2 * crystal.V0
        delta = -crystal.Cw
        trap_config.kx = -(0.5 + delta) * trap_config.kz
        trap_config.ky = -(0.5 - delta) * trap_config.kz
        trap_config.omega = crystal.wrot

        self.trap_config = trap_config

        # initialize particles
        ptcls = Ptcls.Ptcls()
        ptcls.set_nptcls(n_ions)
        x = crystal.uE[:n_ions].astype(np.float32)
        x = x.reshape(1, crystal.Nion)
        y = crystal.uE[n_ions:].astype(np.float32)
        y = y.reshape(1, crystal.Nion)
        ptcls.ptclList[0] = x
        ptcls.ptclList[1] = y
        ptcls.ptclList[2:6, :n_ions] = 0
        ptcls.ptclList[6, :n_ions] = fund_charge
        ptcls.ptclList[7, :n_ions] = ion_mass

        self.ptcls = ptcls

        self.update_loads()

    def run_dynamics_simulation(self,dt=5e-10, tmax=1e-6, num_dump=1000):

        ctx = cl.create_some_context(interactive=True)
        queue = cl.CommandQueue(ctx)
        coulomb_acc = CoulombAcc.CoulombAcc(ctx, queue)
        accelerations = [coulomb_acc]
        trap_acc = TrapAcc.TrapAcc(ctx, queue)
        trap_acc.trapConfiguration = self.trap_config
        accelerations.append(trap_acc)

        updater = BorisUpdater.BorisUpdater(ctx, queue)

        xd = cl_array.to_device(queue, self.ptcls.x())
        yd = cl_array.to_device(queue, self.ptcls.y())
        zd = cl_array.to_device(queue, self.ptcls.z())
        vxd = cl_array.to_device(queue, self.ptcls.vx())
        vyd = cl_array.to_device(queue, self.ptcls.vy())
        vzd = cl_array.to_device(queue, self.ptcls.vz())
        qd = cl_array.to_device(queue, self.ptcls.q())
        md = cl_array.to_device(queue, self.ptcls.m())

        t = 0.0
        i = 0
        while t < tmax:
            np.save('ptcls' + str(i) + '.npy', self.ptcls.ptclList[0:3, :])
            t = updater.update(
                xd, yd, zd, vxd, vyd, vzd, qd, md, accelerations, t, dt,
                num_dump)
            xd.get(queue, self.ptcls.x())
            yd.get(queue, self.ptcls.y())
            zd.get(queue, self.ptcls.z())
            i += 1

        np.savetxt('ptcls' + str(num_dump) + '.npy', self.ptcls.ptclList[0:3, :])

    def setup_itano(self,):

        self.itano = ic.ItanoAnalysis(defaultoff=30.0E-6, defaultdet=-500E6, wr=2 * np.pi * 45.0E3, Tguess=1E-3,
                                     saturation=.5, dens=2.77E9, ywidth=2.0E-6, radius=225.0E-6, quiet=True,
                                     spar=.2, wpar=2E6)

    def set_output_folder(self,dir):
        self.writeto = dir


def main(argv=None):
    try:
        a = penning_trap_master()
        a.setup_crystal_and_trap()
        a.run_dynamics_simulation()

    except UsageError as err:
        print(sys.stderr, err.msg)
        print(sys.stderr, 'For help use --help')
        return 2


if __name__ == '__main__':
    sys.exit(main())

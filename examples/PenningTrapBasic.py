import uci.BorisUpdater as BorisUpdater
import uci.CoulombAcc as CoulombAcc
import uci.Ptcls as Ptcls
import uci.TrapAcc as TrapAcc
import uci.TrapConfiguration as TrapConfiguration
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys

sys.path.append(r'C:\Users\sbt\Desktop\calc\simulation\crystal_mode_code')
import mode_analysis_code as mc

class UsageError(Exception):
    def __init__(self, msg):
        self.msg = msg

fund_charge = 1.602176565e-19
atomic_unit = 1.66053892e-27
ion_mass = 8.96 * atomic_unit


def run_simulation(n_ions=19, Vtrap=[0.0, -1750.0, -2000.0], B=4.4588, frot=180., Vwall=5., dt=5e-10, tmax=1e-6, num_dump = 1):

    crystal = mc.ModeAnalysis(N=n_ions, Vtrap=Vtrap, frot=frot, B=B, Vwall=Vwall)
    crystal.run()
    trap_config = TrapConfiguration.TrapConfiguration()
    trap_config.Bz = B
    trap_config.kz = 2*crystal.V0
    delta = -crystal.Cw
    trap_config.kx = -(0.5 + delta)*trap_config.kz
    trap_config.ky = -(0.5 - delta)*trap_config.kz
    trap_config.omega = crystal.wrot

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

    ctx = cl.create_some_context(interactive = True)
    queue = cl.CommandQueue(ctx)
    coulomb_acc = CoulombAcc.CoulombAcc(ctx, queue)
    accelerations = [coulomb_acc]
    trap_acc=TrapAcc.TrapAcc(ctx,queue)
    trap_acc.trapConfiguration = trap_config
    accelerations.append(trap_acc)

    updater = BorisUpdater.BorisUpdater(ctx, queue)

    xd = cl_array.to_device(queue, ptcls.x())
    yd = cl_array.to_device(queue, ptcls.y())
    zd = cl_array.to_device(queue, ptcls.z())
    vxd = cl_array.to_device(queue, ptcls.vx())
    vyd = cl_array.to_device(queue, ptcls.vy())
    vzd = cl_array.to_device(queue, ptcls.vz())
    qd = cl_array.to_device(queue, ptcls.q())
    md = cl_array.to_device(queue, ptcls.m())

    t = 0.0
    i = 0
    while t < tmax:
        np.save('ptcls' + str(i) + '.npy',ptcls.ptclList[0:3,:])
        t = updater.update(
                xd, yd, zd, vxd, vyd, vzd, qd, md, accelerations, t, dt,
                num_dump)
        xd.get(queue, ptcls.x())
        yd.get(queue, ptcls.y())
        zd.get(queue, ptcls.z())
        i += 1

    np.savetxt('ptcls' + str(num_dump) + '.npy',ptcls.ptclList[0:3,:])

def main(argv=None):
    try:
        run_simulation()

    except UsageError as err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, 'For help use --help'
        return 2

if __name__ == '__main__':
    sys.exit(main())


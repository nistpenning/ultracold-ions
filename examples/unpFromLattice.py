import uci.BorisUpdater as BorisUpdater
import uci.CoulombAcc as CoulombAcc
import uci.Ptcls as Ptcls
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import argparse
import sys

parser = argparse.ArgumentParser(
        description=
        """Simulate evolution of a plasma starting from a lattice
        configuration""")
parser.add_argument('-t', '--tmax',
        nargs=1,
        help='Duration of simulation in s',
        type=float,
        default=1.0e-11)
parser.add_argument('-d', '--dt',
        nargs=1,
        help='Integrator time step size',
        type=float,
        default=1.0e-12)
parser.add_argument('-n', '--n_wells',
        nargs=1,
        help='Number of lattice wells along one dimension',
        type=int,
        default=5)
parser.add_argument('-o', '--num_dump',
        nargs=1,
        help='Number of time steps between dumps of plasma state',
        type=int,
        default=10)

class UsageError(Exception):
    def __init__(self, msg):
        self.msg = msg

fund_charge = 1.602176565e-19
atomic_unit = 1.66053892e-27
ion_mass = 87.9056 * atomic_unit


def run_simulation(n_wells, dt, tmax, num_dump = 1):
    # initialize particles
    n_ions = n_wells**3
    ptcls = Ptcls.Ptcls()
    ptcls.set_nptcls(2 * n_ions)

    # ions
    lattice_spacing = 1.0e-6
    xmin = -0.5 * (n_wells - 1) * lattice_spacing
    for i in range(n_ions):
        ptcls.x()[i] = xmin + lattice_spacing * ((i / (n_wells**0)) % n_wells)
        ptcls.y()[i] = xmin + lattice_spacing * ((i / (n_wells**1)) % n_wells)
        ptcls.z()[i] = xmin + lattice_spacing * ((i / (n_wells**2)) % n_wells)
    ptcls.ptclList[3:6,:n_ions] = 0
    ptcls.ptclList[6,:n_ions] = fund_charge
    ptcls.ptclList[7,:n_ions] = ion_mass

    #electrons
    ptcls.ptclList[0:3,n_ions:] = np.random.normal(
            0.0, 0.5 * abs(xmin), ptcls.ptclList[0:3,n_ions:].shape)
    electron_temperature = 3.0
    electron_mass = 9.10938291e-31
    kB = 1.3806e-23
    vThermal = np.sqrt(kB * electron_temperature / electron_mass)
    ptcls.ptclList[3:6,n_ions:] = np.random.normal(
            0.0, vThermal, ptcls.ptclList[3:6,n_ions:].shape)
    ptcls.ptclList[6,n_ions:] = -fund_charge
    ptcls.ptclList[7,n_ions:] = electron_mass


    ctx = cl.create_some_context(interactive = True)
    queue = cl.CommandQueue(ctx)
    coulomb_acc = CoulombAcc.CoulombAcc(ctx, queue)
    accelerations = [coulomb_acc]
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
    while t < tmax:
        np.save('ptcls' + str(i) + '.npy',ptcls.ptclList[0:3,:])
        t = updater.update(
                xd, yd, zd, vxd, vyd, vzd, qd, md, accelerations, t, dt,
                num_dump)
        xd.get(queue, ptcls.x())
        yd.get(queue, ptcls.y())
        zd.get(queue, ptcls.z())

    np.savetxt('ptcls' + str(num_dump) + '.txt',ptcls.ptclList[0:3,:])

def main(argv=None):
    try:
        try:
            args = parser.parse_args(argv)
        except Exception as msg:
            raise UsageError(msg)

        run_simulation(args.n_wells, args.dt, args.tmax, args.num_dump)

    except UsageError as err:
        print >>sys.stderr, err.msg
        print >>sys.stderr, 'For help use --help'
        return 2

if __name__ == '__main__':
    sys.exit(main())


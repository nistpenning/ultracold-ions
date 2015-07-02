import uci.BendKickUpdater as BendKickUpdater
import uci.CoulombAcc as CoulombAcc
import uci.Ptcls as Ptcls
import uci.TrapAcc as TrapAcc
import uci.TrapConfiguration as TrapConfiguration
import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array
import sys
import os
from matplotlib import pyplot as plt
from matplotlib import animation

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

"""THINGS TO CHANGE/FIX"""
# Make trap parameters between different codes consistent. ex: Nion vs n_ions


class penning_trap_master:
    """
    Contains methods to manage the Itano, Mode, and Dynamics codes all at once,
    and pass information between the three.

    """

    fund_charge = 1.602176565e-19
    ion_mass = 8.96 * 1.66053892e-27

    def __init__(self, n_ions=19, Vtrap=(0.0, -1750.0, -2000.0),
                 B=4.4588, frot=180., Vwall=5.,
                 writeto=None, spinup=True):
        """Initialize Everything - pass all relevent trap parameters now"""

        if writeto is None:
            self.writeto = os.getcwd()
        else:
            self.writeto = writeto
        self.itano = None

        # Setup Zero Temperature Crystal
        self.crystal = mc.ModeAnalysis(N=n_ions, Vtrap=Vtrap, B=B, frot=frot, Vwall=Vwall)

        # Setup Simulation

        self.trap_config = None
        self.ptcls = None
        self.n_ions = n_ions
        self.accelerations = None
        self.delta = None

        self.setup_sim(spinup)

        self.mode_code_loaded = False
        self.itano_code_loaded = False
        self.dynamics_loaded = False

        self.update_loads()

    def update_loads(self):
        if self.crystal is not None:
            self.mode_code_loaded = True

        if self.itano is not None:
            self.itano_code_loaded = True

        if self.ptcls is not None and self.trap_config is not None and self:
            self.dynamics_loaded = True

    def run_everything(self):

        pass  # make this do everything

    def setup_sim(self, spinup=True):
        """Use trap parameters in self.crystal to setup simulation"""
        self.crystal.run()
        self.trap_config = TrapConfiguration.TrapConfiguration()
        self.trap_config.Bz = self.crystal.B
        self.trap_config.kz = 2 * self.crystal.V0
        self.delta = -self.crystal.Cw
        self.trap_config.kx = -(0.5 + self.delta) * self.trap_config.kz
        self.trap_config.ky = -(0.5 - self.delta) * self.trap_config.kz
        self.trap_config.omega = self.crystal.wrot
        self.trap_config.theta = 0

        # initialize particles
        self.ptcls = Ptcls.Ptcls()
        self.ptcls.set_nptcls(self.n_ions)
        x = self.crystal.uE[:self.n_ions].astype(np.float32)
        x = x.reshape(1, self.crystal.Nion)
        y = self.crystal.uE[self.n_ions:].astype(np.float32)
        y = y.reshape(1, self.crystal.Nion)
        self.ptcls.ptclList[0] = x
        self.ptcls.ptclList[1] = y
        self.ptcls.ptclList[2:6, :self.n_ions] = 0
        self.ptcls.ptclList[6, :self.n_ions] = self.crystal.q
        self.ptcls.ptclList[7, :self.n_ions] = self.crystal.m

        self.update_loads()

        if spinup is True:

            radii = np.sqrt(self.ptcls.x() ** 2 + self.ptcls.y() ** 2)
            velocities = self.trap_config.omega * radii
            for i in range(0, self.n_ions):
                v = np.array([-self.ptcls.y()[i], self.ptcls.x()[i]])
                v = v / np.linalg.norm(v)
                v = velocities[i] * v
                self.ptcls.vx()[i] = v[0] + self.ptcls.vx()[i]  # move velocities to rotating frame
                self.ptcls.vy()[i] = v[1] + self.ptcls.vy()[i]

    def run_dynamics_simulation(self, dt=5e-10, tmax=1e-6, num_dump=1000):

        ctx = cl.create_some_context(interactive=True)
        queue = cl.CommandQueue(ctx)

        coulomb_acc = CoulombAcc.CoulombAcc(ctx, queue)
        trap_acc = TrapAcc.TrapAcc(ctx, queue)
        trap_acc.trapConfiguration = self.trap_config

        accelerations = [coulomb_acc, trap_acc]

        updater = BendKickUpdater.BendKickUpdater(ctx, queue)
        updater.trapConfiguration = self.trap_config
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
        np.save('ptclsz' + str(i) + '.npy', self.ptcls.ptclList[0:3, :])

        while t < tmax:
            t = updater.update(
                xd, yd, zd, vxd, vyd, vzd, qd, md, accelerations, t, dt,
                num_dump)

            xd.get(queue, self.ptcls.x())
            yd.get(queue, self.ptcls.y())
            zd.get(queue, self.ptcls.z())
            print(self.ptcls.x())
            vxd.get(queue, self.ptcls.vx())
            vyd.get(queue, self.ptcls.vy())
            vzd.get(queue, self.ptcls.vz())
            i += 1
            np.save('ptclsz' + str(i) + '.npy', self.ptcls.ptclList[0:3, :])

    def setup_itano(self, defaultoff=30.0E-6, defaultdet=-500E6, wr=2 * np.pi * 45.0E3,
                    Tguess=1E-3, saturation=.5, dens=2.77E9, ywidth=2.0E-6, radius=225.0E-6,
                    quiet=True, spar=.2, wpar=2E6):

        self.itano = ic.ItanoAnalysis(defaultoff=defaultoff, defaultdet=defaultdet,
                                      wr=wr, Tguess=Tguess, saturation=saturation,
                                      dens=dens, ywidth=ywidth, radius=radius,
                                      quiet=quiet, spar=spar, wpar=wpar)

    def set_output_folder(self, dir):
        self.writeto = dir

    def exciteMode(self, branch, mode, amp, phase):
        """Excite a particular mode

        Branch = 0: Axial
                 1: Planar

        Mode = 1 to N (or 2N for planar)
                (they are ordered by frequency ascending)

        amp -- amplitude of motion (think lambda^2*A^2)
        phase -- phase angle of excitation (think, displace positions or velocities?)
                However, I don't know what phase = 0, means to the eigensystem solver

        returns column vector with displacements stacked on velocities
        """
        # convert to python indexing
        mode = mode - 1

        if branch == 0:  # axial branch
            D = self.crystal.axialEvals[mode]
            E1 = np.squeeze(self.crystal.axialEvects[:, 2 * mode])
            E1 = np.concatenate((E1[:self.n_ions], E1[self.n_ions:] / D))
            E2 = np.squeeze(self.crystal.axialEvects[:, 2 * mode + 1])
            E2 = np.concatenate((E2[:self.n_ions], E2[self.n_ions:] / D))

        else:  # planar modes
            D = self.crystal.planarEvals[mode]
            E1 = np.squeeze(self.crystal.planarEvals[:, 2 * mode])
            E1 = np.concatenate((E1[:2 * self.n_ions], E1[2 * self.n_ions:] / D))
            E2 = np.squeeze(self.crystal.planarEvals[:, 2 * mode + 1])
            E2 = np.concatenate((E2[:2 * self.n_ions], E2[2 * self.n_ions:] / D))

        A1 = amp * np.exp(1j * (D + phase));
        A2 = amp * np.exp(-1j * (D + phase));

        qqdot = A1 * E1 + A2 * E2
        return qqdot

    def spin_down(self, x, y, vx, vy):
        """Find velocities in rotating frame (move to that frame)"""
        radii = np.sqrt(x ** 2 + y ** 2)
        velocities = self.trap_config.omega * radii
        for i in range(x.size):
            rot = np.hstack((-x[i], y[i]))
            rot = rot / np.linalg.norm(rot)

            # counter velocity to move to rotating frame
            rot = -velocities[i] * rot
            vx[i] += rot[0]
            vy[i] += rot[1]
        return vx, vy

    def make_qqdot_dimensionless(self, qqdot):
        """Take a vector of positions stacked on top of velocities and convert
        to dimensionless quantities

        Assumes first half is positions and second half is velocities
        """
        N = int(u.size / 2)  # N is NOT necessarily n_ions for this function only
        pos = qqdot[0:N]
        vel = qqdot[N:]
        qqdot_dim = np.hstack((pos / self.crystal.l0, vel / self.crystal.v0))
        return qqdot_dim

    def axial_norm_coords(self, ptcls):
        """Project axial motion and velocities into normal coordinates of
        axial modes for a crystal snapshot"""

        z = self.ptcls.ptclList[2, :self.n_ions]
        vz = self.ptcls.ptclList[5, :self.n_ions]
        qqdot = np.hstack((z, vz))
        qqdot = self.make_qqdot_dimensionless(qqdot)

        norm_coords = []
        for mode in self.n_ions:
            # assumes mode pairs are one after the other
            E = np.squeeze(self.crystal.axialEvects[:, 2 * mode])
            D = self.crystal.axialEvals[mode]
            E = np.concatenate((E[:self.n_ions], E[self.n_ions:] / D))
            norm_coords.append(np.absolute(np.dot(qqdot, E)))

        return np.array(norm_coords)

    def linearized_axial_energy(self, ptcls):
        """Compute linearized axial energy"""
        z = self.ptcls.ptclList[2, :self.n_ions]
        vz = self.ptcls.ptclList[5, :self.n_ions]
        qqdot = np.hstack((z, vz))
        qqdot = self.make_qqdot_dimensionless(qqdot)
        return np.sum(qqdot ** 2)

    def axial_mode_energy(self, ptcls):
        """Takes a ptcls snapshot and converts to axial mode energy"""
        norm_coords = self.axial_norm_coords(self.ptcls)

        EnergyAxialMode = np.zeros(self.nions)
        for mode in range(self.n_ions):
            D = self.crystal.axialEvals[mode]
            EnergyAxialMode[mode] = self.crystal.E0 * 2 * D ** 2 * norm_coords[2 * mode] ** 2

        return EnergyAxialMode

    def planar_norm_coords(self, ptcls):
        """Project planar motion (away from equilbirium)
        and velocities into normal coordinates of axial modes for a crystal snapshot"""
        x = self.ptcls.ptclList[0, :self.n_ions]
        y = self.ptcls.ptclList[1, :self.n_ions]
        vx = self.ptcls.ptclList[3, :self.n_ions]
        vy = self.ptcls.ptclList[4, :self.n_ions]

        # Get velocties in rotating frame (this needs to happen first)
        vx, vy = self.spin_down(x, y, vx, vy)

        # Rotate positions into rotating frame
        x, y = self.rotate(x, y, -self.trap_config.theta)  # rotate crystal back
        displacement = np.hstack((x, y)) - self.crystal.uE  # subtract off equilibrium positions
        qqdot = np.hstack((displacement, vx, vy))
        qqdot = self.make_qqdot_dimensionless(qqdot)

        norm_coords = []
        for mode in 2 * self.n_ions:
            # assumes mode pairs are one after the other
            E = np.squeeze(self.crystal.axialEvects[:, 2 * mode])
            D = self.crystal.axialEvals[mode]
            E = np.concatenate((E[:2 * self.n_ions], E[2 * self.n_ions:] / D))
            norm_coords.append(np.absolute(np.dot(qqdot, E)))

        return np.array(norm_coords)

    def planar_mode_energy(self, ptcls):
        """Takes a ptcls snapshot and converts to axial mode energy"""
        norm_coords = self.planar_norm_coords(ptcls)

        norm_coords = self.planar_norm_coords(ptcls)
        EnergyPlanarMode = np.zeros(2 * self.nions)
        for mode in range(self.n_ions):
            D = self.crystal.planarEvals[mode]
            EnergyPlanarMode[mode] = self.crystal.E0 * 2 * D ** 2 * norm_coords[2 * mode] ** 2

        return EnergyPlanarMode

    @staticmethod
    def rotate(x, y, theta):
        """Rotates coordinates by theta"""
        xnew = x * np.cos(theta) - y * np.sin(theta)
        ynew = x * np.sin(theta) + y * np.cos(theta)
        return xnew, ynew


class penning_trap_analyze():
    def __init__(self):

        pass

    @staticmethod
    def unpickle_ptcls(index):
        """Unpickle a ptcls object from a file

        index:"""
        if index[-4:] is not ".npy":
            index.append(".npy")
        ptcls = np.load(index)
        return ptcls

    def unpickle_ptclslist(self, ind):
        """Unpickle a np.array that is a subarray from the ptcls.ptclList
        given by indices = ind from a file and convert to a ptcls object"""
        pass

    def convert_to_ptcls(self, ind):
        """Convert a subarray of a ptcls.ptclList to a ptcls object"""
        # Maybe this could be a method in the ptclsclass

    def load_all_data(self):
        """Loop though all pickled data (either ptcls or ptclsList and convert
        to list of ptcls objects for each time step."""
        self.unpickle_ptcls(self.filelocation)
        # OR
        self.unpickle_ptclslist(self.filelocation)
        # not sure how to distinguish
        return data

    def fourierAnalysis(self):
        """Load all data and compute power spectral densities"""
        data = self.load_all_data()  # list of ptcls?

        self.dt = 5e-10
        self.tmax = 1e-6
        self.num_dump = 1000
        # not even close to ready

        # MATLAB PSD CODE
        #        InSteps = 1e2   # Inner steps in simulation
        #        OutSteps = 1e6  # Outer number of steps in simulation
        #        dt = 5e-10      # time step
        #
        #        freq = np.arange(0, 0.5/(InSteps*dt), 1/(InSteps*dt*OutSteps))
        #        freq = (0:0.5/(InSteps*dt)/(params(5)/2):0.5/(InSteps*dt))
        #        freq = 1.0e-6*freq(1:end-1); % chop of last one, because of Matlab ranges...
        #
        #        # Calculate PSD for Axial Motion
        #        spectra = abs(fft(zs)).^2;
        #        Zpsd = sum(spectra, 2);
        #        Zpsd = Zpsd(1:(length(Zpsd)/2))+Zpsd(end:-1:length(Zpsd)/2+1);
        #
        #        # Calculate PSD for Planar Motion
        #        motion = us - repmat(us(1,:),params(5),1); % subtract off equilibrium positions
        #        spectra = abs(fft(motion)).^2;
        #        Ppsd = sum(spectra, 2);
        #        Ppsd = Ppsd(1:(length(Ppsd)/2))+Ppsd(end:-1:length(Ppsd)/2+1);

    def kinetic_energy(self):
        """Calculate total kinetic energy of all particles"""
        # Note, this should be the full energy, not the linearized approximation
        pass

    def potential_energy(self):
        """Calculate total potential energy of all particles using ModeAnalysis?"""
        # Note, this should be the full energy, not the linearized approximation
        pass

    def make_movie(self,
                   dt=5e-10, tmax=1e-6, num_dump=500, zcolor=False):
        """
        Super experimental! Not sure how this will work out, but let's see anyway!
        """
        numframes = round(tmax / (dt * num_dump))
        toread = ["ptclsz" + str(i) + '.npy' for i in range(numframes)]
        global uniqueanimationcounter, colorextreme
        uniqueanimationcounter = 0
        colorextreme = 1.0E-6
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # I'm still unfamiliar with the following line of code:
        if zcolor is True:
            cm = plt.get_cmap('seismic')
            #line, = ax.plot([], [], 'o', ms=10, c=[], cmap=cm)
            line, = ax.plot([], [], 'o', ms=10, c=cm)
        if zcolor is False:
            line, = ax.plot([], [], 'o', ms=10, color='DeepSkyBlue')
            # fig.backgroundcolor = "black"

        ax.set_ylim(-9E-5, 9E-5)
        ax.set_xlim(-9E-5, 9E-5)
        print('numframes:', numframes)

        def crystalpoints(a):
            print("bang!")
            global uniqueanimationcounter, colorextreme
            data = np.load(toread[uniqueanimationcounter])
            xdata = data[0]
            ydata = data[1]
            zdata = data[2]
            print(xdata)

            if zcolor is False:
                line.set_data(xdata, ydata)
            if zcolor is True:
                if max(abs(zdata)) > colorextreme:
                    colorextreme = max(abs(zdata))
                    cm.set_clim(-colorextreme, colorextreme)
                line.set_data(xdata, ydata)
                if min(zdata) < 0:
                    zdata += min(zdata)
                zdata/=max(zdata)
                zcolors = cm(zdata)
                print(zcolors)

                line.set_c(zcolors)



            uniqueanimationcounter += 1
            print("numframes:", numframes, "counter:", uniqueanimationcounter)
            print(line)
            return line,

        ani = animation.FuncAnimation(fig, func=crystalpoints,
                                      blit=False, fargs=(),
                                      frames=numframes - 1, repeat=False)
        # plt.show()
        ani.save('frankenstein.mp4', fps=20, writer='ffmpeg', extra_args=['-vcodec', 'libx264'])


def main(argv=None):
    try:
        # a = penning_trap_master()
        # a.setup_sim()
        # a.run_dynamics_simulation(tmax=8E-6,num_dump=50)
        b = penning_trap_analyze()
        b.make_movie(tmax=8E-6, num_dump=100, zcolor=False)

    except UsageError as err:
        print(sys.stderr, err.msg)
        print(sys.stderr, 'For help use --help')
        return 2


if __name__ == '__main__':
    sys.exit(main())

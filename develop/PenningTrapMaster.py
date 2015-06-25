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
                               writeto=None):
        """Initialize Everything - pass all relevent trap parameters now"""                            

        if writeto is None:
            self.writeto = os.getcwd()

        self.itano = None
        
        # Setup Zero Temperature Crystal
        self.crystal = mc.ModeAnalysis(N=n_ions, Vtrap=Vtrap, frot=frot, B=B,
                                  Vwall=Vwall)      
        self.crystal.run()

        # Setup Simulation
        self.setup_sim()
        
        self.trap_config = None
        self.ptcls = None
        self.n_ions = n_ions
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

    def setup_sim(self):
        """Use trap parameters in self.crystal to setup simulation"""
        self.trap_config = TrapConfiguration.TrapConfiguration()
        self.trap_config.Bz = self.crystal.B
        self.trap_config.kz = 2 * self.crystal.V0
        self.delta = -self.crystal.Cw
        self.trap_config.kx = -(0.5 + self.delta) * trap_config.kz
        self.trap_config.ky = -(0.5 - self.delta) * trap_config.kz
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

    def unpickle_ptcls():
        """Unpickle a ptcls object from a file"""
        pass
    
    def unpickle_ptclslist(ind):
        """Unpickle a np.array that is a subarray from the ptcls.ptclList 
        given by indices = ind from a file and convert to a ptcls object"""
        pass
    
    def convert_to_ptcls(ind):
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
    
    def fourierAnalysis():
        """Load all data and compute power spectral densities"""
        data = self.load_all_data()
        
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
        
    def kinetic_energy():
        """Calculate kinetic energy of all particles"""
        pass
    
    def potential_energy():
        """Calculate potential energy of all particles using ModeAnalysis?"""
        pass
        
    def exciteMode(branch, mode):
        """Excite a particular mode
        
        Branch = 0: Magnetron
                 1: Axial
                 2: Cyclotron
        """
        pass
    

    
    def rotate(x, y, theta):
        """Rotates coordinates by theta"""
        xnew = x*np.cos(theta) - y*np.sin(theta)
        ynew = x*np.sin(theta) + y*np.cos(theta)    
        return xnew, ynew
        
    def spin_down(self, x, y, vx, vy):
        """Find velocities in rotating frame (move to that frame)"""
        radii = np.sqrt(x**2 + y**2)
        velocities = self.trap_config.omega*radii
        for i in range(x.size):
            rot = np.hstack((-x[i], y[i]))
            rot = rot/ny.linalg.norm(rot)
            
            # counter velocity to move to rotating frame
            rot = -velocities[i]*rot;
            vx[i] += rot[0]
            vy[i] += rot[1]
        return vx, vy  

    def make_qqdot_dimensionless(self, qqdot):
        """Take a vector of positions stacked on top of velocities and convert
        to dimensionless quantities
        
        Assumes first half is positions and second half is velocities        
        """
        N = int(u.size/2)  # N is NOT necessarily n_ions for this function only
        pos = qqot[0:N]
        vel = qqdot[N:]
        qqdot_dim = np.hstack((pos/self.crystal.l0, vel/self.crystal.v0))
        return qqdot_dim
        
    def axial_norm_coords(self, ptcls):
        """Project axial motion and velocities into normal coordinates of 
        axial modes for a crystal snapshot"""
        #STILL WORKING ON THIS - AK
        z = self.ptcls.ptclList[2, :self.n_ions]
        vz = self.ptcls.ptclList[5, :self.n_ions]
        qqdot = np.hstack((z,vz))
        qqdot = self.make_qqdot_dimensionless(qqdot)
        self.crystal.axialEvals
          
        # not sure which one works better
        #a_norm_coords = np.linalg.inv(self.crystal.axialEvects)*qqdot 
        #a_norm_coords = qqdot/self.crystal.axialEvects
        
        
    def planar_norm_coords(self, ptcls):
        """Project planar motion (away from equilbirium) 
        and velocities into normal coordinates of axial modes for a crystal snapshot"""
        #STILL WORKING ON THIS - AK
        x = self.ptcls.ptclList[0, :self.n_ions]
        y = self.ptcls.ptclList[1, :self.n_ions]
        vx = self.ptcls.ptclList[3, :self.n_ions]
        vy = self.ptcls.ptclList[4, :self.n_ions]
        
        # Get velocties in rotation frame (this needs to happen first)
        vx, vy = self.spin_down(x, y, vx, vy)
        
        # Rotate positions into rotating frame        
        x, y  = self.rotate(x, y, -self.trap_config.theta)  # rotate crystal back
        displacement = np.hstack((x,y)) - self.crystal.uE   # subtract off equilibrium positions  
        

        qqdot = np.hstack((x, y, vx, vy))
        
        qqdot = np.array([PlanarMotion[s, :], vrot[s,:])
        # need to get these dimensions right so matrix product works
        norm_coords_planar[i,:] = numpy.linalg.inv(aTrap.planarEvects)*qqdot 
        N = int(u.size/2)
        norm_coords = zeros(1,N);    
    
    def axial_mode_energy(self, ptcls)
        """Takes a ptcls snapshot and converts to axial mode energy"""
        #STILL WORKING ON THIS - AK
        norm_coords = self.axial_norm_coords(ptcls)
        
        EnergyAxialMode = np.zeros(self.nions)
        for i in range(self.n_ions):
            EnergyAxialMode[i] = self.crystal.E0*(np.absolute(norm_coords[2*i])**2+
                                                 np.absolute(norm_coords[2*i+1])**2)

    def planar_mode_energy(self, ptcls)
        """Takes a ptcls snapshot and converts to axial mode energy"""
        
        #STILL WORKING ON THIS - AK
        norm_coords = self.planar_norm_coords(ptcls)
        
        EnergyPlanarMode = np.zeros(self.nions
        for i in range(self.n_ions):
            EnergyPlanarMode[i] = self.crystal.E0*(np.absolute(norm_coords_planar[2*i])**2+
                                                 np.absolute(norm_coords_planar[2*i+1])**2)


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

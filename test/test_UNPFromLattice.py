from os.path import dirname, realpath, sep, pardir
import sys
sys.path.append(dirname(realpath(__file__)) + sep + pardir + sep +
         "examples")
import unpFromLattice

def test_sim():
    unpFromLattice.run_simulation(4, 1.0e-12, 1.0e-13, 1)



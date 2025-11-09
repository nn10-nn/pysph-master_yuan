"""Simulate the modified Sod Shocktube problem (as per the reference) in 1D.
"""
from pysph.examples.gas_dynamics.shocktube_setup import ShockTubeSetup
from pysph.sph.scheme import (ADKEScheme, GasDScheme, GSPHScheme,
                              SchemeChooser, add_bool_argument)
from pysph.sph.gas_dynamics.psph import PSPHScheme
from pysph.sph.gas_dynamics.tsph import TSPHScheme
from pysph.sph.gas_dynamics.magma2 import MAGMA2Scheme
from pysph.sph.wc.crksph import CRKSPHScheme
from pysph.base.nnps import DomainManager

# Numerical constants
dim = 1
gamma = 5/3  # 修改：绝热指数改为5/3
gamma1 = gamma - 1.0  # 同步更新gamma-1

# solution parameters
dt = 1e-4
tf = 0.5    # 修改：总模拟时间改为0.5

class ModifiedSodShockTube(ShockTubeSetup):

    def initialize(self):
        self.xmin = 0.0   # 修改：左边界改为0
        self.xmax = 1.0   # 修改：右边界改为1
        self.x0 = 0.5     # 修改：分隔位移改为0.5
        self.rhol = 1.0   
        self.rhor = 0.125 
        self.pl = 1.0     
        self.pr = 0.1     
        self.ul = 0.5     # 修改：左侧速度改为0.5
        self.ur = 0.5     # 修改：右侧速度改为0.5
        self.bx = 0.00    

    def add_user_options(self, group):
        group.add_argument(
            "--hdx", action="store", type=float,
            dest="hdx", default=1.2,
            help="Ratio h/dx."
        )
        group.add_argument(
            "--nl", action="store", type=float, dest="nl", default=640,
            help="Number of particles in left region"
        )
        group.add_argument(
            "--dscheme", choices=["constant_mass", "constant_volume"],
            dest="dscheme", default="constant_mass",
            help="Spatial discretization scheme."
        )
        add_bool_argument(group, 'smooth-ic', dest='smooth_ic', default=False,
                          help="Smooth the initial condition.")

    def consume_user_options(self):
        self.nl = self.options.nl
        self.hdx = self.options.hdx
        self.smooth_ic = self.options.smooth_ic
        self.dscheme = self.options.dscheme
        self.dxl = (self.x0 - self.xmin) / self.nl  
        if self.dscheme == 'constant_mass':
            ratio = self.rhor / self.rhol  
            self.dxr = self.dxl / ratio  
        else:
            self.dxr = self.dxl  
        self.h0 = self.hdx * self.dxr  
        self.dt = dt  
        self.tf = tf  

    def create_particles(self):
        f, b = self.generate_particles(
            xmin=self.xmin, xmax=self.xmax, x0=self.x0, rhol=self.rhol,
            rhor=self.rhor, pl=self.pl, pr=self.pr, bx=self.bx, gamma1=gamma1,
            ul=self.ul, ur=self.ur, dxl=self.dxl, dxr=self.dxr, h0=self.h0
        )
        self.scheme.setup_properties([f, b])  
        return [f]  

    def create_domain(self):
        return DomainManager(
            xmin=self.xmin, xmax=self.xmax, mirror_in_x=True,
            n_layers=2  
        )

    def configure_scheme(self):
        scheme = self.scheme
        if self.options.scheme in ['gsph', 'mpm']:
            scheme.configure(kernel_factor=self.hdx)
        elif self.options.scheme in ['psph', 'tsph']:
            scheme.configure(hfact=self.hdx)
        scheme.configure_solver(tf=self.tf, dt=self.dt)  

    def create_scheme(self):
        adke = ADKEScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,  # 同步修改gamma
            alpha=1, beta=1.0, k=0.3, eps=0.5, g1=0.2, g2=0.4)

        mpm = GasDScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,  # 同步修改gamma
            kernel_factor=None, alpha1=1.0, alpha2=0.1,
            beta=2.0, update_alpha1=True, update_alpha2=True,
        )
        gsph = GSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,  # 同步修改gamma
            kernel_factor=None,
            g1=0.2, g2=0.4, rsolver=2, interpolation=1, monotonicity=1,
            interface_zero=True, hybrid=True, blend_alpha=2.0,
            niter=20, tol=1e-6
        )
        crk = CRKSPHScheme(
            fluids=['fluid'], dim=dim, rho0=0, c0=0,
            nu=0, h0=0, p0=0, gamma=gamma, cl=3  # 同步修改gamma
        )

        psph = PSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,  # 同步修改gamma
            hfact=None
        )

        tsph = TSPHScheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,  # 同步修改gamma
            hfact=None
        )

        magma2 = MAGMA2Scheme(
            fluids=['fluid'], solids=[], dim=dim, gamma=gamma,  # 同步修改gamma
            has_ghosts=True, ndes=7
        )

        s = SchemeChooser(
            default='adke', adke=adke, mpm=mpm, gsph=gsph, crk=crk, psph=psph,
            tsph=tsph, magma2=magma2)
        return s  


if __name__ == '__main__':
    app = ModifiedSodShockTube()
    app.run()
    app.post_process()
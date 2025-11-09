""" This is a setup example that will be used other gas_dynamics problem
    like SodShockTube, BlastWave
"""

import os
import numpy
from math import sqrt

from pysph.base.utils import get_particle_array as gpa
from pysph.solver.application import Application
from pysph.examples.gas_dynamics import riemann_solver


class ShockTubeSetup_rel(Application):
    c = 1.0

    def generate_particles(self, xmin, xmax, dxl, dxr, rhol, rhor, pl, pr, bx,
                           gamma1, h0=None, x0=0.0, ul=0, ur=0, constants={}):
        xt1 = numpy.arange(xmin - bx + 0.5 * dxl, x0, dxl)
        xt2 = numpy.arange(x0 + 0.5 * dxr, xmax + bx, dxr)
        xt = numpy.concatenate([xt1, xt2])
        leftb_indices = numpy.where(xt <= xmin)[0]
        left_indices = numpy.where((xt > xmin) & (xt < x0))[0]
        right_indices = numpy.where((xt >= x0) & (xt < xmax))[0]
        rightb_indices = numpy.where(xt >= xmax)[0]
        x1 = xt[left_indices]
        x2 = xt[right_indices]
        b1 = xt[leftb_indices]
        b2 = xt[rightb_indices]

        x = numpy.concatenate([x1, x2])
        b = numpy.concatenate([b1, b2])
        right_indices = numpy.where(x > x0)[0]

        smooth_ic = self.smooth_ic if hasattr(self, 'smooth_ic') else False
        if smooth_ic:
            deltax = 1.5 * numpy.mean(x[1:] - x[:-1])
            p = (pl - pr) / (1 + numpy.exp((x - x0) / deltax)) + pr
            u = (ul - ur) / (1 + numpy.exp((x - x0) / deltax)) + ur
            rho = (rhol - rhor) / (1 + numpy.exp((x - x0) / deltax)) + rhor
            gamma = 1.0 / numpy.sqrt(1 - numpy.clip(u**2 / self.c**2, 0, 0.999))
            D = rho * gamma
        else:
            rho = numpy.ones_like(x) * rhol
            rho[right_indices] = rhor
            p = numpy.ones_like(x) * pl
            p[right_indices] = pr
            u = numpy.ones_like(x) * ul
            u[right_indices] = ur
            gamma_left = 1.0 / numpy.sqrt(1 - numpy.clip(ul**2 / self.c**2, 0, 0.999))
            gamma_right = 1.0 / numpy.sqrt(1 - numpy.clip(ur**2 / self.c**2, 0, 0.999))
            gamma = numpy.ones_like(x) * gamma_left
            gamma[right_indices] = gamma_right
            D = rho * gamma

        dx = numpy.ones_like(x) * dxl
        dx[right_indices] = dxr
        m = D * dx

        if h0 is None:
            dx = numpy.ones_like(x) * dxl
            dx[right_indices] = dxr
            h = dx * self.hdx
        else:
            h = numpy.ones_like(x) * h0

        e = p / (gamma1 * rho)
        wij = numpy.ones_like(x)

        bwij = numpy.ones_like(b)
        rho_b = numpy.ones_like(b)
        gamma_b = 1.0 / numpy.sqrt(1 - numpy.clip(ul**2 / self.c**2, 0, 0.999))
        D_b = rho_b * gamma_b
        bp = numpy.ones_like(b)
        be = bp / (gamma1 * rho_b)
        bm = numpy.ones_like(b) * dxl
        bh = numpy.ones_like(b) * 4 * h0 if h0 is not None else numpy.ones_like(b) * 4 * dx[0]
        bhtmp = numpy.ones_like(b)
        
        fluid = gpa(
            constants=constants, name='fluid', x=x, rho=rho,
            D=D, gamma=gamma,
            p=p, e=e, h=h, m=m, u=u, wij=wij, h0=h.copy()
        )

        boundary = gpa(
            constants=constants, name='boundary', x=b, rho=rho_b,
            D=D_b, gamma=gamma_b,
            p=bp, e=be, h=bh, m=bm, wij=bwij, h0=bh.copy(), htmp=bhtmp
        )

        self.scheme.setup_properties([fluid, boundary])
        print("1D Shocktube with %d particles" %
              (fluid.get_number_of_particles()))
        return [fluid, boundary]

    def post_process(self):
        try:
            import matplotlib
            matplotlib.use('Agg')
            from matplotlib import pyplot as plt
        except ImportError:
            print("Post processing requires matplotlib.")
            return
        if self.rank > 0 or len(self.output_files) == 0:
            return

        last_output = self.output_files[-1]
        from pysph.solver.utils import load
        data = load(last_output)
        pa = data['arrays']['fluid']
        gamma = self.options.gamma if self.options.gamma else 1.4
        riemann_solver.set_gamma(gamma)

        rho_e, u_e, p_e, e_e, x_e = riemann_solver.solve(
            x_min=self.xmin, x_max=self.xmax, x_0=self.x0,
            t=self.tf, p_l=self.pl, p_r=self.pr, rho_l=self.rhol,
            rho_r=self.rhor, u_l=self.ul, u_r=self.ur, N=101
        )
        gamma_e = 1.0 / numpy.sqrt(1 - numpy.clip(u_e**2 / self.c**2, 0, 0.999))
        D_e = rho_e * gamma_e

        x = pa.x
        D = pa.D
        rho = pa.rho
        e = pa.e
        h_rel = 1 + e + pa.p / rho
        cs = self.c * numpy.sqrt(gamma * pa.p / (D * h_rel))
        u = pa.u
        p = pa.p
        h = pa.h

        plt.plot(x, D, label='pysph (' + str(self.options.scheme) + ')')
        plt.plot(x_e, D_e, label='exact')
        plt.xlabel('x')
        plt.ylabel('D')
        plt.legend()
        fig = os.path.join(self.output_dir, "density_D.png")
        plt.savefig(fig, dpi=300)
        plt.clf()

        plt.plot(x, e, label='pysph (' + str(self.options.scheme) + ')')
        plt.plot(x_e, e_e, label='exact')
        plt.xlabel('x')
        plt.ylabel('e')
        plt.legend()
        fig = os.path.join(self.output_dir, "energy.png")
        plt.savefig(fig, dpi=300)
        plt.clf()

        plt.plot(x, D * u, label='pysph (' + str(self.options.scheme) + ')')
        plt.plot(x_e, D_e * u_e, label='exact')
        plt.xlabel('x')
        plt.ylabel('D*u')
        plt.legend()
        fig = os.path.join(self.output_dir, "Machno.png")
        plt.savefig(fig, dpi=300)
        plt.clf()

        plt.plot(x, p, label='pysph (' + str(self.options.scheme) + ')')
        plt.plot(x_e, p_e, label='exact')
        plt.xlabel('x')
        plt.ylabel('p')
        plt.legend()
        fig = os.path.join(self.output_dir, "pressure.png")
        plt.savefig(fig, dpi=300)
        plt.clf()

        fname = os.path.join(self.output_dir, 'results.npz')
        numpy.savez(fname, x=x, u=u, e=e, cs=cs, rho=rho, D=D, gamma=pa.gamma, p=p, h=h)

        fname = os.path.join(self.output_dir, 'exact.npz')
        numpy.savez(fname, x=x_e, u=u_e, e=e_e, p=p_e, rho=rho_e, D=D_e, gamma=gamma_e)

    def configure_scheme(self):
        s = self.scheme
        kernel_factor = self.options.hdx
        if self.options.scheme == 'mpm':
            s.configure(kernel_factor=kernel_factor)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=True, pfreq=50)
        elif self.options.scheme == 'adke':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'gsph':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'crk':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=1)
        elif self.options.scheme in ['tsph', 'psph']:
            s.configure(hfact=kernel_factor)
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
        elif self.options.scheme == 'magma2':
            s.configure_solver(dt=self.dt, tf=self.tf,
                               adaptive_timestep=False, pfreq=50)
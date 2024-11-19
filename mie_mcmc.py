import numpy as np
import columnplots as clp

class mcmc:

   def __init__(self, 
                nmol, ncycle, nmicro, nwrite, 
                density, delta_max, 
                lattice_type, n, 
                T, pot_param, 
                rdf_point):
      
      self.nmol         = nmol
      self.ncycle      = ncycle
      self.nmicro       = nmicro
      self.nwrite       = nwrite
      self.density      = density
      self.boxlength    = (self.nmol / self.density) ** (1/3)
      self.delta_max    = delta_max
      self.lattice_type = lattice_type
      self.nx           = n[0]
      self.ny           = n[1]
      self.nz           = n[2]
      self.T            = T
      self.pot_param    = pot_param
      self.rdf_point    = rdf_point
      print(f'nmol = {nmol:3d}, ncycle = {ncycle:4d}, nmicro = {nmicro:3d}, nwrite = {nwrite:3d}')
      print(f'system density = {density:7.4f}, random move max distance = {delta_max:6.4f}')
      print(f'lattice_type = {lattice_type}, lattice index = {n}')
      print(f'simulation temperature = {T:9.4f} K, potential paramete = {pot_param}, RDF grid number = {rdf_point:4d}')

   def Mie_potential(self, distance):
      # Mie potential U = C * epsilon * ((sigma/r)^n - (sigma/r)^m)
      # C = n / (n - m) * (n / m)^(m/(n-m))
      sigma, epsilon, n, m = self.pot_param
      C = n / (n - m) * (n / m) ** (m / (n - m))
      return C * epsilon * ((sigma/distance) ** n - (sigma/distance) ** m)

   def single_energy(self, current_position, test_position, imol):
      net_energy = 0
      for i in range(self.nmol):
         if i != imol : 
            distance = np.linalg.norm(current_position[i,:] - test_position)
            net_energy += self.Mie_potential(distance)
      return net_energy

   def total_energy(self, current_position):
      net_energy = 0
      for imol in range(self.nmol-1):
         for jmol in range(imol+1,self.nmol):
            distance = np.linalg.norm(current_position[imol,:]-current_position[jmol,:])
            net_energy += self.Mie_potential(distance)
      return net_energy

   def bulid_lattice(self):
      # Only support SC, BCC, FCC
      if self.lattice_type == 'SC' :    return 1, np.array([0.0,0.0,0.0])
      elif self.lattice_type == 'BCC' : return 2, np.array([[0.0,0.0,0.0], [0.5,0.5,0.5]])
      elif self.lattice_type == 'FCC' : return 4, np.array([[0.0,0.0,0.0], [0.5,0.5,0.0], [0.5,0.0,0.5], [0.0,0.5,0.5]])
      else : return None

   def initial_box(self):
      trajectory = np.zeros((self.nmol,3,self.ncycle+1))
      nmol_per_lattice, lattice_position = self.bulid_lattice()
      assert self.nmol // nmol_per_lattice == self.nx * self.ny * self.nz

      times = -1
      for i in range(self.nx):
         for j in range(self.ny):
            for k in range(self.nz):
               times += 1
               if self.lattice_type == 'SC' : 
                  trajectory[nmol_per_lattice*times:nmol_per_lattice*(times+1),:,0] = lattice_position + np.array([i,j,k])
               else : 
                  trajectory[nmol_per_lattice*times:nmol_per_lattice*(times+1),:,0] = lattice_position + np.einsum('ij,j->ij', np.ones_like(lattice_position), np.array([i,j,k]))

      trajectory *= self.boxlength
      self.trajectory = trajectory
      self.energy = np.zeros(self.ncycle+1)
      self.energy[0] = self.total_energy(self.trajectory[:,:,0])

   def random_move(self, current_position):

      new_position = current_position

      for imol in range(self.nmol):
         for itest in range(self.nmicro):

            random_position = (np.random.random((1,3)) - 0.5) * self.delta_max
            test_position  = new_position[imol,:] + random_position
            test_position -= (test_position // self.boxlength) * self.boxlength

            energy_old = self.single_energy(new_position, new_position[imol,:], imol)
            energy_new = self.single_energy(new_position, test_position, imol)

            if energy_new < energy_old : 
               new_position[imol,:] = test_position
               break
            else :
               ref_possibility = np.random.random()
               new_possibility = np.exp(-(energy_new - energy_old) / self.T)
               if new_possibility > ref_possibility : 
                  new_position[imol,:] = test_position
                  break
      
      return new_position

   def get_distance(self, position):
      ref_position = position[0,:]
      new_position = position - ref_position
      row = np.shape(position)[0]
      full_distance = np.zeros(row*(9**3))
      idx = 0
      search_bound = 4
      for nx in range(-search_bound,search_bound+1):
         for ny in range(-search_bound,search_bound+1):
            for nz in range(-search_bound,search_bound+1):
               vector = np.array([nx, ny, nz])
               lattice_position = new_position + np.einsum('ij,j->ij', np.ones_like(position), vector) * self.boxlength
               full_distance[idx*row:(idx+1)*row] = np.linalg.norm(lattice_position, axis=1)
               idx += 1
      return full_distance

   def get_pcf(self):
      # g(r) = V / N * (n(r)) / 4*pi*r^2*dr

      self.possibility = np.array([np.exp(-energy/self.T) for energy in self.energy])
      self.partial_function = np.sum(self.possibility)
      #min_idx = np.where(self.energy==self.energy.min())[0][0]
      position = np.zeros_like(self.trajectory[:,:,0])
      for i in range(len(self.energy)):
         position += self.possibility[i] / self.partial_function * self.trajectory[:,:,i]

      distance = self.get_distance(position)
      r  = np.linspace(0,self.boxlength*4,self.rdf_point+1)
      nr = np.zeros(self.rdf_point)
      dr = self.boxlength / self.rdf_point
      distance = np.array(distance)
      for i in range(self.rdf_point):
         largelist = np.where(distance>r[i])[0]
         smalllist = np.where(distance[largelist]<r[i]+dr)[0]
         nr[i] = len(distance[largelist][smalllist])
      dv = np.array([3.1415926575 * (r[i] +r[i+1])**2 * dr for i in range(self.rdf_point)])
      gr = np.array([ nr[i] / dv[i] for i in range(self.rdf_point)]) / self.density
      ax = clp.initialize(1, 1, width=4.3, LaTeX=True, fontsize=12)
      ax.axhline(y=1, linestyle='-.', alpha=0.5, color="k")
      ax.set_yticks([1,2,4,6,8,10,12])
      clp.plotone([r[1:]/self.boxlength], [gr], ax, labels=[r"$g(r)$"], colors=["r"], lw=1.5, xlim=[0,4], ylim=[0,12], xlabel=r"$r/a_0$", ylabel=r"$g(r)$")
      clp.adjust(savefile=f"./Ar_84K_333.png")

   def kernel(self):
      self.initial_box()
      for icycle in range(self.ncycle):
         if icycle % self.nwrite == 0 :  print(f'No. {icycle:4d} cycle, total energy = {self.energy[icycle]:14.8f}')
         self.trajectory[:,:,icycle+1] = self.random_move(self.trajectory[:,:,icycle])
         self.energy[icycle+1] = self.total_energy(self.trajectory[:,:,icycle+1])
      print(f'No. {self.ncycle:4d} cycle, total energy = {self.energy[-1]:14.8f}')
      self.get_pcf()

if __name__ == "__main__":
   
   test = mcmc(nmol=108, ncycle=1000, nmicro=50, nwrite=100, 
               density=0.02103, delta_max=1, 
               lattice_type='FCC', n=[3,3,3], 
               T=84, pot_param=[3.404, 117.84, 12, 6.0], # sigma (in angstrom), epsilon (in K^-1, normalized by k_B), n, m 
               rdf_point=100)
   
   # Ar : m.p. = 83.81K; b.p. = 87.302K; teiple point 83.8058K
   # density at b.p. : 1.3954 g/cm^3 / 39.95 g/mol * 6.02214076*10^23 mol^-1 * 10^(-24) cm^3/ang^3 = 0.02103 ang^(-3)
   # FCC a=5.4691 angstrom density : 4 / (5.4691)^3 = 0.02445 ang^(-3) in teiple point 83.8058K
   # mie potential parameter [3.404, 117.84, 12.085, 6.0]
   
   test.kernel()
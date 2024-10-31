import numpy as np
import scipy.sparse as sparse

class delta_rad():
    '''
    Class to compute radiance from a given atmospheric profile using the delta-Eddington radiative transfer 
    approximation. Based on the "deddington" code from Hayne et al. (2012), adapted for Python by R. W. Stevens.

    Last updated: 10/30/24

    Attributes
    ----------
    wl : array
        Wavelengths of interest :: [um]
        1D array of shape (number of wavelengths). Can be a float if only considering one wavlength
    ztan : array
        Tangent altitudes of each observation (negative for on-planet observations) :: [km]
        1D array of shape (number of observational tangent points). Can be a float if only considering one tangent 
        altitude
    alt : array
        Altitude of each atmospheric layer :: [km]
        1D array of shape (number of atmospheric layers)
    t : array
        Temperature within each atmospheric layer :: [K]
        1D array of same shape as alt
    opac : array
        Opacity within each atmospheric layer--NOT optical thickness!! :: [km^-1]
        1D array of same shape as alt
    dz : array
        Thickness of each atmospheric layer :: [km]
        1D array of same shape as alt
    qext : array
        Extinction efficiencies in each atmospheric layer at each wavelength. 
        2D array of shape (number of wavelengths, number of atmospheric layers). If 1D, assumes only one wavelength is 
        provided.
    w : array
        Single scattering albedo in each atmospheric layer at each wavelength
        2D array of same shape as qext.
    g : array
        Asymmetry parameter in each atmospheric layer at each wavelength
        2D array of same shape as qext.
    qext_ref : float
        Extinction efficiency at the reference wavelength opacities are provided in
    zsc : float
        Spacecraft altitude :: [km]
    t_surf : float
        Surface temperature :: [K]
    e_surf : array
        Surface emissivity
        1D array of same shape as wl. Can be a float if only one wavelength is provided.
    depth : float
        Fractional depth within each layer. Tells you how close the grid point containing all the atmospheric 
        information is to the top or bottom of the layer. Default is 0.5 which means the grid point is in the middle 
        of the layer. Probably don't need to mess with this, but it's here because I'm paranoid.
    
    Methods
    -------
    planck_wn(wl, t)
        Computes the Planck function at a given wavelength and temperature.
    transform_g(g)
        Transforms g to g' for use in the delta-Eddington approximation (see appendix in Hayne et al. (2012)).
    transform_w(w, g)
        Transforms w to w' for use in the delta-Eddington approximation.
    transform_tau(tau, w, g)
        Transforms tau to tau' for use in the delta-Eddington approximation.
    path_length(zbot, ztop, ztan)
        Computes the path length through the atmosphere between two altitudes for a given tangent altitude.
    determine_ray_geom(ztan, alt, dz)
        Computes the differential path length within each atmospheric layer for the observational ray associated 
        with each tangent altitude.
    get_ctau(opac, dz)
        Computes the normal optical depth (NOT thickness or opacity) within each atmospheric layer.
    get_sftau(opac, ds, depth)
        Computes the optical depth (NOT thickness or opacity) within each atmospheric layer for each tangent point. 
        This is different from the 'get_ctau' method in that it accounts for the true observational geometry instead 
        of assuming nadir viewing.
    compute_c(ctau_prime, bb, w_prime, g_prime, bb_surf, e_surf)
        Computes the coefficients used to calculate the local intensity field within each atmospheric layer for each 
        tangent altitude by solving the 2N system of equations outlined in Hayne et al. (2012).
    get_mu(ztan, alt)
        Computes the cosine of the emission angle within each atmospheric layer for each tangent altitude.
    get_sf(c, ctau_prime, w_prime, g_prime, mu, p, k, bb)
        Computes the source function within each atmospheric layer for each tangent altitude.
    int_sf(c, ctau_prime, sftau_prime, w_prime, g_prime, mu, p, k, bb, extra, bb_surf, e_surf)
        Integrates the source function over the full optical depth to compute the radiance associated with each 
        tangent altitude.
    compute_radiance()
        Calls all the methods above to compute the radiance associated with each tangent altitude at each 
        wavelength provided.
    '''

    def __init__(self, wl, ztan, alt, t, opac, dz, qext, w, g, qext_ref, zsc, t_surf, e_surf, depth=.5):
        '''
        Defines all necessary parameters to be used.

        Parameters
        ----------
        wl : array
            Wavelengths of interest :: [um]
            1D array of shape (number of wavelengths). Can be a float if only considering one wavlength
        ztan : array
            Tangent altitudes of each observation (negative for on-planet observations) :: [km]
            1D array of shape (number of observational tangent points). Can be a float if only considering one 
            tangent altitude
        alt : array
            Altitude of each atmospheric layer :: [km]
            1D array of shape (number of atmospheric layers)
        t : array
            Temperature within each atmospheric layer :: [K]
            1D array of same shape as alt
        opac : array
            Opacity within each atmospheric layer--NOT optical thickness!! :: [km^-1]
            1D array of same shape as alt
        dz : array
            Thickness of each atmospheric layer :: [km]
            1D array of same shape as alt
        qext : array
            Extinction efficiencies in each atmospheric layer at each wavelength.
            2D array of shape (number of wavelengths, number of atmospheric layers). If 1D, assumes only one 
            wavelength is provided.
        w : array
            Single scattering albedo in each atmospheric layer at each wavelength
            2D array of same shape as qext.
        g : array
            Asymmetry parameter in each atmospheric layer at each wavelength
            2D array of same shape as qext.
        qext_ref : float
            Extinction efficiency at the reference wavelength opacities are provided in
        zsc : float
            Spacecraft altitude :: [km]
        t_surf : float
            Surface temperature :: [K]
        e_surf : array
            Surface emissivity
            1D array of same shape as wl. Can be a float if only one wavelength is provided.
        depth : float
            Fractional depth within each layer. Tells you how close the grid point containing all the atmospheric 
            information is to the top or bottom of the layer. Default is 0.5 which means the grid point is in the 
            middle of the layer. Probably don't need to mess with this, but it's here because I'm paranoid.
        '''

        self.wl  = np.array([wl])
        if len(self.wl.shape) > 1:
            # Allows code to handle both single wavelengths and arrays of wavelengths
            self.wl = self.wl[0]
            qext = np.array(qext)[:, None]
        
        self.ztan = np.array([ztan])
        if len(self.ztan.shape) > 1:
            # Allows code to handle both single tangent altitudes and arrays of tangent altitudes
            self.ztan = self.ztan[0]
            qext = np.resize(qext, (self.wl.shape[0], self.ztan.shape[0]))[:, :, None]
        
        # Resize arrays to have shape (number of wavelengths, number of tangent altitudes, number of atmospheric 
        # layers) to allow for vectorization of code
        self.t    = np.resize(t, (self.wl.shape[0], self.ztan.shape[0], t.shape[0]))
        self.opac = np.transpose(np.resize(opac / qext_ref * qext, (self.ztan.shape[0], self.wl.shape[0], t.shape[0])), 
                                 axes=(1, 0, 2))
        self.alt  = np.resize(alt, (self.wl.shape[0], self.ztan.shape[0], t.shape[0]))
        self.dz   = np.resize(dz, (self.wl.shape[0], self.ztan.shape[0], t.shape[0]))
        self.w    = np.resize(w, (t.shape[0], self.ztan.shape[0], self.wl.shape[0])).T
        self.g    = np.resize(g, (t.shape[0], self.ztan.shape[0], self.wl.shape[0])).T
        
        self.qext     = qext
        self.qext_ref = qext_ref
        self.zsc      = zsc
        self.t_surf   = t_surf
        self.e_surf   = np.array([e_surf])
        if len(self.e_surf.shape) > 1:
            self.e_surf = self.e_surf[0]
        self.depth = depth

        # Constants
        self.h     = 6.62607015e-34 # Planck constant [J s]
        self.c     = 2.99792458e10  # speed of light [cm / s]
        self.kB    = 1.380649e-23   # Boltzmann constant [J / K]
        self.Rmars = 3398.0         # radius of Mars [km]
    
    def planck_wn(self, wl, t):
        '''
        Computes the Planck function at a given wavelength and temperature.

        Parameters
        ----------
        wl : array
            Wavelengths of interest :: [um]
            1D array of shape (number of wavelengths). Can be a float if only considering one wavlength
        t : array
            Temperature within each atmospheric layer :: [K]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Returns
        -------
        B : array
            Black body radiance :: [mW / (m^2 sr cm^-1)]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        '''

        wn = 10**4 / wl # Convert wavelength to wavenumber

        # Constants
        alpha1 = 1.191066e-005
        alpha2 = 1.438833

        return(alpha1 * (wn * wn * wn)[:, None, None] / (np.exp(alpha2 * wn[:, None, None] / t) - 1.0))
    
    def transform_g(self, g):
        '''
        Transforms g to g' for use in the delta-Eddington approximation (see appendix in Hayne et al. (2012)).

        Parameters
        ----------
        g : array
            Asymmetry parameter in each atmospheric layer at each wavelength
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Returns
        -------
        g_prime : array
            Delta-Eddington asymmetry parameter
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        '''

        g_prime = g / (1.0 + g)

        return(g_prime)
    
    def transform_w(self, w, g):
        '''
        Transforms w to w' for use in the delta-Eddington approximation (see appendix in Hayne et al. (2012)).

        Parameters
        ----------
        w : array
            Single scattering albedo in each atmospheric layer at each wavelength
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Returns
        -------
        w_prime : array
            Delta-Eddington single scattering albedo
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        '''

        w_prime = (1.0 - g**2) * w / (1.0 - (g**2 * w))

        return(w_prime)
    
    def transform_tau(self, tau, w, g):
        '''
        Transforms tau to tau' for use in the delta-Eddington approximation (see appendix in Hayne et al. (2012)).

        Parameters
        ----------
        tau : array
            Optical depth in each atmospheric layer at each wavelength
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Returns
        -------
        tau_prime : array
            Delta-Eddington optical depth
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        '''

        tau_prime = (1.0 - (g**2 * w)) * tau

        return(tau_prime)
    
    def path_length(self, zbot, ztop, ztan):
        '''
        Computes the path length through the atmosphere between two altitudes for a given tangent altitude.

        Parameters
        ----------
        zbot : array
            Altitude of the bottom of the atmospheric layer :: [km]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        ztop : array
            Altitude of the top of the atmospheric layer :: [km]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        ztan : array
            Tangent altitudes of each observation (negative for on-planet observations) :: [km]
            1D array of shape (number of observational tangent points)
        
        Returns
        -------
        ds : array
            Differential path lengths between zbot and ztop for tangent altitude associated with each 
            observation :: [km]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        '''

        short_path = np.sqrt((self.Rmars + zbot)**2 - (self.Rmars + ztan[None, :, None])**2)
        long_path = np.sqrt((self.Rmars + ztop)**2 - (self.Rmars + ztan[None, :, None])**2)

        ds = long_path - short_path

        return(ds)
    
    def determine_ray_geom(self, ztan, alt, dz):
        '''
        Compuetes the differential path length within each atmospheric layer for the observational ray associated
        with each tangent altitude.

        Parameters
        ----------
        ztan : array
            Tangent altitudes of each observation (negative for on-planet observations) :: [km]
            1D array of shape (number of observational tangent points)
        alt : array
            Altitude of each atmospheric layer :: [km]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        dz : array
            Thickness of each atmospheric layer :: [km]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Returns
        -------
        ds : array
            Differential path lengths within each atmospheric layer for each tangent altitude :: [km]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Notes
        -----
        The differential path length is computed by determining the altitudes at which the ray intersects the
        bottom and top of each atmospheric layer. The path length is then computed as the difference between
        these two altitudes. The path length is adjusted for limb observations by doubling the path length
        at the lowest (tangent) altitude.
        '''

        zb = alt - .5 * dz
        zt = alt + .5 * dz

        ztan_temp = np.ones(alt.shape) * ztan[None, :, None] # Create a temporary array for ztan with same shape as alt

        zb[zb < ztan_temp] = ztan_temp[zb < ztan_temp] # Set altitude of bottom of layer to ztan if below
        zt[zt > self.zsc] = self.zsc                   # Set altitude of top of layer to zsc if above

        ds = self.path_length(zb, zt, ztan)

        # Double path length of bottommost layer (tangent layer) for limb observations
        if self.limb_flag == True:
            ds[:, :, np.argmin(alt)] = 2.0 * ds[:, :, np.argmin(alt)]
        
        return(ds)
    
    def get_ctau(self, opac, dz):
        '''
        Computes the normal optical depth (NOT thickness or opacity) within each atmospheric layer.

        Parameters
        ----------
        opac : array
            Opacity within each atmospheric layer--NOT optical thickness!! :: [km^-1]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        dz : array
            Thickness of each atmospheric layer :: [km]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Returns
        -------
        ctau : array
            Normal optical depth within each atmospheric layer
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Notes
        -----
        The normal optical depth is calculated by summing the normal optical thickness of each layer, which is 
        defined as the product of the opacity and the thickness of the layer.
        '''

        ctau = np.zeros(opac.shape)
        
        ot = opac * dz # Convert from opacity to normal optical thickness
        
        # The topmost layer is in contact with space, so optical thickness is defined seperately
        ctau[:, :, 0] = .5 * ot[:, :, 0]
        
        # Calculate optical thickness in each subsequent layer
        for i, element in enumerate(ctau[0, 0, 1:]):
            ctau[:, :, i + 1] = ctau[:, :, i] + (.5 * ot[:, :, i + 1]) + (.5 * ot[:, :, i])
        
        return(ctau)
    
    def get_sftau(self, opac, ds, depth=.5):
        '''
        Competes the optical depth (NOT thickness or opacity) within each atmospheric layer for each tangent point.
        This is different from the 'get_ctau' method in that it accounts for the true observational geometry instead
        of assuming nadir viewing.

        Parameters
        ----------
        opac : array
            Opacity within each atmospheric layer--NOT optical thickness!! :: [km^-1]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        ds : array
            Differential path lengths within each atmospheric layer for each tangent altitude :: [km]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        depth : float
            Fractional depth within each layer. Tells you how close the grid point containing all the atmospheric 
            information is to the top or bottom of the layer. Default is 0.5 which means the grid point is in the 
            middle of the layer. Probably don't need to mess with this, but it's here because I'm paranoid.
        
        Returns
        -------
        sftau : array
            Optical depth within each atmospheric layer for each tangent point
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Notes
        -----
        The optical depth is calculated by summing the optical thickness of each layer, which is defined as the
        product of the opacity and the differential path length through the layer.
        '''

        sftau = np.zeros(opac.shape)
        
        ot = opac * ds # Convert from opacity to optical thickness

        # The topmost layer is in contact with space, so optical thickness is defined separately
        sftau[:, :, 0] = depth * ot[:, :, 0]

        # Calculate optical thickness in each subsequent layer
        for i, element in enumerate(sftau[0, 0, 1:]):
            sftau[:, :, i + 1] = sftau[:, :, i] + ((1.0 - depth) * ot[:, :, i]) + (depth * ot[:, :, i + 1])
        
        # We lose any contribution from the bottom fraction of the lowest layer, so add it back in
        sftau[:, :, -1] = sftau[:, :, -1] + ((1.0 - depth) * ot[:, :, -1])
        
        return(sftau)
    
    def compute_c(self, ctau_prime, bb, w_prime, g_prime, bb_surf, e_surf):
        '''
        Competes the coefficients used to calculate the local intensity field within each atmospheric layer for each
        tangent altitude by solving the 2N system of equations outlined in Hayne et al. (2012).

        Parameters
        ----------
        ctau_prime : array
            Normal optical depth within each atmospheric layer at each wavelength
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        bb : array
            Black body radiance within each atmospheric layer at each wavelength
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        w_prime : array
            Delta-Eddington single scattering albedo
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        g_prime : array
            Delta-Eddington asymmetry parameter
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        bb_surf : array
            Black body radiance at the surface
            1D array of shape (number of wavelengths)
        e_surf : array
            Surface emissivity
            1D array of shape (number of wavelengths)
        
        Returns
        -------
        c_arr : array
            C-coefficients used to calculate the local intensity field within each atmospheric layer for each tangent 
            altitude
            4D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers, 2)
            Index[:, :, :, 0] contains all C_1 coefficients and Index[:, :, :, 1] contains all C_2 coefficients
        p : array
            P-coefficient used in the delta-Eddington approximation
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        k : array
            kappa-coefficients used in the delta-Eddington approximation
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Notes
        -----
        The C-coefficients are computed by solving a sparse linear system of equations Ax=B. The system is constructed
        by requiring continuity of the intensity field at the boundaries of each layer and accounting for the
        boundary conditions at the top and bottom of the atmosphere (see Hayne et al. (2012) for details).
        '''

        # Compute the P and kappa coefficients
        p = np.sqrt((3.0 * (1.0 - w_prime)) / (1.0 - (w_prime * g_prime)))
        k = np.sqrt((3.0 * (1.0 - w_prime)) * (1.0 - (w_prime * g_prime)))
        
        # Create the x and y indices for the sparse matrix
        x1 = np.array([i for i in range(1, (2 * ctau_prime.shape[2]) - 1) for _ in range(4)])
        x = np.zeros(x1.shape[0] + 4)
        x[2:-2] = x1
        x[-2:] = x[-3] + 1
        
        y1 = np.array([(2 * int(i / 8)) + i % 4 for i in range((2 * (ctau_prime.shape[2] - 1)) * 4)])
        y = np.zeros(y1.shape[0] + 4)
        y[2:-2] = y1
        y[1] = 1
        y[-2:] = [y1[-2], y1[-1]]

        # Useful exponential functions
        emkt = np.exp(-k * ctau_prime)
        epkt = np.exp(k * ctau_prime)
        pemkt = p * emkt
        pepkt = p * epkt

        c_arr = np.zeros((ctau_prime.shape[0], ctau_prime.shape[1], ctau_prime.shape[2], 2))

        # Loop over wavelengths
        for i in range(ctau_prime.shape[0]):
            # Inner continuity between layers condition, looping over tangent altitudes
            v1 = np.array([[emkt[i, 0, j], epkt[i, 0, j], -emkt[i, 0, j], -epkt[i, 0, j], pemkt[i, 0, j], 
                            -pepkt[i, 0, j], -pemkt[i, 0, j], pepkt[i, 0, j]] 
                            for j in range(ctau_prime.shape[2] - 1)]).ravel()
            
            v = np.zeros(v1.shape[0] + 4)
            v[2:-2] = v1

            # Top of atmosphere boundary condition
            v[:2] = [1.0 + (p[i, 0, 0] * 2.0 / 3.0), 1.0 - (p[i, 0, 0] * 2.0 / 3.0)]
            # Surface boundary condition
            v[-2:] = [emkt[i, 0, -1] * (1.0 - ((2.0 / 3.0) * p[i, 0, -1])) - ((1.0 - e_surf[i]) * 
                                                                              (1.0 + ((2.0 / 3.0) * p[i, 0, -1]))), 
                      epkt[i, 0, -1] * (1.0 + ((2.0 / 3.0) * p[i, 0, -1])) - ((1.0 - e_surf[i]) * 
                                                                              (1.0 - ((2.0 / 3.0) * p[i, 0, -1])))]
            
            # Construct the A matrix
            a1_arr = sparse.csr_matrix((v, (x, y)), shape=(2 * w_prime.shape[2], 2 * w_prime.shape[2]))

            # Construct the B matrix
            b1_arr = np.zeros(2 * ctau_prime.shape[2])
            b1_arr[1:-1:2] = bb[i, 0, 1:] - bb[i, 0, :-1]        # Inner continuity condition
            b1_arr[0] = -bb[i, 0, 0]                             # Top of atmosphere boundary condition
            b1_arr[-1] = e_surf[i] * (bb_surf[i] - bb[i, 0, -1]) # Surface boundary condition

            c = sparse.linalg.spsolve(a1_arr, b1_arr)
            c_arr[i, :, :, :] = np.reshape(c, (ctau_prime.shape[2], 2))

        return(c_arr, p, k)
    
    def get_mu(self, ztan, alt):
        '''
        Competes the cosine of the emission angle within each atmospheric layer for each tangent altitude.

        Parameters
        ----------
        ztan : array
            Tangent altitudes of each observation (negative for on-planet observations) :: [km]
            1D array of shape (number of observational tangent points)
        alt : array
            Altitude of each atmospheric layer :: [km]
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Returns
        -------
        mu : array
            Cosine of the emission angle within each atmospheric layer for each tangent altitude
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Notes
        -----
        This mu corresponds to mu' in Hayne et al. (2012), which is positive for downward radiance and negative for 
        upward radiance.
        '''

        geom = (ztan[None, :, None] + self.Rmars) / (alt + self.Rmars)
        geom[geom > 1.0] = 1.0 # Geom > 1.0 is not physical
        
        theta = np.arcsin(geom)
        
        mu = -np.cos(theta)
        
        # Adjust mu for limb observations where emission angles are flipped for everything past the tangent layer
        if self.limb_flag == True:
            mu[:, :, np.argmin(alt) + 1:] = -mu[:, :, np.argmin(alt) + 1:]
        
        return(mu)
    
    def get_sf(self, c, ctau_prime, w_prime, g_prime, mu, p, k, bb):
        '''
        Computes the source function within each atmospheric layer for each tangent altitude.

        Parameters
        ----------
        c : array
            C-coefficients used to calculate the local intensity field within each atmospheric layer for each tangent 
            altitude
            4D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers, 2)
        ctau_prime : array
            Normal optical depth within each atmospheric layer at each wavelength
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        w_prime : array
            Delta-Eddington single scattering albedo
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        g_prime : array
            Delta-Eddington asymmetry parameter
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        mu : array
            Cosine of the emission angle within each atmospheric layer for each tangent altitude
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        p : array
            P-coefficient used in the delta-Eddington approximation
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        k : array
            kappa-coefficients used in the delta-Eddington approximation
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        bb : array
            Black body radiance within each atmospheric layer at each wavelength
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        
        Returns
        -------
        sf : array
            Source function within each atmospheric layer for each tangent altitude
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        '''

        I0 = (c[:, :, :, 0] * np.exp(-k * ctau_prime)) + (c[:, :, :, 1] * np.exp(k * ctau_prime)) + bb
        I1 = p * ((c[:, :, :, 0] * np.exp(-k * ctau_prime)) - (c[:, :, :, 1] * np.exp(k * ctau_prime)))
        
        return((w_prime * I0) - (w_prime * g_prime * mu * I1) + ((1.0 - w_prime) * bb))
    
    def int_sf(self, c, ctau_prime, sftau_prime, w_prime, g_prime, mu, p, k, bb, extra, bb_surf, e_surf):
        '''
        Itegrates the source function over the full optical depth to compute the radiance associated with each
        tangent altitude.

        Parameters
        ----------
        c : array
            C-coefficients used to calculate the local intensity field within each atmospheric layer for each tangent 
            altitude
            4D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers, 2)
        ctau_prime : array
            Delta-Eddington normal optical depth within each atmospheric layer at each wavelength
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        sftau_prime : array
            Delta-Eddington optical depth within each atmospheric layer for each tangent point
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        w_prime : array
            Delta-Eddington single scattering albedo
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        g_prime : array
            Delta-Eddington asymmetry parameter
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        mu : array
            Cosine of the emission angle within each atmospheric layer for each tangent altitude
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        p : array
            P-coefficient used in the delta-Eddington approximation
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        k : array
            kappa-coefficients used in the delta-Eddington approximation
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        bb : array
            Black body radiance within each atmospheric layer at each wavelength
            3D array of shape (number of wavelengths, number of tangent altitudes, number of atmospheric layers)
        extra : array
            Extra normal optical depth from the bottom 1/2 of the lowest atmospheric layer
            2D array of shape (number of wavelengths, number of tangent altitudes)
        bb_surf : array
            Black body radiance at the surface
            1D array of shape (number of wavelengths)
        e_surf : array
            Surface emissivity
            1D array of shape (number of wavelengths)
        
        Returns
        -------
        R : array
            Radiance associated with each tangent altitude at each wavelength :: [mW / (m^2 sr cm^-1)]
            2D array of shape (number of wavelengths, number of tangent altitudes)
        '''

        # Calculate source function
        sf = self.get_sf(c, ctau_prime, w_prime, g_prime, mu, p, k, bb)

        # If limb observation, neglect surface contribution
        if self.limb_flag == False:
            surface_intensity = (np.exp(-sftau_prime[:, :, -1]) * 
                                 ((1.0 - e_surf[:, None]) * 
                                  ((c[:, :, -1, 0] * np.exp(-k[:, :, -1] * (ctau_prime[:, :, -1] + extra)) * 
                                    (1.0 + ((2.0 / 3.0) * p[:, :, -1]))) + 
                                    (c[:, :, -1, 1] * np.exp(k[:, :, -1] * (ctau_prime[:, :, -1] + extra)) * 
                                     (1.0 - ((2.0 / 3.0) * p[:, :, -1]))) + bb[:, :, -1]) + 
                                     (e_surf * bb_surf)[:, None]))
            
            return(np.trapz(sf * np.exp(-sftau_prime), sftau_prime) + surface_intensity)
        
        else:
            return(np.trapz(sf * np.exp(-sftau_prime), sftau_prime))
    
    def compute_radiance(self):
        '''
        Wrapper function to calculate the radiance associated with each tangent altitude at each wavelength by using 
        the delta-Eddington approximation to solve the radiative transfer equation.

        Notes
        -----
        This function calls all the necessary methods to compute the radiance, and can handle nadir, off-nadir, and 
        limb viewing geometries.

        The final output is stored in the 'rad' attribute of the class with units [mW / (m^2 sr cm^-1)].
        '''

        self.rad = np.zeros((self.wl.shape[0], self.ztan.shape[0]))
        
        # Compute black body radiance from both atmosphere and surface
        self.bb      = self.planck_wn(self.wl, self.t)
        self.bb_surf = self.planck_wn(self.wl, self.t_surf).ravel()
        
        # Transform Mie parameters for delta-Eddington approximation
        self.g_prime = self.transform_g(self.g)
        self.w_prime = self.transform_w(self.w, self.g)
        
        # Compute and transform normal optical depths
        self.ctau       = self.get_ctau(self.opac, self.dz)
        self.ctau_prime = self.transform_tau(self.ctau, self.w, self.g)
        
        # Calculate delta-Eddington intensity coefficients
        self.c, self.p, self.k = self.compute_c(self.ctau_prime, self.bb, self.w_prime, self.g_prime, self.bb_surf, 
                                                self.e_surf)
        
        # Split tangent altitudes into on-planet and limb observations
        self.ztan_op   = self.ztan[self.ztan <= 0.0]
        self.ztan_limb = self.ztan[self.ztan > 0.0]

        # Calculate radiance for on-planet observations
        if self.ztan_op.shape[0] > 0:
            self.limb_flag = False

            # Extract atmospheric quantities associated with on-planet observations
            self.alt_op        = self.alt[:, self.ztan <= 0.0]
            self.dz_op         = self.dz[:, self.ztan <= 0.0]
            self.opac_op       = self.opac[:, self.ztan <= 0.0]
            self.w_op          = self.w[:, self.ztan <= 0.0]
            self.g_op          = self.g[:, self.ztan <= 0.0]
            self.bb_op         = self.bb[:, self.ztan <= 0.0]
            self.g_prime_op    = self.g_prime[:, self.ztan <= 0.0]
            self.w_prime_op    = self.w_prime[:, self.ztan <= 0.0]
            self.ctau_prime_op = self.ctau_prime[:, self.ztan <= 0.0]
            self.c_op          = self.c[:, self.ztan <= 0.0]
            self.p_op          = self.p[:, self.ztan <= 0.0]
            self.k_op          = self.k[:, self.ztan <= 0.0]

            # Calculate viewing geometry and path lengths
            self.mu_op = self.get_mu(self.ztan_op, self.alt_op + .5 * self.dz_op)
            self.ds_op = self.determine_ray_geom(self.ztan_op, self.alt_op, self.dz_op)
            
            # Compute and transform observational optical depths
            self.sftau_op = self.get_sftau(self.opac_op, self.ds_op, depth=self.depth)
            self.sftau_prime_op = self.transform_tau(self.sftau_op, self.w_op, self.g_op)
            
            # Extra optical depth from bottom 1/2 of lowermost layer
            self.extra_op = self.transform_tau(.5 * self.opac_op[:, :, -1] * self.dz_op[:, :, -1], self.w_op[:, :, -1], 
                                               self.g_op[:, :, -1])
            
            # Calculate atmospheric radiance
            self.rad_op = self.int_sf(self.c_op, self.ctau_prime_op, self.sftau_prime_op, self.w_prime_op, 
                                      self.g_prime_op, self.mu_op, self.p_op, self.k_op, self.bb_op, self.extra_op, 
                                      self.bb_surf, self.e_surf)
            
            self.rad[:, :self.ztan_op.shape[0]] = self.rad_op
        
        # Calculate radiance for limb observations
        if self.ztan_limb.shape[0] > 0:
            self.limb_flag = True
            self.rad_limb = np.zeros((self.wl.shape[0], self.ztan_limb.shape[0]))

            # Extract atmospheric quantities associated with limb observations
            self.alt_limb        = self.alt[:, self.ztan > 0.0]
            self.dz_limb         = self.dz[:, self.ztan > 0.0]
            self.opac_limb       = self.opac[:, self.ztan > 0.0]
            self.w_limb          = self.w[:, self.ztan > 0.0]
            self.g_limb          = self.g[:, self.ztan > 0.0]
            self.bb_limb         = self.bb[:, self.ztan > 0.0]
            self.g_prime_limb    = self.g_prime[:, self.ztan > 0.0]
            self.w_prime_limb    = self.w_prime[:, self.ztan > 0.0]
            self.ctau_prime_limb = self.ctau_prime[:, self.ztan > 0.0]
            self.c_limb          = self.c[:, self.ztan > 0.0]
            self.p_limb          = self.p[:, self.ztan > 0.0]
            self.k_limb          = self.k[:, self.ztan > 0.0]
            
            # This loop can't be fully vectorized because we need to exclude all atmospheric layers below the tangent 
            # layer, which produces arrays of different sizes for each tangent altitude.
            for i, element in enumerate(self.ztan_limb):
                # Determine the indices of the atmospheric layers that contribute to the limb observation
                self.limb_ind = np.where(np.logical_and(self.alt[0, i] + (.5 * self.dz[0, i]) > self.ztan_limb[i], 
                                                        self.alt[0, i] - (.5 * self.dz[0, i]) <= self.zsc))[0]
                shape = self.bb_limb[:, i, self.limb_ind].shape[1] - 1

                # Pad all atmospheric quantities to account for the limb observation geometry such that all layers 
                # after the tangent layer mirror the layers before (if you plot alt_limb1, it should look like a V)
                self.bb_limb1         = np.pad(self.bb_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.alt_limb1        = np.pad(self.alt_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.dz_limb1         = np.pad(self.dz_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.opac_limb1       = np.pad(self.opac_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.g_limb1          = np.pad(self.g_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.w_limb1          = np.pad(self.w_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.g_prime_limb1    = np.pad(self.g_prime_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.w_prime_limb1    = np.pad(self.w_prime_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.c_limb1          = np.pad(self.c_limb[:, i, self.limb_ind], ((0, 0), (0, shape), (0, 0)), 
                                               'reflect')[:, None, :]
                self.p_limb1          = np.pad(self.p_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.k_limb1          = np.pad(self.k_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                self.ctau_prime_limb1 = np.pad(self.ctau_prime_limb[:, i, self.limb_ind], ((0, 0), (0, shape)), 
                                               'reflect')[:, None, :]
                
                # Calculate viewing geometry and path lengths
                self.mu_limb = self.get_mu(np.array([self.ztan_limb[i]]), self.alt_limb1 + .5 * self.dz_limb1)
                self.ds_limb = self.determine_ray_geom(np.array([self.ztan_limb[i]]), self.alt_limb1, self.dz_limb1)

                # Compute and transform observational optical depths
                self.sftau_limb = self.get_sftau(self.opac_limb1, self.ds_limb, depth=self.depth)
                self.sftau_prime_limb = self.transform_tau(self.sftau_limb, self.w_limb1, self.g_limb1)

                # Extra optical depth from bottom 1/2 of lowermost layer (this isn't used in the limb calculation)
                self.extra_limb = self.transform_tau(.5 * self.opac_limb1[:, -1] * self.dz_limb1[:, -1], 
                                                     self.w_limb1[:, -1], self.g_limb1[:, -1])
                
                # Calculate atmospheric radiance
                self.rad_limb[:, i] = self.int_sf(self.c_limb1, self.ctau_prime_limb1, self.sftau_prime_limb, 
                                                  self.w_prime_limb1, self.g_prime_limb1, self.mu_limb, self.p_limb1, 
                                                  self.k_limb1, self.bb_limb1, self.extra_limb, self.bb_surf, 
                                                  self.e_surf)[:, 0]
            
            self.rad[:, self.ztan_op.shape[0]:] = self.rad_limb
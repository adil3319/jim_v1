from abc import ABC

import jax.numpy as jnp
from jaxtyping import Array, Float
from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc, gen_IMRPhenomD
from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc
from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc

def gen_IMRPhenomD_hphchb(f: Array, params: Array, f_ref: float):
    """
    Generate PhenomD frequency domain waveform following 1508.07253.
    vars array contains both intrinsic and extrinsic variables
    theta = [Mchirp, eta, chi1, chi2, D, tc, phic]
    Mchirp: Chirp mass of the system [solar masses]
    eta: Symmetric mass ratio [between 0.0 and 0.25]
    chi1: Dimensionless aligned spin of the primary object [between -1 and 1]
    chi2: Dimensionless aligned spin of the secondary object [between -1 and 1]
    D: Luminosity distance to source [Mpc]
    tc: Time of coalesence. This only appears as an overall linear in f contribution to the phase
    phic: Phase of coalesence
    inclination: Inclination angle of the binary [between 0 and PI]

    f_ref: Reference frequency for the waveform

    Returns:
    --------
      hp (array): Strain of the plus polarization
      hc (array): Strain of the cross polarization
    """
    iota = params[7]
    h0 = gen_IMRPhenomD(f, params, f_ref)

    hp = h0 * (1 / 2 * (1 + jnp.cos(iota) ** 2))
    hc = -1j * h0 * jnp.cos(iota)
    hb = jnp.sqrt(3/2)* h0 * (jnp.sin(iota))**2

    return hp, hc, hb

class Waveform(ABC):
    def __init__(self):
        return NotImplemented

    def __call__(
        self, axis: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        return NotImplemented


class RippleIMRPhenomD_ScalarTensor(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0, **kwargs):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
                params["alpha"],
                params["beta"],
                params["alphaB"],
            ]
        )
        hp, hc, hb = gen_IMRPhenomD_hphchb(frequency, theta, self.f_ref)
        cf =1.4765e3
        c=2.998e8
        u=(jnp.pi*(theta[0]/theta[1]**0.6)*frequency*cf/c)**(1/3) # cf =1.4765e3 m
        a,b=-2,-7
        hpT = hp*(1+theta[-3]*u**a)*jnp.exp(1.0j*theta[-2]*u**b)
        hcT = hc*(1+theta[-3]*u**a)*jnp.exp(1.0j*theta[-2]*u**b)
        hbT = hb*theta[-1]*jnp.exp(1.0j*theta[-2]*u**b)
        
        output["p"] = hpT
        output["c"] = hcT
        output["b"] = hbT
        return output

    def __repr__(self):
        return f"RippleIMRPhenomD_scalarTensor(f_ref={self.f_ref})"

class RippleIMRPhenomD(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0, **kwargs):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output
    def __repr__(self):
        return f"RippleIMRPhenomD(f_ref={self.f_ref})"


class RippleIMRPhenomPv2(Waveform):
    f_ref: float

    def __init__(self, f_ref: float = 20.0, **kwargs):
        self.f_ref = f_ref

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}
        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_x"],
                params["s1_y"],
                params["s1_z"],
                params["s2_x"],
                params["s2_y"],
                params["s2_z"],
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"RippleIMRPhenomPv2(f_ref={self.f_ref})"


class RippleTaylorF2(Waveform):

    f_ref: float
    use_lambda_tildes: bool

    def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False):
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}

        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )
        hp, hc = gen_TaylorF2_hphc(
            frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes
        )
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"RippleTaylorF2(f_ref={self.f_ref})"


class RippleIMRPhenomD_NRTidalv2(Waveform):

    f_ref: float
    use_lambda_tildes: bool

    def __init__(
        self,
        f_ref: float = 20.0,
        use_lambda_tildes: bool = False,
        no_taper: bool = False,
    ):
        """
        Initialize the waveform.

        Args:
            f_ref (float, optional): Reference frequency in Hz. Defaults to 20.0.
            use_lambda_tildes (bool, optional): Whether we sample over lambda_tilde and delta_lambda_tilde, as defined for instance in Equation (5) and Equation (6) of arXiv:1402.5156, rather than lambda_1 and lambda_2. Defaults to False.
            no_taper (bool, optional): Whether to remove the Planck taper in the amplitude of the waveform, which we use for relative binning runs. Defaults to False.
        """
        self.f_ref = f_ref
        self.use_lambda_tildes = use_lambda_tildes
        self.no_taper = no_taper

    def __call__(
        self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
    ) -> dict[str, Float[Array, " n_dim"]]:
        output = {}

        if self.use_lambda_tildes:
            first_lambda_param = params["lambda_tilde"]
            second_lambda_param = params["delta_lambda_tilde"]
        else:
            first_lambda_param = params["lambda_1"]
            second_lambda_param = params["lambda_2"]

        theta = jnp.array(
            [
                params["M_c"],
                params["eta"],
                params["s1_z"],
                params["s2_z"],
                first_lambda_param,
                second_lambda_param,
                params["d_L"],
                0,
                params["phase_c"],
                params["iota"],
            ]
        )

        hp, hc = gen_IMRPhenomD_NRTidalv2_hphc(
            frequency,
            theta,
            self.f_ref,
            use_lambda_tildes=self.use_lambda_tildes,
            no_taper=self.no_taper,
        )
        output["p"] = hp
        output["c"] = hc
        return output

    def __repr__(self):
        return f"RippleIMRPhenomD_NRTidalv2(f_ref={self.f_ref})"


waveform_preset = {
    "RippleIMRPhenomD": RippleIMRPhenomD,
    "RippleIMRPhenomPv2": RippleIMRPhenomPv2,
    "RippleTaylorF2": RippleTaylorF2,
    "RippleIMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
    "RippleIMRPhenomD_ScalarTensor":RippleIMRPhenomD_ScalarTensor,
}




# from abc import ABC

# import jax.numpy as jnp
# from jaxtyping import Array, Float
# from ripplegw.waveforms.IMRPhenomD import gen_IMRPhenomD_hphc
# from ripplegw.waveforms.IMRPhenomPv2 import gen_IMRPhenomPv2_hphc
# from ripplegw.waveforms.TaylorF2 import gen_TaylorF2_hphc
# from ripplegw.waveforms.IMRPhenomD_NRTidalv2 import gen_IMRPhenomD_NRTidalv2_hphc


# class Waveform(ABC):
#     def __init__(self):
#         return NotImplemented

#     def __call__(
#         self, axis: Float[Array, " n_dim"], params: dict[str, Float]
#     ) -> dict[str, Float[Array, " n_dim"]]:
#         return NotImplemented


# class RippleIMRPhenomD(Waveform):
#     f_ref: float

#     def __init__(self, f_ref: float = 20.0, **kwargs):
#         self.f_ref = f_ref

#     def __call__(
#         self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
#     ) -> dict[str, Float[Array, " n_dim"]]:
#         output = {}
#         theta = jnp.array(
#             [
#                 params["M_c"],
#                 params["eta"],
#                 params["s1_z"],
#                 params["s2_z"],
#                 params["d_L"],
#                 0,
#                 params["phase_c"],
#                 params["iota"],
#             ]
#         )
#         hp, hc = gen_IMRPhenomD_hphc(frequency, theta, self.f_ref)
#         output["p"] = hp
#         output["c"] = hc
#         return output

#     def __repr__(self):
#         return f"RippleIMRPhenomD(f_ref={self.f_ref})"


# class RippleIMRPhenomPv2(Waveform):
#     f_ref: float

#     def __init__(self, f_ref: float = 20.0, **kwargs):
#         self.f_ref = f_ref

#     def __call__(
#         self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
#     ) -> dict[str, Float[Array, " n_dim"]]:
#         output = {}
#         theta = jnp.array(
#             [
#                 params["M_c"],
#                 params["eta"],
#                 params["s1_x"],
#                 params["s1_y"],
#                 params["s1_z"],
#                 params["s2_x"],
#                 params["s2_y"],
#                 params["s2_z"],
#                 params["d_L"],
#                 0,
#                 params["phase_c"],
#                 params["iota"],
#             ]
#         )
#         hp, hc = gen_IMRPhenomPv2_hphc(frequency, theta, self.f_ref)
#         output["p"] = hp
#         output["c"] = hc
#         return output

#     def __repr__(self):
#         return f"RippleIMRPhenomPv2(f_ref={self.f_ref})"


# class RippleTaylorF2(Waveform):

#     f_ref: float
#     use_lambda_tildes: bool

#     def __init__(self, f_ref: float = 20.0, use_lambda_tildes: bool = False):
#         self.f_ref = f_ref
#         self.use_lambda_tildes = use_lambda_tildes

#     def __call__(
#         self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
#     ) -> dict[str, Float[Array, " n_dim"]]:
#         output = {}

#         if self.use_lambda_tildes:
#             first_lambda_param = params["lambda_tilde"]
#             second_lambda_param = params["delta_lambda_tilde"]
#         else:
#             first_lambda_param = params["lambda_1"]
#             second_lambda_param = params["lambda_2"]

#         theta = jnp.array(
#             [
#                 params["M_c"],
#                 params["eta"],
#                 params["s1_z"],
#                 params["s2_z"],
#                 first_lambda_param,
#                 second_lambda_param,
#                 params["d_L"],
#                 0,
#                 params["phase_c"],
#                 params["iota"],
#             ]
#         )
#         hp, hc = gen_TaylorF2_hphc(
#             frequency, theta, self.f_ref, use_lambda_tildes=self.use_lambda_tildes
#         )
#         output["p"] = hp
#         output["c"] = hc
#         return output

#     def __repr__(self):
#         return f"RippleTaylorF2(f_ref={self.f_ref})"


# class RippleIMRPhenomD_NRTidalv2(Waveform):

#     f_ref: float
#     use_lambda_tildes: bool

#     def __init__(
#         self,
#         f_ref: float = 20.0,
#         use_lambda_tildes: bool = False,
#         no_taper: bool = False,
#     ):
#         """
#         Initialize the waveform.

#         Args:
#             f_ref (float, optional): Reference frequency in Hz. Defaults to 20.0.
#             use_lambda_tildes (bool, optional): Whether we sample over lambda_tilde and delta_lambda_tilde, as defined for instance in Equation (5) and Equation (6) of arXiv:1402.5156, rather than lambda_1 and lambda_2. Defaults to False.
#             no_taper (bool, optional): Whether to remove the Planck taper in the amplitude of the waveform, which we use for relative binning runs. Defaults to False.
#         """
#         self.f_ref = f_ref
#         self.use_lambda_tildes = use_lambda_tildes
#         self.no_taper = no_taper

#     def __call__(
#         self, frequency: Float[Array, " n_dim"], params: dict[str, Float]
#     ) -> dict[str, Float[Array, " n_dim"]]:
#         output = {}

#         if self.use_lambda_tildes:
#             first_lambda_param = params["lambda_tilde"]
#             second_lambda_param = params["delta_lambda_tilde"]
#         else:
#             first_lambda_param = params["lambda_1"]
#             second_lambda_param = params["lambda_2"]

#         theta = jnp.array(
#             [
#                 params["M_c"],
#                 params["eta"],
#                 params["s1_z"],
#                 params["s2_z"],
#                 first_lambda_param,
#                 second_lambda_param,
#                 params["d_L"],
#                 0,
#                 params["phase_c"],
#                 params["iota"],
#             ]
#         )

#         hp, hc = gen_IMRPhenomD_NRTidalv2_hphc(
#             frequency,
#             theta,
#             self.f_ref,
#             use_lambda_tildes=self.use_lambda_tildes,
#             no_taper=self.no_taper,
#         )
#         output["p"] = hp
#         output["c"] = hc
#         return output

#     def __repr__(self):
#         return f"RippleIMRPhenomD_NRTidalv2(f_ref={self.f_ref})"


# waveform_preset = {
#     "RippleIMRPhenomD": RippleIMRPhenomD,
#     "RippleIMRPhenomPv2": RippleIMRPhenomPv2,
#     "RippleTaylorF2": RippleTaylorF2,
#     "RippleIMRPhenomD_NRTidalv2": RippleIMRPhenomD_NRTidalv2,
# }

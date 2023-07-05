#!/usr/bin/env python3


class BiomassComponent:
    """
    Convenience class to store properties associated with a biomass component,
    e.g. macromolecule pseudo-metabolite

    Attributes
    ----------
    metabolite_label : string
        Name of metabolite.
    metabolite_id : string
        Metabolite ID of metabolite in cobrapy model.
    pseudoreaction : string
        Reaction ID of producing pseudoreaction in cobrapy model.
    molecular_mass : float
        Molecular mass of metabolite, in g/mol.
    ablated_flux : float
        Flux of ablated biomass reaction (to prioritise the metabolite of
        interest), to be computed, in h-1.
    est_time : float
        Doubling time estimated from flux of ablated biomass reaction, to be
        computed, in hours.
    """

    def __init__(self, metabolite_label, metabolite_id, pseudoreaction, molecular_mass):
        self.metabolite_label = metabolite_label
        self.metabolite_id = metabolite_id
        self.pseudoreaction = pseudoreaction
        self.molecular_mass = molecular_mass

        self.ablated_flux = None  # h-1
        self.est_time = None  # h

    def get_est_time(self):
        """Estimate doubling time based on growth rate"""
        self.est_time = (self.molecular_mass / MW_BIOMASS) * (
            np.log(2) / self.ablated_flux
        )


Lipids = BiomassComponent(
    metabolite_label="lipid",
    metabolite_id="s_1096[c]",
    pseudoreaction="r_2108",
    molecular_mass=MW_LIPID,
)

Proteins = BiomassComponent(
    metabolite_label="protein",
    metabolite_id="s_3717[c]",
    pseudoreaction="r_4047",
    molecular_mass=MW_PROTEIN,
)

Carbohydrates = BiomassComponent(
    metabolite_label="carbohydrate",
    metabolite_id="s_3718[c]",
    pseudoreaction="r_4048",
    molecular_mass=MW_CARB,
)

DNA = BiomassComponent(
    metabolite_label="DNA",
    metabolite_id="s_3720[c]",
    pseudoreaction="r_4050",
    molecular_mass=MW_DNA,
)

RNA = BiomassComponent(
    metabolite_label="RNA",
    metabolite_id="s_3719[c]",
    pseudoreaction="r_4049",
    molecular_mass=MW_RNA,
)

Cofactors = BiomassComponent(
    metabolite_label="cofactor",
    metabolite_id="s_4205[c]",
    pseudoreaction="r_4598",
    molecular_mass=MW_COFACTOR,
)

Ions = BiomassComponent(
    metabolite_label="ion",
    metabolite_id="s_4206[c]",
    pseudoreaction="r_4599",
    molecular_mass=MW_ION,
)

# original, non-enzyme constrained model
# defined because it annoyingly uses different naming conventions

Lipids_orig = BiomassComponent(
    metabolite_label="lipid",
    metabolite_id="s_1096",
    pseudoreaction="r_2108",
    molecular_mass=MW_LIPID,
)

Proteins_orig = BiomassComponent(
    metabolite_label="protein",
    metabolite_id="s_3717",
    pseudoreaction="r_4047",
    molecular_mass=MW_PROTEIN,
)

Carbohydrates_orig = BiomassComponent(
    metabolite_label="carbohydrate",
    metabolite_id="s_3718",
    pseudoreaction="r_4048",
    molecular_mass=MW_CARB,
)

DNA_orig = BiomassComponent(
    metabolite_label="DNA",
    metabolite_id="s_3720",
    pseudoreaction="r_4050",
    molecular_mass=MW_DNA,
)

RNA_orig = BiomassComponent(
    metabolite_label="RNA",
    metabolite_id="s_3719",
    pseudoreaction="r_4049",
    molecular_mass=MW_RNA,
)

Cofactors_orig = BiomassComponent(
    metabolite_label="cofactor",
    metabolite_id="s_4205",
    pseudoreaction="r_4598",
    molecular_mass=MW_COFACTOR,
)

Ions_orig = BiomassComponent(
    metabolite_label="ion",
    metabolite_id="s_4206",
    pseudoreaction="r_4599",
    molecular_mass=MW_ION,
)

biomass_component_list_orig = [
    Lipids_orig,
    Proteins_orig,
    Carbohydrates_orig,
    DNA_orig,
    RNA_orig,
    Cofactors_orig,
    Ions_orig,
]

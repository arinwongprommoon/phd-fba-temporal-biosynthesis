#!/usr/bin/env python3
# TODO: Test whether these things actually work
# TODO: Document functions & methods

import cobra
import numpy as np
import pandas as pd

from collections import namedtuple
from wrap_timeout_decorator import *

CELL_DRY_MASS = 15e-12  # g
MW_CARB = 368.03795704972003  # g/mol
MW_DNA = 3.9060196439999997
MW_RNA = 64.04235752722991
MW_PROTEIN = 504.3744234012359
MW_COFACTOR = 4.832782477018401
MW_ION = 2.4815607543700002
MW_LIPID = 31.5659867112958
MW_BIOMASS = 979.24108756487

GROWTH_ID = "r_2111"
BIOMASS_ID = "r_4041"

AuxotrophProperties = namedtuple(
    "AuxotrophProperties", ["genes_to_delete", "exch_to_add"]
)

# List genes to delete and exchange reactions to add for each auxotroph
AUXOTROPH_DICT = {
    "BY4741": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YLR303W", "YEL021W"],
        [
            "r_1893",
            "r_1899",
            "r_1902",
            "r_2090",
            "r_1893_REV",
            "r_1899_REV",
            "r_1902_REV",
            "r_2090_REV",
        ],
    ),
    "BY4742": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YBR115C", "YEL021W"],
        [
            "r_1893",
            "r_1899",
            "r_1900",
            "r_2090",
            "r_1893_REV",
            "r_1899_REV",
            "r_1900_REV",
            "r_2090_REV",
        ],
    ),
}

# TODO: Add lookup tables:
# - Carbon sources/other nutrients
# - Genes (systematic vs common)


class BiomassComponent:
    def __init__(self, metabolite_label, metabolite_id, pseudoreaction, molecular_mass):
        self.metabolite_label = metabolite_label
        self.metabolite_id = metabolite_id
        self.pseudoreaction = pseudoreaction
        self.molecular_mass = molecular_mass  # g/mol

        self.ablated_flux = None  # h-1
        self.est_time = None  # h

    def get_est_time(self):
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


class Yeast8Model:
    def __init__(self, model_filepath):
        self.model_filepath = model_filepath
        # Load wild-type model
        if model_filepath.endswith(".xml"):
            self.model = cobra.io.read_sbml_model(model_filepath)
        else:
            raise Exception(
                "Invaild file format for model. Please use SBML model as .xml file."
            )
        # Unrestrict growth
        self.model.reactions.get_by_id(GROWTH_ID).bounds = (0, 1000)

        self.solution = None
        # TODO: Add
        # - way to store what's been done, e.g.
        #   auxotrophy, deletions, media (w/ some cross-validation)
        # - way to store fluxes/solution
        # - way to store results from ablation
        self.auxotrophy = None
        self.deleted_genes = []
        # TODO: getters/setters?

    def reset(self):
        self.model = cobra.io.read_sbml_model(self.model_filepath)

    def knock_out_list(self, genes_to_delete):
        for gene_id in genes_to_delete:
            self.model.genes.get_by_id(gene_id).knock_out()
        self.deleted_genes.append(genes_to_delete)

    def add_media_components(self, exch_to_unbound):
        for exch_id in exch_to_unbound:
            self.model.reactions.get_by_id(exch_id).bounds = (-1000, 0)

    def remove_media_components(self, exch_to_unbound):
        for exch_id in exch_to_unbound:
            self.model.reactions.get_by_id(exch_id).bounds = (0, 0)

    def make_auxotroph(self, auxo_strain, supplement_media=True):
        if auxo_strain in AUXOTROPH_DICT.keys():
            # Knock out genes to create auxotroph
            self.knock_out_list(AUXOTROPH_DICT[auxo_strain].genes_to_delete)
            # By default, supplements nutrients
            if supplement_media:
                self.add_media_components(
                    self.model, AUXOTROPH_DICT[auxo_strain].exch_to_add
                )
            self.auxotrophy = auxo_strain
        else:
            raise Exception("Invalid string for auxotroph strain background.")

    def optimize(self, model, timeout_time=60):
        @timeout(timeout_time)
        def _optimize_internal(model):
            return model.optimize()

        try:
            if model is None:
                model = self.model
            solution = _optimize_internal(model)
        except TimeoutError():
            raise Exception(f"Model optimisation timeout, {timeout_time} s")

        return solution

    def ablate(self):
        # Copy model -- needed to restore the un-ablated model to work with
        # in successive loops
        model_working = self.model.copy()

        # UN-ABLATED
        print("Original")
        fba_solution = self.optimize(model_working)
        original_flux = fba_solution.fluxes[GROWTH_ID]
        original_est_time = np.log(2) / original_flux
        # ABLATED
        # Set up lists
        biomass_component_list = [
            Lipids,
            Proteins,
            Carbohydrates,
            DNA,
            RNA,
            Cofactors,
            Ions,
        ]
        all_metabolite_ids = [
            biomass_component.metabolite_id
            for biomass_component in biomass_component_list
        ]
        all_pseudoreaction_ids = [
            (biomass_component.metabolite_label, biomass_component.pseudoreaction)
            for biomass_component in biomass_component_list
        ]
        all_pseudoreaction_ids.append(("biomass", BIOMASS_ID))
        all_pseudoreaction_ids.append(("objective", GROWTH_ID))

        for biomass_component in biomass_component_list:
            print(f"Prioritising {biomass_component.metabolite_label}")
            model_working = self.model.copy()

            # boilerplate: lookup
            to_ablate = all_metabolite_ids.copy()
            to_ablate.remove(biomass_component.metabolite_id)
            to_ablate_keys = [
                model_working.metabolites.get_by_id(metabolite_id)
                for metabolite_id in to_ablate
            ]
            to_ablate_dict = dict(zip(to_ablate_keys, [-1] * len(to_ablate_keys)))

            # ablate metabolites from biomass reaction
            model_working.reactions.get_by_id(BIOMASS_ID).subtract_metabolites(
                to_ablate_dict
            )
            # optimise model
            fba_solution = model_working.optimize()
            # store outputs
            biomass_component.ablated_flux = fba_solution.fluxes[GROWTH_ID]
            biomass_component.get_est_time()

        # construct output dataframe
        d = {
            "priority_component": ["original"]
            + [
                biomass_component.metabolite_label
                for biomass_component in biomass_component_list
            ],
            "flux": [original_flux]
            + [
                biomass_component.ablated_flux
                for biomass_component in biomass_component_list
            ],
            "est_time": [original_est_time]
            + [
                biomass_component.est_time
                for biomass_component in biomass_component_list
            ],
        }
        return pd.Dataframe(data=d)


# Takes dataframe output from ablation function as input
def ablation_barplot(ablated_df, ax):
    ax.bar(
        ablated_df.priority_component,
        ablated_df.est_time,
    )


def compare_fluxes(model1, model2):
    # Check if fluxes have already been computed.
    # If not, compute automatically.
    for model in [model1, model2]:
        if model.solution is None:
            model.solution = model.optimize()

    diff_fluxes = model2.solution.fluxes - model1.solution.fluxes
    nonzero_idx = diff_fluxes.to_numpy().nonzero()[0]
    diff_fluxes_nonzero = diff_fluxes[nonzero_idx]
    diff_fluxes_sorted = diff_fluxes_nonzero[
        diff_fluxes_nonzero.abs().sort_values(ascending=False).index
    ]

    return diff_fluxes_sorted

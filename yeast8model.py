#!/usr/bin/env python3

import cobra
from collections import namedtuple

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


class Yeast8Model:
    def __init__(self, model_filepath):
        # Load wild-type model
        if model_filepath.endswith(".xml"):
            self.wtmodel = cobra.io.read_sbml_model(model_filepath)
        else:
            raise Exception(
                "Invaild file format for model. Please use SBML model as .xml file."
            )
        # Copy model, to work with
        self.model = self.wtmodel.copy()

        # TODO: Add
        # - way to store what's been done, e.g.
        #   auxotrophy, deletions, media (w/ some cross-validation)
        # - way to store fluxes/solution
        # - way to store results from ablation

    def knock_out_list(model, genes_to_delete):
        for gene_id in genes_to_delete:
            model.genes.get_by_id(gene_id).knock_out()

    def add_media_components(model, exch_to_unbound):
        for exch_id in exch_to_unbound:
            model.reactions.get_by_id(exch_id).bounds = (-1000, 0)

    def remove_media_components(model, exch_to_unbound):
        for exch_id in exch_to_unbound:
            model.reactions.get_by_id(exch_id).bounds = (0, 0)

    def make_auxotroph(self, auxo_strain, supplement_media=True):
        if auxo_strain in AUXOTROPH_DICT.keys():
            # Knock out genes to create auxotroph
            self.knock_out_list(self.model, AUXOTROPH_DICT[auxo_strain].genes_to_delete)
            # By default, supplements nutrients
            if supplement_media:
                self.add_media_components(
                    self.model, AUXOTROPH_DICT[auxo_strain].exch_to_add
                )
        else:
            raise Exception("Invalid string for auxotroph strain background.")

    def wtoptimize(self):
        pass

    def optimize(self):
        pass

    def compare_fluxes(self):
        # Must have a way to check if fluxes have already been computed.
        # If not, compute automatically.
        pass

    def ablate(self):
        pass


# Should take some sort of output from ablation function as input
def ablation_barplot():
    pass

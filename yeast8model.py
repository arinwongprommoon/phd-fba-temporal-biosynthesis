#!/usr/bin/env python3

import cobra
import numpy as np
import os
import pandas as pd

from collections import namedtuple
from wrapt_timeout_decorator import *

# Constants needed for ablation calculations
# From Bionumbers
CELL_DRY_MASS = 15e-12  # g
# Computed from isa reactions in Yeast8 model,
# see molecular weights notebook
MW_CARB = 368.03795704972003  # g/mol
MW_DNA = 3.9060196439999997
MW_RNA = 64.04235752722991
MW_PROTEIN = 504.3744234012359
MW_COFACTOR = 4.832782477018401
MW_ION = 2.4815607543700002
MW_LIPID = 31.5659867112958
MW_BIOMASS = 979.24108756487

# Defaut IDs for growth and biomass reactions for batch Yeast8 model
GROWTH_ID = "r_2111"
BIOMASS_ID = "r_4041"

# List genes to delete and exchange reactions to add for each auxotroph
AuxotrophProperties = namedtuple(
    "AuxotrophProperties", ["genes_to_delete", "exch_to_add"]
)

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
    "BY4741_orig": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YLR303W", "YEL021W"],
        [
            "EX_his__L_e",
            "EX_met__L_e",
            "EX_leu__L_e",
            "EX_ura_e",
        ],
    ),
    "BY4742_orig": AuxotrophProperties(
        ["YOR202W", "YCL018W", "YBR115C", "YEL021W"],
        [
            "EX_his__L_e",
            "EX_met__L_e",
            "EX_lys__L_e",
            "EX_ura_e",
        ],
    ),
}

# TODO: Add lookup tables:
# - Carbon sources/other nutrients
# - Genes (systematic vs common)


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
    metabolite_id="protein_c",
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
    metabolite_id="dna_c",
    pseudoreaction="r_4050",
    molecular_mass=MW_DNA,
)

RNA_orig = BiomassComponent(
    metabolite_label="RNA",
    metabolite_id="rna_c",
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


class Yeast8Model:
    """
    Yeast8-derived model (cobrapy) with functions for modification.

    Attributes
    ----------
    model_filepath : string
        Filepath of model.  Should refer to a SBML-formatted .xml file.
    model : cobra.Model object
        Model.
    growth_id : string
        Reaction ID of growth reaction.
    biomass_id : string
        Reaction ID of biomass reaction.
    biomass_component_list : list of BiomassComponent objects
        List of biomass components in the growth subsystem of the model.
    solution : cobra.Solution object
        Optimisation (FBA) solution of model.
    auxotrophy : string
        Name of auxotrophic background strain, if applicable.
    deleted_genes : list of string
        List of genes deleted in model, beyond auxotrophy, if applicable.
    ablation_result : pandas.DataFrame object
        Results of ablation study.  Columns: 'priority component' (biomass
        component being prioritised), 'flux' (flux of ablated biomass reaction),
        'est_time' (estimated doubling time based on flux).  Rows: 'original'
        (un-ablated biomass), other rows indicate biomass component.
    """

    def __init__(self, model_filepath, growth_id=GROWTH_ID, biomass_id=BIOMASS_ID):
        # Needed for reset method
        self.model_filepath = model_filepath
        # Load wild-type model
        if model_filepath.endswith(".xml"):
            self.model = cobra.io.read_sbml_model(model_filepath)
        else:
            raise Exception(
                "Invaild file format for model. Please use SBML model as .xml file."
            )
        self.growth_id = growth_id
        self.biomass_id = biomass_id
        # Unrestrict growth
        self.model.reactions.get_by_id(growth_id).bounds = (0, 1000)
        print(f"Growth ({growth_id}) unrestricted.")

        # Biomass components
        self.biomass_component_list = [
            Lipids,
            Proteins,
            Carbohydrates,
            DNA,
            RNA,
            Cofactors,
            Ions,
        ]

        # TODO: getters/setters?
        self.solution = None
        self.auxotrophy = None
        # TODO: Validation of deleted genes?  i.e. check that bounds are truly
        # zero before defining
        self.deleted_genes = []
        self.ablation_result = None

    def reset(self):
        """Reset model to filepath"""
        self.model = cobra.io.read_sbml_model(self.model_filepath)

    def knock_out_list(self, genes_to_delete):
        """Knock out list of genes

        Parameters
        ----------
        genes_to_delete : list of string
            List of genes to delete, as systematic gene names.
        """
        for gene_id in genes_to_delete:
            try:
                self.model.genes.get_by_id(gene_id).knock_out()
            except KeyError as e:
                print(f"Error-- Cannot knock out. Gene not found: {gene_id}")
        self.deleted_genes.append(genes_to_delete)

    def add_media_components(self, exch_to_unbound):
        """Add media components

        Parameters
        ----------
        exch_to_unbound : list of string
            List of exchange reactions (associated with nutrient components) to
            remove bounds for.
        """
        for exch_id in exch_to_unbound:
            try:
                self.model.reactions.get_by_id(exch_id).bounds = (-1000, 0)
            except KeyError as e:
                print(
                    f"Error-- Cannot add media component. Exchange reaction not found: {exch_id}"
                )

    def remove_media_components(self, exch_to_unbound):
        """Remove media components

        Parameters
        ----------
        exch_to_unbound : list of string
            List of exchange reactions (associated with nutrient components) to
            fix bounds for.  Bounds fixed to (0, 0) to remove influence of
            exchange reaction.
        """
        for exch_id in exch_to_unbound:
            try:
                self.model.reactions.get_by_id(exch_id).bounds = (0, 0)
            except KeyError as e:
                print(
                    f"Error-- Cannot remove media component. Exchange reaction not found: {exch_id}"
                )

    def make_auxotroph(self, auxo_strain, supplement_media=True):
        """Make the model an auxotrophic strain

        Make the model an auxotrophic strain, optionally supplement media
        composition to allow for strain to grow. Available strains include:
        BY4741, BY4742.

        Parameters
        ----------
        auxo_strain : string
            Name of auxotrophic strain.
        supplement_media : bool
            If true, add exchange reactions to simulate adding supplements
            needed for chosen auxotrophic strain to grow.

        Raises
        ------
        Exception
            Error raised if supplied auxotrophic strain is not in list of valid
            strains.
        """
        if self.auxotrophy is not None:
            # This warning is necessary because running successive rounds of
            # this method will delete the union of genes to be deleted among all
            # auxotrophies applied to model so far.
            print(
                f"Warning-- strain has existing auxotrophy: {self.auxotrophy}",
                f"Invoking make_auxotroph may cause unintended effects.",
                f"For best practise, reset the model to its source file (Yeast8Model.reset())",
                f"before proceeding.",
                sep=os.linesep,
            )
        if auxo_strain in AUXOTROPH_DICT.keys():
            # Knock out genes to create auxotroph
            self.knock_out_list(AUXOTROPH_DICT[auxo_strain].genes_to_delete)
            # By default, supplements nutrients
            if supplement_media:
                self.add_media_components(AUXOTROPH_DICT[auxo_strain].exch_to_add)
            self.auxotrophy = auxo_strain
        else:
            raise Exception(
                f"Invalid string for auxotroph strain background: {auxo_strain}"
            )

    def optimize(self, model=None, timeout_time=60):
        # Unlike previous methods, takes a model object as input because I need
        # to re-use this in ablate().
        """Simulate model with FBA

        Parameters
        ----------
        model : cobra.Model object
            Model.  If this is not specified, uses the class's model attribute.
        timeout_time : int
            Time till timeout in seconds. Default 60.

        Returns
        -------
        cobra.Solution object
            Solution of model simulation.

        Examples
        --------
        y = Yeast8Model('./path/to/model.xml')
        y.optimize()       # simulates y.model
        y.optimize(m)      # simulates model m, defined elsewhere
        sol = y.optimize() # saves output somewhere
        """

        @timeout(timeout_time)
        def _optimize_internal(model):
            return model.optimize()

        try:
            if model is None:
                model = self.model
            solution = _optimize_internal(model)
            return solution
        except TimeoutError as e:
            print(f"Model optimisation timeout, {timeout_time} s")

    def ablate(self):
        """Ablate biomass components and get growth rates & doubling times

        Ablate components in biomass reaction (i.e. macromolecules like lipids,
        proteins, carbohydrates, DNA, RNA, and also cofactors & ions) that have
        pseudoreactions associated with the 'Growth' subsystem, in turn. This
        means changing the stoichiometric matrix. In each round, the model is
        simulated and the flux is recorded. Doubling time is computed, taking
        into account the mass fraction of each biomass component -- i.e. if the
        component is a smaller fraction of the cell, it takes less time.

        Returns
        -------
        pandas.DataFrame object
            Results of ablation study.  Columns: 'priority component' (biomass
            component being prioritised), 'ablated_flux' (flux of ablated
            biomass reaction), 'ablated_est_time' (estimated doubling time based
            on flux), 'proportional_est_time' (estimated biomass synthesis time,
            proportional to mass fraction).  Rows: 'original' (un-ablated
            biomass), other rows indicate biomass component.
        """
        # Copy model -- needed to restore the un-ablated model to work with
        # in successive loops
        model_working = self.model.copy()

        print("Biomass component ablation...")

        # UN-ABLATED
        print("Original")
        fba_solution = self.optimize(model_working)
        original_flux = fba_solution.fluxes[self.growth_id]
        original_est_time = np.log(2) / original_flux
        # ABLATED
        # Set up lists
        all_metabolite_ids = [
            biomass_component.metabolite_id
            for biomass_component in self.biomass_component_list
        ]
        all_pseudoreaction_ids = [
            (biomass_component.metabolite_label, biomass_component.pseudoreaction)
            for biomass_component in self.biomass_component_list
        ]
        all_pseudoreaction_ids.append(("biomass", BIOMASS_ID))
        all_pseudoreaction_ids.append(("objective", self.growth_id))
        # Loop
        for biomass_component in self.biomass_component_list:
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
            model_working.reactions.get_by_id(self.biomass_id).subtract_metabolites(
                to_ablate_dict
            )
            # optimise model
            fba_solution = model_working.optimize()
            # store outputs
            biomass_component.ablated_flux = fba_solution.fluxes[self.growth_id]
            biomass_component.get_est_time()

        # construct output dataframe
        d = {
            "priority_component": ["original"]
            + [
                biomass_component.metabolite_label
                for biomass_component in self.biomass_component_list
            ],
            # Flux through ablated biomass reactions
            "ablated_flux": [original_flux]
            + [
                biomass_component.ablated_flux
                for biomass_component in self.biomass_component_list
            ],
            # Estimated doubling time, taking into account the ablated content
            # of the virtual cell's biomass
            "ablated_est_time": [original_est_time]
            + [
                biomass_component.est_time
                for biomass_component in self.biomass_component_list
            ],
            # Estimated time for each biomass component, assuming that it is
            # proportional to mass fraction
            "proportional_est_time": [original_est_time]
            + [
                (biomass_component.molecular_mass / MW_BIOMASS)
                * (np.log(2) / original_flux)
                for biomass_component in self.biomass_component_list
            ],
        }
        print("Ablation done.")
        return pd.DataFrame(data=d)

    # Takes dataframe output from ablation function as input
    def ablation_barplot(self, ax, ablation_result=None):
        """Draws bar plot showing synthesis times from ablation study

        Parameters
        ----------
        ax : matplotlib.pyplot.Axes object
            Axes to draw bar plot on.
        ablation_result : pandas.DataFrame object
            Results of ablation study.  Columns: 'priority component' (biomass
            component being prioritised), 'ablated_flux' (flux of ablated
            biomass reaction), 'ablated_est_time' (estimated doubling time based
            on flux), 'proportional_est_time' (estimated biomass synthesis time,
            proportional to mass fraction).  Rows: 'original' (un-ablated
            biomass), other rows indicate biomass component.

        Examples
        --------
        # Initialise model
        y = Yeast8Model('./path/to/model.xml')

        # Ablate
        y.ablation_result = y.ablate()

        # Draw bar plot
        fig, ax = plt.subplots()
        y.ablation_barplot(ax)
        plt.show()
        """
        # Default argument
        if ablation_result is None:
            ablation_result = self.ablation_result
        # Check if ablation already done.  If not, then the value should still
        # be None despite above if statement.  If ablation already done, draw.
        if ablation_result is not None:
            # Get values for each bar plot series from ablation result DataFrame
            values_ablated, values_proportion = _bar_vals_from_ablation_df(
                ablation_result
            )

            # Draw bar plot
            # https://www.python-graph-gallery.com/8-add-confidence-interval-on-barplot
            barwidth = 0.4

            bar_labels = ablation_result.priority_component.to_list()
            bar_labels[0] = "all biomass"

            x_ablated = np.arange(len(bar_labels))
            x_proportion = [x + barwidth for x in x_ablated]

            ax.bar(
                x=x_ablated,
                height=values_ablated,
                width=barwidth,
                color="#3714b0",
                label="From ablating components\n in the biomass reaction",
            )
            ax.bar(
                x=x_proportion,
                height=values_proportion,
                width=barwidth,
                color="#cb0077",
                label="From mass fractions\n of each biomass component",
            )
            ax.set_xticks(
                ticks=[x + barwidth / 2 for x in range(len(x_ablated))],
                labels=bar_labels,
                rotation=45,
            )
            ax.set_xlabel("Biomass component")
            ax.set_ylabel("Estimated synthesis time (hours)")
            ax.legend()
        else:
            print(
                "No ablation result. Please run ablate() to generate results before plotting."
            )


def compare_fluxes(ymodel1, ymodel2):
    """Compare fluxes between two models

    Compare fluxes between two models.  If fluxes aren't already computed and
    stored, run simulations first.

    Parameters
    ----------
    ymodel1 : Yeast8Model object
        Model 1.
    ymodel2 : Yeast8Model object
        Model 2.

    Returns
    -------
    pandas.DataFrame object
        Fluxes, sorted by magnitude, large changes on top.  Indices show
        reaction ids, values show the fluxes (with signs, positive or negative)

    Examples
    --------
    # Initialise model-handling objects
    y = Yeast8Model("./models/ecYeastGEM_batch.xml")
    z = Yeast8Model("./models/ecYeastGEM_batch.xml")

    # Make z different
    z.make_auxotroph("BY4741")

    # Compare fluxes
    dfs = compare_fluxes(y, z)
    """
    # Check if fluxes have already been computed.
    # If not, compute automatically.
    for idx, ymodel in enumerate([ymodel1, ymodel2]):
        if ymodel.solution is None:
            print(f"Model {idx+1} has no stored solution, optimising...")
            ymodel.solution = ymodel.optimize()

    diff_fluxes = ymodel2.solution.fluxes - ymodel1.solution.fluxes
    nonzero_idx = diff_fluxes.to_numpy().nonzero()[0]
    diff_fluxes_nonzero = diff_fluxes[nonzero_idx]
    # Sort by absolute flux value, large changes on top.
    diff_fluxes_sorted = diff_fluxes_nonzero[
        diff_fluxes_nonzero.abs().sort_values(ascending=False).index
    ]

    return diff_fluxes_sorted


def compare_ablation_times(ablation_result1, ablation_result2, ax):
    # Compute fold changes
    values_ablated1, values_proportion1 = _bar_vals_from_ablation_df(ablation_result1)
    values_ablated2, values_proportion2 = _bar_vals_from_ablation_df(ablation_result2)

    foldchange_ablated = np.array(values_ablated2) / np.array(values_ablated1)
    foldchange_proportion = np.array(values_proportion2) / np.array(values_proportion1)

    # Draw bar plot
    barwidth = 0.4
    bar_labels = ablation_result1.priority_component.to_list()
    bar_labels[0] = "all biomass"
    x_ablated = np.arange(len(bar_labels))
    x_proportion = [x + barwidth for x in x_ablated]
    ax.bar(
        x=x_ablated,
        height=np.log2(foldchange_ablated),
        width=barwidth,
        color="#3714b0",
        label="From ablating components\n in the biomass reaction",
    )
    ax.bar(
        x=x_proportion,
        height=np.log2(foldchange_proportion),
        width=barwidth,
        color="#cb0077",
        label="From mass fractions\n of each biomass component",
    )
    ax.set_xticks(
        ticks=[x + barwidth / 2 for x in range(len(x_ablated))],
        labels=bar_labels,
        rotation=45,
    )
    ax.set_xlabel("Biomass component")
    ax.set_ylabel("log fold change of estimated time")
    ax.legend()


def _bar_vals_from_ablation_df(ablation_result):
    # sum of times
    sum_of_times = ablation_result.loc[
        ablation_result.priority_component != "original",
        ablation_result.columns == "ablated_est_time",
    ].sum()
    # get element
    sum_of_times = sum_of_times[0]
    # ablated
    values_ablated = ablation_result.ablated_est_time.to_list()
    values_ablated[0] = sum_of_times
    # proportion
    values_proportion = ablation_result.proportional_est_time.to_list()

    return values_ablated, values_proportion
